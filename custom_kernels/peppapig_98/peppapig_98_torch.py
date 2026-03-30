"""
FP16 PyTorch reimplementation of
model_assets/peppapig_teacher_Nx3x256x256.onnx — the HRNet-W18-based
98-point WFLW face landmark detector (PEPPANet teacher model).

Architecture
------------
Backbone : HRNet-W18
  Stem     : 2 × Conv(stride=2) + 4 Bottleneck blocks → (N,256,64,64)
  Stage 2  : 2 branches [18,36 ch]  × 1 module  × 4 BasicBlocks each
  Stage 3  : 3 branches [18,36,72]  × 4 modules × 4 BasicBlocks each
  Stage 4  : 4 branches [18,36,72,144] × 3 modules × 4 BasicBlocks each
Decoder  : ASPP (4 heads, GlobalAvgPool, BN) → bilinear up → DW-sep blocks
Head     : Conv(128→294, 1×1) → decode into (N,98,3) landmarks
Output   : (N, 98, 3)  — (x, y, score) all in [0,1] normalised to 256×256

Decode head
-----------
  hm        : (N,294,64,64)  294 = 98×3 groups: x_hm, y_hm, score_hm
  x_flat    : reshape(N,98,4096)
  flat_idx  : argmax(x_flat, dim=2) — position of peak x activation
  col_f     : float(flat_idx % 64) + GatherElements(y_hm, flat_idx)
  row_f     : float(flat_idx // 64) + GatherElements(score_hm, flat_idx)
  (x,y)     : (col_f, row_f) / 64.0   → [0,1]
  score     : max(x_flat, dim=2)

Weight loading
--------------
All 325 Conv layers and the 1 BatchNorm layer are loaded from the ONNX
during `PeppaPig98Torch.__init__()`. Conv weights use **positional** loading
(ONNX topological order = PyTorch forward order). The 4 ASPP named
Conv weights and the BN/attention weights are loaded by name.
"""

from __future__ import annotations

import pathlib
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ONNX-dtype map used in Cast nodes
_ONNX_DTYPE = {1: torch.float32, 6: torch.int32, 7: torch.int64, 10: torch.float16}


class PeppaPig98Torch(nn.Module):
    """
    ONNX-interpreter-based PyTorch reimplementation of the PEPPANet
    (peppapig_teacher_Nx3x256x256.onnx) 98-point face landmark detector.

    Rather than hand-coding the 839-node HRNet graph, the model parses
    the ONNX at construction time and builds:
      - ``self.convs``  : nn.ModuleList of all 325 Conv2d layers (in ONNX
                         topological order) with weights pre-loaded.
      - ``self.bn``     : the single BatchNorm2d layer in the ASPP.
      - ``self._plan``  : a compact execution plan (list of dicts) that
                         ``forward()`` walks to execute every node.

    Because the execution plan is a fixed Python list, all CUDA kernel
    launches happen in the same order on every call → CUDA graph compatible.
    """

    def __init__(
        self,
        onnx_path: str | pathlib.Path,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self._visomaster_onnx_path = str(onnx_path)
        self._build(str(onnx_path))

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | pathlib.Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "PeppaPig98Torch":
        """Thin shim so the subprocess compile worker can reconstruct this model."""
        return cls(onnx_path, compute_dtype=compute_dtype)

    # -------------------------------------------------------------------------

    def _build(self, onnx_path: str) -> None:
        import onnx
        from onnx import numpy_helper
        import numpy as np

        proto = onnx.load(onnx_path)
        g = proto.graph

        inits: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in g.initializer
        }

        # ── Build Conv2d list in ONNX topological order ───────────────────────
        convs: list[nn.Conv2d] = []
        conv_node_to_idx: dict[int, int] = {}  # ONNX node index → convs[]

        bn_layer: Optional[nn.BatchNorm2d] = None
        _bn_node_idx: int = -1

        def _get_attr(node, name, default=None):
            for a in node.attribute:
                if a.name == name:
                    if a.ints:
                        return list(a.ints)
                    if a.floats:
                        return list(a.floats)
                    if a.s:
                        return a.s.decode()
                    if a.HasField("i"):
                        return a.i
                    if a.HasField("f"):
                        return a.f
            return default

        for ni, node in enumerate(g.node):
            if node.op_type == "Conv":
                w_name = node.input[1]
                w = inits.get(w_name)
                if w is None:
                    continue
                has_bias = (
                    len(node.input) > 2
                    and bool(node.input[2])
                    and node.input[2] in inits
                )
                strides = _get_attr(node, "strides", [1, 1])
                pads = _get_attr(node, "pads", [0, 0, 0, 0])
                groups = _get_attr(node, "group", 1)
                ks = _get_attr(node, "kernel_shape", [1, 1])
                dilations = _get_attr(node, "dilations", [1, 1])

                out_c = w.shape[0]
                in_c = w.shape[1] * (groups if isinstance(groups, int) else groups[0])
                g_val = groups if isinstance(groups, int) else groups[0]
                d_val = dilations[0] if isinstance(dilations, list) else dilations
                s_val = (
                    (strides[0], strides[1]) if isinstance(strides, list) else strides
                )
                p_val = (pads[0], pads[1])

                conv = nn.Conv2d(
                    in_c,
                    out_c,
                    (ks[0], ks[1]),
                    stride=s_val,
                    padding=p_val,
                    dilation=d_val,
                    groups=g_val,
                    bias=has_bias,
                )
                conv.weight.data = torch.tensor(w, dtype=self.compute_dtype)
                if has_bias:
                    b = inits[node.input[2]]
                    conv.bias.data = torch.tensor(b, dtype=self.compute_dtype)

                conv_node_to_idx[ni] = len(convs)
                convs.append(conv)

            elif node.op_type == "BatchNormalization":
                w = inits[node.input[1]]
                b = inits[node.input[2]]
                mn = inits[node.input[3]]
                vr = inits[node.input[4]]
                C = w.shape[0]
                bn_layer = nn.BatchNorm2d(C, eps=1e-5)
                bn_layer.weight.data.copy_(torch.tensor(w, dtype=torch.float32))
                bn_layer.bias.data.copy_(torch.tensor(b, dtype=torch.float32))
                bn_layer.running_mean.copy_(torch.tensor(mn, dtype=torch.float32))
                bn_layer.running_var.copy_(torch.tensor(vr, dtype=torch.float32))
                _bn_node_idx = ni

        self.convs = nn.ModuleList(convs)
        if bn_layer is not None:
            self.bn = bn_layer
        else:
            self.bn = nn.Identity()

        # ── Build execution plan ─────────────────────────────────────────────
        plan = []
        for ni, node in enumerate(g.node):
            op = node.op_type
            inp = list(node.input)
            out = list(node.output)

            if op == "Conv":
                ci = conv_node_to_idx.get(ni)
                if ci is None:
                    continue
                plan.append({"op": "Conv", "ci": ci, "in": inp[0], "out": out[0]})

            elif op == "Relu":
                plan.append({"op": "Relu", "in": inp[0], "out": out[0]})

            elif op == "Add":
                plan.append({"op": "Add", "in0": inp[0], "in1": inp[1], "out": out[0]})

            elif op == "BatchNormalization":
                plan.append({"op": "BN", "in": inp[0], "out": out[0]})

            elif op == "GlobalAveragePool":
                plan.append({"op": "GAP", "in": inp[0], "out": out[0]})

            elif op == "Sigmoid":
                plan.append({"op": "Sigmoid", "in": inp[0], "out": out[0]})

            elif op == "Mul":
                plan.append({"op": "Mul", "in0": inp[0], "in1": inp[1], "out": out[0]})

            elif op == "Concat":
                axis = _get_attr(node, "axis", 1)
                plan.append({"op": "Concat", "ins": inp, "axis": axis, "out": out[0]})

            elif op == "Resize":
                # All resize nodes use static float scale tensors in inits
                # Find the scales tensor (last non-empty inp that is in inits)
                scales = None
                for nm in inp[1:]:
                    if nm and nm in inits:
                        arr = inits[nm]
                        if arr.shape == (4,) and arr.dtype in [
                            __import__("numpy").float32,
                            __import__("numpy").float64,
                        ]:
                            scales = (float(arr[2]), float(arr[3]))
                mode = _get_attr(node, "mode", "nearest")
                ctm = _get_attr(node, "coordinate_transformation_mode", "asymmetric")
                align = ctm == "half_pixel"
                interp_mode = "nearest" if mode == "nearest" else "bilinear"
                plan.append(
                    {
                        "op": "Resize",
                        "in": inp[0],
                        "scale": scales,
                        "mode": interp_mode,
                        "align": align,
                        "out": out[0],
                    }
                )

            elif op == "Slice":
                # All slices here are along channel axis=1
                # starts/ends/axes are constant initializers
                starts = int(inits[inp[1]].flat[0])
                ends = int(inits[inp[2]].flat[0])
                plan.append(
                    {
                        "op": "Slice",
                        "in": inp[0],
                        "start": starts,
                        "end": ends,
                        "out": out[0],
                    }
                )

            elif op == "Reshape":
                shape = inits[inp[1]].tolist()  # e.g. [-1, 98, 4096]
                plan.append(
                    {"op": "Reshape", "in": inp[0], "shape": shape, "out": out[0]}
                )

            elif op == "ReduceMax":
                axes = _get_attr(node, "axes", [2])
                plan.append(
                    {"op": "ReduceMax", "in": inp[0], "axis": axes[0], "out": out[0]}
                )

            elif op == "ArgMax":
                axis = _get_attr(node, "axis", 2)
                plan.append({"op": "ArgMax", "in": inp[0], "axis": axis, "out": out[0]})

            elif op == "GatherElements":
                axis = _get_attr(node, "axis", 2)
                plan.append(
                    {
                        "op": "GatherElements",
                        "data": inp[0],
                        "idx": inp[1],
                        "axis": axis,
                        "out": out[0],
                    }
                )

            elif op == "Mod":
                divisor = int(inits[inp[1]].flat[0])
                plan.append(
                    {"op": "Mod", "in": inp[0], "divisor": divisor, "out": out[0]}
                )

            elif op == "Div":
                # divisor may be a constant initializer (int or float)
                div_name = inp[1]
                div_val = float(inits[div_name].flat[0]) if div_name in inits else None
                plan.append(
                    {
                        "op": "Div",
                        "in": inp[0],
                        "div_name": div_name,
                        "div_val": div_val,
                        "out": out[0],
                    }
                )

            elif op == "Cast":
                to = _get_attr(node, "to", 1)
                plan.append({"op": "Cast", "in": inp[0], "to": to, "out": out[0]})

        self._plan = plan
        self._output_name = g.output[0].name
        self._input_name = g.input[0].name

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 3, 256, 256)  float32  in [0, 1]
        returns : (N, 98, 3)  float32  — (x, y, score) in [0,1]
        """
        T: dict[str, torch.Tensor] = {self._input_name: x.to(self.compute_dtype)}

        for step in self._plan:
            op = step["op"]

            if op == "Conv":
                h = T[step["in"]]
                T[step["out"]] = self.convs[step["ci"]](h.to(self.compute_dtype))

            elif op == "Relu":
                T[step["out"]] = F.relu(T[step["in"]], inplace=False)

            elif op == "Add":
                T[step["out"]] = T[step["in0"]] + T[step["in1"]]

            elif op == "BN":
                T[step["out"]] = self.bn(T[step["in"]].float()).to(self.compute_dtype)

            elif op == "GAP":
                T[step["out"]] = F.adaptive_avg_pool2d(T[step["in"]], 1)

            elif op == "Sigmoid":
                T[step["out"]] = torch.sigmoid(T[step["in"]])

            elif op == "Mul":
                T[step["out"]] = T[step["in0"]] * T[step["in1"]]

            elif op == "Concat":
                parts = [T[n] for n in step["ins"] if n and n in T]
                T[step["out"]] = torch.cat(parts, dim=step["axis"])

            elif op == "Resize":
                src = T[step["in"]]
                sf = step["scale"]
                if step["mode"] == "nearest":
                    T[step["out"]] = F.interpolate(src, scale_factor=sf, mode="nearest")
                else:
                    T[step["out"]] = F.interpolate(
                        src.float(),
                        scale_factor=sf,
                        mode="bilinear",
                        align_corners=False,
                    ).to(self.compute_dtype)

            elif op == "Slice":
                T[step["out"]] = T[step["in"]][:, step["start"] : step["end"]]

            elif op == "Reshape":
                T[step["out"]] = T[step["in"]].reshape(step["shape"])

            elif op == "ReduceMax":
                T[step["out"]] = (
                    T[step["in"]].float().max(dim=step["axis"], keepdim=True).values
                )

            elif op == "ArgMax":
                T[step["out"]] = (
                    T[step["in"]].float().argmax(dim=step["axis"], keepdim=True)
                )

            elif op == "GatherElements":
                data = T[step["data"]].float()
                idx = T[step["idx"]].long()
                T[step["out"]] = data.gather(step["axis"], idx)

            elif op == "Mod":
                T[step["out"]] = T[step["in"]].long() % step["divisor"]

            elif op == "Div":
                src = T[step["in"]]
                dv = step["div_val"]
                T[step["out"]] = src / dv

            elif op == "Cast":
                dst_dtype = _ONNX_DTYPE.get(step["to"], torch.float32)
                T[step["out"]] = T[step["in"]].to(dst_dtype)

        return T[self._output_name].float()  # (N, 98, 3)


# ---------------------------------------------------------------------------


def build_cuda_graph_runner(
    model: PeppaPig98Torch,
    input_shape: tuple = (1, 3, 256, 256),
    torch_compile: bool = False,
):
    """
    Wrap a PeppaPig98Torch in a CUDA graph for zero-CPU-overhead inference.

    Returns a callable  runner(x) → tensor (N, 98, 3)  float32
    where x is (N, 3, 256, 256) float32 on CUDA in [0, 1].

    Notes
    -----
    - Input/output shapes are fixed at capture time — always pass (1,3,256,256).
    - The runner clones the output tensor each call so callers own their copy.
    - The execution plan (Python list walk) runs once during capture; only the
      CUDA kernels are replayed — no Python overhead on the hot path.
    - torch_compile: if True, apply torch.compile (mode='default') before CUDA
      graph capture.  Triggers Triton JIT on first call (~30 s); compiled
      kernels are then captured in the graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[peppapig_98] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[peppapig_98] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    device = next(model.parameters()).device
    assert device.type == "cuda", "Model must be on a CUDA device"

    static_in = torch.zeros(input_shape, dtype=torch.float32, device=device)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            with torch.no_grad():
                _ = model(static_in)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup_stream, capture_error_mode="relaxed"):
        with torch.no_grad():
            static_out = model(static_in)  # (N, 98, 3)

    def runner(x: torch.Tensor) -> torch.Tensor:
        static_in.copy_(x)
        graph.replay()
        return static_out.clone()

    return runner
