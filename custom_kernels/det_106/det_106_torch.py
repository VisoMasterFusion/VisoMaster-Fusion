"""
FP16 PyTorch reimplementation of model_assets/2d106det.onnx — the MobileNetV1-style
106-point face landmark detector.

Architecture summary (verified from ONNX node inspection)
----------------------------------------------------------
Input  : (N, 3, 192, 192)  float32  in [0, 255]
Pre-   : (x - 127.5) * 0.0078125  →  [-1, 1]  (baked into forward)
Stem   : Conv(3→16, 3×3, s=2, pad=1) + PReLU          → (N, 16,96,96)
Block 1: DW(16,s=1) + PReLU + PW(16→32)  + PReLU      → (N, 32,96,96)
Block 2: DW(32,s=2) + PReLU + PW(32→64)  + PReLU      → (N, 64,48,48)
Block 3: DW(64,s=1) + PReLU + PW(64→64)  + PReLU      → (N, 64,48,48)
Block 4: DW(64,s=2) + PReLU + PW(64→128) + PReLU      → (N,128,24,24)
Block 5: DW(128,s=1)+ PReLU + PW(128→128)+ PReLU      → (N,128,24,24)
Block 6: DW(128,s=2)+ PReLU + PW(128→256)+ PReLU      → (N,256,12,12)
×5     : DW(256,s=1)+ PReLU + PW(256→256)+ PReLU      → (N,256,12,12)
Block 7: DW(256,s=2)+ PReLU + PW(256→512)+ PReLU      → (N,512, 6, 6)
Block 8: DW(512,s=1)+ PReLU + PW(512→512)+ PReLU      → (N,512, 6, 6)
Final  : Conv(512→64, 3×3, s=2, pad=1) + PReLU        → (N, 64, 3, 3)
FC     : Flatten → (N,576) → Linear(576,212)           → (N,212)
Output : (N, 212)  =  106 × (x, y)  in model space
         post-proc: (pred + 1) * 96.0  →  pixel coords in 192×192 crop
"""

from __future__ import annotations

import pathlib
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------


class Det106Torch(nn.Module):
    """MobileNetV1-style 106-point face landmark detector."""

    NUM_REPEATS = 5  # DW+PW blocks at 12×12 feature map (256-ch)

    def __init__(self, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype

        # Pre-processing constants (kept as buffers → move to device with .cuda())
        self.register_buffer("_sub", torch.tensor(127.5, dtype=torch.float32))
        self.register_buffer("_mul", torch.tensor(0.0078125, dtype=torch.float32))

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem_conv = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True)
        self.stem_prelu = nn.PReLU(16)

        # ── Block 1: DW(16,s=1) + PW(16→32) ─────────────────────────────────
        self.dw16 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16, bias=True)
        self.prelu_dw16 = nn.PReLU(16)
        self.pw16 = nn.Conv2d(16, 32, 1, bias=True)
        self.prelu_pw16 = nn.PReLU(32)

        # ── Block 2: DW(32,s=2) + PW(32→64) ─────────────────────────────────
        self.dw32 = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=True)
        self.prelu_dw32 = nn.PReLU(32)
        self.pw32 = nn.Conv2d(32, 64, 1, bias=True)
        self.prelu_pw32 = nn.PReLU(64)

        # ── Block 3: DW(64,s=1) + PW(64→64) ─────────────────────────────────
        self.dw64a = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=True)
        self.prelu_dw64a = nn.PReLU(64)
        self.pw64a = nn.Conv2d(64, 64, 1, bias=True)
        self.prelu_pw64a = nn.PReLU(64)

        # ── Block 4: DW(64,s=2) + PW(64→128) ────────────────────────────────
        self.dw64b = nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=True)
        self.prelu_dw64b = nn.PReLU(64)
        self.pw64b = nn.Conv2d(64, 128, 1, bias=True)
        self.prelu_pw64b = nn.PReLU(128)

        # ── Block 5: DW(128,s=1) + PW(128→128) ──────────────────────────────
        self.dw128a = nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=True)
        self.prelu_dw128a = nn.PReLU(128)
        self.pw128a = nn.Conv2d(128, 128, 1, bias=True)
        self.prelu_pw128a = nn.PReLU(128)

        # ── Block 6: DW(128,s=2) + PW(128→256) ──────────────────────────────
        self.dw128b = nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=True)
        self.prelu_dw128b = nn.PReLU(128)
        self.pw128b = nn.Conv2d(128, 256, 1, bias=True)
        self.prelu_pw128b = nn.PReLU(256)

        # ── 5× DW+PW repeats at 12×12, 256 channels ──────────────────────────
        self.rep_dw = nn.ModuleList(
            [
                nn.Conv2d(256, 256, 3, stride=1, padding=1, groups=256, bias=True)
                for _ in range(self.NUM_REPEATS)
            ]
        )
        self.rep_pdw = nn.ModuleList([nn.PReLU(256) for _ in range(self.NUM_REPEATS)])
        self.rep_pw = nn.ModuleList(
            [nn.Conv2d(256, 256, 1, bias=True) for _ in range(self.NUM_REPEATS)]
        )
        self.rep_ppw = nn.ModuleList([nn.PReLU(256) for _ in range(self.NUM_REPEATS)])

        # ── Block 7: DW(256,s=2) + PW(256→512) ──────────────────────────────
        self.dw256_s2 = nn.Conv2d(
            256, 256, 3, stride=2, padding=1, groups=256, bias=True
        )
        self.prelu_dw256_s2 = nn.PReLU(256)
        self.pw256 = nn.Conv2d(256, 512, 1, bias=True)
        self.prelu_pw256 = nn.PReLU(512)

        # ── Block 8: DW(512,s=1) + PW(512→512) ──────────────────────────────
        self.dw512 = nn.Conv2d(512, 512, 3, stride=1, padding=1, groups=512, bias=True)
        self.prelu_dw512 = nn.PReLU(512)
        self.pw512 = nn.Conv2d(512, 512, 1, bias=True)
        self.prelu_pw512 = nn.PReLU(512)

        # ── Final regular conv 512→64, stride=2, 6→3 ─────────────────────────
        self.conv_final = nn.Conv2d(512, 64, 3, stride=2, padding=1, bias=True)
        self.prelu_final = nn.PReLU(64)

        # ── Fully-connected head ──────────────────────────────────────────────
        self.fc = nn.Linear(576, 212, bias=True)  # 64×3×3 = 576

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 3, 192, 192)  float32  in [0, 255]
        returns : (N, 212)  float32 — raw network outputs in model space
        """
        # Pre-processing: [0,255] → [-1,1]
        h = x.to(self.compute_dtype)
        h = (h - self._sub.to(self.compute_dtype)) * self._mul.to(self.compute_dtype)

        # Stem
        h = self.stem_prelu(self.stem_conv(h))

        # Block 1
        h = self.prelu_dw16(self.dw16(h))
        h = self.prelu_pw16(self.pw16(h))

        # Block 2
        h = self.prelu_dw32(self.dw32(h))
        h = self.prelu_pw32(self.pw32(h))

        # Block 3
        h = self.prelu_dw64a(self.dw64a(h))
        h = self.prelu_pw64a(self.pw64a(h))

        # Block 4
        h = self.prelu_dw64b(self.dw64b(h))
        h = self.prelu_pw64b(self.pw64b(h))

        # Block 5
        h = self.prelu_dw128a(self.dw128a(h))
        h = self.prelu_pw128a(self.pw128a(h))

        # Block 6
        h = self.prelu_dw128b(self.dw128b(h))
        h = self.prelu_pw128b(self.pw128b(h))

        # 5× repeat at 12×12
        for dw, pdw, pw, ppw in zip(
            self.rep_dw, self.rep_pdw, self.rep_pw, self.rep_ppw
        ):
            h = ppw(pw(pdw(dw(h))))

        # Block 7
        h = self.prelu_dw256_s2(self.dw256_s2(h))
        h = self.prelu_pw256(self.pw256(h))

        # Block 8
        h = self.prelu_dw512(self.dw512(h))
        h = self.prelu_pw512(self.pw512(h))

        # Final conv → 3×3
        h = self.prelu_final(self.conv_final(h))

        # FC — fc.weight is compute_dtype; keep h in compute_dtype for the matmul,
        # then return float32 for downstream compatibility.
        h = h.reshape(h.shape[0], -1)
        return self.fc(h).float()

    # -------------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | pathlib.Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> "Det106Torch":
        """
        Construct and initialise a Det106Torch from the 2d106det.onnx file.

        Weight-loading strategy
        -----------------------
        All 28 Conv layers and 28 PReLU layers are visited in ONNX topological
        order and assigned sequentially to the corresponding PyTorch parameters.

        The Gemm (fc) weights are loaded by inspecting the final Gemm node's
        input names directly.
        """
        import onnx
        from onnx import numpy_helper

        proto = onnx.load(str(onnx_path))
        inits: dict = {
            i.name: numpy_helper.to_array(i) for i in proto.graph.initializer
        }

        # ── Collect positional weights in topological order ───────────────────
        conv_params: list[tuple] = []  # (weight_arr, bias_arr | None)
        prelu_slopes: list = []

        gemm_node = None

        for node in proto.graph.node:
            if node.op_type == "Conv":
                w = inits[node.input[1]]
                b = inits[node.input[2]] if len(node.input) > 2 else None
                conv_params.append((w, b))
            elif node.op_type == "PRelu":
                prelu_slopes.append(inits[node.input[1]])
            elif node.op_type == "Gemm":
                gemm_node = node

        assert len(conv_params) == 28, f"Expected 28 Conv nodes, got {len(conv_params)}"
        assert len(prelu_slopes) == 28, (
            f"Expected 28 PReLU nodes, got {len(prelu_slopes)}"
        )

        # ── Build model and assign weights ────────────────────────────────────
        m = cls(compute_dtype=compute_dtype)

        ci = 0  # conv index
        pi = 0  # prelu index

        def _c(layer: nn.Conv2d) -> None:
            nonlocal ci
            w, b = conv_params[ci]
            ci += 1
            layer.weight.data = torch.tensor(w, dtype=compute_dtype)
            if b is not None and layer.bias is not None:
                layer.bias.data = torch.tensor(b, dtype=compute_dtype)

        def _p(layer: nn.PReLU) -> None:
            nonlocal pi
            slope = prelu_slopes[pi]
            pi += 1
            # ONNX slope shape (C,1,1) → PyTorch PReLU weight shape (C,)
            layer.weight.data = torch.tensor(slope.reshape(-1), dtype=compute_dtype)

        # Stem
        _c(m.stem_conv)
        _p(m.stem_prelu)

        # Block 1
        _c(m.dw16)
        _p(m.prelu_dw16)
        _c(m.pw16)
        _p(m.prelu_pw16)
        # Block 2
        _c(m.dw32)
        _p(m.prelu_dw32)
        _c(m.pw32)
        _p(m.prelu_pw32)
        # Block 3
        _c(m.dw64a)
        _p(m.prelu_dw64a)
        _c(m.pw64a)
        _p(m.prelu_pw64a)
        # Block 4
        _c(m.dw64b)
        _p(m.prelu_dw64b)
        _c(m.pw64b)
        _p(m.prelu_pw64b)
        # Block 5
        _c(m.dw128a)
        _p(m.prelu_dw128a)
        _c(m.pw128a)
        _p(m.prelu_pw128a)
        # Block 6
        _c(m.dw128b)
        _p(m.prelu_dw128b)
        _c(m.pw128b)
        _p(m.prelu_pw128b)

        # 5× repeats at 12×12
        for i in range(cls.NUM_REPEATS):
            _c(m.rep_dw[i])
            _p(m.rep_pdw[i])
            _c(m.rep_pw[i])
            _p(m.rep_ppw[i])

        # Block 7
        _c(m.dw256_s2)
        _p(m.prelu_dw256_s2)
        _c(m.pw256)
        _p(m.prelu_pw256)

        # Block 8
        _c(m.dw512)
        _p(m.prelu_dw512)
        _c(m.pw512)
        _p(m.prelu_pw512)

        # Final conv
        _c(m.conv_final)
        _p(m.prelu_final)

        assert ci == 28, f"Conv weight cursor mismatch: {ci}"
        assert pi == 28, f"PReLU slope cursor mismatch: {pi}"

        # ── Gemm (fc) — load by name from Gemm node inputs ───────────────────
        assert gemm_node is not None, "Gemm node not found"
        fc_w = inits[
            gemm_node.input[1]
        ]  # (212, 576)  transB=1 matches nn.Linear layout
        fc_b = inits[gemm_node.input[2]]  # (212,)
        m.fc.weight.data = torch.tensor(fc_w, dtype=compute_dtype)
        m.fc.bias.data = torch.tensor(fc_b, dtype=compute_dtype)

        m._visomaster_onnx_path = str(onnx_path)
        return m


# ---------------------------------------------------------------------------


def build_cuda_graph_runner(
    model: Det106Torch,
    input_shape: tuple = (1, 3, 192, 192),
    torch_compile: bool = False,
):
    """
    Wrap a Det106Torch in a CUDA graph for zero-CPU-overhead repeated inference.

    Returns a callable  runner(x) → tensor (N,212) float32
    where x is (N,3,192,192) float32 on CUDA.

    Notes
    -----
    - Input shape is fixed at capture time; always pass (1,3,192,192).
    - The runner clones the output tensor each call so callers own their copy.
    - torch_compile: if True, apply torch.compile (mode='default') before CUDA
      graph capture.  Triggers Triton JIT on first call (~30 s); compiled
      kernels are then captured in the graph.
    """
    if torch_compile:
        try:
            from custom_kernels.compile_utils import apply_torch_compile
            device = next(model.parameters()).device
            example_inp = torch.zeros((1, 3, 192, 192), dtype=torch.float32, device=device)
            compiled = apply_torch_compile(model, example_inp)
            print("[det_106] torch.compile warmup done.")
            return compiled  # CUDA graph on top of torch.compile fails on Windows
        except Exception as e:
            print(f"[det_106] torch.compile failed ({e!s:.120}), falling back to CUDA graph.")
    device = next(model.parameters()).device
    assert device.type == "cuda", "Model must be on a CUDA device"

    static_in = torch.zeros(input_shape, dtype=torch.float32, device=device)

    # Warm up outside the graph to initialise cuDNN auto-tuning etc.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            with torch.no_grad():
                _ = model(static_in)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup_stream, capture_error_mode="relaxed"):
        with torch.no_grad():
            static_out = model(static_in)  # (N, 212)

    def runner(x: torch.Tensor) -> torch.Tensor:
        static_in.copy_(x)
        graph.replay()
        return static_out.clone()

    return runner
