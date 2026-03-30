"""
Quick numerical accuracy test: compare GPENTorch FP32 vs ORT CUDA for GPEN-BFR-256.
Run:
    .venv/Scripts/python custom_kernels/gpen_bfr/test_accuracy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL_PATH = str(ROOT / "model_assets" / "GPEN-BFR-256.onnx")
if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

# ── ORT reference ────────────────────────────────────────────────────────────
import onnxruntime as ort  # noqa: E402

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(
    MODEL_PATH,
    so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"device_id": "0"}, {}],
)
inp_name = sess.get_inputs()[0].name
rng = np.random.default_rng(42)
inp_np = rng.random((1, 3, 256, 256)).astype("float32")
ort_out = sess.run(None, {inp_name: inp_np})[0]

# ── GPENTorch FP32 ───────────────────────────────────────────────────────────
from custom_kernels.gpen_bfr.gpen_torch import GPENTorch  # noqa: E402

inp_t = torch.from_numpy(inp_np).cuda()

for label, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
    model = GPENTorch.from_onnx(MODEL_PATH, compute_dtype=dtype).cuda().eval()
    with torch.no_grad():
        pt_out = model(inp_t).cpu().numpy()
    diff = np.abs(pt_out - ort_out)
    rel = diff / (np.abs(ort_out) + 1e-6)
    print(f"\n=== {label} ===")
    print(f"  ORT range: [{ort_out.min():.6f}, {ort_out.max():.6f}]")
    print(f"  PT  range: [{pt_out.min():.6f}, {pt_out.max():.6f}]")
    print(
        f"  max|diff|={diff.max():.6f}  mean|diff|={diff.mean():.6f}  mean_rel={rel.mean():.4f}"
    )
