"""
Correctness test for Triton kernels in triton_ops.py.
Compares Triton outputs vs PyTorch reference for all three kernels.

Run:
    .venv/Scripts/python custom_kernels/test_triton_ops.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from custom_kernels.triton_ops import (  # noqa: E402
    TRITON_AVAILABLE,
    triton_demod,
    triton_adain,
    triton_fused_gpen_act,
    triton_fused_gfpgan_act,
)

if not TRITON_AVAILABLE:
    print("Triton not available — skipping tests.")
    sys.exit(0)

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

PASS = "[PASS]"
FAIL = "[FAIL]"


def allclose(a, b, atol=1e-2, rtol=1e-2, name=""):
    a = a.float()
    b = b.float()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}  max={max_diff:.5f}  mean={mean_diff:.5f}")
    return ok


all_ok = True

# ===========================================================================
# Test 1: triton_demod
# ===========================================================================
print("\n=== triton_demod ===")

for C_out, C_in, kH, kW in [(64, 64, 3, 3), (256, 256, 3, 3), (512, 256, 3, 3)]:
    w = torch.randn(C_out, C_in, kH, kW, dtype=dtype, device=device)
    style = torch.randn(C_in, device=device)  # float32

    # PyTorch reference
    w_f = w.float()
    s_f = style.float()
    ws = w_f * s_f[None, :, None, None]
    norm = (ws.pow(2).sum(dim=[1, 2, 3], keepdim=True) + 1e-8).rsqrt()
    ref = (ws * norm).to(dtype)

    out = triton_demod(w, style)
    ok = allclose(
        out, ref, atol=5e-3, rtol=5e-3, name=f"C_out={C_out} C_in={C_in} k={kH}"
    )
    all_ok = all_ok and ok

# ===========================================================================
# Test 2: triton_fused_gpen_act
# ===========================================================================
print("\n=== triton_fused_gpen_act ===")

for C_out, H, W in [(64, 64, 64), (256, 16, 16), (512, 8, 8)]:
    conv_out = torch.randn(1, C_out, H, W, dtype=dtype, device=device)
    noise_term = torch.randn(1, C_out, H, W, dtype=dtype, device=device)
    act_b = torch.randn(1, 2 * C_out, 1, 1, dtype=dtype, device=device)
    neg_slope = 0.2
    scale = 2.0**0.5

    # PyTorch reference
    cat_out = torch.cat([conv_out, noise_term], dim=1)  # [1, 2*C_out, H, W]
    ref = F.leaky_relu(cat_out + act_b, neg_slope) * scale

    out = triton_fused_gpen_act(conv_out, noise_term, act_b, neg_slope, scale)
    ok = allclose(out, ref, atol=5e-3, rtol=5e-3, name=f"C_out={C_out} H={H} W={W}")
    all_ok = all_ok and ok

# ===========================================================================
# Test 3: triton_fused_gfpgan_act  (with and without noise)
# ===========================================================================
print("\n=== triton_fused_gfpgan_act (with noise) ===")

for C_out, H, W in [(64, 64, 64), (128, 32, 32), (512, 8, 8)]:
    x = torch.randn(1, C_out, H, W, dtype=dtype, device=device)
    noise = torch.randn(1, C_out, H, W, dtype=dtype, device=device)
    bias = torch.randn(1, C_out, 1, 1, dtype=dtype, device=device)
    neg_slope = 0.2
    scale = 2.0**0.5

    # PyTorch reference
    ref = F.leaky_relu(x * scale + noise + bias, neg_slope)

    out = triton_fused_gfpgan_act(x, noise, bias, neg_slope, scale)
    ok = allclose(out, ref, atol=5e-3, rtol=5e-3, name=f"C_out={C_out} H={H} W={W}")
    all_ok = all_ok and ok

print("\n=== triton_fused_gfpgan_act (no noise) ===")

for C_out, H, W in [(64, 64, 64), (128, 32, 32), (512, 8, 8)]:
    x = torch.randn(1, C_out, H, W, dtype=dtype, device=device)
    bias = torch.randn(1, C_out, 1, 1, dtype=dtype, device=device)
    neg_slope = 0.2
    scale = 2.0**0.5

    # PyTorch reference
    ref = F.leaky_relu(x * scale + bias, neg_slope)

    out = triton_fused_gfpgan_act(x, None, bias, neg_slope, scale)
    ok = allclose(
        out, ref, atol=5e-3, rtol=5e-3, name=f"C_out={C_out} H={H} W={W} no_noise"
    )
    all_ok = all_ok and ok

# ===========================================================================
# Test 4: triton_adain
# ===========================================================================
print("\n=== triton_adain ===")

eps = 1e-5
for C, HW in [(1024, 1024), (512, 1024), (128, 4096), (256, 256)]:
    x = torch.randn(C, HW, dtype=dtype, device=device)
    scale = torch.randn(C, dtype=dtype, device=device)
    bias = torch.randn(C, dtype=dtype, device=device)

    # PyTorch reference (FP32 for numerical stability)
    xf = x.float()
    mean = xf.mean(dim=1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
    x_n = (xf - mean) / (var + eps).sqrt()
    ref = (x_n * scale.float().unsqueeze(1) + bias.float().unsqueeze(1)).to(dtype)

    out = triton_adain(x, scale, bias, eps)
    ok = allclose(out, ref, atol=5e-3, rtol=5e-3, name=f"C={C} HW={HW}")
    all_ok = all_ok and ok

print()
if all_ok:
    print("All Triton kernel tests PASSED.")
else:
    print("Some tests FAILED.")
    sys.exit(1)
