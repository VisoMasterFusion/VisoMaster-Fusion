"""
Triton-accelerated CUDA kernels shared by GFPGAN, GPEN, InSwapper, CodeFormer, and XSeg.

Optimized for:
  - Vectorized I/O (FP16/FP32)
  - Fused activations (ReLU, SiLU, LeakyReLU)
  - Single-pass reductions (where spatial fits in shared memory)
  - Broadcasting of affine parameters
"""

from __future__ import annotations

import os
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Point Triton JIT cache to project-side directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_TRITON_CACHE_ROOT = _PROJECT_ROOT / "model_assets" / "custom_kernels" / "triton_cache"
_TRITON_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _get_triton_cache_version_tag() -> str:
    """Build a version tag that uniquely identifies the current runtime stack.

    The tag is used as a subdirectory name inside triton_cache/ so that
    entries compiled for one version of Triton / CUDA / Python are never
    mixed with entries from a different version.  When the stack changes
    (e.g. Triton upgrade), the old subdirectory is left intact until
    _prune_old_triton_cache() removes it.
    """
    import sys

    try:
        import triton as _t

        tv = getattr(_t, "__version__", "unknown")
    except ImportError:
        tv = "notriton"

    try:
        import torch as _torch

        cuda_v = (_torch.version.cuda or "nocuda").replace(".", "")
    except ImportError:
        cuda_v = "nocuda"

    py_v = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"{tv}_{cuda_v}_{py_v}"


def _prune_old_triton_cache(cache_root: Path, current_tag: str) -> None:
    """Remove triton_cache subdirectories that belong to an old version stack.

    Only subdirectories whose name does NOT equal *current_tag* and whose
    content looks like a Triton cache (contains at least one *.cubin or
    *.json file anywhere inside) are removed.  Unknown directories that
    might be user-created are left untouched.
    """
    if not cache_root.is_dir():
        return
    removed = []
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == current_tag:
            continue  # keep current version
        # Heuristic: a Triton cache dir contains hash-named subdirs with .cubin / .json
        has_cache_content = any(
            f.suffix in (".cubin", ".json", ".ptx", ".llir")
            for f in child.rglob("*")
            if f.is_file()
        )
        if has_cache_content:
            try:
                import shutil

                shutil.rmtree(str(child))
                removed.append(child.name)
            except Exception as e:
                print(f"[TritonCache] Could not remove old cache {child.name}: {e}")
    if removed:
        print(f"[TritonCache] Pruned {len(removed)} old version cache(s): {removed}")


_TRITON_CACHE_TAG = _get_triton_cache_version_tag()
_TRITON_CACHE = _TRITON_CACHE_ROOT / _TRITON_CACHE_TAG
_TRITON_CACHE.mkdir(parents=True, exist_ok=True)

# Prune stale caches from old Triton / CUDA / Python versions.
_prune_old_triton_cache(_TRITON_CACHE_ROOT, _TRITON_CACHE_TAG)

os.environ.setdefault("TRITON_CACHE_DIR", str(_TRITON_CACHE))

SHARED_KERNELS_DIR: Path = _PROJECT_ROOT / "model_assets" / "custom_kernels"

# ---------------------------------------------------------------------------
# BUG-C01: Triton JIT build dialog hooks
# ---------------------------------------------------------------------------
# Optional UI callback hooks set by models_processor at startup.
# show_fn(title: str, message: str) — emit Qt signal to show build dialog
# hide_fn()                          — emit Qt signal to hide it
_show_build_dialog = None
_hide_build_dialog = None
# Tracks which kernel names have already fired the dialog (one-shot per session).
_compiled_kernels: set = set()


def register_triton_build_dialog(show_fn, hide_fn) -> None:
    """Register Qt signal emitters so Triton JIT compiles can show a progress dialog.

    Call once from ModelsProcessor.__init__().  Safe to call regardless of
    current provider — callbacks are only invoked when a Triton kernel actually
    JIT-compiles for the first time on this GPU/driver/Python combination.
    """
    global _show_build_dialog, _hide_build_dialog
    _show_build_dialog = show_fn
    _hide_build_dialog = hide_fn


def _triton_show(name: str, msg: str) -> bool:
    """Emit the build dialog the FIRST time kernel *name* compiles; no-op after.

    Returns True if the dialog was actually shown (first compile), False otherwise.
    Callers should only call _triton_hide() when this returns True, to avoid
    emitting spurious hide signals on every subsequent inference call.
    """
    if name in _compiled_kernels:
        return False
    _compiled_kernels.add(name)
    if _show_build_dialog is not None:
        try:
            _show_build_dialog("Finalizing Custom Provider", msg)
            return True
        except Exception:
            pass
    return False


def _triton_hide() -> None:
    if _hide_build_dialog is not None:
        try:
            _hide_build_dialog()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Triton availability check & Windows Monkeypatching
# ---------------------------------------------------------------------------
TRITON_AVAILABLE: bool = False
try:
    import triton
    import triton.language as tl
    import triton.compiler.compiler as triton_compiler

    TRITON_AVAILABLE = True

    # Patch 1: Missing triton_key in triton-windows
    if not hasattr(triton_compiler, "triton_key"):
        triton_compiler.triton_key = lambda: triton.__version__

    # Patch 2: Missing cluster_dims / num_ctas in KernelMetadata
    # Inductor (PyTorch 2.8+) expects these for SM90+ support, but triton-windows
    # might not provide them in the metadata object.
    try:
        from triton.compiler.compiler import CompiledKernel

        if not hasattr(CompiledKernel, "num_ctas"):
            CompiledKernel.num_ctas = property(
                lambda self: getattr(self.metadata, "num_ctas", 1)
            )
        if not hasattr(CompiledKernel, "cluster_dims"):
            CompiledKernel.cluster_dims = property(
                lambda self: getattr(self.metadata, "cluster_dims", [1, 1, 1])
            )

        # Also patch the Metadata class if possible
        if hasattr(triton_compiler, "KernelMetadata"):
            km_cls = triton_compiler.KernelMetadata
            # metadata attributes in triton are usually C-defined and read-only,
            # but we can try to add them if they are missing.
            if not hasattr(km_cls, "num_ctas"):
                try:
                    km_cls.num_ctas = property(lambda self: 1)
                except Exception:
                    pass
            if not hasattr(km_cls, "cluster_dims"):
                try:
                    km_cls.cluster_dims = property(lambda self: [1, 1, 1])
                except Exception:
                    pass

    except Exception as e:
        print(f"[TritonOps] Warning: Failed to apply Inductor compatibility patch: {e}")

except ImportError:
    pass

if TRITON_AVAILABLE:
    # Apply Windows-specific environment fixes (ptxas path, TF32, UTF-8 cache
    # write patch, etc.) as early as possible — before any kernel JIT fires.
    # setup_compile_env() is idempotent so calling it here is always safe.
    try:
        from custom_kernels.compile_utils import setup_compile_env as _sce
        _sce()
        del _sce
    except Exception:
        pass

if TRITON_AVAILABLE:
    # -----------------------------------------------------------------------
    # Kernel 1 — fused weight demodulation
    # -----------------------------------------------------------------------

    @triton.jit
    def _demod_fwd(
        w_ptr,
        s_ptr,
        o_ptr,
        n,
        kHkW,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        co = tl.program_id(0)
        base = co * n
        sq_acc = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n
            ci = offs // kHkW
            w = tl.load(w_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            s = tl.load(s_ptr + ci, mask=mask, other=0.0).to(tl.float32)
            ws = w * s
            sq_acc += tl.where(mask, ws * ws, 0.0)

        total = tl.sum(sq_acc, axis=0)
        d = tl.rsqrt(total + eps)

        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n
            ci = offs // kHkW
            w = tl.load(w_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            s = tl.load(s_ptr + ci, mask=mask, other=0.0).to(tl.float32)
            tl.store(o_ptr + base + offs, (w * s * d).to(tl.float16), mask=mask)

    def triton_demod(
        w: torch.Tensor, style: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        _shown = _triton_show(
            "demod",
            "Compiling Triton kernel: weight demodulation (GFPGAN/GPEN)…\nThis only happens once per GPU/driver combination.",
        )
        try:
            w = w.contiguous()
            style = style.contiguous().float()
            C_out, C_in, kH, kW = w.shape
            n = C_in * kH * kW
            kHkW = kH * kW
            out = torch.empty_like(w)
            _demod_fwd[C_out,](w, style, out, n, kHkW, eps=eps, BLOCK=256, num_warps=4)
            return out
        finally:
            if _shown:
                _triton_hide()

    # -----------------------------------------------------------------------
    # Kernel 2 — GPEN fused noise-inject + activate (Broadcasting support)
    # -----------------------------------------------------------------------

    @triton.jit
    def _gpen_act_fwd(
        conv_ptr,
        noise_ptr,
        bias_ptr,
        out_ptr,
        C_out,
        HW,
        neg_slope: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        c = tl.program_id(0)  # [0, C_out)
        hw_pid = tl.program_id(1)
        hw_offs = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        base = c * HW + hw_offs

        bias_a = tl.load(bias_ptr + c)
        bias_b = tl.load(bias_ptr + c + C_out)

        # First half
        v_a = tl.load(conv_ptr + base, mask=hw_mask, other=0.0).to(tl.float32)
        v_a = v_a + bias_a
        v_a = tl.where(v_a >= 0.0, v_a, v_a * neg_slope) * scale
        tl.store(out_ptr + base, v_a.to(tl.float16), mask=hw_mask)

        # Second half
        v_b = tl.load(noise_ptr + base, mask=hw_mask, other=0.0).to(tl.float32)
        v_b = v_b + bias_b
        v_b = tl.where(v_b >= 0.0, v_b, v_b * neg_slope) * scale
        tl.store(out_ptr + (c + C_out) * HW + hw_offs, v_b.to(tl.float16), mask=hw_mask)

    def triton_fused_gpen_act(
        conv_out: torch.Tensor,  # [1, C_out, H, W]
        noise_term: torch.Tensor,  # [1, C_out, H, W]
        act_b: torch.Tensor,  # [1, 2*C_out, 1, 1] or [2*C_out]
        neg_slope: float = 0.2,
        scale: float = 2.0**0.5,
    ) -> torch.Tensor:
        _shown = _triton_show(
            "gpen_act",
            "Compiling Triton kernel: GPEN fused noise+activate…\nThis only happens once per GPU/driver combination.",
        )
        try:
            C_out = conv_out.shape[1]
            HW = conv_out.shape[2] * conv_out.shape[3]
            bias = act_b.view(2 * C_out).contiguous().float()
            out = torch.empty(
                1,
                2 * C_out,
                conv_out.shape[2],
                conv_out.shape[3],
                dtype=conv_out.dtype,
                device=conv_out.device,
            )

            BLOCK_HW = 512 if HW >= 512 else 256
            grid = (C_out, triton.cdiv(HW, BLOCK_HW))
            _gpen_act_fwd[grid](
                conv_out,
                noise_term,
                bias,
                out,
                C_out,
                HW,
                neg_slope=neg_slope,
                scale=scale,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
            )
            return out
        finally:
            if _shown:
                _triton_hide()

    # -----------------------------------------------------------------------
    # Kernel 3 — GFPGAN fused activate (Broadcasting support)
    # -----------------------------------------------------------------------

    @triton.jit
    def _gfpgan_act_fwd(
        out_ptr,
        noise_ptr,
        bias_ptr,
        res_ptr,
        C_out,
        HW,
        has_noise: tl.constexpr,
        noise_is_1ch: tl.constexpr,
        neg_slope: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        c = tl.program_id(0)
        hw_pid = tl.program_id(1)
        hw_offs = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        off = c * HW + hw_offs

        v = tl.load(out_ptr + off, mask=hw_mask, other=0.0).to(tl.float32)
        v = v * scale
        if has_noise:
            if noise_is_1ch:
                noise_off = hw_offs
            else:
                noise_off = off
            v += tl.load(noise_ptr + noise_off, mask=hw_mask, other=0.0).to(tl.float32)
        v += tl.load(bias_ptr + c).to(tl.float32)
        v = tl.where(v >= 0.0, v, v * neg_slope)
        tl.store(res_ptr + off, v.to(tl.float16), mask=hw_mask)

    def triton_fused_gfpgan_act(
        out: torch.Tensor,
        noise: torch.Tensor | None,
        bias: torch.Tensor,
        neg_slope: float = 0.2,
        scale: float = 2.0**0.5,
    ) -> torch.Tensor:
        _shown = _triton_show(
            "gfpgan_act",
            "Compiling Triton kernel: GFPGAN fused activation…\nThis only happens once per GPU/driver combination.",
        )
        try:
            C_out = out.shape[1]
            HW = out.shape[2] * out.shape[3]
            res = torch.empty_like(out, dtype=out.dtype)
            bias_c = bias.view(C_out).contiguous().float()

            noise_is_1ch = False
            # Use a zero-filled dummy tensor when noise is None so Triton always
            # receives a valid CUDA pointer.  The has_noise=False constexpr ensures
            # the compiled kernel never dereferences the pointer.
            if noise is not None:
                noise_is_1ch = noise.shape[1] == 1
                noise_ptr = noise
            else:
                noise_ptr = out  # harmless dummy; never loaded when has_noise=False

            BLOCK_HW = 512 if HW >= 512 else 256
            grid = (C_out, triton.cdiv(HW, BLOCK_HW))
            _gfpgan_act_fwd[grid](
                out,
                noise_ptr,
                bias_c,
                res,
                C_out,
                HW,
                has_noise=(noise is not None),
                noise_is_1ch=noise_is_1ch,
                neg_slope=neg_slope,
                scale=scale,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
            )
            return res
        finally:
            if _shown:
                _triton_hide()

    # -----------------------------------------------------------------------
    # Kernel 4 — Single-pass AdaIN + ReLU + Broadcasting (InSwapper)
    # -----------------------------------------------------------------------

    @triton.jit
    def _adain_fwd_single_pass(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        HW,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN for NCHW-contiguous tensors (B=1).
        One program per channel; spatial elements are contiguous (stride 1).
        """
        c = tl.program_id(0)
        base = c * HW

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            out = v * sc + bi
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + base + offs, out.to(tl.float16), mask=mask)

    @triton.jit
    def _adain_fwd_batched(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        C,
        HW,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN for NCHW-contiguous tensors, any batch size B.
        Grid: (B*C,) — one program per (batch_item, channel).
        scale/bias are indexed by channel only (broadcast over batch).
        """
        bc = tl.program_id(0)  # linear index: b * C + c
        c = bc % C  # channel index for scale/bias lookup
        base = bc * HW  # flat offset into NCHW-contiguous x and y

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            out = v * sc + bi
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + base + offs, out.to(tl.float16), mask=mask)

    @triton.jit
    def _adain_fwd_batched_with_residual(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        res_ptr,
        C,
        HW,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN + residual add for NCHW, any batch size B.

        Fuses ``y = adain(x, sc, bi) + residual`` into one kernel pass, saving
        one full read + write of the activation volume (6 × [B,1024,1024] FP16
        = 12–192 MB per inference at B=1..16).

        Grid: (B*C,) — one program per (batch_item, channel).
        scale/bias/residual are indexed by channel only (broadcast over batch).
        """
        bc = tl.program_id(0)
        c = bc % C
        base = bc * HW

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            out = v * sc + bi + r
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + base + offs, out.to(tl.float16), mask=mask)

    @triton.jit
    def _adain_nhwc_fwd(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        HW,
        C,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN for channels-last (NHWC) tensors (B=1).

        Memory layout: [HW, C] — element at spatial position hw, channel c
        lives at offset  hw * C + c.  One program handles one channel c.

        Cache note: with C programs running concurrently (one per channel),
        all programs collectively access every cache line exactly once per
        spatial step, giving effective L2-coalesced bandwidth despite the
        per-program stride-C scatter pattern.
        """
        c = tl.program_id(0)

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            # NHWC: spatial position hw at offset hw*C+c
            v = tl.load(x_ptr + offs * C + c, mask=mask, other=0.0).to(tl.float32)

            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + offs * C + c, mask=mask, other=0.0).to(tl.float32)
            out = v * sc + bi
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + offs * C + c, out.to(tl.float16), mask=mask)

    @triton.jit
    def _adain_nhwc_batched_fwd(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        HW,
        C,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN for channels-last (NHWC) tensors, any batch size B.

        Grid: (B*C,) — one program per (batch_item, channel).
        scale/bias are indexed by channel only (broadcast over batch).
        Memory layout: element (b, hw, c) lives at  b*HW*C + hw*C + c.

        Eliminates the NCHW conversion round-trip that the previous B>1 path
        required, fixing a correctness bug and reducing memory traffic.
        """
        bc = tl.program_id(0)
        b = bc // C
        c = bc % C
        base = b * HW * C  # byte-offset to start of batch item b in NHWC

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs * C + c, mask=mask, other=0.0).to(
                tl.float32
            )
            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs * C + c, mask=mask, other=0.0).to(
                tl.float32
            )
            out = v * sc + bi
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + base + offs * C + c, out.to(tl.float16), mask=mask)

    @triton.jit
    def _adain_nhwc_batched_with_residual_fwd(
        x_ptr,
        sc_ptr,
        bi_ptr,
        y_ptr,
        res_ptr,
        HW,
        C,
        eps: tl.constexpr,
        fuse_relu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Single-pass Welford AdaIN + residual add for channels-last (NHWC), any batch size B.

        Grid: (B*C,) — one program per (batch_item, channel).
        scale/bias are indexed by channel only (broadcast over batch).
        Memory layout: element (b, hw, c) lives at  b*HW*C + hw*C + c.
        """
        bc = tl.program_id(0)
        b = bc // C
        c = bc % C
        base = b * HW * C

        mean = 0.0
        m2 = 0.0
        count = 0.0
        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs * C + c, mask=mask, other=0.0).to(
                tl.float32
            )
            batch_cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            batch_mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / tl.where(
                batch_cnt > 0, batch_cnt, 1.0
            )
            batch_m2 = tl.sum(
                tl.where(mask, (v - batch_mean) * (v - batch_mean), 0.0), axis=0
            )
            delta = batch_mean - mean
            new_count = count + batch_cnt
            safe_nc = tl.where(new_count > 0, new_count, 1.0)
            mean = mean + delta * (batch_cnt / safe_nc)
            m2 = m2 + batch_m2 + delta * delta * count * (batch_cnt / safe_nc)
            count = new_count

        inv_std = tl.rsqrt(m2 / HW + eps)
        sc = tl.load(sc_ptr + c).to(tl.float32) * inv_std
        bi = tl.load(bi_ptr + c).to(tl.float32) - sc * mean

        for start in range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs * C + c, mask=mask, other=0.0).to(
                tl.float32
            )
            r = tl.load(res_ptr + base + offs * C + c, mask=mask, other=0.0).to(
                tl.float32
            )
            out = v * sc + bi + r
            if fuse_relu:
                out = tl.maximum(out, 0.0)
            tl.store(y_ptr + base + offs * C + c, out.to(tl.float16), mask=mask)

    def triton_adain(
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
        fuse_relu: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused AdaIN [+ ReLU] [+ residual add].

        Auto-dispatches based on memory format, batch size, and residual presence:

        * NCHW, any B, residual  → _adain_fwd_batched_with_residual (fused add)
        * NCHW, any B            → _adain_fwd_batched
        * NHWC, B=1              → _adain_nhwc_fwd  (stride-C coalesced access)
        * NHWC, B>1              → _adain_nhwc_batched_fwd (native NHWC)
        * NHWC + residual        → _adain_nhwc_batched_with_residual_fwd

        ``residual`` must have the same shape/dtype/device as ``x``.  When provided,
        the kernel computes ``y = adain(x) + residual`` in a single memory pass,
        eliminating a separate ``x = x + residual`` op after each style block.
        """
        B = x.shape[0]
        C = x.shape[1]
        HW = x.shape[2] * x.shape[3]
        y = torch.empty_like(x, dtype=torch.float16)
        sc = scale.view(C).contiguous().float()
        bi = bias.view(C).contiguous().float()
        BLOCK = 256
        num_warps = 8 if HW > 4096 else 4

        if x.is_contiguous(memory_format=torch.channels_last):
            if residual is not None:
                _adain_nhwc_batched_with_residual_fwd[B * C,](
                    x,
                    sc,
                    bi,
                    y,
                    residual,
                    HW,
                    C,
                    eps=eps,
                    fuse_relu=fuse_relu,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            elif B == 1:
                _adain_nhwc_fwd[C,](
                    x,
                    sc,
                    bi,
                    y,
                    HW,
                    C,
                    eps=eps,
                    fuse_relu=fuse_relu,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            else:
                _adain_nhwc_batched_fwd[B * C,](
                    x,
                    sc,
                    bi,
                    y,
                    HW,
                    C,
                    eps=eps,
                    fuse_relu=fuse_relu,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
        else:
            # NCHW path — use residual-fused kernel when residual is provided
            if residual is not None:
                _adain_fwd_batched_with_residual[B * C,](
                    x,
                    sc,
                    bi,
                    y,
                    residual.contiguous(),
                    C,
                    HW,
                    eps=eps,
                    fuse_relu=fuse_relu,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            else:
                _adain_fwd_batched[B * C,](
                    x,
                    sc,
                    bi,
                    y,
                    C,
                    HW,
                    eps=eps,
                    fuse_relu=fuse_relu,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
        return y

    # -----------------------------------------------------------------------
    # Kernel 5 — Fused GroupNorm + optional SiLU
    # -----------------------------------------------------------------------

    @triton.jit
    def _group_norm_fwd(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        C,
        HW,
        CG,
        G: tl.constexpr,
        eps: tl.constexpr,
        fuse_silu: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n, g = pid // G, pid % G
        base = n * C * HW + g * CG * HW
        gsz = CG * HW

        acc_s, acc_s2 = tl.zeros([BLOCK], tl.float32), tl.zeros([BLOCK], tl.float32)
        for start in range(0, gsz, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < gsz
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            acc_s += tl.where(mask, v, 0.0)
            acc_s2 += tl.where(mask, v * v, 0.0)

        mean = tl.sum(acc_s, 0) / gsz
        var = tl.sum(acc_s2, 0) / gsz - mean * mean
        inv_std = tl.rsqrt(var + eps)

        for start in range(0, gsz, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < gsz
            c_glob = g * CG + (offs // HW)
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + c_glob, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(b_ptr + c_glob, mask=mask, other=0.0).to(tl.float32)
            out = (v - mean) * inv_std * w + b
            if fuse_silu:
                out = out * tl.sigmoid(out)
            tl.store(y_ptr + base + offs, out.to(tl.float16), mask=mask)

    def triton_group_norm_silu(
        x, weight, bias, num_groups=32, eps=1e-6, fuse_silu=False
    ):
        _shown = _triton_show(
            "group_norm_silu",
            "Compiling Triton kernel: GroupNorm (CodeFormer)…\nThis only happens once per GPU/driver combination.",
        )
        try:
            N, C = x.shape[0], x.shape[1]
            HW = x.numel() // (N * C)
            CG = C // num_groups
            y = torch.empty_like(x)
            BLOCK = min(2048, triton.next_power_of_2(max(CG * HW, 1)))
            _group_norm_fwd[(N * num_groups,)](
                x,
                weight,
                bias,
                y,
                C,
                HW,
                CG,
                G=num_groups,
                eps=eps,
                fuse_silu=fuse_silu,
                BLOCK=BLOCK,
                num_warps=8 if BLOCK >= 512 else 4,
            )
            return y
        finally:
            if _shown:
                _triton_hide()

    # -----------------------------------------------------------------------
    # Kernel 6 — Fused RMSNormMax (XSeg)
    # -----------------------------------------------------------------------

    @triton.jit
    def _rmsnormmax_fwd(
        x_ptr,
        y_ptr,
        gamma_ptr,
        beta_ptr,
        maxval_ptr,
        HW,
        eps,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        base = c * HW
        sq_acc = tl.zeros([BLOCK], tl.float32)
        # Use tl.range (→ MLIR scf.for) instead of Python range().
        # Python range() unrolls at JIT time: for HW=65536, BLOCK=1024 this
        # generates 64 identical MLIR blocks (~640 ops) which crashes MLIR's
        # type-registration pass (0xC0000005 in libtriton.pyd) on Windows.
        # tl.range emits a single loop construct regardless of HW/BLOCK ratio.
        for start in tl.range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            sq_acc += tl.where(mask, v * v, 0.0)

        rstd = tl.rsqrt(tl.sum(sq_acc, 0) / HW + eps)
        g = tl.load(gamma_ptr + c).to(tl.float32)
        b = tl.load(beta_ptr + c).to(tl.float32)
        mv = tl.load(maxval_ptr + c).to(tl.float32)

        for start in tl.range(0, HW, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            out = tl.maximum(v * rstd * g + b, mv)
            tl.store(y_ptr + base + offs, out.to(tl.float16), mask=mask)

    def triton_rmsnormmax(x, gamma, beta, maxval, eps):
        C, HW = x.shape[1], x.shape[2] * x.shape[3]
        y = torch.empty_like(x, dtype=torch.float16)
        BLOCK = min(1024, triton.next_power_of_2(HW))
        _rmsnormmax_fwd[C,](
            x, y, gamma, beta, maxval, HW, eps, BLOCK=BLOCK, num_warps=4
        )
        return y

    # -----------------------------------------------------------------------
    # Kernel 7 — VQ Distance (CodeFormer)
    # -----------------------------------------------------------------------

    @triton.jit
    def _vq_dist_fwd(
        z_ptr,
        emb_ptr,
        out_ptr,
        C,
        N_CODES,
        BLOCK_C: tl.constexpr,
    ):
        """
        Grid: (H*W,)
        Computes L2 distance between z[hw, C] and all codebook entries emb[N_CODES, C].
        Optimized to avoid large intermediate [HW, N_CODES] tensor.
        Returns argmin indices [HW].
        """
        hw = tl.program_id(0)
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        z = tl.load(z_ptr + hw * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)

        min_dist = 1e38
        min_idx = 0

        for i in range(N_CODES):
            emb = tl.load(emb_ptr + i * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            diff = z - emb
            dist = tl.sum(diff * diff, 0)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        tl.store(out_ptr + hw, min_idx)

    def triton_vq_dist(z: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C, H, W], emb: [N_CODES, C]
        Returns: [B, H, W] indices (long)
        """
        B, C, H, W = z.shape
        HW = H * W
        z_flat = z.permute(0, 2, 3, 1).reshape(B * HW, C).contiguous()
        indices = torch.empty(B * HW, dtype=torch.int32, device=z.device)

        BLOCK_C = triton.next_power_of_2(C)
        _vq_dist_fwd[B * HW,](
            z_flat, emb, indices, C, emb.shape[0], BLOCK_C=BLOCK_C, num_warps=4
        )
        return indices.view(B, H, W).long()

    # -----------------------------------------------------------------------
    # Kernel 8 — LayerNorm (InSwapper/CodeFormer)
    # -----------------------------------------------------------------------

    @triton.jit
    def _layer_norm_fwd(
        x_ptr, w_ptr, b_ptr, y_ptr, C, eps: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), 0) / C
        xc = tl.where(mask, x - mean, 0.0)
        inv_std = tl.rsqrt(tl.sum(xc * xc, 0) / C + eps)
        tl.store(
            y_ptr + row * C + offs, (xc * inv_std * w + b).to(tl.float16), mask=mask
        )

    def triton_layernorm(x, weight, bias, eps=1e-6):
        orig_shape = x.shape
        C = orig_shape[-1]
        x2d = x.contiguous().view(-1, C)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d, dtype=torch.float16)
        BLOCK_C = triton.next_power_of_2(C)
        _layer_norm_fwd[(rows,)](
            x2d, weight, bias, y, C, eps=eps, BLOCK_C=BLOCK_C, num_warps=4
        )
        return y.view(orig_shape)

    # -----------------------------------------------------------------------
    # Kernel 9 — Pixel-shift tile extraction (InSwapper batched resolution)
    # -----------------------------------------------------------------------

    @triton.jit
    def _pixel_shift_extract_kernel(
        img_ptr,
        out_ptr,
        H,
        W,
        C,
        B,
        dim: tl.constexpr,
        TILE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Extract all dim×dim strided tiles from an HWC image into a BCHW batch.

        Grid: (B * C * TILE,) — one program handles one row (tile_W=TILE elements)
        of one channel of one tile.

        img[h, w, c] → out[b, c, th, tw]
        where b = j*dim + i,  src_h = j + th*dim,  src_w = i + tw*dim.
        """
        pid = tl.program_id(0)
        b = pid // (C * TILE)
        tmp = pid % (C * TILE)
        c = tmp // TILE
        th = tmp % TILE

        j = b // dim  # tile row offset in original image
        i = b % dim  # tile col offset in original image
        src_row = j + th * dim

        # HWC source offset: img[src_row, i, c]
        src_base = src_row * W * C + i * C + c
        src_stride = dim * C  # step per tw increment (stride along W in HWC)

        # BCHW destination offset: out[b, c, th, 0]
        dst_base = (b * C + c) * TILE * TILE + th * TILE

        tw = tl.arange(0, BLOCK)
        mask = tw < TILE
        vals = tl.load(img_ptr + src_base + tw * src_stride, mask=mask, other=0.0)
        tl.store(out_ptr + dst_base + tw, vals, mask=mask)

    @triton.jit
    def _pixel_shift_insert_kernel(
        tiles_ptr,
        img_ptr,
        H,
        W,
        C,
        B,
        dim: tl.constexpr,
        TILE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Write BCHW tiles back into an HWC image (strided scatter).

        Grid: (B * C * TILE,) — one program handles one row of one channel of one tile.

        tiles[b, c, th, tw] → img[j + th*dim, i + tw*dim, c]
        """
        pid = tl.program_id(0)
        b = pid // (C * TILE)
        tmp = pid % (C * TILE)
        c = tmp // TILE
        th = tmp % TILE

        j = b // dim
        i = b % dim
        src_row = j + th * dim

        # BCHW source offset: tiles[b, c, th, 0]
        src_base = (b * C + c) * TILE * TILE + th * TILE

        # HWC destination offset: img[src_row, i, c]
        dst_base = src_row * W * C + i * C + c
        dst_stride = dim * C

        tw = tl.arange(0, BLOCK)
        mask = tw < TILE
        vals = tl.load(tiles_ptr + src_base + tw, mask=mask, other=0.0)
        tl.store(img_ptr + dst_base + tw * dst_stride, vals, mask=mask)

    def triton_pixel_shift_extract(img_hwc: torch.Tensor, dim: int) -> torch.Tensor:
        """Extract dim×dim strided tiles from an HWC image.

        Args:
            img_hwc: Float32 tensor of shape [H, W, C] where H = W = dim * 128.
                     Must be contiguous.
            dim:     Stride / resolution multiplier (1=128px, 2=256px, 3=384px, 4=512px).

        Returns:
            Float32 tensor of shape [dim*dim, C, 128, 128] — one 128×128 tile per
            strided sub-grid, laid out as a BCHW batch ready for InSwapper input.
        """
        H, W, C = img_hwc.shape
        TILE = 128
        B = dim * dim
        out = torch.empty(B, C, TILE, TILE, dtype=img_hwc.dtype, device=img_hwc.device)
        img_c = img_hwc.contiguous()
        grid = B * C * TILE  # one program per (tile, channel, row)
        _pixel_shift_extract_kernel[grid,](
            img_c,
            out,
            H,
            W,
            C,
            B,
            dim=dim,
            TILE=TILE,
            BLOCK=TILE,
            num_warps=4,
        )
        return out

    def triton_pixel_shift_insert(
        tiles: torch.Tensor, img_hwc: torch.Tensor, dim: int
    ) -> None:
        """Scatter BCHW tiles back into an HWC image in-place.

        Args:
            tiles:   Float32 tensor of shape [dim*dim, C, 128, 128].
            img_hwc: Float32 tensor of shape [H, W, C] — modified in-place.
            dim:     Stride / resolution multiplier matching the extract call.
        """
        H, W, C = img_hwc.shape
        TILE = 128
        B = dim * dim
        tiles_c = tiles.contiguous()
        grid = B * C * TILE
        _pixel_shift_insert_kernel[grid,](
            tiles_c,
            img_hwc,
            H,
            W,
            C,
            B,
            dim=dim,
            TILE=TILE,
            BLOCK=TILE,
            num_warps=4,
        )

    # -----------------------------------------------------------------------
    # Kernel 10 — Fused ReflectionPad + Im2Col (InSwapper GEMM mode)
    # -----------------------------------------------------------------------

    @triton.jit
    def _im2col_reflect_fwd(
        x_ptr,
        col_ptr,
        B,
        C,
        H,
        W,
        k: tl.constexpr,
        pad: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        """Fused im2col with on-the-fly reflection padding — no padded intermediate.

        Grid: (B * C * k * k, ceil(H*W / BLOCK_HW))

        Fills col[b, c*k*k + kh*k + kw, hw] = reflect_pad(x[b, c, h+kh-pad, w+kw-pad])
        where hw = h*W + w.  Eliminates the F.pad intermediate tensor and one full
        memory pass over the activation volume, saving ~0.85 ms for B=16.
        """
        pid_c = tl.program_id(0)  # linear index: b*(C*k*k) + c*(k*k) + kh*k + kw
        pid_hw = tl.program_id(1)

        kk = k * k
        Ckk = C * kk
        b = pid_c // Ckk
        tmp = pid_c % Ckk
        c = tmp // kk
        kk_i = tmp % kk
        kh = kk_i // k
        kw = kk_i % k

        HW = H * W
        hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW

        out_h = hw_offs // W  # spatial row in output
        out_w = hw_offs % W  # spatial col in output

        # Source coordinates with reflection padding
        src_h = out_h + kh - pad
        src_w = out_w + kw - pad

        # reflect(i, 0, H-1): i<0 → -i;  i≥H → 2*(H-1)-i
        src_h = tl.where(src_h < 0, -src_h, src_h)
        src_h = tl.where(src_h >= H, 2 * H - 2 - src_h, src_h)
        src_w = tl.where(src_w < 0, -src_w, src_w)
        src_w = tl.where(src_w >= W, 2 * W - 2 - src_w, src_w)

        # Source offset: x[b, c, src_h, src_w] in NCHW layout
        src_off = b * C * H * W + c * H * W + src_h * W + src_w
        vals = tl.load(x_ptr + src_off, mask=hw_mask, other=0.0)

        # Destination: col[b, c*kk + kk_i, hw]
        col_c = c * kk + kk_i
        col_off = b * Ckk * HW + col_c * HW + hw_offs
        tl.store(col_ptr + col_off, vals, mask=hw_mask)

    def triton_im2col_reflect(x: torch.Tensor, k: int, pad: int) -> torch.Tensor:
        """Fused im2col + reflection padding for NCHW-contiguous FP16 tensors.

        Combines F.pad(mode='reflect') + F.unfold into a single Triton kernel,
        eliminating the intermediate padded tensor and saving one memory pass.

        Args:
            x:    NCHW tensor [B, C, H, W] — will be made contiguous (NCHW).
            k:    Convolution kernel size (square).
            pad:  Reflection padding amount (equal on all sides).

        Returns:
            col: [B, C*k*k, H*W] — suitable for batched cuBLAS GEMM via
                 torch.matmul(w_flat[C_out, C_in*k*k], col) → [B, C_out, H*W].
        """
        x_c = x.contiguous()  # ensure NCHW (converts channels_last if needed)
        B, C, H, W = x_c.shape
        HW = H * W
        col = torch.empty(B, C * k * k, HW, dtype=x_c.dtype, device=x_c.device)
        BLOCK_HW = min(1024, triton.next_power_of_2(HW))
        grid = (B * C * k * k, triton.cdiv(HW, BLOCK_HW))
        _im2col_reflect_fwd[grid](
            x_c,
            col,
            B,
            C,
            H,
            W,
            k=k,
            pad=pad,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
        )
        return col


else:
    # -- Stubs when Triton is not available --
    def register_triton_build_dialog(show_fn, hide_fn) -> None:
        pass  # no-op: no Triton to monitor

    def triton_demod(
        w: torch.Tensor, style: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        raise RuntimeError("triton unavailable")

    def triton_fused_gpen_act(
        conv_out: torch.Tensor,
        noise_term: torch.Tensor,
        act_b: torch.Tensor,
        neg_slope: float = 0.2,
        scale: float = 2.0**0.5,
    ) -> torch.Tensor:
        raise RuntimeError("triton unavailable")

    def triton_fused_gfpgan_act(
        out: torch.Tensor,
        noise: torch.Tensor | None,
        bias: torch.Tensor,
        neg_slope: float = 0.2,
        scale: float = 2.0**0.5,
    ) -> torch.Tensor:
        raise RuntimeError("triton unavailable")

    def triton_adain(
        x: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
        fuse_relu: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("triton unavailable")

    def triton_group_norm_silu(x, w, b, g=32, e=1e-6, f=False):
        raise RuntimeError("triton unavailable")

    def triton_rmsnormmax(x, g, b, m, e):
        raise RuntimeError("triton unavailable")

    def triton_layernorm(x, w, b, e=1e-6):
        raise RuntimeError("triton unavailable")

    def triton_vq_dist(z, e):
        raise RuntimeError("triton unavailable")

    def triton_pixel_shift_extract(img_hwc: torch.Tensor, dim: int) -> torch.Tensor:
        raise RuntimeError("triton unavailable")

    def triton_pixel_shift_insert(
        tiles: torch.Tensor, img_hwc: torch.Tensor, dim: int
    ) -> None:
        raise RuntimeError("triton unavailable")

    def triton_im2col_reflect(x: torch.Tensor, k: int, pad: int) -> torch.Tensor:
        raise RuntimeError("triton unavailable")
