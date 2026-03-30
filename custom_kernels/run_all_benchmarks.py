"""
run_all_benchmarks.py
=====================
Runs every custom-kernel benchmark script in sequence, captures output,
and writes per-model result files to custom_kernels/<model>/benchmark_results.txt.

Usage (from repo root):
    .venv/Scripts/python custom_kernels/run_all_benchmarks.py
"""

from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output even on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"

BENCHMARKS = [
    "det_10g/benchmark_det10g.py",
    "yoloface_8n/benchmark_yoloface8n.py",
    "det_106/benchmark_det_106.py",
    "inswapper_128/benchmark_inswapper.py",
    "w600k_r50/benchmark_w600k_r50.py",
    "faceparser_resnet34/benchmark_faceparser.py",
    "gfpgan_v1_4/benchmark_gfpgan.py",
    "gfpgan_1024/benchmark_gfpgan1024.py",
    "gpen_bfr/benchmark_gpen.py",
    "codeformer/benchmark_codeformer.py",
    "restoreformer/benchmark_restoreformer.py",
    "landmark_203/benchmark_landmark_203.py",
    "landmark_1k3d68/benchmark_1k3d68.py",
    "fan_2dfan4/benchmark_fan_2dfan4.py",
    "face_landmark478/benchmark_face_landmark478.py",
    "face_blendshapes/benchmark_face_blendshapes.py",
    "peppapig_98/benchmark_peppapig_98.py",
    "occluder/benchmark_occluder.py",
    "xseg/benchmark_xseg.py",
    "res50/benchmark_res50.py",
    "vgg_combo/benchmark_vgg_combo.py",
    "ref_ldm/benchmark_ref_ldm.py",
]

CKDIR = ROOT / "custom_kernels"


def run_benchmark(rel_path: str) -> tuple[bool, str]:
    script = CKDIR / rel_path
    out_file = script.parent / "benchmark_results.txt"
    print(f"\n{'=' * 60}")
    print(f"  Running: {rel_path}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    trt_libs = ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    if trt_libs.exists():
        env["PATH"] = str(trt_libs) + __import__("os").pathsep + env.get("PATH", "")
    result = subprocess.run(
        [str(VENV_PY), str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(3)  # allow GPU driver to release VRAM between benchmark subprocesses
    elapsed = time.perf_counter() - t0
    output = result.stdout + (
        "\n[STDERR]\n" + result.stderr if result.stderr.strip() else ""
    )
    out_file.write_text(output, encoding="utf-8")
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (rc={result.returncode})"
    print(output[-3000:] if len(output) > 3000 else output)
    print(f"\n  [{status}]  {elapsed:.1f}s  -> {out_file}")
    return ok, str(out_file)


def main():
    print("\nVisoMaster-Fusion — Master Benchmark Runner")
    print(f"GPU: {_gpu_name()}")
    print(f"Running {len(BENCHMARKS)} benchmarks...\n")

    results: list[tuple[str, bool, str]] = []
    for rel in BENCHMARKS:
        ok, out = run_benchmark(rel)
        results.append((rel, ok, out))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for rel, ok, out in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {rel}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} benchmarks completed successfully.")


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
