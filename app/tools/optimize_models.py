"""
optimize_models.py — ONNX model optimizer for VisoMaster Fusion
----------------------------------------------------------------
Optimizes eligible ONNX models in the model_assets directory using:
  1. onnxsim (ONNX Simplifier) — constant folding, dead node removal
  2. symbolic_shape_infer.py   — symbolic shape inference (auto-merge)

Originals are backed up to model_assets/unopt-backup/ before replacement.

Run from the application root directory (where model_assets/ lives):
    python app/tools/optimize_models.py

Called by the launcher via "Optimize Models (onnxsim)" maintenance action.
After completion the launcher sets USE_OPTIMIZED_MODELS=true in portable.cfg
so that download_models.py skips hash verification for existing files.
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

# --- Configuration ---

# Directory containing the ONNX models, relative to cwd (repo root)
MODEL_DIR = "model_assets"

# Subdirectory for backing up original models
BACKUP_DIR_NAME = "unopt-backup"

# Log file written to cwd (repo root)
LOG_FILE = "optimize_models.log"

# Models that are known to be incompatible with onnxsim / shape inference
MODEL_EXCEPTIONS = {
    "w600k_r50.onnx",
    "yunet_n_640_640.onnx",
    "2dfan4.onnx",
    "codeformer_fp16.onnx",
    "cscs_arcface_model.onnx",
    "cscs_id_adapter.onnx",
    "faceparser_fp16.onnx",
    "det_10g.onnx",
    "inswapper_128.fp16.onnx",
    "ghost_arcface_backbone.onnx",
    "ghost_unet_2_block.onnx",
    "ghost_unet_3_block.onnx",
    "RealESRGAN_x2plus.fp16.onnx",
    "realesr-general-x4v3.onnx",
    "BSRGANx2.fp16.onnx",
}

# --- Script Logic ---


def setup_logging() -> None:
    """Configure logging to both a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def find_onnxruntime_script(script_name: str) -> str | None:
    """Locate a helper script inside the installed onnxruntime package."""
    try:
        import onnxruntime

        script_path = Path(onnxruntime.__file__).parent / "tools" / script_name
        if script_path.is_file():
            return str(script_path)
        logging.error(f"Could not find '{script_name}' at '{script_path}'")
        return None
    except ImportError:
        logging.error(
            "onnxruntime is not installed. Please install it to run this script."
        )
        return None


def run_command(command: list[str]) -> bool:
    """Execute a command and log its output; return True on success."""
    process_str = " ".join(f'"{c}"' if " " in c else c for c in command)
    logging.info(f"Executing: {process_str}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.stdout:
            logging.info(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"STDERR:\n{result.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR:\n{e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error(
            f"Command not found: '{command[0]}'. "
            "Make sure the required tools are installed and in your PATH."
        )
        return False


def main() -> None:
    """Orchestrate the model optimization process."""
    setup_logging()

    # --- Pre-flight Checks ---
    if not Path(MODEL_DIR).is_dir():
        logging.error(
            f"Model directory not found: '{MODEL_DIR}'. "
            "Please run this script from the application's main directory."
        )
        return

    shape_infer_script = find_onnxruntime_script("symbolic_shape_infer.py")
    if not shape_infer_script:
        logging.error("Aborting: missing symbolic_shape_infer.py script.")
        return

    # --- Setup Backup Directory ---
    backup_path = Path(MODEL_DIR) / BACKUP_DIR_NAME
    backup_path.mkdir(exist_ok=True)
    logging.info(f"Backup directory: '{backup_path}'")

    # --- Collect models ---
    model_files = [
        f
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".onnx") and os.path.isfile(os.path.join(MODEL_DIR, f))
    ]

    if not model_files:
        logging.warning(f"No .onnx models found in '{MODEL_DIR}'.")
        return

    logging.info(f"Found {len(model_files)} ONNX model(s) to process.")

    for model_file in model_files:
        logging.info("-" * 60)

        if model_file in MODEL_EXCEPTIONS:
            logging.info(f"Skipping (exception list): {model_file}")
            continue

        logging.info(f"Processing: {model_file}")

        base_name = Path(model_file).stem
        original_path = Path(MODEL_DIR) / model_file
        opt_path = Path(MODEL_DIR) / f"{base_name}_opt.onnx"
        opt_sym_path = Path(MODEL_DIR) / f"{base_name}_opt-sym.onnx"
        final_backup_path = backup_path / model_file
        temp_files = [opt_path, opt_sym_path]

        try:
            # Step 1: onnxsim
            logging.info(f"Step 1: ONNX Simplifier on {model_file}")
            if not run_command(
                [sys.executable, "-m", "onnxsim", str(original_path), str(opt_path)]
            ):
                raise RuntimeError("ONNX Simplifier failed.")

            # Step 2: Symbolic shape inference
            logging.info(f"Step 2: Symbolic Shape Inference on {opt_path.name}")
            if not run_command(
                [
                    sys.executable,
                    shape_infer_script,
                    "--input",
                    str(opt_path),
                    "--output",
                    str(opt_sym_path),
                    "--auto_merge",
                ]
            ):
                raise RuntimeError("Symbolic shape inference failed.")

            # Step 3: Backup original and replace with optimized version
            logging.info("Step 3: Backing up original and replacing with optimized.")
            shutil.move(str(original_path), str(final_backup_path))
            logging.info(
                f"  -> Backed up '{original_path.name}' to '{final_backup_path}'"
            )
            shutil.move(str(opt_sym_path), str(original_path))
            logging.info(f"  -> Placed optimized model at '{original_path.name}'")
            logging.info(f"Successfully optimized: {model_file}")

        except Exception as e:
            logging.error(f"Failed to process {model_file}: {e}")
            logging.error("Original model left in place.")

        finally:
            for temp_file in temp_files:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        logging.info(f"Cleaned up temp file: {temp_file.name}")
                    except OSError as err:
                        logging.error(
                            f"Error removing temp file {temp_file.name}: {err}"
                        )

    logging.info("-" * 60)
    logging.info("Optimization process finished.")


if __name__ == "__main__":
    main()
