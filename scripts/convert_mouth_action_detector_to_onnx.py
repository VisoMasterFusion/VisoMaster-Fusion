"""Convert the mouth action TensorFlow frozen graph to simplified ONNX.

This is a development utility. Runtime inference uses only ONNX Runtime; the
TensorFlow and tf2onnx imports are intentionally kept out of the app path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "model_assets" / "mouth_action_detector" / "model.pb"
DEFAULT_OUTPUT = ROOT / "model_assets" / "mouth_action_detector" / "model.onnx"


def run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert mouth_action_detector/model.pb to simplified ONNX."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=18)
    ns = parser.parse_args()

    input_path = ns.input.resolve()
    output_path = ns.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--graphdef",
            str(input_path),
            "--inputs",
            "image_tensor:0",
            "--outputs",
            "detected_boxes:0,detected_scores:0,detected_classes:0",
            "--opset",
            str(ns.opset),
            "--output",
            str(output_path),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "onnxsim",
            str(output_path),
            str(output_path),
            "--overwrite-input-shape",
            "image_tensor:0:1,320,320,3",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
