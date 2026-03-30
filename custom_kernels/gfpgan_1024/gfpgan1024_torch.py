"""
GFPGAN-1024 PyTorch wrapper.
Re-exports GFPGANTorch configured for the gfpgan-1024.onnx model.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Re-use the shared implementation from gfpgan_v1_4
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from custom_kernels.gfpgan_v1_4.gfpgan_torch import GFPGANTorch, build_cuda_graph_runner

__all__ = ["GFPGANTorch", "build_cuda_graph_runner"]
