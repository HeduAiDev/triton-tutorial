"""
CUDA SIMT Naive GEMM Extension
================================
Compiles and wraps the 00simt_naive.cu kernel for use in Python benchmarks.

Provides ``simt_naive(a, b) -> c`` with the same signature expected by
``benchmark_utils.benchmark_method()``.

Kernel characteristics (matching 00simt_naive.cu):
- One thread per output element (no tiling, no shared memory)
- Block tile: 16 × 8  (M × N)
- FP16 inputs, FP16 intermediate accumulation (lowest precision baseline)
- This is the "zero optimization" reference point for the entire Part 3
"""

from __future__ import annotations
import os
import torch
from torch.utils.cpp_extension import load

# Compile the CUDA kernel as a PyTorch extension
# Source: scripts/simt_naive_kernel.cu — faithful reproduction of
#         src/simt/00simt_naive.cu with a PyTorch host wrapper.
_src_dir = os.path.dirname(os.path.abspath(__file__))
_simt_module = load(
    name="simt_naive_ext",
    sources=[os.path.join(_src_dir, "simt_naive_kernel.cu")],
    verbose=False,
)


def simt_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    SIMT Naive FP16 GEMM — one thread per output element.

    Equivalent to ``src/simt/00simt_naive.cu``.
    Block tile: 16×8.  No shared memory, no tiling, FP16 accumulation.

    Args:
        a: (M, K) float16 CUDA tensor
        b: (K, N) float16 CUDA tensor

    Returns:
        c: (M, N) float16 CUDA tensor
    """
    return _simt_module.simt_naive_cuda(a.contiguous(), b.contiguous())
