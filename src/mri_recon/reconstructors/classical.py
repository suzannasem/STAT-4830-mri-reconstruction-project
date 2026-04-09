"""
Classical (non-learned or optimization-only) reconstructors.

Significance:
- zero_filled: IFFT of masked k-space (baseline).
- gaussian_kernel: sparse coefficients in Gaussian kernel basis + k-space data
  fidelity + optional TV, as in Week 10 MultiSliceKernelOptimizer.
- laplacian_kernel: same structure with Laplacian-shaped atoms (Week 5 / week 6).

These do not require training labels; they may still run Adam per slice.
"""

from __future__ import annotations

# Placeholders — wire to MultiSliceKernelOptimizer logic from Week 10 notebook.
ZERO_FILLED_ID = "zero_filled"
GAUSSIAN_KERNEL_ID = "gaussian_kernel"
LAPLACIAN_KERNEL_ID = "laplacian_kernel"
