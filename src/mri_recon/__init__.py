"""
MRI reconstruction experiment package.

This package implements the comparative study pipeline: classical optimizers
(kernel / zero-filled), supervised deep models (ResidualCNN, U-Net, SRCNN+DC),
and self-supervised baselines (e.g. Noise2Void), wired through shared config,
data loading, and experiment runners.

Subpackages are intentionally split so sweeps over acceleration R and training
budget N can reuse the same interfaces without duplicating training code.
"""

__version__ = "0.1.0"
