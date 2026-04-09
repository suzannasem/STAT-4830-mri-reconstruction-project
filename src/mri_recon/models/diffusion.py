"""Re-export Week 10 ZF diffusion model (alias matches experiment spec naming)."""

from mri_recon.reconstructors.diffusion import DiffusionZFDenoiser as ZFConditionedDiffusionDC

__all__ = ["ZFConditionedDiffusionDC"]
