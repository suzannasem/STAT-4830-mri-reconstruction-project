"""K-space helpers; data consistency lives in ``mri_recon.shared`` (single source)."""

from __future__ import annotations

import torch

from mri_recon.shared.data_consistency import data_consistency

__all__ = ["data_consistency", "kspace_fftshift_from_image", "zero_filled_image"]


def kspace_fftshift_from_image(image: torch.Tensor) -> torch.Tensor:
    """
    image: [..., H, W] real -> k-space [..., H, W] complex, fftshifted.
    """
    return torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)))


def zero_filled_image(k_obs: torch.Tensor) -> torch.Tensor:
    """
    k_obs: complex k-space, fftshift layout [..., H, W].
    Returns real image [..., H, W].
    """
    return torch.fft.ifft2(torch.fft.ifftshift(k_obs, dim=(-2, -1)), dim=(-2, -1)).real
