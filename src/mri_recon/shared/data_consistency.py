"""
Single implementation of k-space data consistency (Experiment Spec §7, §9).

Fixes notebook bug: always use dim=(-2, -1) for ifftshift/ifft2 (never dim=(-2, 1)).
"""

from __future__ import annotations

import torch


def data_consistency(
    x: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, 1, H, W] real image.
    y_obs: [B, 1, H, W] complex undersampled k-space (fftshift layout).
    mask: [H, W] or [1, H, W] real in {0, 1} — 1 = sampled.
    """
    k = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

    if mask.dim() == 2:
        mask_exp = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask_exp = mask.unsqueeze(0)
    else:
        mask_exp = mask

    mask_exp = mask_exp.to(device=x.device, dtype=k.real.dtype)
    y_obs = y_obs.to(device=x.device, dtype=k.dtype)

    k_dc = (1.0 - mask_exp) * k + mask_exp * y_obs
    x_dc = torch.fft.ifft2(torch.fft.ifftshift(k_dc, dim=(-2, -1)), dim=(-2, -1))
    return torch.abs(x_dc)
