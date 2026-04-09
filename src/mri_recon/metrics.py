"""
PSNR and SSIM (Week 10 notebook–aligned).

Pred/target are torch tensors; typical shape [B, 1, H, W] with values in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["psnr", "ssim"]


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Peak SNR in dB. Scalar tensor.

    Uses 10 * log10(data_range^2 / MSE), matching the notebook when data_range=1.
    """
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


_ssim_window_cache: dict[tuple, torch.Tensor] = {}


def _gaussian_window(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    SSIM mean over the SSIM map (Week 10 `compute_ssim`).

    pred, target: [B, 1, H, W] float.
    Returns scalar tensor.
    """
    x = pred.float()
    y = target.float()
    channels = x.size(1)
    key = (window_size, sigma, channels, str(x.device), str(x.dtype))
    if key not in _ssim_window_cache:
        _ssim_window_cache[key] = _gaussian_window(
            window_size, sigma, channels, x.device, x.dtype
        )
    window = _ssim_window_cache[key]

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    )

    return ssim_map.mean()
