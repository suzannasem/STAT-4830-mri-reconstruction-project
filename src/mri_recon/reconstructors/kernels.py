"""Sparse kernel-basis reconstruction with optional TV (Week 10 style)."""

from __future__ import annotations

import numpy as np
import torch

from mri_recon.data_pipeline import kspace_fftshift_from_image

ZERO_FILLED_ID = "zero_filled"
GAUSSIAN_KERNEL_ID = "gaussian_kernel"
LAPLACIAN_KERNEL_ID = "laplacian_kernel"


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV on image [B,1,H,W] or [H,W]."""
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def create_kernel_basis(
    grid_size: int,
    num_kernels: int,
    sigma: float,
    kind: str = "gaussian",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Phi with shape [grid_size * grid_size, K] where columns are vectorized kernels.

    Centers lie on a square grid (Week 10). K must be a perfect square.
    """
    side = int(round(np.sqrt(num_kernels)))
    num_kernels = side * side
    coords = torch.linspace(0, grid_size - 1, side, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    centers_x = xx.reshape(-1)
    centers_y = yy.reshape(-1)

    ys = torch.arange(grid_size, device=device, dtype=torch.float32)
    xs = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    basis: list[torch.Tensor] = []
    for i in range(num_kernels):
        cx, cy = centers_x[i], centers_y[i]
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        if kind == "gaussian":
            kernel = torch.exp(-dist_sq / (2 * sigma**2))
        elif kind == "laplacian":
            kernel = torch.exp(-torch.sqrt(dist_sq + 1e-12) / sigma)
        else:
            raise ValueError(f"unknown kind: {kind}")
        basis.append(kernel.reshape(-1))

    return torch.stack(basis, dim=1)


def reconstruct_sparse_kernel(
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float,
    num_kernels: int,
    lambda_tv: float,
    max_iter: int,
    lr: float,
    kind: str = "gaussian",
) -> torch.Tensor:
    """
    Optimize sparse coefficients c in x = Phi @ c to match observed k-space.

    y_obs, mask: [H, W] complex / real on same device.
    Returns reconstructed image [H, W] real in [0, 1].
    """
    device = y_obs.device
    h, w = y_obs.shape
    grid_size = h
    assert h == w

    Phi = create_kernel_basis(grid_size, num_kernels, sigma, kind=kind, device=device)
    k = Phi.shape[1]
    c = (torch.randn(k, device=device, dtype=torch.float32) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    m = mask.to(dtype=torch.float32)

    for _ in range(max_iter):
        opt.zero_grad()
        x = (Phi @ c).reshape(grid_size, grid_size)
        x_4 = x.unsqueeze(0).unsqueeze(0)
        k_pred = kspace_fftshift_from_image(x_4).squeeze(0).squeeze(0)
        data_loss = torch.mean(torch.abs(m * (k_pred - y_obs)) ** 2)
        t = tv_loss(x_4)
        loss = data_loss + float(lambda_tv) * t
        loss.backward()
        opt.step()

    with torch.no_grad():
        x = (Phi @ c).reshape(grid_size, grid_size).clamp(0.0, 1.0)
    return x
