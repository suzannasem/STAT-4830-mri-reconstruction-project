"""
Learned Proximal Gradient Descent (LPGD) — Experiment Spec §2 Tier 1.5.

Unrolled proximal gradient with learned proximal maps (Monga et al. 2021 style).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mri_recon.shared.data_consistency import data_consistency


def _data_fidelity_loss(x: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    k = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    diff = mask * (k - y_obs)
    return 0.5 * (diff.real**2 + diff.imag**2).sum()


def grad_data_fidelity(
    x: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Gradient of 0.5 || M ⊙ (F x - y) ||_F^2 w.r.t. real image x."""
    g = torch.autograd.grad(
        _data_fidelity_loss(x, y_obs, mask),
        x,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return g


class ProxNet(nn.Module):
    """Conv residual denoiser (spec: 1→32→32→1 with skip from input)."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class LPGD(nn.Module):
    """
    K unrolled steps: z = x - η ∇f(x); x = ProxNet(z). DC at end.

    ``forward`` expects zero-filled magnitude image x0 and observed k-space.
    """

    def __init__(self, num_steps: int = 5, prox_ch: int = 32) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.prox_nets = nn.ModuleList(ProxNet(ch=prox_ch) for _ in range(num_steps))
        self.eta = nn.Parameter(torch.full((num_steps,), 0.05))

    def forward(self, x0: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Nested enable_grad allows analytic ∇f steps under an outer torch.no_grad() inference.
        with torch.enable_grad():
            x = x0.clone().requires_grad_(True)
            for k in range(self.num_steps):
                g = grad_data_fidelity(x, y_obs, mask, create_graph=self.training)
                x = x - self.eta[k] * g
                x = self.prox_nets[k](x)
                x = torch.clamp(x, 0.0, 1.0)
        x = data_consistency(x, y_obs, mask)
        return torch.clamp(x, 0.0, 1.0)
