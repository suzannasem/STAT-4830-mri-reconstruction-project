"""
ZF-conditioned DDPM denoiser + sampling with k-space data consistency (Week 10 notebook).

Matches ``DiffusionZFDenoiser``, ``q_sample``, ``extract``, cosine/linear schedule,
and ``diffusion_sample_with_dc_zf`` from ``Week_10_Notebook_Multi_Image.ipynb``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mri_recon.shared.data_consistency import data_consistency


class SinusoidalTimeEmbedding(nn.Module):
    """Timestep embedding (notebook)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class DiffusionZFDenoiser(nn.Module):
    """
    Predicts noise ε given (x_t, x_zf, t). Conditioned on zero-filled magnitude image.
    """

    def __init__(self, time_dim: int = 64, ch: int = 64) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.conv1 = nn.Conv2d(2, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv4 = nn.Conv2d(ch, 1, 3, padding=1)
        self.time_proj1 = nn.Linear(time_dim, ch)
        self.time_proj2 = nn.Linear(time_dim, ch)
        self.time_proj3 = nn.Linear(time_dim, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_t: torch.Tensor, x_zf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        h = torch.cat([x_t, x_zf], dim=1)
        h = self.conv1(h)
        h = h + self.time_proj1(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = h + self.time_proj2(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.conv3(h)
        h = h + self.time_proj3(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        return self.conv4(h)


class DDPMSchedule(nn.Module):
    """Linear β schedule and derived α, ᾱ tensors (registered buffers)."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather schedule values for batch timesteps; broadcast to x shape."""
    out = a.gather(0, t.long())
    return out.view(-1, 1, 1, 1)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(extract(alpha_bars, t, x0.shape))
    sqrt_1mab = torch.sqrt(1.0 - extract(alpha_bars, t, x0.shape))
    return sqrt_ab * x0 + sqrt_1mab * noise, noise


@torch.no_grad()
def diffusion_sample_with_dc_zf(
    model: nn.Module,
    x_zf: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    schedule: DDPMSchedule,
    *,
    num_steps: int | None = None,
    start_from_zf: bool = True,
) -> torch.Tensor:
    """
    Reverse DDPM with data consistency after each step (Week 10).

    ``x_zf``: conditioning [B,1,H,W]; ``y_obs``: observed k-space (fftshift layout).
    """
    model.eval()
    T = schedule.T
    num_steps = num_steps if num_steps is not None else T
    num_steps = min(num_steps, T)
    betas = schedule.betas
    alphas = schedule.alphas
    alpha_bars = schedule.alpha_bars
    bsz = x_zf.size(0)
    device = x_zf.device

    if start_from_zf:
        x = x_zf + 0.05 * torch.randn_like(x_zf)
    else:
        x = torch.randn_like(x_zf)

    for step in reversed(range(num_steps)):
        t = torch.full((bsz,), step, device=device, dtype=torch.long)
        beta_t = extract(betas, t, x.shape)
        alpha_t = extract(alphas, t, x.shape)
        alpha_bar_t = extract(alpha_bars, t, x.shape)
        eps_pred = model(x, x_zf, t)
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )
        if step > 0:
            z = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * z
        x = data_consistency(x, y_obs, mask)
    return torch.clamp(x, 0.0, 1.0)
