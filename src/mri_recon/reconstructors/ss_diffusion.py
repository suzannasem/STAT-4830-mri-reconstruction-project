"""
Self-supervised diffusion-style reconstruction (Week 12 notebook, Korkmaz et al. 2024).

Mask split M → Mp / Mr, unrolled SSDiffBlock + MapperNetwork, cosine time schedule,
and frequency-weighted loss on Mr (``freq_weight`` avoids shadowing image width W).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from mri_recon.kspace_ops import zero_filled_image
from mri_recon.shared.data_consistency import data_consistency


def create_ss_diff_masks(
    H: int,
    W: int,
    center_fraction: float = 0.08,
    accel: float = 4.0,
    ss_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build M, Mr, Mp with Mp + Mr = M on the outer region (Week 12 notebook).
    Returns tensors shaped [1, 1, H, W].
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )
    rr = torch.sqrt(xx**2 + yy**2)
    prob = 1.0 / accel + (1.0 - 1.0 / accel) * torch.exp(-4.0 * rr**2)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    random_field = torch.rand((H, W), generator=g)
    mask_m = (random_field < prob).float()

    c_h, c_w = int(H * center_fraction), int(W * center_fraction)
    h0, w0 = (H // 2) - (c_h // 2), (W // 2) - (c_w // 2)

    mask_m[h0 : h0 + c_h, w0 : w0 + c_w] = 1.0

    center_mask = torch.zeros_like(mask_m)
    center_mask[h0 : h0 + c_h, w0 : w0 + c_w] = 1.0

    candidate_mask = (mask_m > 0.5) & (center_mask < 0.5)
    candidate_indices = torch.where(candidate_mask)
    num_candidates = candidate_indices[0].shape[0]
    num_sampled_total = torch.sum(mask_m > 0.5).item()
    num_mr = int(num_sampled_total * ss_fraction)

    perm = torch.randperm(num_candidates, generator=g)
    mr_idx_h = candidate_indices[0][perm[:num_mr]]
    mr_idx_w = candidate_indices[1][perm[:num_mr]]

    mask_r = torch.zeros_like(mask_m)
    mask_r[mr_idx_h, mr_idx_w] = 1.0
    mask_p = mask_m - mask_r

    return (
        mask_m.view(1, 1, H, W),
        mask_r.view(1, 1, H, W),
        mask_p.view(1, 1, H, W),
    )


def frequency_weighted_loss(
    k_pred: torch.Tensor,
    k_target: torch.Tensor,
    Mr: torch.Tensor,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spectral weighting by radial frequency (Week 12). ``freq_weight`` is the radial
    weight map (renamed from ``W`` in the notebook to avoid shadowing width).
    """
    device = k_pred.device
    k_target = k_target.to(device)
    Mr = Mr.to(device)
    _B, _C, H_img, W_img = k_pred.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H_img, device=device),
        torch.linspace(-1.0, 1.0, W_img, device=device),
        indexing="ij",
    )
    freq_weight = (torch.sqrt(xx**2 + yy**2) ** alpha).to(device)

    diff = torch.abs(k_pred - k_target) * Mr
    weighted_diff = diff * freq_weight
    loss = weighted_diff.sum() / (Mr.sum() + 1e-8)
    return loss, diff, freq_weight


class SSDiffBlock(nn.Module):
    """Unrolled denoising block with mapper-driven modulation + data consistency."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.to_features = nn.Conv2d(1, channels, 1)
        self.to_image = nn.Conv2d(channels, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        wl: torch.Tensor,
        wg: torch.Tensor,
        y_p: torch.Tensor,
        mask_p: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.to_features(x)
        feat = feat * wg.view(-1, feat.shape[1], 1, 1)
        feat = self.act(self.conv1(feat))
        b, c, h, w = feat.shape
        feat_flat = feat.view(b, c, h * w).permute(0, 2, 1)
        context, _ = self.attn(query=feat_flat, key=wl, value=wl)
        feat = context.permute(0, 2, 1).view(b, c, h, w)
        x_img = self.to_image(feat)
        return data_consistency(x_img, y_p, mask_p)


class MapperNetwork(nn.Module):
    """Maps (time, acceleration) to local/global latents for SSDiff blocks."""

    def __init__(self, input_dim: int = 2, latent_dim: int = 32, num_layers: int = 12) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_f = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_f, latent_dim))
            layers.append(nn.ReLU())
            in_f = latent_dim
        self.net = nn.Sequential(*layers)
        self.to_local = nn.Linear(latent_dim, 64)
        self.to_global = nn.Linear(latent_dim, 64)

    def forward(self, t: torch.Tensor, accel_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        meta = torch.cat([t.unsqueeze(-1), accel_rate.unsqueeze(-1)], dim=-1)
        latent = self.net(meta)
        wl = self.to_local(latent).unsqueeze(1)
        wg = self.to_global(latent)
        return wl, wg


class SSDiffReconModel(nn.Module):
    """Unrolled self-supervised recon (Week 12)."""

    def __init__(self, num_blocks: int = 5) -> None:
        super().__init__()
        self.mapper = MapperNetwork()
        self.blocks = nn.ModuleList([SSDiffBlock() for _ in range(num_blocks)])

    def forward(
        self,
        xt_up: torch.Tensor,
        y_p: torch.Tensor,
        mask_p: torch.Tensor,
        t: torch.Tensor,
        accel_rate: torch.Tensor,
    ) -> torch.Tensor:
        wl, wg = self.mapper(t, accel_rate)
        current_x = xt_up
        for block in self.blocks:
            current_x = block(current_x, wl, wg, y_p, mask_p)
        return torch.sigmoid(current_x)


class _SSDiffTensorDataset(Dataset):
    """Serves full-resolution slices as (image, fftshifted k-space)."""

    def __init__(self, target_images: torch.Tensor, indices: list[int]) -> None:
        self.stack = target_images[indices].float()

    def __len__(self) -> int:
        return self.stack.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.stack[idx]
        k_full = torch.fft.fftshift(torch.fft.fft2(y, norm="ortho"), dim=(-2, -1))
        return y, k_full


@torch.no_grad()
def reconstruct_inference(
    model: SSDiffReconModel,
    y_obs: torch.Tensor,
    mask_full: torch.Tensor,
    *,
    num_steps: int = 5,
    accel_rate: float = 8.0,
) -> torch.Tensor:
    """
    Coarse reverse schedule from zero-filled init (Week 12 notebook).

    ``y_obs``: undersampled k-space [B,1,H,W] complex (fftshift layout).
    ``mask_full``: binary mask M [H,W] or [1,1,H,W] matching training acceleration.
    """
    model.eval()
    device = y_obs.device
    bsz = y_obs.size(0)
    xf = zero_filled_image(y_obs)
    if xf.dim() == 3:
        xt = xf.unsqueeze(1)
    else:
        xt = xf
    if mask_full.dim() == 2:
        mask_b = mask_full.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).to(device)
    else:
        mask_b = mask_full.to(device)

    time_steps = torch.linspace(800, 0, num_steps, device=device)
    accel_tensor = torch.full((bsz,), accel_rate, device=device, dtype=torch.float32)

    for i in range(num_steps):
        t_val = (time_steps[i] / 1000.0).item()
        t_tensor = torch.full((bsz,), t_val, device=device, dtype=torch.float32)
        x0_hat = model(xt, y_obs, mask_b, t_tensor, accel_tensor)
        if i < num_steps - 1:
            sigma_t = 0.01 * (1.0 - t_val)
            z = torch.randn_like(x0_hat)
            xt = x0_hat + sigma_t * z
        else:
            xt = x0_hat
    return torch.clamp(xt, 0.0, 1.0)


def train_ssdiff_recon(
    model: SSDiffReconModel,
    target_images: torch.Tensor,
    train_indices: list[int],
    device: torch.device,
    *,
    num_epochs: int = 100,
    batch_size: int = 1,
    lr: float = 5e-5,
    accel: float = 8.0,
    center_fraction: float = 0.08,
    mask_accel: float = 4.0,
    warmup_epochs: int = 20,
    seed: int = 42,
) -> None:
    """
    Self-supervised training (no GT in the loss — only Mr k-space terms).

    ``target_images``: [N,1,H,W] on CPU; used only to synthesize k-space and diagnostics.
    """
    model = model.to(device)
    ds = _SSDiffTensorDataset(target_images, train_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    warmup_e = min(warmup_epochs, max(0, num_epochs - 1))
    if num_epochs <= 1 or warmup_e >= num_epochs:
        scheduler = LinearLR(opt, start_factor=0.1, total_iters=max(1, num_epochs))
    else:
        warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_e)
        cosine_scheduler = CosineAnnealingLR(opt, T_max=max(1, num_epochs - warmup_e))
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_e],
        )

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, k_full) in enumerate(loader):
            images = images.to(device)
            k_full = k_full.to(device)
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
            k_full = torch.fft.fftshift(
                torch.fft.fft2(images, norm="ortho"),
                dim=(-2, -1),
            )

            _B, _c, H_dim, W_dim = images.shape
            mask_seed = int(seed + batch_idx + epoch * 10007)
            M, Mr, Mp = create_ss_diff_masks(
                H_dim,
                W_dim,
                center_fraction=center_fraction,
                accel=mask_accel,
                seed=mask_seed,
            )
            M, Mr, Mp = M.to(device), Mr.to(device), Mp.to(device)

            y_p = k_full * Mp.to(k_full.dtype)
            x_complex_zf = torch.fft.ifft2(
                torch.fft.ifftshift(y_p, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            )
            x_up = x_complex_zf.abs()
            x_up = (x_up - x_up.min()) / (x_up.max() - x_up.min() + 1e-8)
            input_phase = torch.angle(x_complex_zf)

            t = torch.rand(images.size(0), device=device)
            a_bar = torch.cos(t * 0.5 * math.pi) ** 2
            noise = torch.randn_like(x_up)
            xt_up = torch.sqrt(a_bar).view(-1, 1, 1, 1) * x_up + torch.sqrt(1.0 - a_bar).view(-1, 1, 1, 1) * noise

            accel_tensor = torch.full((images.size(0),), accel, device=device, dtype=torch.float32)

            x0_hat_mag = model(xt_up, y_p, Mp, t, accel_tensor)
            x0_hat_mag = torch.clamp(x0_hat_mag, 0.0, 1.0)

            x0_hat_complex = x0_hat_mag * torch.exp(1j * input_phase)
            k_hat = torch.fft.fftshift(
                torch.fft.fft2(x0_hat_complex, norm="ortho"),
                dim=(-2, -1),
            )
            k_final = y_p + (k_hat * (1.0 - Mp.to(k_hat.dtype)))
            x0_hat_mag = torch.fft.ifft2(
                torch.fft.ifftshift(k_final, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).abs()

            target_mr = k_full * Mr.to(k_full.dtype)
            ss_loss, _diff, _fw = frequency_weighted_loss(k_final, target_mr, Mr, alpha=1.5)

            opt.zero_grad()
            ss_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        scheduler.step()


__all__ = [
    "MapperNetwork",
    "SSDiffReconModel",
    "SSDiffBlock",
    "create_ss_diff_masks",
    "frequency_weighted_loss",
    "reconstruct_inference",
    "train_ssdiff_recon",
]
