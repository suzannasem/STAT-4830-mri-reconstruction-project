"""Training loops for supervised and self-supervised reconstructors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mri_recon.reconstructors.diffusion import q_sample

def train_residual_cnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()


def train_dc_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    k_obs_train: torch.Tensor,
    mask: torch.Tensor,
    train_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train U-Net or SRCNN_DC with data consistency."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mask = mask.to(device)
    n = len(train_indices)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bi = [train_indices[i] for i in idx.tolist()]
            xb = x_train[bi].to(device)
            yb = y_train[bi].to(device)
            kb = k_obs_train[bi].to(device)
            opt.zero_grad()
            pred = model(xb, kb, mask)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()


def train_lpgd(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    k_obs_train: torch.Tensor,
    mask: torch.Tensor,
    train_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train LPGD (same batching contract as ``train_dc_model``)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mask = mask.to(device)
    n = len(train_indices)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bi = [train_indices[i] for i in idx.tolist()]
            xb = x_train[bi].to(device)
            yb = y_train[bi].to(device)
            kb = k_obs_train[bi].to(device)
            opt.zero_grad()
            pred = model(xb, kb, mask)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()


def train_diffusion_zf(
    model: torch.nn.Module,
    schedule: torch.nn.Module,
    x_zf_all: torch.Tensor,
    y_all: torch.Tensor,
    train_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """
    Train ``DiffusionZFDenoiser``: predict noise ε in forward diffusion (Week 10).

    ``schedule`` must provide ``T`` and ``alpha_bars`` buffer.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = int(schedule.T)
    alpha_bars = schedule.alpha_bars
    n = len(train_indices)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bi = [train_indices[i] for i in idx.tolist()]
            x_zf = x_zf_all[bi].to(device)
            y_true = y_all[bi].to(device)
            bsz = y_true.size(0)
            t = torch.randint(0, T, (bsz,), device=device)
            x_t, noise = q_sample(y_true, t, alpha_bars)
            opt.zero_grad()
            eps_pred = model(x_t, x_zf, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            opt.step()


def train_self_supervised_denoise(
    model: torch.nn.Module,
    x_zf: torch.Tensor,
    train_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    noise_std: float = 0.05,
    batch_size: int = 4,
) -> None:
    """Denoising: predict x_zf from (x_zf + noise); no ground truth."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train_indices)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bi = [train_indices[i] for i in idx.tolist()]
            xz = x_zf[bi].to(device)
            noisy = xz + noise_std * torch.randn_like(xz)
            opt.zero_grad()
            pred = model(noisy)
            loss = F.mse_loss(pred, xz)
            loss.backward()
            opt.step()
