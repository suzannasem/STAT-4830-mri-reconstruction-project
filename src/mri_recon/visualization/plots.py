"""Figures for benchmark reports — bar charts, grids, and combined dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_psnr_bar_chart(
    methods: Sequence[str],
    psnrs: Sequence[float],
    out_path: Path,
    title: str = "PSNR by method (dB)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(methods))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(methods)))
    ax.bar(x, psnrs, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_reconstruction_grid(
    panels: list[tuple[str, torch.Tensor]],
    out_path: Path,
    ncols: int = 4,
    dpi: int = 160,
    suptitle: str = "Reconstructions (test slice)",
) -> None:
    """
    panels: list of (title, image tensor [1,H,W] or [H,W]).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    for i, (title, t) in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        im = t.detach().cpu().float()
        if im.dim() == 3:
            im = im.squeeze(0)
        arr = im.numpy().clip(0, 1)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_benchmark_dashboard(
    methods: Sequence[str],
    psnrs: Sequence[float],
    panels: list[tuple[str, torch.Tensor]],
    out_path: Path,
    *,
    title: str = "MRI reconstruction — notebook-aligned benchmark",
    subtitle: str = "",
    ncols_grid: int = 5,
    dpi: int = 170,
) -> None:
    """
    Single cohesive figure: PSNR bar (top) + reconstruction grid (bottom).
    Matches a one-slide / poster summary layout.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.28)
    ax_bar = fig.add_subplot(gs[0])

    x = np.arange(len(methods))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.95, len(methods)))
    ax_bar.bar(x, psnrs, color=colors, edgecolor="black", linewidth=0.35, alpha=0.92)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods, rotation=28, ha="right", fontsize=9)
    ax_bar.set_ylabel("PSNR (dB)", fontsize=11)
    ax_bar.set_title("Test PSNR by method", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.35)

    n = len(panels)
    nrows = (n + ncols_grid - 1) // ncols_grid
    gs_sub = gridspec.GridSpecFromSubplotSpec(nrows, ncols_grid, subplot_spec=gs[1], wspace=0.12, hspace=0.35)

    for i, (ptitle, t) in enumerate(panels):
        r, c = divmod(i, ncols_grid)
        ax = fig.add_subplot(gs_sub[r, c])
        im = t.detach().cpu().float()
        if im.dim() == 3:
            im = im.squeeze(0)
        arr = im.numpy().clip(0, 1)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ptitle, fontsize=8)
        ax.axis("off")

    for j in range(n, nrows * ncols_grid):
        r, c = divmod(j, ncols_grid)
        ax = fig.add_subplot(gs_sub[r, c])
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, style="italic", color="0.35")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
