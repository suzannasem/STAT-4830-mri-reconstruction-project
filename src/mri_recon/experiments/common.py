"""Shared helpers for spec experiment runners (save/load metrics and tensors)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from mri_recon.shared.metrics import mse, psnr, ssim


def save_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fn = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fn})


def save_predictions_pt(path: Path, preds: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"preds": preds.cpu()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def evaluate_volume(
    pred: torch.Tensor,
    gt: torch.Tensor,
    slice_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    pred, gt: [S, 1, H, W]. Returns per-slice rows and means for mse, psnr, ssim.
    """
    rows: list[dict[str, Any]] = []
    n = pred.shape[0]
    m_mse = m_m_psnr = m_m_ssim = 0.0
    for i in range(n):
        p = pred[i : i + 1]
        t = gt[i : i + 1]
        ms = mse(p, t).item()
        mp = psnr(p, t, data_range=1.0).item()
        msim = ssim(p, t, data_range=1.0).item()
        row = {"slice": slice_indices[i] if slice_indices else i, "mse": ms, "psnr": mp, "ssim": msim}
        rows.append(row)
        m_mse += ms
        m_m_psnr += mp
        m_m_ssim += msim
    inv = 1.0 / max(n, 1)
    means = {"mse": m_mse * inv, "psnr": m_m_psnr * inv, "ssim": m_m_ssim * inv}
    return rows, means


def timed_per_slice(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call ``fn`` and return (result, seconds_per_call) for single-slice timing."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return out, elapsed
