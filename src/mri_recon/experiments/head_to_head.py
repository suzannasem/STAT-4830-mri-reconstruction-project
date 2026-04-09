"""
Experiment 3: head-to-head at 4× acceleration, 67 train / 14 test (Experiment Spec §5).

Writes ``results/head_to_head/{method}_metrics.csv``, ``{method}_predictions.pt``,
and ``test_ground_truth.pt`` for figure generation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from mri_recon.config import (
    GLOBAL_SEED,
    KernelConfig,
    RESULTS_HEAD_TO_HEAD,
    SPEC_N_TEST,
    SPEC_N_TRAIN,
    TrainConfig,
    get_device,
)
from mri_recon.data_pipeline import (
    build_mask,
    spec_train_val_test_indices,
    synthetic_phantom_stack,
    undersample_stack,
)
from mri_recon.experiments.common import evaluate_volume, save_metrics_csv, save_predictions_pt
from mri_recon.models.learned_pgd import LPGD
from mri_recon.reconstructors.kernels import reconstruct_sparse_kernel
from mri_recon.reconstructors.networks import UNet, ZFResidualCNN_DC
from mri_recon.training.loops import train_dc_model, train_lpgd


def run_head_to_head(
    *,
    acceleration: int = 4,
    num_slices: int = 96,
    h: int = 64,
    w: int = 64,
    train_cfg: TrainConfig | None = None,
    kernel_cfg: KernelConfig | None = None,
    fast: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    train_cfg = train_cfg or TrainConfig()
    kernel_cfg = kernel_cfg or KernelConfig()
    device = get_device()
    torch.manual_seed(GLOBAL_SEED)

    if h % 8 != 0 or w % 8 != 0:
        raise ValueError("h and w must be multiples of 8 for U-Net")

    out_dir = Path(out_dir) if out_dir is not None else RESULTS_HEAD_TO_HEAD
    out_dir.mkdir(parents=True, exist_ok=True)

    y = synthetic_phantom_stack(num_slices, h, w, seed=GLOBAL_SEED)
    mask = build_mask(h, w, acceleration=acceleration, seed=GLOBAL_SEED)
    x_zf, k_obs, _ = undersample_stack(y, mask)
    mask_d = mask.to(device)

    train_idx, _val_idx, test_idx = spec_train_val_test_indices(
        num_slices, n_train=SPEC_N_TRAIN, n_test=SPEC_N_TEST, seed=GLOBAL_SEED
    )

    y_tr = y[train_idx].to(device)
    x_tr = x_zf[train_idx].to(device)
    k_tr = k_obs[train_idx].to(device)

    y_te = y[test_idx].to(device)
    x_te = x_zf[test_idx].to(device)
    k_te = k_obs[test_idx].to(device)

    epochs = 3 if fast else train_cfg.num_epochs
    lr = train_cfg.learning_rate
    bs = train_cfg.batch_size

    summary_rows: list[dict[str, Any]] = []

    # --- Zero-filled ---
    pred_zf = x_te.clone()
    rows_zf, mean_zf = evaluate_volume(pred_zf, y_te, test_idx)
    save_metrics_csv(out_dir / "zero_filled_metrics.csv", rows_zf)
    save_predictions_pt(out_dir / "zero_filled_predictions.pt", pred_zf, {"test_indices": test_idx})
    summary_rows.append(
        {
            "method": "Zero-Filled FFT",
            "category": "baseline",
            "psnr": mean_zf["psnr"],
            "ssim": mean_zf["ssim"],
            "time_per_slice_s": 0.0,
            "params": 0,
        }
    )

    # --- Gaussian kernel (per test slice; slow) ---
    preds_k: list[torch.Tensor] = []
    t_kernel = 0.0
    for j in range(len(test_idx)):
        ti = test_idx[j]
        yo = k_obs[ti, 0].to(device)
        t0 = time.perf_counter()
        x_g = reconstruct_sparse_kernel(
            yo,
            mask_d,
            sigma=kernel_cfg.sigma,
            num_kernels=kernel_cfg.num_kernels,
            lambda_tv=kernel_cfg.lambda_tv,
            max_iter=50 if fast else kernel_cfg.max_iter,
            lr=kernel_cfg.lr,
            kind="gaussian",
        )
        t_kernel += time.perf_counter() - t0
        preds_k.append(x_g.unsqueeze(0).unsqueeze(0))
    pred_k = torch.cat(preds_k, dim=0)
    rows_k, mean_k = evaluate_volume(pred_k, y_te, test_idx)
    save_metrics_csv(out_dir / "gaussian_kernel_metrics.csv", rows_k)
    save_predictions_pt(out_dir / "gaussian_kernel_predictions.pt", pred_k)
    summary_rows.append(
        {
            "method": "Kernel Recon",
            "category": "classical",
            "psnr": mean_k["psnr"],
            "ssim": mean_k["ssim"],
            "time_per_slice_s": t_kernel / max(len(test_idx), 1),
            "params": 0,
        }
    )

    # --- ZF + Residual CNN + DC ---
    res = ZFResidualCNN_DC(ch=32).to(device)
    train_dc_model(
        res,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs,
        lr,
        bs,
    )
    res.eval()
    with torch.no_grad():
        pred_r = res(x_te, k_te, mask_d)
    rows_r, mean_r = evaluate_volume(pred_r, y_te, test_idx)
    save_metrics_csv(out_dir / "zf_residual_cnn_metrics.csv", rows_r)
    save_predictions_pt(out_dir / "zf_residual_cnn_predictions.pt", pred_r)
    n_res = sum(p.numel() for p in res.parameters())
    summary_rows.append(
        {
            "method": "ZF + Residual CNN",
            "category": "supervised",
            "psnr": mean_r["psnr"],
            "ssim": mean_r["ssim"],
            "time_per_slice_s": 0.001,
            "params": n_res,
        }
    )

    # --- U-Net + DC ---
    unet = UNet(ch=32).to(device)
    train_dc_model(
        unet,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs,
        lr,
        bs,
    )
    unet.eval()
    with torch.no_grad():
        pred_u = unet(x_te, k_te, mask_d)
    rows_u, mean_u = evaluate_volume(pred_u, y_te, test_idx)
    save_metrics_csv(out_dir / "unet_metrics.csv", rows_u)
    save_predictions_pt(out_dir / "unet_predictions.pt", pred_u)
    n_unet = sum(p.numel() for p in unet.parameters())
    summary_rows.append(
        {
            "method": "ZF + U-Net",
            "category": "supervised",
            "psnr": mean_u["psnr"],
            "ssim": mean_u["ssim"],
            "time_per_slice_s": 0.001,
            "params": n_unet,
        }
    )

    # --- LPGD ---
    lpgd = LPGD(num_steps=5, prox_ch=32).to(device)
    train_lpgd(
        lpgd,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs,
        lr,
        bs,
    )
    lpgd.eval()
    with torch.no_grad():
        pred_l = lpgd(x_te, k_te, mask_d)
    rows_l, mean_l = evaluate_volume(pred_l, y_te, test_idx)
    save_metrics_csv(out_dir / "lpgd_metrics.csv", rows_l)
    save_predictions_pt(out_dir / "lpgd_predictions.pt", pred_l)
    n_lpgd = sum(p.numel() for p in lpgd.parameters())
    summary_rows.append(
        {
            "method": "Learned Prox. GD",
            "category": "supervised",
            "psnr": mean_l["psnr"],
            "ssim": mean_l["ssim"],
            "time_per_slice_s": 0.001,
            "params": n_lpgd,
        }
    )

    save_metrics_csv(
        out_dir / "summary_metrics.csv",
        summary_rows,
        fieldnames=["method", "category", "psnr", "ssim", "time_per_slice_s", "params"],
    )

    torch.save(
        {"y_test": y[test_idx].cpu(), "test_indices": test_idx, "h": h, "w": w},
        out_dir / "test_ground_truth.pt",
    )

    return {"out_dir": str(out_dir), "summary": summary_rows}


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 3: head-to-head (spec §5)")
    p.add_argument("--fast", action="store_true", help="Fewer epochs / kernel iters")
    p.add_argument("--accel", type=int, default=4)
    args = p.parse_args()
    out = run_head_to_head(acceleration=args.accel, fast=args.fast)
    print(out)


if __name__ == "__main__":
    main()
