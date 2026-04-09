"""
Experiment 2: data efficiency at fixed 4× (Experiment Spec §4).

Retrains supervised methods for each training slice budget; writes under
``results/data_sweep/{8slices,16slices,...}/``.
"""

from __future__ import annotations

import argparse

import torch

from mri_recon.config import (
    GLOBAL_SEED,
    KernelConfig,
    RESULTS_DATA_SWEEP,
    SPEC_N_TEST,
    SPEC_N_TRAIN,
    TRAIN_SLICE_COUNTS,
    TrainConfig,
    get_device,
)
from mri_recon.data_pipeline import (
    build_mask,
    spec_train_val_test_indices,
    synthetic_phantom_stack,
    take_n_shuffled_train,
    undersample_stack,
)
from mri_recon.experiments.common import evaluate_volume, save_metrics_csv, save_predictions_pt
from mri_recon.models.learned_pgd import LPGD
from mri_recon.reconstructors.kernels import reconstruct_sparse_kernel
from mri_recon.reconstructors.networks import UNet, ZFResidualCNN_DC
from mri_recon.training.loops import train_dc_model, train_lpgd


def run_data_efficiency_sweep(*, fast: bool = False) -> dict[str, str]:
    device = get_device()
    torch.manual_seed(GLOBAL_SEED)

    h, w = 64, 64
    num_slices = 96
    acceleration = 4

    y = synthetic_phantom_stack(num_slices, h, w, seed=GLOBAL_SEED)
    mask = build_mask(h, w, acceleration=acceleration, seed=GLOBAL_SEED)
    x_zf, k_obs, _ = undersample_stack(y, mask)
    mask_d = mask.to(device)

    full_train, _val, test_idx = spec_train_val_test_indices(
        num_slices, n_train=SPEC_N_TRAIN, n_test=SPEC_N_TEST, seed=GLOBAL_SEED
    )

    y_te = y[test_idx].to(device)
    x_te = x_zf[test_idx].to(device)
    k_te = k_obs[test_idx].to(device)

    train_cfg = TrainConfig()
    epochs = 3 if fast else train_cfg.num_epochs
    lr = train_cfg.learning_rate
    bs = train_cfg.batch_size
    kernel_cfg = KernelConfig()

    RESULTS_DATA_SWEEP.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for n in TRAIN_SLICE_COUNTS:
        tag = f"{n}slices"
        sub = RESULTS_DATA_SWEEP / tag
        sub.mkdir(parents=True, exist_ok=True)
        train_idx = take_n_shuffled_train(full_train, min(n, len(full_train)), seed=GLOBAL_SEED)

        pred_zf = x_te.clone()
        rows_zf, _ = evaluate_volume(pred_zf, y_te, test_idx)
        save_metrics_csv(sub / "zero_filled_metrics.csv", rows_zf)

        preds_k = []
        for j in range(len(test_idx)):
            ti = test_idx[j]
            yo = k_obs[ti, 0].to(device)
            x_g = reconstruct_sparse_kernel(
                yo,
                mask_d,
                sigma=kernel_cfg.sigma,
                num_kernels=kernel_cfg.num_kernels,
                lambda_tv=kernel_cfg.lambda_tv,
                max_iter=40 if fast else kernel_cfg.max_iter,
                lr=kernel_cfg.lr,
                kind="gaussian",
            )
            preds_k.append(x_g.unsqueeze(0).unsqueeze(0))
        pred_k = torch.cat(preds_k, dim=0)
        rows_k, _ = evaluate_volume(pred_k, y_te, test_idx)
        save_metrics_csv(sub / "gaussian_kernel_metrics.csv", rows_k)

        res = ZFResidualCNN_DC(ch=32).to(device)
        train_dc_model(res, x_zf, y, k_obs, mask, train_idx, device, epochs, lr, bs)
        res.eval()
        with torch.no_grad():
            pred_r = res(x_te, k_te, mask_d)
        rows_r, _ = evaluate_volume(pred_r, y_te, test_idx)
        save_metrics_csv(sub / "zf_residual_cnn_metrics.csv", rows_r)
        save_predictions_pt(sub / "zf_residual_cnn_predictions.pt", pred_r)

        unet = UNet(ch=32).to(device)
        train_dc_model(unet, x_zf, y, k_obs, mask, train_idx, device, epochs, lr, bs)
        unet.eval()
        with torch.no_grad():
            pred_u = unet(x_te, k_te, mask_d)
        rows_u, _ = evaluate_volume(pred_u, y_te, test_idx)
        save_metrics_csv(sub / "unet_metrics.csv", rows_u)
        save_predictions_pt(sub / "unet_predictions.pt", pred_u)

        lpgd = LPGD(num_steps=5, prox_ch=32).to(device)
        train_lpgd(lpgd, x_zf, y, k_obs, mask, train_idx, device, epochs, lr, bs)
        lpgd.eval()
        with torch.no_grad():
            pred_l = lpgd(x_te, k_te, mask_d)
        rows_l, _ = evaluate_volume(pred_l, y_te, test_idx)
        save_metrics_csv(sub / "lpgd_metrics.csv", rows_l)
        save_predictions_pt(sub / "lpgd_predictions.pt", pred_l)

        paths[tag] = str(sub.resolve())

    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 2: data efficiency sweep")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    print(run_data_efficiency_sweep(fast=args.fast))


if __name__ == "__main__":
    main()
