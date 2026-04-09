"""
End-to-end benchmark aligned with Week 10 notebook + experiment spec methods.

Runs: zero-filled, Gaussian/Laplacian kernel, ZF+Residual CNN+DC, U-Net+DC,
SRCNN+DC, LPGD, ZF-conditioned diffusion + DC (Week 10), Noise2Void-style
denoising, and **SSDiffRecon** (Week 12 self-supervised Mr/Mp). Writes
``results/benchmark.csv`` and cohesive figures under ``figures/``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from mri_recon.config import FIGURES_DIR, RESULTS_DIR, SPLIT_SEED, get_device
from mri_recon.data_pipeline import (
    build_mask,
    load_dicom_series_from_dir,
    subsample_train_indices,
    synthetic_phantom_stack,
    train_val_test_indices,
    undersample_stack,
)
from mri_recon.metrics import psnr, ssim
from mri_recon.models.learned_pgd import LPGD
from mri_recon.reconstructors.diffusion import (
    DDPMSchedule,
    DiffusionZFDenoiser,
    diffusion_sample_with_dc_zf,
)
from mri_recon.reconstructors.kernels import reconstruct_sparse_kernel
from mri_recon.reconstructors.networks import SRCNN_DC, UNet, ZFResidualCNN_DC
from mri_recon.reconstructors.noise2void_style import SmallDenoiseNet
from mri_recon.reconstructors.ss_diffusion import SSDiffReconModel, reconstruct_inference, train_ssdiff_recon
from mri_recon.training.loops import (
    train_dc_model,
    train_diffusion_zf,
    train_lpgd,
    train_self_supervised_denoise,
)
from mri_recon.visualization.plots import (
    save_benchmark_dashboard,
    save_psnr_bar_chart,
    save_reconstruction_grid,
)


@dataclass
class BenchmarkConfig:
    """Defaults balance Week 10 notebook fidelity with runnable CPU/GPU time."""

    h: int = 64
    w: int = 64
    num_slices: int = 48
    acceleration: int = 6
    seed: int = 42
    n_train_cap: int = 20
    # Kernel (spec / Week 10 Bayesian-tuned)
    kernel_sigma: float = 2.5
    num_kernels: int = 3844
    kernel_iter: int = 500
    lambda_tv: float = 0.0
    kernel_lr: float = 0.01
    # Supervised CNNs
    cnn_epochs: int = 20
    cnn_lr: float = 1e-3
    unet_ch: int = 32
    res_cnn_ch: int = 32
    # LPGD
    lpgd_unroll: int = 5
    lpgd_ch: int = 32
    # Diffusion (Week 10 notebook)
    diffusion_T: int = 100
    diffusion_epochs: int = 50
    diffusion_time_dim: int = 64
    diffusion_width: int = 64
    diffusion_sample_steps: int | None = None
    # Self-supervised
    n2v_epochs: int = 12
    batch_size: int = 4
    # Week 12 SSDiffRecon (Korkmaz et al. style; training uses Mr k-space loss only)
    ssdiff_epochs: int = 20
    ssdiff_num_blocks: int = 5
    ssdiff_accel: float = 8.0
    ssdiff_split_accel: float = 4.0
    ssdiff_infer_steps: int = 5
    ssdiff_lr: float = 5e-5
    # Real data (notebook-style): directory of DICOMs; if set, ``num_slices`` caps how many are used
    dicom_dir: str | None = None
    dicom_middle_slice_fraction: Tuple[float, float] = (0.1, 0.9)


def benchmark_config_for_presentation(cfg: BenchmarkConfig) -> BenchmarkConfig:
    """Stronger training for demos and final presentations (longer GPU wall time)."""
    return replace(
        cfg,
        kernel_iter=700,
        cnn_epochs=50,
        cnn_lr=8e-4,
        unet_ch=48,
        res_cnn_ch=48,
        lpgd_unroll=8,
        lpgd_ch=48,
        diffusion_T=200,
        diffusion_epochs=80,
        diffusion_time_dim=96,
        diffusion_width=96,
        n2v_epochs=30,
        ssdiff_epochs=50,
        ssdiff_num_blocks=6,
        ssdiff_infer_steps=10,
    )


def _eval_pair(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    p = psnr(pred, gt, data_range=1.0).item()
    s = ssim(pred, gt, data_range=1.0).item()
    return p, s


def run_benchmark(
    cfg: BenchmarkConfig | None = None,
    *,
    quick: bool = False,
    presentation: bool = False,
) -> dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    if presentation and quick:
        raise ValueError("Choose either presentation-quality training (--presentation) or a fast smoke run (--quick), not both.")
    if presentation:
        cfg = benchmark_config_for_presentation(cfg)
    if quick:
        cfg = replace(
            cfg,
            num_slices=16,
            n_train_cap=8,
            num_kernels=400,
            kernel_iter=48,
            cnn_epochs=4,
            n2v_epochs=4,
            diffusion_T=24,
            diffusion_epochs=4,
            batch_size=4,
            ssdiff_epochs=2,
        )

    device = get_device()
    torch.manual_seed(cfg.seed)

    h, w = cfg.h, cfg.w
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError("h and w must be multiples of 8 for U-Net / LPGD grid")

    if cfg.dicom_dir:
        root = Path(cfg.dicom_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"DICOM directory does not exist or is not a folder: {root}\n"
                "  Create it and add .dcm files, e.g. on the server:\n"
                f"    mkdir -p {root} && rsync -avz your_mac:/path/to/dicoms/ {root}/\n"
                "  Then pass --dicom to that same path."
            )
        _datasets, y_full = load_dicom_series_from_dir(root)
        ntot = y_full.shape[0]
        lo = max(0, int(cfg.dicom_middle_slice_fraction[0] * ntot))
        hi = min(ntot, int(cfg.dicom_middle_slice_fraction[1] * ntot))
        if hi <= lo + 1:
            raise ValueError("DICOM series too short for middle-slice selection")
        pool = list(range(lo, hi))
        ns = min(cfg.num_slices, len(pool))
        if len(pool) > ns:
            step = max(1, len(pool) // ns)
            sel = pool[::step][:ns]
        else:
            sel = pool
        y = y_full[sel]
        # Older loaders stacked [S,1,1,H,W]; current loader uses [S,1,H,W]
        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)
        if y.shape[2] != h or y.shape[3] != w:
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    else:
        y = synthetic_phantom_stack(cfg.num_slices, h, w, seed=cfg.seed)

    mask = build_mask(h, w, acceleration=cfg.acceleration, seed=cfg.seed + 1)
    x_zf, k_obs, _k_full = undersample_stack(y, mask)

    n_vol = y.shape[0]
    tr, _va, te = train_val_test_indices(n_vol, seed=SPLIT_SEED)
    train_idx = subsample_train_indices(tr, n_train=min(cfg.n_train_cap, len(tr)), seed=42)
    test_i = te[0]

    y_te = y[test_i : test_i + 1].to(device)
    x_te = x_zf[test_i : test_i + 1].to(device)
    k_te = k_obs[test_i : test_i + 1].to(device)
    mask_d = mask.to(device)

    rows: list[dict[str, Any]] = []
    recon_panels: list[tuple[str, torch.Tensor]] = [
        ("Ground truth", y_te[0, 0]),
        ("Zero-filled", x_te[0, 0]),
    ]

    # --- zero-filled ---
    p, s = _eval_pair(x_te, y_te)
    rows.append(
        {"method": "zero_filled", "label": "Zero-filled FFT", "psnr": p, "ssim": s, "R": cfg.acceleration}
    )

    yo = k_obs[test_i, 0].to(device)
    m_cpu = mask.to(yo.device)

    # --- Gaussian kernel ---
    t0 = time.perf_counter()
    x_g = reconstruct_sparse_kernel(
        yo,
        m_cpu,
        sigma=cfg.kernel_sigma,
        num_kernels=cfg.num_kernels,
        lambda_tv=cfg.lambda_tv,
        max_iter=cfg.kernel_iter,
        lr=cfg.kernel_lr,
        kind="gaussian",
    )
    t_kernel_g = time.perf_counter() - t0
    x_gauss = x_g.unsqueeze(0).unsqueeze(0)
    p, s = _eval_pair(x_gauss, y_te)
    rows.append(
        {
            "method": "gaussian_kernel",
            "label": "Gaussian kernel",
            "psnr": p,
            "ssim": s,
            "R": cfg.acceleration,
            "time_s": t_kernel_g,
        }
    )
    recon_panels.append(("Gaussian kernel", x_gauss[0, 0]))

    # --- Laplacian kernel ---
    t0 = time.perf_counter()
    x_l = reconstruct_sparse_kernel(
        yo,
        m_cpu,
        sigma=cfg.kernel_sigma * 0.8,
        num_kernels=cfg.num_kernels,
        lambda_tv=cfg.lambda_tv,
        max_iter=cfg.kernel_iter,
        lr=cfg.kernel_lr,
        kind="laplacian",
    )
    t_kernel_l = time.perf_counter() - t0
    x_lap = x_l.unsqueeze(0).unsqueeze(0)
    p, s = _eval_pair(x_lap, y_te)
    rows.append(
        {
            "method": "laplacian_kernel",
            "label": "Laplacian kernel",
            "psnr": p,
            "ssim": s,
            "R": cfg.acceleration,
            "time_s": t_kernel_l,
        }
    )
    recon_panels.append(("Laplacian kernel", x_lap[0, 0]))

    # --- ZF + Residual CNN + DC (Week 10 style) ---
    zf_res = ZFResidualCNN_DC(ch=cfg.res_cnn_ch).to(device)
    train_dc_model(
        zf_res,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs=cfg.cnn_epochs,
        lr=cfg.cnn_lr,
        batch_size=cfg.batch_size,
    )
    zf_res.eval()
    with torch.no_grad():
        pred_zr = zf_res(x_te, k_te, mask_d)
    p, s = _eval_pair(pred_zr, y_te)
    rows.append(
        {"method": "zf_residual_cnn_dc", "label": "ZF + ResCNN + DC", "psnr": p, "ssim": s, "R": cfg.acceleration}
    )
    recon_panels.append(("ZF + ResCNN + DC", pred_zr[0, 0]))

    # --- U-Net + DC ---
    unet = UNet(ch=cfg.unet_ch).to(device)
    train_dc_model(
        unet,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs=cfg.cnn_epochs,
        lr=cfg.cnn_lr,
        batch_size=cfg.batch_size,
    )
    unet.eval()
    with torch.no_grad():
        pred_u = unet(x_te, k_te, mask_d)
    p, s = _eval_pair(pred_u, y_te)
    rows.append({"method": "unet_dc", "label": "U-Net + DC", "psnr": p, "ssim": s, "R": cfg.acceleration})
    recon_panels.append(("U-Net + DC", pred_u[0, 0]))

    # --- SRCNN + DC ---
    srcnn = SRCNN_DC().to(device)
    train_dc_model(
        srcnn,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        epochs=cfg.cnn_epochs + max(2, cfg.cnn_epochs // 6),
        lr=cfg.cnn_lr * 0.5,
        batch_size=cfg.batch_size,
    )
    srcnn.eval()
    with torch.no_grad():
        pred_s = srcnn(x_te, k_te, mask_d)
    p, s = _eval_pair(pred_s, y_te)
    rows.append({"method": "srcnn_dc", "label": "SRCNN + DC", "psnr": p, "ssim": s, "R": cfg.acceleration})
    recon_panels.append(("SRCNN + DC", pred_s[0, 0]))

    # --- LPGD ---
    lpgd = LPGD(num_steps=cfg.lpgd_unroll, prox_ch=cfg.lpgd_ch).to(device)
    train_lpgd(
        lpgd,
        x_zf,
        y,
        k_obs,
        mask,
        train_idx,
        device,
        cfg.cnn_epochs,
        cfg.cnn_lr,
        cfg.batch_size,
    )
    lpgd.eval()
    with torch.no_grad():
        pred_l = lpgd(x_te, k_te, mask_d)
    p, s = _eval_pair(pred_l, y_te)
    rows.append({"method": "lpgd", "label": "LPGD", "psnr": p, "ssim": s, "R": cfg.acceleration})
    recon_panels.append(("LPGD", pred_l[0, 0]))

    # --- ZF + Diffusion + DC (Week 10) ---
    schedule = DDPMSchedule(T=cfg.diffusion_T).to(device)
    diff = DiffusionZFDenoiser(
        time_dim=cfg.diffusion_time_dim,
        ch=cfg.diffusion_width,
    ).to(device)
    train_diffusion_zf(
        diff,
        schedule,
        x_zf,
        y,
        train_idx,
        device,
        epochs=cfg.diffusion_epochs,
        lr=cfg.cnn_lr,
        batch_size=cfg.batch_size,
    )
    diff.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_d = diffusion_sample_with_dc_zf(
            diff,
            x_te,
            k_te,
            mask_d,
            schedule,
            num_steps=cfg.diffusion_sample_steps,
            start_from_zf=True,
        )
    t_diff = time.perf_counter() - t0
    p, s = _eval_pair(pred_d, y_te)
    rows.append(
        {
            "method": "zf_diffusion_dc",
            "label": "ZF + Diffusion + DC",
            "psnr": p,
            "ssim": s,
            "R": cfg.acceleration,
            "time_s": t_diff,
        }
    )
    recon_panels.append(("ZF + Diffusion + DC", pred_d[0, 0]))

    # --- Noise2Void-style (no GT in training) ---
    n2v = SmallDenoiseNet(ch=32).to(device)
    train_self_supervised_denoise(
        n2v,
        x_zf,
        train_idx,
        device,
        epochs=cfg.n2v_epochs,
        lr=cfg.cnn_lr,
        batch_size=cfg.batch_size,
    )
    n2v.eval()
    with torch.no_grad():
        pred_n = n2v(x_te)
    p, s = _eval_pair(pred_n, y_te)
    rows.append(
        {"method": "noise2void_style", "label": "Noise2Void-style", "psnr": p, "ssim": s, "R": cfg.acceleration}
    )
    recon_panels.append(("Noise2Void-style", pred_n[0, 0]))

    # --- SSDiffRecon (Week 12: Mp/Mr, unrolled blocks, no pixel-GT loss) ---
    ss_model = SSDiffReconModel(num_blocks=cfg.ssdiff_num_blocks).to(device)
    train_ssdiff_recon(
        ss_model,
        y.cpu(),
        train_idx,
        device,
        num_epochs=cfg.ssdiff_epochs,
        batch_size=1,
        lr=cfg.ssdiff_lr,
        accel=cfg.ssdiff_accel,
        center_fraction=0.08,
        mask_accel=cfg.ssdiff_split_accel,
        warmup_epochs=min(20, max(1, cfg.ssdiff_epochs // 2)),
        seed=cfg.seed,
    )
    ss_model.eval()
    t1 = time.perf_counter()
    with torch.no_grad():
        pred_ss = reconstruct_inference(
            ss_model,
            k_te,
            mask_d,
            num_steps=cfg.ssdiff_infer_steps,
            accel_rate=cfg.ssdiff_accel,
        )
    t_ss_infer = time.perf_counter() - t1
    p, s = _eval_pair(pred_ss, y_te)
    rows.append(
        {
            "method": "ssdiffrecon",
            "label": "SSDiffRecon (Week 12)",
            "psnr": p,
            "ssim": s,
            "R": cfg.acceleration,
            "time_s": t_ss_infer,
        }
    )
    recon_panels.append(("SSDiffRecon (Week 12)", pred_ss[0, 0]))

    # --- save CSV ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["method", "label", "R", "psnr", "ssim", "time_s"]
        wtr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        wtr.writeheader()
        for row in rows:
            wtr.writerow({k: row.get(k, "") for k in fieldnames})

    # --- figures (cohesive dashboard + components) ---
    labels = [r.get("label", r["method"]) for r in rows]
    psnrs = [r["psnr"] for r in rows]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data_note = f"DICOM: {cfg.dicom_dir}" if cfg.dicom_dir else "Synthetic phantom"
    subtitle = (
        f"{data_note}  |  R={cfg.acceleration}×  |  train slices={len(train_idx)}  |  "
        f"diffusion T={cfg.diffusion_T}, epochs={cfg.diffusion_epochs}  |  "
        f"SSDiff epochs={cfg.ssdiff_epochs}"
        + ("  |  **quick**" if quick else "")
        + ("  |  **presentation**" if presentation else "")
    )
    save_benchmark_dashboard(
        labels,
        psnrs,
        recon_panels,
        FIGURES_DIR / "benchmark_dashboard.png",
        title="MRI reconstruction — full method comparison",
        subtitle=subtitle,
        ncols_grid=5,
    )
    save_psnr_bar_chart(
        labels,
        psnrs,
        FIGURES_DIR / "benchmark_psnr.png",
        title=f"PSNR (dB) — R={cfg.acceleration}×, test slice",
    )
    save_reconstruction_grid(
        recon_panels,
        FIGURES_DIR / "benchmark_reconstructions.png",
        ncols=5,
        dpi=165,
        suptitle=f"Reconstructions (same test slice, R={cfg.acceleration}×)",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snap_panels = {lbl: t.detach().cpu() for lbl, t in recon_panels}
    torch.save(
        {
            "ground_truth": recon_panels[0][1].detach().cpu(),
            "panels": snap_panels,
            "acceleration": cfg.acceleration,
            "train_slices": len(train_idx),
            "dicom_dir": cfg.dicom_dir,
            "quick": quick,
            "presentation": presentation,
            "rows": rows,
        },
        RESULTS_DIR / "benchmark_snapshot.pt",
    )

    return {
        "device": str(device),
        "csv": str(csv_path),
        "snapshot": str(RESULTS_DIR / "benchmark_snapshot.pt"),
        "figures": [
            str(FIGURES_DIR / "benchmark_dashboard.png"),
            str(FIGURES_DIR / "benchmark_psnr.png"),
            str(FIGURES_DIR / "benchmark_reconstructions.png"),
        ],
        "rows": rows,
        "quick": quick,
        "presentation": presentation,
    }
