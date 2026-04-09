"""
Publication figures (Experiment Spec §6) — wired to ``results/`` artifacts.

Reads ``benchmark_snapshot.pt``, ``head_to_head/``, ``accel_sweep/``, ``data_sweep/``
and writes PNGs under ``results/figures/`` and ``figures/`` (project root).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mri_recon.config import (
    ACCELERATION_FACTORS,
    FIGURES_DIR,
    RESULTS_ACCEL_SWEEP,
    RESULTS_DATA_SWEEP,
    RESULTS_DIR,
    RESULTS_FIGURES,
    RESULTS_HEAD_TO_HEAD,
)
from mri_recon.visualization.results_io import (
    discover_metrics_files,
    load_torch,
    mean_std_psnr,
    parse_data_sweep_n,
    read_metrics_csv,
)

# --- Tier colors (spec §6) ---
COLOR_BASELINE = "#E07A5F"
COLOR_SUPERVISED = "#3D5A80"
COLOR_DIFFUSION = "#2A9D8F"
COLOR_SS = "#9B5DE5"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_dual(out_results: Path, out_legacy: Path | None, fig) -> None:
    import matplotlib.pyplot as plt

    out_results.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_results, dpi=300, bbox_inches="tight")
    if out_legacy is not None:
        out_legacy.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_legacy, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_gallery_from_results(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "gallery.png",
    slice_index: int = 0,
) -> Path | None:
    """
    Figure 1: rows = acceleration (4×, 6×, 8×), columns = GT + methods (from saved preds).
    Second band per R: error maps (hot). Falls back to ``benchmark_snapshot.pt`` if no sweep.
    """
    plt = _plt()
    acc_dirs = [results_dir / "accel_sweep" / f"{r}x" for r in ACCELERATION_FACTORS]
    if not any(d.is_dir() for d in acc_dirs):
        snap = load_torch(results_dir / "benchmark_snapshot.pt")
        if not snap or "panels" not in snap:
            return None
        panels = [
            (k, v[0, 0] if v.dim() == 4 else v)
            for k, v in snap["panels"].items()
            if "ground" not in k.lower()
        ]
        gt = snap["ground_truth"]
        gtn = gt[0].numpy() if gt.dim() == 3 else gt.numpy()
        ncols = min(5, len(panels) + 1)
        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6))
        axes = np.atleast_2d(axes)
        axes[0, 0].imshow(np.clip(gtn, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("GT")
        axes[0, 0].axis("off")
        for j, (title, t) in enumerate(panels[: ncols - 1]):
            arr = t.numpy() if isinstance(t, torch.Tensor) else t
            axes[0, j + 1].imshow(np.clip(arr, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[0, j + 1].set_title(str(title)[:28], fontsize=7)
            axes[0, j + 1].axis("off")
            err = np.abs(arr - gtn)
            emax = err.max() + 1e-8
            axes[1, j + 1].imshow(err, cmap="hot", vmin=0, vmax=emax)
            axes[1, j + 1].axis("off")
        axes[1, 0].axis("off")
        fig.suptitle("Gallery (benchmark snapshot — run experiments for full accel grid)")
        outp = RESULTS_FIGURES / out_name
        _save_dual(outp, FIGURES_DIR / out_name, fig)
        return outp

    gt_tensor: torch.Tensor | None = None
    for d in acc_dirs:
        gt_path = d / "test_ground_truth.pt"
        if gt_path.is_file():
            o = load_torch(gt_path)
            if "y_test" in o:
                gt_tensor = o["y_test"][slice_index, 0]
                break
    if gt_tensor is None:
        snap = load_torch(results_dir / "benchmark_snapshot.pt")
        if snap and "ground_truth" in snap:
            gt_tensor = snap["ground_truth"]
            if gt_tensor.dim() == 3:
                gt_tensor = gt_tensor[0]

    pred_files = [
        "zero_filled_predictions.pt",
        "gaussian_kernel_predictions.pt",
        "zf_residual_cnn_predictions.pt",
        "unet_predictions.pt",
        "lpgd_predictions.pt",
    ]
    titles = ["Zero-filled", "Kernel", "ZF+ResCNN", "U-Net", "LPGD"]
    ncols = 1 + len(pred_files)
    nrows_acc = len(ACCELERATION_FACTORS)
    fig_h = 2.8 * nrows_acc * 2
    fig_w = 2.5 * ncols
    fig, axes = plt.subplots(nrows_acc * 2, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    gtn = gt_tensor.numpy() if gt_tensor is not None else None

    for ri, R in enumerate(ACCELERATION_FACTORS):
        sub = results_dir / "accel_sweep" / f"{R}x"
        row_recon = ri * 2
        row_err = ri * 2 + 1
        if gtn is not None:
            axes[row_recon, 0].imshow(np.clip(gtn, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row_recon, 0].set_ylabel(f"{R}×", fontsize=10, rotation=0, labelpad=12)
        axes[row_recon, 0].set_title("GT" if ri == 0 else "")
        axes[row_recon, 0].axis("off")

        emax = 1e-8
        preds_row: list[np.ndarray] = []
        hw = (64, 64)
        if gtn is not None:
            hw = gtn.shape
        for pf in pred_files:
            pt_path = sub / pf
            if not pt_path.is_file():
                preds_row.append(np.zeros(hw, dtype=np.float32))
                continue
            data = load_torch(pt_path)
            pr = data["preds"][slice_index, 0].numpy()
            preds_row.append(pr)
            if gtn is not None:
                emax = max(emax, float(np.abs(pr - gtn).max()))
        if emax < 1e-8:
            emax = 1.0

        for j, pr in enumerate(preds_row):
            axes[row_recon, j + 1].imshow(np.clip(pr, 0, 1), cmap="gray", vmin=0, vmax=1)
            if ri == 0:
                axes[row_recon, j + 1].set_title(titles[j], fontsize=8)
            axes[row_recon, j + 1].axis("off")
            ref = gtn if gtn is not None else pr
            err = np.abs(pr - ref)
            axes[row_err, j + 1].imshow(err, cmap="hot", vmin=0, vmax=emax)
            axes[row_err, j + 1].axis("off")

        axes[row_err, 0].axis("off")

    fig.suptitle("Reconstruction gallery (one test slice)", fontsize=12)
    plt.tight_layout()
    outp = RESULTS_FIGURES / out_name
    _save_dual(outp, FIGURES_DIR / out_name, fig)
    return outp


def figure_degradation_curves(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "degradation_curves.png",
) -> Path | None:
    """Figure 2: PSNR vs acceleration with ±1 std from per-slice CSVs."""
    plt = _plt()
    root = results_dir / "accel_sweep"
    if not root.is_dir():
        return None

    # slug -> list of (R, mean_psnr, std_psnr)
    series: dict[str, list[tuple[int, float, float]]] = {}

    for R in ACCELERATION_FACTORS:
        sub = root / f"{R}x"
        if not sub.is_dir():
            continue
        for slug, csvp in discover_metrics_files(sub).items():
            rows = read_metrics_csv(csvp)
            m, s = mean_std_psnr(rows)
            series.setdefault(slug, []).append((R, m, s))

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for slug in sorted(series.keys()):
        items = sorted(series[slug], key=lambda x: x[0])
        if not items:
            continue
        xs = [t[0] for t in items]
        means = [t[1] for t in items]
        stds = [t[2] for t in items]
        label = slug.replace("_", " ").title()
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label, linewidth=1.2)

    ax.set_xticks(list(ACCELERATION_FACTORS))
    ax.set_xlabel("Acceleration factor")
    ax.set_ylabel("Mean test PSNR (dB)")
    ax.set_title("PSNR vs acceleration (±1 std over test slices)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outp = RESULTS_FIGURES / out_name
    _save_dual(outp, FIGURES_DIR / out_name, fig)
    return outp


def figure_data_efficiency(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "data_efficiency.png",
) -> Path | None:
    """Figure 3: supervised lines + horizontal baselines (ZF / kernel / SSDiff from head_to_head)."""
    plt = _plt()
    ds_root = results_dir / "data_sweep"
    if not ds_root.is_dir():
        return None

    folders = sorted(
        [d for d in ds_root.iterdir() if d.is_dir() and parse_data_sweep_n(d.name)],
        key=lambda d: parse_data_sweep_n(d.name) or 0,
    )
    if not folders:
        return None

    xs = [parse_data_sweep_n(d.name) for d in folders]
    supervised_slugs = ["zf_residual_cnn", "unet", "lpgd"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for slug in supervised_slugs:
        ys = []
        for fd in folders:
            csvp = fd / f"{slug}_metrics.csv"
            rows = read_metrics_csv(csvp)
            m, _ = mean_std_psnr(rows)
            ys.append(m)
        if any(math.isfinite(y) for y in ys):
            ax.plot(xs, ys, marker="o", label=slug.replace("_", " ").title())

    h2h = results_dir / "head_to_head"
    zf_m = _mean_label_psnr(h2h / "zero_filled_metrics.csv")
    gk_m = _mean_label_psnr(h2h / "gaussian_kernel_metrics.csv")
    if math.isfinite(zf_m):
        ax.axhline(zf_m, color=COLOR_BASELINE, linestyle="--", label="ZF (baseline)")
    if math.isfinite(gk_m):
        ax.axhline(gk_m, color=COLOR_BASELINE, linestyle=":", label="Kernel")

    summ = h2h / "summary_metrics.csv"
    if summ.is_file():
        for row in read_metrics_csv(summ):
            mname = row.get("method", "")
            if mname and ("SSDiff" in mname or "Week 12" in mname):
                try:
                    v = float(row["psnr"])
                    ax.axhline(v, color=COLOR_SS, linestyle="--", label="SSDiff (ref)")
                except (KeyError, ValueError):
                    pass
                break

    ax.set_xticks(xs)
    ax.set_xlabel("Training slices")
    ax.set_ylabel("Mean test PSNR (dB)")
    ax.set_title("Data efficiency (fixed acceleration in data_sweep runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pth = RESULTS_FIGURES / out_name
    _save_dual(pth, FIGURES_DIR / out_name, fig)
    return pth


def _mean_label_psnr(path: Path) -> float:
    rows = read_metrics_csv(path)
    m, _ = mean_std_psnr(rows)
    return m


def figure_compute_tradeoff(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "compute_tradeoff.png",
) -> Path | None:
    """Figure 4: PSNR vs time/slice (log x), point size ∝ params, Pareto frontier."""
    plt = _plt()
    summ = results_dir / "head_to_head" / "summary_metrics.csv"
    rows = read_metrics_csv(summ)
    if not rows:
        summ = results_dir / "benchmark.csv"
        rows = read_metrics_csv(summ)
    if not rows:
        return None

    times = []
    psnrs = []
    params = []
    labels = []
    _default_n = {
        "zero_filled": 0,
        "gaussian_kernel": 0,
        "laplacian_kernel": 0,
        "zf_residual_cnn_dc": 75_000,
        "unet_dc": 1_200_000,
        "srcnn_dc": 50_000,
        "lpgd": 50_000,
        "zf_diffusion_dc": 130_000,
        "noise2void_style": 20_000,
        "ssdiffrecon": 800_000,
    }
    for r in rows:
        try:
            t = float(r.get("time_per_slice_s", r.get("time_s", 1e-3)) or 1e-3)
            p = float(r["psnr"])
            mid = r.get("method", "")
            n = float(r.get("params") or _default_n.get(mid, 50_000.0))
        except (TypeError, ValueError, KeyError):
            continue
        times.append(max(t, 1e-4))
        psnrs.append(p)
        params.append(max(n, 100.0))
        labels.append(r.get("label", r.get("method", "?"))[:22])

    if not times:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    sz = [300 * math.log10(p / 100 + 1) for p in params]
    ax.scatter(times, psnrs, s=sz, alpha=0.75, c=range(len(times)), cmap="tab10")
    for i, lb in enumerate(labels):
        ax.annotate(lb, (times[i], psnrs[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    pts = sorted(zip(times, psnrs), key=lambda x: x[0])
    pareto: list[tuple[float, float]] = []
    best_p = -1e9
    for t, p in pts:
        if p > best_p:
            pareto.append((t, p))
            best_p = p
    if len(pareto) >= 2:
        ax.plot([x[0] for x in pareto], [x[1] for x in pareto], "k--", alpha=0.5, label="Pareto")

    ax.set_xscale("log")
    ax.set_xlabel("Time / slice (s, log)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("PSNR vs compute (point area ∝ parameters)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = RESULTS_FIGURES / out_name
    _save_dual(p, FIGURES_DIR / out_name, fig)
    return p


def figure_error_maps(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "error_maps.png",
) -> Path | None:
    """Figure 5: |recon − GT| for each method (head_to_head or benchmark snapshot)."""
    plt = _plt()
    h2h = results_dir / "head_to_head"
    gt = None
    snap = load_torch(results_dir / "benchmark_snapshot.pt")
    if snap and "ground_truth" in snap:
        gt = snap["ground_truth"]
        if gt.dim() == 2:
            gt = gt.numpy()
        else:
            gt = gt[0].numpy()

    methods: list[tuple[str, Path]] = []
    if h2h.is_dir():
        for p in sorted(h2h.glob("*_predictions.pt")):
            if "ground" in p.name.lower():
                continue
            slug = p.name.replace("_predictions.pt", "")
            methods.append((slug, p))

    if not methods and snap and "panels" in snap:
        gt_arr = snap["ground_truth"].numpy()
        if gt_arr.ndim == 3:
            gt_arr = gt_arr[0]
        panels = [(k, v[0, 0].numpy()) for k, v in snap["panels"].items() if k != "Ground truth"]
        emax = max(np.abs(pr - gt_arr).max() for _, pr in panels) + 1e-8
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, (name, pr) in zip(axes, panels):
            ax.imshow(np.abs(pr - gt_arr), cmap="hot", vmin=0, vmax=emax)
            ax.set_title(name[:24], fontsize=8)
            ax.axis("off")
        fig.suptitle("|Error| maps (benchmark snapshot)")
        p = RESULTS_FIGURES / out_name
        _save_dual(p, FIGURES_DIR / out_name, fig)
        return p

    if gt is None:
        o = load_torch(h2h / "test_ground_truth.pt")
        if o and "y_test" in o:
            gt = o["y_test"][0, 0].numpy()

    if gt is None:
        return None

    arrs = []
    titles = []
    emax = 0.0
    for name, pth in methods[:12]:
        d = load_torch(pth)
        if "preds" not in d:
            continue
        pr = d["preds"][0, 0].numpy()
        err = np.abs(pr - gt)
        emax = max(emax, err.max())
        arrs.append(err)
        titles.append(name)

    if not arrs:
        return None

    emax += 1e-8
    n = len(arrs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, er, tt in zip(axes, arrs, titles):
        ax.imshow(er, cmap="hot", vmin=0, vmax=emax)
        ax.set_title(tt.replace("_", " ")[:20], fontsize=8)
        ax.axis("off")
    fig.suptitle("|reconstruction − GT| (test slice 0)")
    fig.tight_layout()
    p = RESULTS_FIGURES / out_name
    _save_dual(p, FIGURES_DIR / out_name, fig)
    return p


def figure_boxplots(
    results_dir: Path = RESULTS_DIR,
    out_name: str = "boxplots.png",
) -> Path | None:
    """Figure 6: PSNR distribution per method (head_to_head per-slice CSVs)."""
    plt = _plt()
    h2h = results_dir / "head_to_head"
    if not h2h.is_dir():
        return None

    data = []
    labels = []
    for p in sorted(h2h.glob("*_metrics.csv")):
        if "summary" in p.name.lower():
            continue
        rows = read_metrics_csv(p)
        ps = []
        for r in rows:
            if r.get("psnr"):
                try:
                    ps.append(float(r["psnr"]))
                except ValueError:
                    pass
        if ps:
            data.append(ps)
            labels.append(p.name.replace("_metrics.csv", "").replace("_", "\n")[:18])

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 5))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-slice PSNR distributions (test set)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    p = RESULTS_FIGURES / out_name
    _save_dual(p, FIGURES_DIR / out_name, fig)
    return p


def generate_publication_figures(
    results_dir: Path = RESULTS_DIR,
    *,
    also_write_legacy_figures: bool = True,
) -> dict[str, str | None]:
    """
    Build all six spec figures from on-disk experiment outputs.

    Safe to call after ``run_benchmark`` only: figures that need sweeps are skipped
    until those directories exist.
    """
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    _ = also_write_legacy_figures

    out: dict[str, str | None] = {}
    g = figure_gallery_from_results(results_dir)
    out["gallery"] = str(g) if g else None
    d = figure_degradation_curves(results_dir)
    out["degradation_curves"] = str(d) if d else None
    de = figure_data_efficiency(results_dir)
    out["data_efficiency"] = str(de) if de else None
    ct = figure_compute_tradeoff(results_dir)
    out["compute_tradeoff"] = str(ct) if ct else None
    em = figure_error_maps(results_dir)
    out["error_maps"] = str(em) if em else None
    bx = figure_boxplots(results_dir)
    out["boxplots"] = str(bx) if bx else None
    return out


# Back-compat
generate_all_placeholder = generate_publication_figures

__all__ = [
    "figure_boxplots",
    "figure_compute_tradeoff",
    "figure_data_efficiency",
    "figure_degradation_curves",
    "figure_error_maps",
    "figure_gallery_from_results",
    "generate_all_placeholder",
    "generate_publication_figures",
]
