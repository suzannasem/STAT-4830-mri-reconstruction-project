#!/usr/bin/env python3
"""
Build comparison figures from ``combined_summary.json`` produced by ``run_all_ipynb.py``.

  python scripts/notebook_pipeline/visualize_notebook_runs.py \\
      --run results/notebook_runs/20260409_120000

  # or latest run under results/notebook_runs/
  python scripts/notebook_pipeline/visualize_notebook_runs.py --latest
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import notebook_batch_utils as _nbu  # noqa: E402


def _load_summary(run_root: Path) -> list[dict]:
    path = run_root / "combined_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("combined_summary.json must be a list")
    return data


def render_comparison_figures(run_root: Path) -> None:
    rows = _load_summary(run_root)
    out_dir = run_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [_nbu.methodology_label_from_row(r) for r in rows]
    ok = [r.get("exit_code") == 0 for r in rows]
    durations = [float(r.get("duration_s") or 0) for r in rows]

    # --- 1) Duration + success ---
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(rows))))
    colors = ["#2ca02c" if o else "#d62728" for o in ok]
    y = np.arange(len(labels))
    ax.barh(y, durations, color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Runtime (s)")
    ax.set_title("Run duration by method (green=exit 0, red=error/timeout)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_status.png", dpi=150)
    plt.close(fig)

    def _row_merged_metrics(r: dict) -> dict[str, float]:
        m: dict[str, float] = dict(r.get("scraped_metrics") or {})
        mf = r.get("metrics_file")
        if isinstance(mf, dict):
            for key in ("psnr", "ssim", "nmse", "loss", "psnr_db", "mse", "ss_loss"):
                v = mf.get(key)
                if isinstance(v, (int, float)):
                    m.setdefault(key, float(v))
            pdb = mf.get("psnr_db")
            if isinstance(pdb, (int, float)) and "psnr" not in m:
                m["psnr"] = float(pdb)
        return m

    merged_json_metrics: list[dict[str, float]] = []
    for r in rows:
        merged_json_metrics.append(_row_merged_metrics(r))

    sanitized_metrics: list[dict[str, float]] = [
        _nbu.sanitize_reconstruction_metrics(m) for m in merged_json_metrics
    ]

    # Per-method bars when notebooks emit ``methods`` (see notebook_batch_utils)
    method_labels: list[str] = []
    method_metrics: list[dict[str, float]] = []
    for r in rows:
        slug = r.get("slug") or "nb"
        disp = _nbu.methodology_label_from_row(r)
        for md in r.get("methods") or []:
            if not isinstance(md, dict):
                continue
            mid = md.get("method_id") or md.get("name") or "method"
            method_labels.append(f"{disp} — {mid}")
            mm: dict[str, float] = {}
            for k, v in md.items():
                if k in ("method_id", "name"):
                    continue
                if isinstance(v, (int, float)):
                    mm[k] = float(v)
            pdb = md.get("psnr_db")
            if isinstance(pdb, (int, float)) and "psnr" not in mm:
                mm["psnr"] = float(pdb)
            method_metrics.append(mm)

    quality_keys = ["psnr", "ssim"]
    q_metrics = [k for k in quality_keys if any(k in m for m in sanitized_metrics)]

    if q_metrics:
        n_notebooks = len(rows)
        n_met = len(q_metrics)
        x = np.arange(n_notebooks)
        w = 0.8 / max(n_met, 1)
        fig, ax = plt.subplots(figsize=(max(9, n_notebooks * 1.0), 5.2))
        for i, mn in enumerate(q_metrics):
            vals = [m.get(mn) for m in sanitized_metrics]
            heights = [v if v is not None else np.nan for v in vals]
            ax.bar(
                x + (i - (n_met - 1) / 2) * w,
                np.nan_to_num(heights, nan=0.0),
                width=w,
                label=mn.upper(),
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("PSNR (dB) / SSIM (0–1)")
        ax.set_title(
            "Reconstruction quality (sanitized: PSNR 8–55 dB, SSIM 0–1; missing = bar absent)"
        )
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_by_notebook.png", dpi=150)
        plt.close(fig)

    loss_any = any("loss" in m for m in sanitized_metrics)
    if loss_any:
        n_notebooks = len(rows)
        x = np.arange(n_notebooks)
        fig, ax = plt.subplots(figsize=(max(9, n_notebooks * 1.0), 4))
        vals = [m.get("loss") for m in sanitized_metrics]
        heights = [v if v is not None else np.nan for v in vals]
        ax.bar(x, np.nan_to_num(heights, nan=0.0), color="#9467bd", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("Training / val loss (sanitized ≤5)")
        ax.set_title("Loss (separate scale from PSNR/SSIM — only small losses kept)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_loss_only.png", dpi=150)
        plt.close(fig)

    n_notebooks = len(rows)

    if method_labels and method_metrics:
        met_names = sorted({k for m in method_metrics for k in m.keys()})
        if met_names:
            n_m = len(method_labels)
            n_met = len(met_names)
            x = np.arange(n_m)
            w = 0.8 / max(n_met, 1)
            fig, ax = plt.subplots(figsize=(max(10, n_m * 0.55), 5))
            for i, mn in enumerate(met_names):
                vals = [m.get(mn) for m in method_metrics]
                heights = [v if v is not None else 0.0 for v in vals]
                ax.bar(
                    x + (i - (n_met - 1) / 2) * w,
                    heights,
                    width=w,
                    label=mn.upper(),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Value")
            ax.set_title("Per-method metrics (from methods[] / MRI_METHOD_RESULT)")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "metrics_by_method.png", dpi=150)
            plt.close(fig)

    hm_cols = [c for c in ("psnr", "ssim") if any(c in m for m in sanitized_metrics)]
    if hm_cols:
        n_hm = len(hm_cols)
        mat = np.full((n_notebooks, n_hm), np.nan)
        for i, m in enumerate(sanitized_metrics):
            for j, mn in enumerate(hm_cols):
                v = m.get(mn)
                if v is not None:
                    mat[i, j] = float(v)
        col_min = np.nanmin(mat, axis=0)
        col_max = np.nanmax(mat, axis=0)
        denom = np.where(col_max - col_min < 1e-12, 1.0, col_max - col_min)
        norm = (mat - col_min) / denom
        norm = np.clip(np.nan_to_num(norm, nan=0.5), 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(max(4.5, n_hm * 1.4), max(4, n_notebooks * 0.5)))
        im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_hm))
        ax.set_xticklabels([c.upper() for c in hm_cols])
        ax.set_yticks(np.arange(n_notebooks))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Quality heatmap (column min–max; PSNR & SSIM only; higher = better)")
        fig.colorbar(im, ax=ax, fraction=0.03, label="normalized within column")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_heatmap.png", dpi=150)
        plt.close(fig)

    # --- Runtime vs PSNR (sanitized) ---
    psnrs = [m.get("psnr") for m in sanitized_metrics]
    ssims = [m.get("ssim") for m in sanitized_metrics]

    def _is_num(x: object) -> bool:
        return isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x))

    if any(_is_num(v) for v in psnrs):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(len(rows)):
            px = psnrs[i]
            py = durations[i]
            if not _is_num(px):
                continue
            ssim = ssims[i]
            size = 80 + 420 * float(ssim) if _is_num(ssim) else 120
            c = "#2ca02c" if ok[i] else "#d62728"
            ax.scatter(float(px), py, s=size, alpha=0.55, c=c, edgecolors="k", linewidths=0.4)
            ax.annotate(labels[i], (float(px), py), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("PSNR (dB, sanitized 8–55)")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime vs PSNR (bubble size ∝ SSIM when available)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "compute_vs_quality.png", dpi=150)
        plt.close(fig)

    # --- PSNR / SSIM horizontal bars (readable alternative to radar) ---
    if any("psnr" in m for m in sanitized_metrics) or any("ssim" in m for m in sanitized_metrics):
        fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.32 * n_notebooks)))
        y = np.arange(n_notebooks)
        psnr_vals = [sanitized_metrics[i].get("psnr") for i in range(n_notebooks)]
        ssim_vals = [sanitized_metrics[i].get("ssim") for i in range(n_notebooks)]
        axes[0].barh(y, [v if v is not None else np.nan for v in psnr_vals], color="#ff7f0e", height=0.65)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_xlabel("PSNR (dB)")
        axes[0].set_title("PSNR (sanitized)")
        axes[0].grid(True, axis="x", alpha=0.3)
        axes[1].barh(y, [v if v is not None else np.nan for v in ssim_vals], color="#2ca02c", height=0.65)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("SSIM")
        axes[1].set_title("SSIM (sanitized)")
        axes[1].grid(True, axis="x", alpha=0.3)
        fig.suptitle("Quality by methodology", y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_bars_psnr_ssim.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Polar chart: PSNR & SSIM only (same data as quality bars; one subplot per method)
    rcols = [m for m in ("psnr", "ssim") if any(m in x for x in sanitized_metrics)]
    if len(rcols) >= 2:
        row_cap = min(n_notebooks, 8)
        range_lo = {mn: 1e9 for mn in rcols}
        range_hi = {mn: -1e9 for mn in rcols}
        for i in range(row_cap):
            for mn in rcols:
                v = sanitized_metrics[i].get(mn)
                if v is not None:
                    range_lo[mn] = min(range_lo[mn], float(v))
                    range_hi[mn] = max(range_hi[mn], float(v))
        for mn in rcols:
            if range_hi[mn] < range_lo[mn] + 1e-12:
                range_lo[mn], range_hi[mn] = 0.0, 1.0
        ncols_fig = min(4, row_cap)
        nrows_fig = int(np.ceil(row_cap / ncols_fig))
        fig, axes = plt.subplots(
            nrows_fig,
            ncols_fig,
            figsize=(3.2 * ncols_fig, 3.0 * nrows_fig),
            subplot_kw=dict(polar=True),
        )
        if row_cap == 1:
            axes = np.array([axes])
        axes_flat = np.atleast_1d(axes).flatten()
        angles = np.linspace(0, 2 * np.pi, len(rcols), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        for idx in range(row_cap):
            ax = axes_flat[idx]
            m = sanitized_metrics[idx]
            vals = []
            for mn in rcols:
                v = m.get(mn)
                lo, hi = range_lo[mn], range_hi[mn]
                if v is None or hi - lo < 1e-12:
                    vals.append(0.0)
                else:
                    vals.append(float(np.clip((float(v) - lo) / (hi - lo), 0, 1)))
            vals.append(vals[0])
            ax.plot(angles, vals, "o-", linewidth=1.5)
            ax.fill(angles, vals, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([s.upper() for s in rcols], size=7)
            ax.set_title(labels[idx][:42] + ("…" if len(labels[idx]) > 42 else ""), size=8)
        for j in range(row_cap, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("PSNR vs SSIM (min–max normalized across methods in this run)", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "radar_multimetric.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    csv_path = out_dir / "runs_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["notebook", "slug", "methodology", "exit_code", "duration_s", "scraped_json", "methods_json"]
        )
        for r in rows:
            w.writerow(
                [
                    r.get("notebook", ""),
                    r.get("slug", ""),
                    _nbu.methodology_label_from_row(r),
                    r.get("exit_code", ""),
                    r.get("duration_s", ""),
                    json.dumps(r.get("scraped_metrics") or {}, sort_keys=True),
                    json.dumps(r.get("methods") or [], sort_keys=True),
                ]
            )
    print(f"Wrote figures under {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot notebook batch comparison figures")
    ap.add_argument("--run", type=Path, default=None, help="run directory with combined_summary.json")
    ap.add_argument("--latest", action="store_true", help="use newest results/notebook_runs/*")
    args = ap.parse_args(argv)

    if args.latest:
        base = _REPO_ROOT / "results" / "notebook_runs"
        if not base.is_dir():
            print(f"No {base}", file=sys.stderr)
            return 1
        candidates = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No run folders found", file=sys.stderr)
            return 1
        run_root = candidates[-1]
    elif args.run is not None:
        run_root = args.run.expanduser().resolve()
    else:
        ap.print_help()
        return 1

    # Re-scrape executed notebooks so viz stays useful if summary lacked scrapes
    summary_path = run_root / "combined_summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        updated = False
        for row in data:
            ep = row.get("executed_path")
            if ep:
                p = Path(ep)
                if p.is_file():
                    fresh = _nbu.scrape_metrics_from_executed_notebook(p)
                    if fresh and fresh != row.get("scraped_metrics"):
                        row["scraped_metrics"] = fresh
                        updated = True
                    merged_m: list[dict] = []
                    mf = _nbu.load_methods_summary_json(p.parent / "methods_summary.json")
                    if mf:
                        merged_m.extend(mf)
                    from_nb = _nbu.scrape_methods_from_executed_notebook(p)
                    seen = {m.get("method_id") for m in merged_m if m.get("method_id")}
                    for m in from_nb:
                        mid = m.get("method_id")
                        if mid and mid in seen:
                            continue
                        if mid:
                            seen.add(mid)
                        merged_m.append(m)
                    norm = [_nbu.normalize_method_dict(m) for m in merged_m]
                    if norm != row.get("methods"):
                        row["methods"] = norm
                        updated = True
        if updated:
            summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    render_comparison_figures(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
