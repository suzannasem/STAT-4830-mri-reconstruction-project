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


def _short_label(row: dict) -> str:
    return row.get("slug") or row.get("notebook", "?")[:28]


def render_comparison_figures(run_root: Path) -> None:
    rows = _load_summary(run_root)
    out_dir = run_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [_short_label(r) for r in rows]
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
    ax.set_title("Notebook runs: duration (green=ok, red=failed)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_status.png", dpi=150)
    plt.close(fig)

    merged_json_metrics: list[dict[str, float]] = []
    for r in rows:
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
        merged_json_metrics.append(m)

    metric_names = sorted({k for m in merged_json_metrics for k in m.keys()})

    if metric_names:
        n_notebooks = len(rows)
        n_met = len(metric_names)
        x = np.arange(n_notebooks)
        w = 0.8 / max(n_met, 1)
        fig, ax = plt.subplots(figsize=(max(8, n_notebooks * 0.9), 5))
        for i, mn in enumerate(metric_names):
            vals = [m.get(mn) for m in merged_json_metrics]
            heights = [v if v is not None else 0.0 for v in vals]
            ax.bar(
                x + (i - (n_met - 1) / 2) * w,
                heights,
                width=w,
                label=mn.upper(),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Value")
        ax.set_title("Metrics (scraped from outputs + summary JSON)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_by_notebook.png", dpi=150)
        plt.close(fig)

        # --- 3) Normalized heatmap ---
        mat = np.full((n_notebooks, n_met), np.nan)
        for i, m in enumerate(merged_json_metrics):
            for j, mn in enumerate(metric_names):
                v = m.get(mn)
                if v is not None:
                    mat[i, j] = v
        col_min = np.nanmin(mat, axis=0)
        col_max = np.nanmax(mat, axis=0)
        denom = np.where(col_max - col_min < 1e-12, 1.0, col_max - col_min)
        norm = (mat - col_min) / denom
        # Higher is better for psnr/ssim; lower for loss — flip loss column
        lower_better = {"loss", "mse", "ss_loss", "nmse"}
        for j, mn in enumerate(metric_names):
            if mn in lower_better and not np.all(np.isnan(norm[:, j])):
                norm[:, j] = 1.0 - np.nan_to_num(norm[:, j], nan=0.5)

        fig, ax = plt.subplots(figsize=(max(5, n_met * 1.2), max(4, n_notebooks * 0.45)))
        im = ax.imshow(np.nan_to_num(norm, nan=0.0), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_met))
        ax.set_xticklabels([m.upper() for m in metric_names])
        ax.set_yticks(np.arange(n_notebooks))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title("Relative metric strength (row-wise min–max; loss inverted)")
        fig.colorbar(im, ax=ax, fraction=0.03, label="normalized")
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_heatmap.png", dpi=150)
        plt.close(fig)

    # --- 4) Bubble: duration vs PSNR ---
    psnrs = [m.get("psnr") for m in merged_json_metrics]
    ssims = [m.get("ssim") for m in merged_json_metrics]
    if any(v is not None for v in psnrs):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, row in enumerate(rows):
            px = psnrs[i]
            py = durations[i]
            if px is None:
                continue
            ssim = ssims[i]
            size = 80 + 400 * float(ssim) if ssim is not None else 120
            c = "#2ca02c" if ok[i] else "#d62728"
            ax.scatter(px, py, s=size, alpha=0.55, c=c, edgecolors="k", linewidths=0.4)
            ax.annotate(labels[i], (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("PSNR")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime vs quality (bubble area ∝ SSIM when present)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "compute_vs_quality.png", dpi=150)
        plt.close(fig)

    # --- 5) Radar (up to 6 metrics, cap notebook count for layout) ---
    if len(metric_names) >= 2:
        radar_metrics = metric_names[:6]
        row_cap = min(len(rows), 10)
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        fig, axes = plt.subplots(
            1,
            row_cap,
            figsize=(3.0 * row_cap, 3.6),
            subplot_kw=dict(polar=True),
        )
        if row_cap == 1:
            axes = [axes]
        range_lo: dict[str, float] = {}
        range_hi: dict[str, float] = {}
        for mn in radar_metrics:
            vals = [m.get(mn) for m in merged_json_metrics[:row_cap]]
            vals = [float(v) for v in vals if v is not None]
            if len(vals) == 0:
                range_lo[mn] = 0.0
                range_hi[mn] = 1.0
            else:
                range_lo[mn] = min(vals)
                range_hi[mn] = max(vals)
        slice_rows = rows[:row_cap]
        slice_merged = merged_json_metrics[:row_cap]
        slice_labels = labels[:row_cap]
        for ax, row, m, label in zip(axes, slice_rows, slice_merged, slice_labels):
            vals = []
            for mn in radar_metrics:
                v = m.get(mn)
                lo, hi = range_lo[mn], range_hi[mn]
                if v is None or hi - lo < 1e-12:
                    vals.append(0.0)
                else:
                    t = (float(v) - lo) / (hi - lo)
                    if mn in ("loss", "mse", "ss_loss", "nmse"):
                        t = 1.0 - t
                    vals.append(float(np.clip(t, 0, 1)))
            vals += [vals[0]]
            ax.plot(angles, vals, "o-", linewidth=1.5)
            ax.fill(angles, vals, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([s.upper() for s in radar_metrics], size=8)
            ax.set_title(label, size=9)
        fig.suptitle("Notebook radar (per-metric min–max; loss inverted)", y=1.06)
        fig.tight_layout()
        fig.savefig(out_dir / "radar_multimetric.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    csv_path = out_dir / "runs_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["notebook", "slug", "exit_code", "duration_s", "scraped_json"])
        for r in rows:
            w.writerow(
                [
                    r.get("notebook", ""),
                    r.get("slug", ""),
                    r.get("exit_code", ""),
                    r.get("duration_s", ""),
                    json.dumps(r.get("scraped_metrics") or {}, sort_keys=True),
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
        if updated:
            summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    render_comparison_figures(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
