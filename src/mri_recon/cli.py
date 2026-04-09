"""
Command-line entry: cohesive notebook-style pipeline (benchmark + figures + optional sweeps).

Usage:
  python run.py                      # full benchmark + publication figures
  python run.py --quick              # fast smoke
  python run.py --dicom /path/to/dcm # real slices: 256×256, spec train budget (override with flags)
  python run.py --experiments        # also head-to-head + accel + data sweeps (long)
  python run.py visualize            # regenerate figures from results/ only
  python run.py demo                 # zero-filled sweep only
  python run.py test                 # pytest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace

import torch

from mri_recon.config import (
    ACCELERATION_FACTORS,
    DATA_CACHE_DIR,
    FIGURES_DIR,
    OUT_SIZE,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPEC_N_TRAIN,
    TRAIN_SLICE_COUNTS,
    get_device,
)
from mri_recon.data_pipeline import (
    build_mask,
    subsample_train_indices,
    synthetic_phantom_stack,
    train_val_test_indices,
    undersample_stack,
)
from mri_recon.metrics import psnr, ssim


def ensure_directories() -> None:
    import os

    # Cloud images sometimes disallow ~/.config/matplotlib; use repo-local or /tmp.
    for mpl_dir in (PROJECT_ROOT / ".matplotlib",):
        try:
            mpl_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
            break
        except OSError:
            pass
    else:
        tmp_mpl = os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib-mri-recon")
        os.makedirs(tmp_mpl, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", tmp_mpl)

    for p in (DATA_CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)


def run_visualize_only() -> int:
    ensure_directories()
    from mri_recon.visualize import generate_publication_figures

    print()
    print("  Regenerating publication figures from results/")
    print("  " + "-" * 52)
    paths = generate_publication_figures()
    for k, v in paths.items():
        print(f"    {k}: {v or '(skipped — missing inputs)'}")
    print()
    return 0


def run_experiment_sweeps(*, quick: bool = False) -> None:
    from mri_recon.experiments.acceleration_sweep import run_acceleration_sweep
    from mri_recon.experiments.data_efficiency_sweep import run_data_efficiency_sweep
    from mri_recon.experiments.head_to_head import run_head_to_head

    print("  Running head-to-head (spec Exp 3) …")
    run_head_to_head(fast=quick)
    print("  Running acceleration sweep (spec Exp 1) …")
    run_acceleration_sweep(fast=quick)
    print("  Running data-efficiency sweep (spec Exp 2) …")
    run_data_efficiency_sweep(fast=quick)


def run_full(
    *,
    quick: bool = False,
    dicom_dir: str | None = None,
    experiments: bool = False,
    benchmark_height: int | None = None,
    benchmark_width: int | None = None,
    n_train: int | None = None,
    num_slices: int | None = None,
) -> int:
    """Train/eval all reconstructors; save snapshot + CSV; build publication figures."""
    ensure_directories()
    from mri_recon.benchmark import BenchmarkConfig, run_benchmark
    from mri_recon.visualize import generate_publication_figures

    cfg = BenchmarkConfig()
    if dicom_dir:
        # Notebook-style real data: 256², spec train cap, enough slices for ~67 train indices.
        h0, w0 = OUT_SIZE
        cfg = replace(
            cfg,
            dicom_dir=dicom_dir,
            h=h0,
            w=w0,
            n_train_cap=SPEC_N_TRAIN,
            num_slices=120,
        )
    if benchmark_height is not None:
        cfg = replace(cfg, h=benchmark_height)
    if benchmark_width is not None:
        cfg = replace(cfg, w=benchmark_width)
    if n_train is not None:
        cfg = replace(cfg, n_train_cap=n_train)
    if num_slices is not None:
        cfg = replace(cfg, num_slices=num_slices)

    print()
    print("  MRI reconstruction — cohesive run (benchmark + figures)")
    print("  " + "-" * 52)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Device:       {get_device()}")
    print(f"  Results:      {RESULTS_DIR / 'benchmark.csv'}")
    print(f"  Snapshot:     {RESULTS_DIR / 'benchmark_snapshot.pt'}")
    print(f"  Figures:      {FIGURES_DIR}/  and  results/figures/")
    if dicom_dir:
        print(f"  Data:         DICOM dir {dicom_dir}")
        print(f"  Grid:         {cfg.h}×{cfg.w}  |  train cap {cfg.n_train_cap}  |  up to {cfg.num_slices} slices")
    if experiments:
        print("  Also running: head-to-head + accel + data sweeps (long).")
    print()
    out = run_benchmark(cfg, quick=quick)
    print("  Publication figures (from available results) …")
    fig_paths = generate_publication_figures()
    for k, v in sorted(fig_paths.items()):
        print(f"    {k}: {v or '(skipped)'}")

    if experiments:
        print()
        run_experiment_sweeps(quick=quick)
        print("  Refreshing publication figures after sweeps …")
        fig_paths = generate_publication_figures()
        for k, v in sorted(fig_paths.items()):
            print(f"    {k}: {v or '(skipped)'}")

    print()
    print("  Methods evaluated:")
    for row in out["rows"]:
        print(
            f"    {row['method']:<22}  PSNR {row['psnr']:7.2f} dB   SSIM {row['ssim']:.4f}"
        )
    print()
    print("  Done.")
    print()
    return 0


def run_demo(h: int = 128, w: int = 128, num_slices: int = 8, seed: int = 0) -> int:
    ensure_directories()
    print()
    print("  MRI reconstruction — pipeline demo (zero-filled sweep only)")
    print("  " + "-" * 52)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Output dirs:  {RESULTS_DIR}, {FIGURES_DIR}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device (info): {device}")
    print()

    y = synthetic_phantom_stack(num_slices, h, w, seed=seed)
    print(f"  Phantom stack: {tuple(y.shape)}  (min={y.min():.4f}, max={y.max():.4f})")
    print()

    n = num_slices
    tr, va, te = train_val_test_indices(n, seed=42)
    print(f"  Train/val/test split: {len(tr)} / {len(va)} / {len(te)} slices")
    sub = subsample_train_indices(tr, n_train=min(8, len(tr)), seed=0)
    print(f"  Example subsample (n_train=8): {len(sub)} indices")
    print()

    print("  Acceleration sweep (zero-filled vs GT):")
    print(f"  {'R':>4}  {'PSNR (dB)':>12}  {'SSIM':>10}")
    print("  " + "-" * 32)
    for r in ACCELERATION_FACTORS:
        mask = build_mask(h, w, acceleration=r, seed=seed + r)
        x_zf, _, _ = undersample_stack(y, mask)
        p = psnr(x_zf, y, data_range=1.0).item()
        s = ssim(x_zf, y, data_range=1.0).item()
        print(f"  {r:>4}  {p:>12.4f}  {s:>10.4f}")

    print()
    print("  Spec training slice counts:", list(TRAIN_SLICE_COUNTS))
    print()
    print("  Run `python run.py` for the full benchmark + publication figures.")
    print()
    return 0


def run_tests() -> int:
    ensure_directories()
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.is_dir():
        print("No tests/ directory found.", file=sys.stderr)
        return 1
    print()
    print("  Running pytest …")
    print("  " + "-" * 52)
    print()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mri-recon",
        description="MRI reconstruction: benchmark, DICOM, experiment sweeps, publication figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Faster benchmark and shorter experiment sweeps",
    )
    parser.add_argument(
        "--dicom",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory of DICOMs (middle slices). Defaults to 256×256, "
            f"n_train_cap={SPEC_N_TRAIN}, num_slices=120 (spec-style); override with --height/--width/--n-train/--num-slices"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="H",
        help="Benchmark image height (default: 64 synthetic; with --dicom: 256 unless set)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="W",
        help="Benchmark image width (default: 64 synthetic; with --dicom: 256 unless set)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        metavar="N",
        help=f"Max training slices (default: 20 synthetic; with --dicom: {SPEC_N_TRAIN})",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=None,
        metavar="N",
        help="Slices to load from volume (default: 48 synthetic; with --dicom: 120)",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="After benchmark, run head-to-head + acceleration + data-efficiency sweeps",
    )
    sub = parser.add_subparsers(dest="command", help="Command (default: full pipeline)")

    sub.add_parser("full", help="Same as default: benchmark + figures")

    sub.add_parser("visualize", help="Regenerate publication figures from results/ only")

    sub.add_parser("demo", help="Zero-filled acceleration sweep only (fast)")

    sub.add_parser("test", help="Run pytest on tests/")

    sub.add_parser("all", help="Run pytest, then full pipeline (no --quick)")

    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "visualize":
        return run_visualize_only()
    if cmd == "demo":
        return run_demo()
    if cmd == "test":
        return run_tests()
    if cmd == "all":
        code = run_tests()
        if code != 0:
            return code
        return run_full(quick=False, dicom_dir=None, experiments=False)

    if cmd is None or cmd == "full":
        return run_full(
            quick=bool(args.quick),
            dicom_dir=args.dicom,
            experiments=bool(args.experiments),
            benchmark_height=args.height,
            benchmark_width=args.width,
            n_train=args.n_train,
            num_slices=args.num_slices,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
