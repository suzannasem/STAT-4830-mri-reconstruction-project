"""
Command-line entry: demo pipeline, run tests, ensure output directories.

Usage:
  mri-recon              # same as demo
  mri-recon demo
  mri-recon test
  mri-recon all          # tests then demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from mri_recon.config import (
    ACCELERATION_FACTORS,
    DATA_CACHE_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TRAIN_SLICE_COUNTS,
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
    """Create cache, results, and figures dirs from config."""
    for p in (DATA_CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)


def run_demo(h: int = 128, w: int = 128, num_slices: int = 8, seed: int = 0) -> int:
    """
    Synthetic end-to-end check: mask per R, undersample, PSNR/SSIM vs ground truth.
    """
    ensure_directories()
    print()
    print("  MRI reconstruction — pipeline demo (synthetic data)")
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
    print("  Done. Experiment runners (acceleration / data_budget / leaderboard)")
    print("  are next to wire to trained models — see src/mri_recon/experiments/")
    print()
    return 0


def run_tests() -> int:
    """Run pytest against the project tests/ directory."""
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
        description="MRI reconstruction project: demo pipeline, run tests.",
    )
    sub = parser.add_subparsers(dest="command", help="Command (default: demo)")

    sub.add_parser("demo", help="Synthetic k-space demo + metrics table")
    sub.add_parser("test", help="Run pytest on tests/")
    sub.add_parser("all", help="Run tests, then demo")

    args = parser.parse_args(argv)
    cmd = args.command
    if cmd is None or cmd == "demo":
        return run_demo()
    if cmd == "test":
        return run_tests()
    if cmd == "all":
        code = run_tests()
        if code != 0:
            return code
        return run_demo()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
