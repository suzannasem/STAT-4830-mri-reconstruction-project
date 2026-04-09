"""
Experiment 1: acceleration sweep 4× / 6× / 8× (Experiment Spec §3).

Writes under ``results/accel_sweep/{4x,6x,8x}/`` (same artifacts as head-to-head).
"""

from __future__ import annotations

import argparse

from mri_recon.config import ACCELERATION_FACTORS, RESULTS_ACCEL_SWEEP, get_device
from mri_recon.experiments.head_to_head import run_head_to_head


def run_acceleration_sweep(*, fast: bool = False) -> dict[str, str]:
    _ = get_device()
    RESULTS_ACCEL_SWEEP.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for r in ACCELERATION_FACTORS:
        sub = RESULTS_ACCEL_SWEEP / f"{r}x"
        sub.mkdir(parents=True, exist_ok=True)
        run_head_to_head(acceleration=r, fast=fast, out_dir=sub)
        out[str(r)] = str(sub.resolve())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 1: acceleration sweep")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    print(run_acceleration_sweep(fast=args.fast))


if __name__ == "__main__":
    main()
