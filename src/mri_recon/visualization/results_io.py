"""Load metrics CSV and prediction tensors from ``results/`` trees."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path
from typing import Any

import torch


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_std_psnr(rows: list[dict[str, Any]]) -> tuple[float, float]:
    ps = []
    for r in rows:
        if "psnr" in r and r["psnr"] not in ("", None):
            try:
                ps.append(float(r["psnr"]))
            except (TypeError, ValueError):
                continue
    if not ps:
        return float("nan"), 0.0
    m = sum(ps) / len(ps)
    s = statistics.stdev(ps) if len(ps) > 1 else 0.0
    return m, s


def load_torch(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def method_slug_from_metrics_name(name: str) -> str:
    return name.replace("_metrics.csv", "").replace(".csv", "")


def discover_metrics_files(directory: Path) -> dict[str, Path]:
    """Map slug -> path for files matching ``*_metrics.csv`` excluding summary."""
    out: dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for p in directory.iterdir():
        if p.suffix == ".csv" and "metrics" in p.name.lower() and "summary" not in p.name.lower():
            slug = p.name.replace("_metrics.csv", "").replace("-metrics.csv", "")
            out[slug] = p
    return out


def parse_data_sweep_n(folder_name: str) -> int | None:
    m = re.match(r"(\d+)slices", folder_name)
    if m:
        return int(m.group(1))
    return None
