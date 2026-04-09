"""
Central experiment configuration (MRI Reconstruction Experiment Specification).

Single source of truth for paths, device, acceleration factors, mask center
fractions per R, training slice budgets, image size, seeds, and defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# --- Spec axes ---
ACCELERATION_FACTORS: Tuple[int, ...] = (4, 6, 8)
TRAIN_SLICE_COUNTS: Tuple[int, ...] = (8, 16, 32, 67)

# Per-R center fraction (spec table: 4× 8%, 6× 6%, 8× 4%)
ACCELERATION_CENTER_FRACTION: Dict[int, float] = {4: 0.08, 6: 0.06, 8: 0.04}

# Fixed split sizes (spec §3–5)
SPEC_N_TRAIN: int = 67
SPEC_N_TEST: int = 14
# Remaining slices go to validation after train+test are taken from shuffled order.
GLOBAL_SEED: int = 42

# --- Data defaults (UPENN-GBM style, 256×256) ---
DEFAULT_COLLECTION: str = "UPENN-GBM"
OUT_SIZE: Tuple[int, int] = (256, 256)
PERCENTILE_CLIP: Tuple[float, float] = (1.0, 99.0)

# Legacy default: 4× mask (used when acceleration not in map)
MASK_CENTER_FRACTION: float = ACCELERATION_CENTER_FRACTION[4]

# --- Train / val / test ratios (fallback when not using spec split) ---
TRAIN_VAL_TEST_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_SEED: int = GLOBAL_SEED

# --- I/O (spec §7) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_ACCEL_SWEEP: Path = RESULTS_DIR / "accel_sweep"
RESULTS_DATA_SWEEP: Path = RESULTS_DIR / "data_sweep"
RESULTS_HEAD_TO_HEAD: Path = RESULTS_DIR / "head_to_head"
RESULTS_FIGURES: Path = RESULTS_DIR / "figures"


def get_device() -> "torch.device":
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Training loop defaults shared by supervised runs."""

    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 50
    seed: int = GLOBAL_SEED


@dataclass
class KernelConfig:
    """Gaussian kernel sparse reconstruction (spec: σ=2.5, K=3844, λ_tv=0)."""

    num_kernels: int = 3844
    sigma: float = 2.5
    lambda_tv: float = 0.0
    max_iter: int = 500
    lr: float = 0.01


@dataclass
class LPGDConfig:
    """Learned proximal gradient descent (spec §2 Tier 1.5)."""

    num_unroll: int = 5
    prox_channels: int = 32


@dataclass
class ExperimentConfig:
    """One full run: acceleration R, training slice budget, optional fast-dev."""

    acceleration: int = 4
    n_train_slices: int = SPEC_N_TRAIN
    train: TrainConfig = field(default_factory=TrainConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    lpgd: LPGDConfig = field(default_factory=LPGDConfig)
    fast_dev_run: bool = False
