"""
Central experiment configuration.

Significance: Single source of truth for paths, device, acceleration factors
(4×, 6×, 8×), training slice counts (8, 16, 32, 67), image size, seeds, and
default hyperparameters. Experiment scripts import from here so sweeps stay
reproducible and CLI flags can override defaults without scattering magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# --- Spec axes (MRI Reconstruction experiment specification) ---
ACCELERATION_FACTORS: Tuple[int, ...] = (4, 6, 8)
TRAIN_SLICE_COUNTS: Tuple[int, ...] = (8, 16, 32, 67)

# --- Data defaults (aligned with notebooks: UPENN-GBM, 256×256) ---
DEFAULT_COLLECTION: str = "UPENN-GBM"
OUT_SIZE: Tuple[int, int] = (256, 256)
PERCENTILE_CLIP: Tuple[float, float] = (1.0, 99.0)

# --- k-space mask (Week 10 variable-density; center fully sampled) ---
# Effective undersampling: outer k-space sampled with radius-dependent prob ~ 1/accel.
MASK_CENTER_FRACTION: float = 0.08

# --- Train / val / test slice fractions (Week 10 notebook) ---
TRAIN_VAL_TEST_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_SEED: int = 42

# --- I/O ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


@dataclass
class TrainConfig:
    """Training loop defaults shared by supervised and self-supervised runs."""

    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 50
    seed: int = 42


@dataclass
class KernelConfig:
    """Hyperparameters for Gaussian / Laplacian kernel sparse reconstruction."""

    num_kernels: int = 1600
    sigma: float = 8.0
    lambda_tv: float = 1e-5
    max_iter: int = 500
    lr: float = 0.01


@dataclass
class ExperimentConfig:
    """
    Binds one full experiment run: which R, how many training slices, optional
    fast-dev mode for CI/smoke tests.
    """

    acceleration: int = 6
    n_train_slices: int = 32
    train: TrainConfig = field(default_factory=TrainConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    fast_dev_run: bool = False
