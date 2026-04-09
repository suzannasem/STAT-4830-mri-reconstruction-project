"""
DICOM loading, k-space undersampling (Week 10 variable-density mask), datasets.

Tensors are [S or B, 1, H, W] float32 in image domain unless noted; k-space is
complex with fftshift layout (low freq at center).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pydicom.dataset import FileDataset

from mri_recon.config import (
    ACCELERATION_CENTER_FRACTION,
    DATA_CACHE_DIR,
    DEFAULT_COLLECTION,
    MASK_CENTER_FRACTION,
    OUT_SIZE,
    PERCENTILE_CLIP,
    SPLIT_SEED,
    TRAIN_VAL_TEST_RATIOS,
)

__all__ = [
    "PairDataset",
    "build_mask",
    "create_variable_density_mask",
    "dicom_to_target_tensor",
    "download_series_if_needed",
    "kspace_fftshift_from_image",
    "load_dicom_series_from_dir",
    "normalize_percentile_01",
    "spec_train_val_test_indices",
    "subsample_train_indices",
    "synthetic_phantom_stack",
    "take_n_shuffled_train",
    "train_val_test_indices",
    "undersample_stack",
    "zero_filled_image",
]


def dicom_to_target_tensor(
    ds: "FileDataset",
    out_size: Tuple[int, int] | None = None,
    percentile_clip: Tuple[float, float] | None = None,
) -> torch.Tensor:
    """
    Single slice: clip outliers by percentile, normalize to [0, 1], resize.

    Returns [1, H, W] float32 CPU.
    """
    out_size = out_size or OUT_SIZE
    percentile_clip = percentile_clip or PERCENTILE_CLIP

    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope"):
        slope = float(ds.RescaleSlope)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    p_low, p_high = np.percentile(arr, percentile_clip)
    arr = np.clip(arr, p_low, p_high)
    denom = p_high - p_low + 1e-12
    arr = (arr - p_low) / denom

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=False)
    return t.squeeze(0).float()


def load_dicom_series_from_dir(root: Path | str) -> Tuple[List["FileDataset"], torch.Tensor]:
    """
    Load all *.dcm under root, sort by InstanceNumber (fallback: filename).

    Returns (datasets, target_images) with target_images ``[S, 1, H, W]`` float32 CPU.
    """
    import pydicom

    root = Path(root)
    paths: List[Path] = []
    for walk_root, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                paths.append(Path(walk_root) / f)

    datasets = [pydicom.dcmread(str(p)) for p in paths]

    def _sort_key(ds: pydicom.dataset.FileDataset) -> int:
        if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
            return int(ds.InstanceNumber)
        return 0

    datasets.sort(key=_sort_key)
    target_list = [dicom_to_target_tensor(ds) for ds in datasets]
    if not target_list:
        raise FileNotFoundError(f"No DICOM files found under {root}")
    stacked = torch.stack(target_list, dim=0)  # [S, 1, H, W]
    return datasets, stacked


def download_series_if_needed(
    collection: str = DEFAULT_COLLECTION,
    download_path: Path | str | None = None,
    number: int = 1,
) -> Path:
    """
    Download one series from TCIA via tcia_utils (requires network).

    Returns path to directory containing DICOMs (unzipped under download_path).
    """
    from tcia_utils import nbia

    download_path = Path(download_path or DATA_CACHE_DIR)
    download_path.mkdir(parents=True, exist_ok=True)

    data = nbia.getSeries(collection=collection)
    if data is None:
        raise RuntimeError(f"TCIA getSeries returned no data for collection={collection!r}")
    nbia.downloadSeries(data, number=number, path=str(download_path))
    return download_path


def normalize_percentile_01(x: torch.Tensor, p_low: float = 1.0, p_high: float = 99.0) -> torch.Tensor:
    """
    Clip to [p_low, p_high] percentiles per tensor (or per-image if 4D), scale to [0, 1].
    Spec §10: uniform 99th-percentile style normalization across methods.
    """
    if x.dim() == 4:
        out = []
        for i in range(x.shape[0]):
            out.append(normalize_percentile_01(x[i : i + 1], p_low, p_high))
        return torch.cat(out, dim=0)
    flat = x.flatten()
    lo = torch.quantile(flat, p_low / 100.0)
    hi = torch.quantile(flat, p_high / 100.0)
    z = (x - lo) / (hi - lo + 1e-12)
    return torch.clamp(z, 0.0, 1.0)


def create_variable_density_mask(
    h: int,
    w: int,
    center_fraction: float = MASK_CENTER_FRACTION,
    accel: float = 4.0,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    2D variable-density binary mask (real-valued in {0, 1}), Week 10 logic.

    - Fully samples a low-frequency center block.
    - Outer k-space: Bernoulli with radius-dependent probability.

    `accel` is the notebook's acceleration factor (higher => fewer samples overall).
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    rr = torch.sqrt(xx**2 + yy**2)
    prob = 1.0 / accel + (1.0 - 1.0 / accel) * torch.exp(-6.0 * rr**2)

    center_h = max(1, int(h * center_fraction))
    center_w = max(1, int(w * center_fraction))
    h0 = h // 2 - center_h // 2
    h1 = h0 + center_h
    w0 = w // 2 - center_w // 2
    w1 = w0 + center_w

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    random_field = torch.rand((h, w), generator=g)
    mask_real = (random_field < prob).float()
    mask_real[h0:h1, w0:w1] = 1.0
    return mask_real.to(device=device)


def build_mask(
    h: int,
    w: int,
    acceleration: int,
    seed: int,
    center_fraction: float | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Build undersampling mask for R ∈ {4, 6, 8}.

    Uses spec §3 center fractions when ``center_fraction`` is None:
    4× → 8%, 6× → 6%, 8× → 4%.
    """
    cf = center_fraction if center_fraction is not None else ACCELERATION_CENTER_FRACTION.get(
        int(acceleration), MASK_CENTER_FRACTION
    )
    return create_variable_density_mask(
        h, w, center_fraction=cf, accel=float(acceleration), seed=seed, device=device
    )


def kspace_fftshift_from_image(image: torch.Tensor) -> torch.Tensor:
    """
    image: [..., H, W] real -> k-space [..., H, W] complex, fftshifted.
    """
    return torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)))


def zero_filled_image(k_obs: torch.Tensor) -> torch.Tensor:
    """
    k_obs: complex k-space, fftshift layout [..., H, W].
    Returns real image [..., H, W].
    """
    return torch.fft.ifft2(torch.fft.ifftshift(k_obs, dim=(-2, -1)), dim=(-2, -1)).real


def undersample_stack(
    target_images: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    target_images: [S, 1, H, W] real in [0, 1].
    mask: [H, W] real {0,1} or broadcastable; applied to fftshifted k-space.

    Returns:
      x_zf: [S, 1, H, W] zero-filled recon
      k_obs: [S, 1, H, W] complex observed k-space
      k_full: [S, 1, H, W] complex full k-space (for consistency / DC models)
    """
    if target_images.dim() != 4:
        raise ValueError("target_images must be [S, 1, H, W]")
    s, c, h, w = target_images.shape
    if c != 1:
        raise ValueError("expected single channel (C=1)")
    img = target_images.squeeze(1)
    k_full = kspace_fftshift_from_image(img)
    m = mask.to(device=k_full.device, dtype=k_full.real.dtype)
    if m.dim() == 2:
        m = m.view(1, h, w)
    k_obs = k_full * m
    x_2d = zero_filled_image(k_obs)
    x_zf = x_2d.unsqueeze(1)
    k_full_c = k_full.unsqueeze(1)
    k_obs_c = k_obs.unsqueeze(1)
    return x_zf, k_obs_c, k_full_c


def train_val_test_indices(
    num_images: int,
    seed: int = SPLIT_SEED,
    ratios: Tuple[float, float, float] = TRAIN_VAL_TEST_RATIOS,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Shuffle all slice indices and split by (train, val, test) ratios (Week 10).
    """
    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    all_indices = list(range(num_images))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    train_end = int(r_train * num_images)
    val_end = int((r_train + r_val) * num_images)
    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]
    return train_indices, val_indices, test_indices


def subsample_train_indices(
    train_indices: Sequence[int],
    n_train: int,
    seed: int,
) -> List[int]:
    """
    For data-budget sweeps: choose `n_train` slices from the train pool.

    If n_train >= len(train_indices), returns all train indices (stable order).
    Otherwise returns a random subset of size n_train (sorted ascending).
    """
    pool = list(train_indices)
    if n_train >= len(pool):
        return sorted(pool)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))[:n_train]
    return sorted(pool[i] for i in perm)


def take_n_shuffled_train(train_indices: Sequence[int], n: int, seed: int) -> List[int]:
    """
    Spec §4: first N indices after shuffling the train pool (fixed seed).

    Order follows the shuffled permutation (not sorted) so the first 8 are
    literally the first 8 in the permuted order.
    """
    pool = list(train_indices)
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool)).tolist()
    return [pool[i] for i in order[:n]]


def spec_train_val_test_indices(
    n_total: int,
    n_train: int = 67,
    n_test: int = 14,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Spec §3–5: fixed train/test sizes from a global shuffle.

    Remaining slices are validation. Requires ``n_total >= n_train + n_test``.
    """
    if n_total < n_train + n_test:
        raise ValueError(f"n_total ({n_total}) must be >= n_train + n_test ({n_train + n_test})")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    test_idx = perm[:n_test].tolist()
    train_idx = perm[n_test : n_test + n_train].tolist()
    val_idx = perm[n_test + n_train :].tolist()
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def synthetic_phantom_stack(
    num_slices: int,
    h: int,
    w: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Smooth random [0, 1] images for tests (no DICOM / network).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.rand(num_slices, 1, h, w, generator=g)
    # light blur
    kernel = torch.ones(1, 1, 5, 5, device=x.device, dtype=x.dtype) / 25.0
    blurred = F.conv2d(x, kernel, padding=2)
    bmin = blurred.amin(dim=(2, 3), keepdim=True)
    bmax = blurred.amax(dim=(2, 3), keepdim=True)
    return (blurred - bmin) / (bmax - bmin + 1e-12)


class PairDataset(Dataset):
    """Pairs (undersampled zero-filled, ground truth) per slice."""

    def __init__(self, x_undersampled: torch.Tensor, y_target: torch.Tensor) -> None:
        if x_undersampled.shape != y_target.shape:
            raise ValueError("x and y must have the same shape [S, 1, H, W]")
        self.x = x_undersampled.float()
        self.y = y_target.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
