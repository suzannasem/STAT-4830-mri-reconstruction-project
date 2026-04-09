"""Tests for mri_recon.data_pipeline (synthetic only; no TCIA)."""

import torch

from mri_recon.data_pipeline import (
    build_mask,
    kspace_fftshift_from_image,
    subsample_train_indices,
    synthetic_phantom_stack,
    train_val_test_indices,
    undersample_stack,
    zero_filled_image,
)


def test_synthetic_shape():
    x = synthetic_phantom_stack(5, 64, 64, seed=1)
    assert x.shape == (5, 1, 64, 64)
    assert x.min() >= 0 and x.max() <= 1 + 1e-6


def test_undersample_shapes_and_k_consistency():
    s, h, w = 4, 32, 32
    y = synthetic_phantom_stack(s, h, w, seed=2)
    mask = build_mask(h, w, acceleration=6, seed=0)
    assert mask.shape == (h, w)
    x_zf, k_obs, k_full = undersample_stack(y, mask)
    assert x_zf.shape == y.shape
    assert k_obs.shape == (s, 1, h, w)
    assert k_full.shape == (s, 1, h, w)
    # y_obs = k_full * mask
    m = mask.to(k_full.device).view(1, h, w)
    expected_obs = k_full * m
    assert torch.allclose(k_obs, expected_obs)


def test_zero_filled_roundtrip_rms():
    """Full k-space -> IFFT should recover image (up to numerical error)."""
    img = synthetic_phantom_stack(1, 16, 16, seed=3).squeeze(0)
    k = kspace_fftshift_from_image(img)
    recon = zero_filled_image(k)
    assert recon.shape == img.shape
    assert torch.allclose(recon, img, atol=1e-5, rtol=1e-4)


def test_train_val_test_split_lengths():
    n = 100
    tr, va, te = train_val_test_indices(n, seed=7)
    assert len(tr) == 70
    assert len(va) == 15
    assert len(te) == 15
    assert set(tr) | set(va) | set(te) == set(range(n))


def test_subsample_train_indices():
    tr, _, _ = train_val_test_indices(50, seed=0)
    sub = subsample_train_indices(tr, n_train=8, seed=99)
    assert len(sub) == 8
    assert set(sub).issubset(set(tr))
    assert subsample_train_indices(tr, n_train=1000, seed=0) == sorted(tr)
