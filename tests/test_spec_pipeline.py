"""Spec-aligned splits, masks, and LPGD smoke."""

import torch

from mri_recon.data_pipeline import (
    build_mask,
    spec_train_val_test_indices,
    take_n_shuffled_train,
)
from mri_recon.models.learned_pgd import LPGD


def test_spec_train_val_test_sizes():
    tr, va, te = spec_train_val_test_indices(100, n_train=67, n_test=14, seed=42)
    assert len(tr) == 67
    assert len(te) == 14
    assert len(va) == 19
    assert len(set(tr) | set(va) | set(te)) == 100


def test_build_mask_center_fraction_per_r():
    h, w = 32, 32
    m4 = build_mask(h, w, 4, seed=0)
    m6 = build_mask(h, w, 6, seed=0)
    assert m4.shape == (h, w)
    assert m6.shape == (h, w)
    assert (m4 >= 0).all() and (m4 <= 1).all()


def test_take_n_shuffled_train():
    pool = list(range(67))
    sub8 = take_n_shuffled_train(pool, 8, seed=42)
    assert len(sub8) == 8
    assert set(sub8).issubset(set(pool))


def test_lpgd_forward_no_grad():
    h, w = 32, 32
    x0 = torch.rand(1, 1, h, w)
    k = torch.fft.fftshift(torch.fft.fft2(x0, dim=(-2, -1)), dim=(-2, -1))
    mask = torch.ones(h, w)
    y_obs = k * mask.unsqueeze(0).unsqueeze(0)
    m = LPGD(num_steps=3, prox_ch=16)
    m.eval()
    with torch.no_grad():
        out = m(x0, y_obs, mask.to(x0.device))
    assert out.shape == x0.shape
