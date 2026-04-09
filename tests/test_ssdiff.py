"""Week 12 SSDiffRecon smoke tests."""

import torch

from mri_recon.reconstructors.ss_diffusion import (
    SSDiffReconModel,
    create_ss_diff_masks,
    frequency_weighted_loss,
    reconstruct_inference,
)


def test_create_masks_sum():
    M, Mr, Mp = create_ss_diff_masks(32, 32, seed=0)
    assert M.shape == (1, 1, 32, 32)
    assert torch.all((Mr + Mp - M).abs() < 1e-5)


def test_frequency_weighted_loss_no_shadow():
    B, H, W = 1, 16, 16
    k_pred = torch.randn(B, 1, H, W, dtype=torch.complex64)
    k_tgt = torch.randn(B, 1, H, W, dtype=torch.complex64)
    Mr = torch.ones(B, 1, H, W)
    loss, diff, freq_weight = frequency_weighted_loss(k_pred, k_tgt, Mr, alpha=1.5)
    assert loss.ndim == 0
    assert freq_weight.shape == (H, W)


def test_model_forward():
    m = SSDiffReconModel(num_blocks=2)
    xt = torch.rand(1, 1, 32, 32)
    yp = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    mp = torch.ones(1, 1, 32, 32)
    t = torch.tensor([0.5])
    ar = torch.tensor([8.0])
    out = m(xt, yp, mp, t, ar)
    assert out.shape == (1, 1, 32, 32)


def test_reconstruct_inference_smoke():
    m = SSDiffReconModel(num_blocks=2)
    k = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    mask = torch.ones(32, 32)
    with torch.no_grad():
        x = reconstruct_inference(m, k, mask, num_steps=3, accel_rate=8.0)
    assert x.shape == (1, 1, 32, 32)
