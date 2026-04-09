"""Tests for mri_recon.metrics."""

import torch

from mri_recon.metrics import psnr, ssim


def test_psnr_identical_high():
    x = torch.rand(2, 1, 32, 32)
    p = psnr(x, x, data_range=1.0)
    assert p > 100.0
    assert torch.isfinite(p)


def test_ssim_identical_one():
    x = torch.rand(2, 1, 32, 32)
    s = ssim(x, x, data_range=1.0)
    assert s > 0.999
    assert torch.isfinite(s)


def test_psnr_known_offset():
    # constant images: mse = delta^2
    a = torch.zeros(1, 1, 16, 16)
    b = torch.ones(1, 1, 16, 16) * 0.1
    p = psnr(a, b, data_range=1.0)
    mse = 0.01
    expected = 10.0 * torch.log10(torch.tensor(1.0 / mse))
    assert torch.isclose(p, expected, rtol=1e-5)
