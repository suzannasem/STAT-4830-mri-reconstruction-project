"""Week 10 diffusion helpers."""

import torch

from mri_recon.reconstructors.diffusion import DDPMSchedule, DiffusionZFDenoiser, q_sample


def test_q_sample_shape():
    device = torch.device("cpu")
    sch = DDPMSchedule(T=20).to(device)
    x0 = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 20, (2,))
    xt, eps = q_sample(x0, t, sch.alpha_bars)
    assert xt.shape == x0.shape
    assert eps.shape == x0.shape


def test_diffusion_zf_denoiser_forward():
    m = DiffusionZFDenoiser(time_dim=32, ch=16)
    x_t = torch.randn(1, 1, 32, 32)
    x_zf = torch.randn(1, 1, 32, 32)
    t = torch.zeros(1, dtype=torch.long)
    out = m(x_t, x_zf, t)
    assert out.shape == (1, 1, 32, 32)
