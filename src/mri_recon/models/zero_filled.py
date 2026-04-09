"""Zero-filled FFT baseline (Tier 0)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mri_recon.data_pipeline import zero_filled_image


class ZeroFilled(nn.Module):
    """Identity pipeline wrapper: k_obs → magnitude IFFT."""

    name = "Zero-Filled FFT"
    category = "baseline"
    requires_ground_truth = False

    def forward(self, y_obs: torch.Tensor) -> torch.Tensor:
        if y_obs.dim() == 3:
            y_obs = y_obs.unsqueeze(0)
        x = zero_filled_image(y_obs[:, 0]).unsqueeze(1)
        return torch.abs(x)
