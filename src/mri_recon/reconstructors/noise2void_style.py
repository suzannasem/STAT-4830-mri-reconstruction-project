"""
Self-supervised denoising: map corrupted zero-filled input back to x_zf (no GT).

Trains a small ResidualCNN on pairs (x_zf + noise, x_zf) so the run does not
use ground-truth labels at train time; evaluation is still vs GT on the test slice.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NOISE2VOID_ID = "noise2void"


class SmallDenoiseNet(nn.Module):
    """Light residual CNN (same family as ResidualCNN, fewer channels for speed)."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.net(x), 0.0, 1.0)
