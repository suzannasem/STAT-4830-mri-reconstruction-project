"""
Supervised deep reconstructors.

Significance:
- residual_cnn: maps undersampled (or optional kernel/ZF input mode) → GT.
- unet: U-Net with k-space data consistency (y_obs, mask) in forward, Week 10.
- srcnn: SRCNN_DC-style 3-layer CNN + data consistency term, Week 10.

Training uses MSE between prediction and ground truth; SSIM can be added later.
"""

from __future__ import annotations

RESIDUAL_CNN_ID = "residual_cnn"
UNET_ID = "unet"
SRCNN_ID = "srcnn"
