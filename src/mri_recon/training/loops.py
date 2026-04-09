"""
Training loops.

Significance: One place for epoch loops, validation PSNR tracking, and optional
gradient clipping. Experiment runners call into here so acceleration and
data-budget sweeps do not duplicate training boilerplate.
"""

from __future__ import annotations

# Implementation will add train_epoch, validate, and N2V-specific step.
