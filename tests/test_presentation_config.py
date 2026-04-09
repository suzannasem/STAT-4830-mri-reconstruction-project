"""Presentation-mode benchmark overrides."""

from __future__ import annotations

import pytest

from mri_recon.benchmark import BenchmarkConfig, benchmark_config_for_presentation, run_benchmark


def test_presentation_config_preserves_paths_and_raises_epochs() -> None:
    base = BenchmarkConfig(dicom_dir="data/cache", h=256, w=256)
    p = benchmark_config_for_presentation(base)
    assert p.dicom_dir == "data/cache"
    assert p.h == 256 and p.w == 256
    assert p.cnn_epochs == 50
    assert p.diffusion_epochs == 80
    assert p.unet_ch == 48


def test_run_benchmark_rejects_presentation_with_quick() -> None:
    with pytest.raises(ValueError, match="presentation"):
        run_benchmark(BenchmarkConfig(h=64, w=64, num_slices=8), quick=True, presentation=True)
