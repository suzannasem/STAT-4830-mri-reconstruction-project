"""
Visualization helpers.

Significance: Reads metrics files from results/, produces line plots (PSNR vs R,
PSNR vs N) and optional image panels (GT, zero-filled, best/worst method) for
the report figures. Keeps matplotlib usage out of experiment runners.
"""

from __future__ import annotations

# Implementation will add load_results_csv, plot_sweep_lines, save_figure.
