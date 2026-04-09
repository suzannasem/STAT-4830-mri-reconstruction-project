"""
Experiment 1: Acceleration sweep.

Significance: Fix training slice count, vary R ∈ {4, 6, 8}, retrain/evaluate
each method per R. Answers “where methods break” as undersampling increases.
Outputs CSV/JSON under results/ for visualization.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry; to be wired to argparse + pipeline."""
    raise NotImplementedError("Acceleration sweep — implement after data_pipeline + models.")
