"""Shared ``Method`` protocol (Experiment Spec §7)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Method(Protocol):
    name: str
    category: str
    requires_ground_truth: bool

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        mask: torch.Tensor,
        config: Any,
    ) -> dict[str, Any]: ...

    def reconstruct(self, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...
