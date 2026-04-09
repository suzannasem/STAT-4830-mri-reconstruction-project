"""
Abstract protocols for reconstructors.

Significance: Classical methods need per-slice optimization (kernel basis);
supervised nets need batched training + forward; self-supervised needs a
custom loss. A shared Protocol (or ABC) keeps experiment runners agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reconstructor(Protocol):
    """Minimal interface evaluated by experiment scripts."""

    method_id: str

    def reconstruct(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """Return reconstructed image tensor(s); signature finalized per subclass."""
        ...
