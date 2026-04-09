"""
Self-supervised reconstructors.

Significance: Noise2Void (or similar blind-spot) training does not use ground
truth in the loss; it predicts masked pixels from neighbors. Used to fill the
“self-supervised tier” in the experiment spec alongside supervised methods.

Not present in Week 10 notebook; implemented to match standard N2V for 1-channel MRI.
"""

from __future__ import annotations

NOISE2VOID_ID = "noise2void"
