#!/usr/bin/env python3
"""
Run from the repo root without installing the package (adds src/ to path).

  python run.py                    # full benchmark + benchmark_snapshot.pt + 6 publication figures
  python run.py --quick            # fast smoke (same outputs, shorter training)
  python run.py --dicom DIR        # load DICOM series from DIR (middle slices → H×W)
  python run.py --experiments      # also run head-to-head + accel + data sweeps, refresh figures
  python run.py visualize          # regenerate figures from results/ only
  python run.py demo               # zero-filled PSNR sweep only
  python run.py test               # pytest

After `pip install -e .`, use: mri-recon [options] [command]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mri_recon.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
