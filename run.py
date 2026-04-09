#!/usr/bin/env python3
"""
Run from the repo root without installing the package (adds src/ to path).

  python run.py           # pipeline demo
  python run.py demo
  python run.py test
  python run.py all

After `pip install -e .`, you can use: mri-recon [demo|test|all]
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
