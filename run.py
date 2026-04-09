#!/usr/bin/env python3
"""
Run Colab-exported notebooks from ``scripts/notebook_pipeline/exports/``.

  python run.py
  python run.py --only week12_self_supervised_colab.py
  python run.py --out results/my_run

(Options are passed straight through to ``run_all_exports.main``.)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    repo = Path(__file__).resolve().parent
    runner = repo / "scripts" / "notebook_pipeline" / "run_all_exports.py"
    mod = runpy.run_path(str(runner), run_name="_run_all_exports")
    argv = sys.argv[1:]
    raise SystemExit(mod["main"](argv))
