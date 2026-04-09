#!/usr/bin/env python3
"""
Run all Jupyter notebooks under a folder (see ``scripts/notebook_pipeline/run_all_ipynb.py``).

  pip install -e ".[notebook-run]"
  python run_ipynb.py
  python run_ipynb.py --dir ~/Downloads/Notebooks\\ Download\\ Apr\\ 9\\ 2026 --dry-run

Options are passed through to the runner.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    repo = Path(__file__).resolve().parent
    runner = repo / "scripts" / "notebook_pipeline" / "run_all_ipynb.py"
    mod = runpy.run_path(str(runner), run_name="_run_all_ipynb")
    raise SystemExit(mod["main"](sys.argv[1:]))
