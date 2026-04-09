#!/usr/bin/env python3
"""
Run Colab-exported notebook .py files in order and merge a small summary.

Primary entry: ``python run.py`` from the repo root (or this module directly).

Usage (from repo root):

  pip install -e ".[notebook-export]"
  python scripts/notebook_pipeline/run_all_exports.py

  # custom output root (default: results/notebook_pipeline/runs/<timestamp>)
  python scripts/notebook_pipeline/run_all_exports.py --out results/my_colab_run

  # only specific scripts
  python scripts/notebook_pipeline/run_all_exports.py --only week12_self_supervised_colab.py

Drop more exports into scripts/notebook_pipeline/exports/ — they run in
lexicographic order unless you pass --only or --manifest path/to/order.txt
(with one filename per line).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> list[Path]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return [_EXPORTS_DIR / n for n in names]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run Colab-export notebooks from scripts/notebook_pipeline/exports/")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory (default: results/notebook_pipeline/runs/<timestamp>)",
    )
    p.add_argument("--only", nargs="*", help="Basenames under exports/ to run (default: all *.py)")
    p.add_argument("--manifest", type=Path, default=None, help="Text file: one script basename per line")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args(argv)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = (args.out or (_REPO_ROOT / "results" / "notebook_pipeline" / "runs" / ts)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    logs.mkdir(exist_ok=True)

    if args.manifest is not None:
        scripts = _load_manifest(args.manifest)
    elif args.only:
        scripts = [_EXPORTS_DIR / n for n in args.only]
    else:
        scripts = sorted(_EXPORTS_DIR.glob("*.py"))

    scripts = [s for s in scripts if s.is_file()]
    if not scripts:
        print(f"No scripts found under {_EXPORTS_DIR}", file=sys.stderr)
        return 1

    summary: list[dict] = []
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MRI_NB_OUT": str(run_root),
    }

    for script in scripts:
        log_path = logs / f"{script.stem}.log"
        print(f"→ {script.name}  (log: {log_path})")
        cmd = [sys.executable, str(script)]
        if args.dry_run:
            print(" ", " ".join(cmd))
            summary.append({"script": script.name, "skipped": True})
            continue
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        row = {
            "script": script.name,
            "exit_code": proc.returncode,
            "log": str(log_path),
        }
        js = run_root / f"{script.stem}_summary.json"
        if js.is_file():
            try:
                row["metrics"] = json.loads(js.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                row["metrics"] = None
        summary.append(row)

    combined = run_root / "combined_summary.json"
    combined.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {combined}")
    failed = [r for r in summary if r.get("exit_code") not in (0, None) and r.get("skipped") is not True]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
