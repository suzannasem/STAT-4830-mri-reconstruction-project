#!/usr/bin/env python3
"""
Execute every Jupyter notebook in a directory (including extensionless ``.ipynb``
downloads), write executed copies + logs, merge summaries, and optionally plot comparisons.

Requires: ``pip install -e ".[notebook-run]"`` (nbconvert + kernel).

Example::

  pip install -e ".[notebook-run]"
  python scripts/notebook_pipeline/run_all_ipynb.py \\
    --dir "/Users/me/Downloads/Notebooks Download Apr 9 2026"

Outputs under ``results/notebook_runs/<timestamp>/``:

- ``<slug>/notebook.ipynb`` — copy of source
- ``<slug>/executed.ipynb`` — executed notebook
- ``<slug>/run.log`` — traceback on failure
- ``combined_summary.json`` — one row per notebook
- ``comparison/`` — figures from ``visualize_notebook_runs`` (unless ``--no-viz``)

Environment passed to the kernel: ``MPLBACKEND=Agg``, ``MRI_NB_OUT=<run>/<slug>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import notebook_batch_utils as _nbu


def _require_nbconvert():
    try:
        import nbformat  # noqa: F401
        from nbconvert.preprocessors import ExecutePreprocessor  # noqa: F401

        return nbformat, ExecutePreprocessor
    except ImportError as e:
        print(
            "Missing nbconvert/nbformat. Install with:\n"
            '  pip install -e ".[notebook-run]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Execute all Jupyter notebooks in a directory")
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Folder containing .ipynb files (default: NOTEBOOK_DIR env or Downloads path)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run directory (default: results/notebook_runs/<timestamp>)",
    )
    p.add_argument("--only", nargs="*", help="Notebook basenames to include (substring match)")
    p.add_argument("--timeout", type=int, default=3600, help="Per-notebook kernel timeout (seconds)")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch on the first notebook error (default: run all)",
    )
    p.add_argument(
        "--allow-errors",
        action="store_true",
        help="Let the kernel continue past cell errors (nbconvert allow_errors)",
    )
    p.add_argument("--dry-run", action="store_true", help="List notebooks only")
    p.add_argument("--no-viz", action="store_true", help="Skip comparison figures")
    args = p.parse_args(argv)

    default_dir = os.environ.get("NOTEBOOK_DIR")
    if default_dir:
        nb_dir = Path(default_dir).expanduser().resolve()
    else:
        nb_dir = (
            Path.home()
            / "Downloads"
            / "Notebooks Download Apr 9 2026"
        ).resolve()

    if args.dir is not None:
        nb_dir = args.dir.expanduser().resolve()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = (args.out or (_REPO_ROOT / "results" / "notebook_runs" / ts)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    paths = _nbu.discover_notebooks(nb_dir)
    if args.only:
        key = [k.lower() for k in args.only]
        paths = [p for p in paths if any(k in p.name.lower() for k in key)]

    if not paths:
        print(f"No notebooks found under {nb_dir}", file=sys.stderr)
        return 1

    slug_map = _nbu.unique_slugs(paths)

    if args.dry_run:
        for path in paths:
            print(f"{slug_map[path]:40}  {path.name}")
        print(f"Would write under {run_root}")
        return 0

    nbformat, ExecutePreprocessor = _require_nbconvert()

    summary: list[dict] = []

    for path in paths:
        slug = slug_map[path]
        sub = run_root / slug
        sub.mkdir(parents=True, exist_ok=True)
        dst_ipynb = sub / "notebook.ipynb"
        shutil.copy2(path, dst_ipynb)
        executed = sub / "executed.ipynb"
        log_path = sub / "run.log"

        env = {
            **os.environ,
            "MPLBACKEND": "Agg",
            "MRI_NB_OUT": str(sub),
        }

        row: dict = {
            "notebook": path.name,
            "slug": slug,
            "source_path": str(path),
            "work_dir": str(sub),
            "executed_path": str(executed),
            "log": str(log_path),
        }

        t0 = time.perf_counter()
        nb = None
        try:
            with dst_ipynb.open(encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(
                timeout=args.timeout,
                kernel_name="python3",
                allow_errors=args.allow_errors,
            )
            prev_env = os.environ.copy()
            try:
                os.environ.update(env)
                ep.preprocess(nb, {"metadata": {"path": str(sub)}})
            finally:
                os.environ.clear()
                os.environ.update(prev_env)

            with executed.open("w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            row["exit_code"] = 0
            row["error"] = None
        except Exception as e:
            row["exit_code"] = 1
            row["error"] = f"{type(e).__name__}: {e}"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            if nb is not None:
                try:
                    with executed.open("w", encoding="utf-8") as f:
                        nbformat.write(nb, f)
                except Exception:
                    pass

        row["duration_s"] = round(time.perf_counter() - t0, 3)

        # Optional JSON written by notebook exports
        row["metrics_file"] = None
        for candidate in sorted(sub.glob("*_summary.json")) + [sub / "summary.json"]:
            if candidate.is_file():
                try:
                    row["metrics_file"] = json.loads(candidate.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    row["metrics_file"] = None

        if executed.is_file():
            row["scraped_metrics"] = _nbu.scrape_metrics_from_executed_notebook(executed)
        else:
            row["scraped_metrics"] = {}

        summary.append(row)

        if row["exit_code"] != 0 and args.fail_fast:
            break

    combined = run_root / "combined_summary.json"
    combined.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {combined}")

    if not args.no_viz:
        try:
            import visualize_notebook_runs as _viz

            _viz.render_comparison_figures(run_root)
        except Exception as e:
            print(f"Warning: visualization failed: {e}", file=sys.stderr)

    failed = [r for r in summary if r.get("exit_code") not in (0, None)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
