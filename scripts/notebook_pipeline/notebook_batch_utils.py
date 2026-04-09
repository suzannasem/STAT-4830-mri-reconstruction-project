"""
Helpers for batch-running local Jupyter notebooks and scraping printed metrics.

Used by ``run_all_ipynb.py`` and ``visualize_notebook_runs.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def is_probably_ipynb(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() == ".ipynb":
        return True
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not raw.lstrip().startswith("{"):
        return False
    if '"cells"' not in raw:
        return False
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(doc, dict) and isinstance(doc.get("cells"), list)


def discover_notebooks(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    for p in sorted(root.iterdir()):
        if is_probably_ipynb(p):
            found.append(p)
    return found


def notebook_slug(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".ipynb"):
        name = name[: -len(".ipynb")]
    s = re.sub(r"[^\w\d]+", "_", name).strip("_").lower()
    return s or "notebook"


def unique_slugs(paths: list[Path]) -> dict[Path, str]:
    """Assign unique slug per path (add _1, _2, … if collisions)."""
    out: dict[Path, str] = {}
    counts: dict[str, int] = {}
    for p in paths:
        base = notebook_slug(p)
        c = counts.get(base, 0)
        slug = base if c == 0 else f"{base}_{c}"
        counts[base] = c + 1
        out[p] = slug
    return out


def _cell_source_text(cell: dict) -> str:
    src = cell.get("source") or ""
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def notebook_to_plaintext(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        parts.append(_cell_source_text(cell))
        for out in cell.get("outputs") or []:
            ot = out.get("output_type")
            if ot == "stream":
                t = out.get("text", "")
                if isinstance(t, list):
                    t = "".join(t)
                parts.append(t)
            elif ot == "error":
                parts.append("\n".join(out.get("traceback") or []))
            elif ot in ("display_data", "execute_result"):
                data = out.get("data") or {}
                plain = data.get("text/plain")
                if isinstance(plain, list):
                    plain = "".join(plain)
                if plain:
                    parts.append(plain)
    return "\n".join(parts)


# Last match wins (usually final epoch / summary line)
_METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "psnr",
        re.compile(r"(?i)psnr\s*[\[=:]\s*([0-9]*\.?[0-9]+)\s*d?b?"),
    ),
    ("ssim", re.compile(r"(?i)ssim\s*[\[=:]\s*([0-9]*\.?[0-9]+)")),
    ("nmse", re.compile(r"(?i)nmse\s*[\[=:]\s*([0-9.eE+-]+)")),
    ("loss", re.compile(r"(?i)(?:val(?:idation)?|test)?\s*loss\s*[\[=:]\s*([0-9.eE+-]+)")),
]


def scrape_metrics_from_text(text: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for name, pat in _METRIC_PATTERNS:
        matches = list(pat.finditer(text))
        if not matches:
            continue
        raw = matches[-1].group(1)
        raw = raw.replace("dB", "").strip()
        try:
            found[name] = float(raw)
        except ValueError:
            continue
    return found


def scrape_metrics_from_executed_notebook(path: Path) -> dict[str, float]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return scrape_metrics_from_text(notebook_to_plaintext(nb))
