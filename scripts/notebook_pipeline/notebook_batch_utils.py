"""
Helpers for batch-running local Jupyter notebooks and scraping printed metrics.

Used by ``run_all_ipynb.py`` and ``visualize_notebook_runs.py``.
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
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


# Regexes for all candidate matches (we pick plausible values; see below)
_METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "psnr",
        re.compile(r"(?i)psnr\s*[\[=:]\s*([0-9]*\.?[0-9]+)\s*d?b?"),
    ),
    ("ssim", re.compile(r"(?i)ssim\s*[\[=:]\s*([0-9]*\.?[0-9]+)")),
    ("nmse", re.compile(r"(?i)nmse\s*[\[=:]\s*([0-9.eE+-]+)")),
    ("loss", re.compile(r"(?i)(?:val(?:idation)?|test)?\s*loss\s*[\[=:]\s*([0-9.eE+-]+)")),
]

# PSNR lines like "Zero-Filled FFT PSNR: 320 dB" are numerical garbage; keep reconstruction-range dB only.
_PSNR_DB_LO, _PSNR_DB_HI = 5.0, 60.0
_SSIM_LO, _SSIM_HI = -0.05, 1.0
_LOSS_HI = 10.0


def scrape_metrics_from_text(text: str) -> dict[str, float]:
    found: dict[str, float] = {}

    psnr_pat = _METRIC_PATTERNS[0][1]
    last_psnr: float | None = None
    for m in psnr_pat.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if _PSNR_DB_LO <= v <= _PSNR_DB_HI:
            last_psnr = v
    if last_psnr is not None:
        found["psnr"] = last_psnr

    ssim_pat = _METRIC_PATTERNS[1][1]
    last_ssim: float | None = None
    for m in ssim_pat.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if _SSIM_LO <= v <= _SSIM_HI:
            last_ssim = v
    if last_ssim is not None:
        found["ssim"] = max(0.0, min(1.0, last_ssim))

    nmse_pat = _METRIC_PATTERNS[2][1]
    for m in nmse_pat.finditer(text):
        try:
            found["nmse"] = float(m.group(1))
        except ValueError:
            continue

    loss_pat = _METRIC_PATTERNS[3][1]
    last_loss: float | None = None
    for m in loss_pat.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0.0 <= v <= _LOSS_HI:
            last_loss = v
    if last_loss is not None:
        found["loss"] = last_loss

    return found


def scrape_metrics_from_executed_notebook(path: Path) -> dict[str, float]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return scrape_metrics_from_text(notebook_to_plaintext(nb))


def scrape_methods_from_text(text: str) -> list[dict]:
    """Parse lines: ``MRI_METHOD_RESULT {"method_id": "...", "psnr": ...}`` (one JSON object per line)."""
    out: list[dict] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("MRI_METHOD_RESULT"):
            continue
        rest = s[len("MRI_METHOD_RESULT") :].strip()
        try:
            obj = json.loads(rest)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def scrape_methods_from_executed_notebook(path: Path) -> list[dict]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return scrape_methods_from_text(notebook_to_plaintext(nb))


def load_methods_summary_json(path: Path) -> list[dict] | None:
    """Load ``methods_summary.json``: either a JSON list of objects or ``{"methods": [...]}``."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        m = data.get("methods")
        if isinstance(m, list):
            return [x for x in m if isinstance(x, dict)]
    return None


def normalize_method_dict(m: dict) -> dict:
    """Copy metric keys for plotting (``psnr_db`` -> ``psnr``)."""
    out = dict(m)
    if "psnr" not in out and out.get("psnr_db") is not None:
        try:
            out["psnr"] = float(out["psnr_db"])
        except (TypeError, ValueError):
            pass
    return out


def sanitize_reconstruction_metrics(m: dict[str, float]) -> dict[str, float]:
    """
    Drop scraped values that are not comparable reconstruction summary stats
    (e.g. PSNR parsed from wrong lines, huge ``loss`` from unrelated prints).
    """
    out: dict[str, float] = {}
    if "psnr" in m:
        try:
            v = float(m["psnr"])
        except (TypeError, ValueError):
            v = float("nan")
        if 8.0 <= v <= 55.0:
            out["psnr"] = v
    if "ssim" in m:
        try:
            v = float(m["ssim"])
        except (TypeError, ValueError):
            v = float("nan")
        if -0.05 <= v <= 1.0:
            out["ssim"] = max(0.0, min(1.0, v))
    if "loss" in m:
        try:
            v = float(m["loss"])
        except (TypeError, ValueError):
            v = float("nan")
        if 0.0 <= v <= 5.0:
            out["loss"] = v
    for k in ("nmse", "mse", "ss_loss"):
        if k in m:
            try:
                out[k] = float(m[k])
            except (TypeError, ValueError):
                pass
    return out


METHOD_DISPLAY_NAME_BY_SLUG: dict[str, str] = {
    "srcnn_mfcnn": "SRCNN / MFCNN (kernel pipeline)",
    "week_10_multi_image_notebook": "Multi-slice supervised (Week 10)",
    "week_12_notebook": "Multi-slice supervised (Week 12)",
    "week_12_self_supervised_notebook_v1": "Self-supervised diffusion (v1)",
    "week_12_self_supervised_notebook_v2": "Self-supervised diffusion (v2)",
    "week_4_notebook": "Gaussian kernel basis + sparse coding",
    "week_5_notebook": "Undersampling mask + residual CNN",
    "week_8_notebook": "Kernel recon + neural residual refine",
}


def methodology_label_from_row(row: dict) -> str:
    slug = row.get("slug") or ""
    if slug in METHOD_DISPLAY_NAME_BY_SLUG:
        return METHOD_DISPLAY_NAME_BY_SLUG[slug]
    name = (row.get("notebook") or "").replace(".ipynb", "")
    return name or slug or "?"


def extract_png_images_from_notebook(path: Path) -> list:
    """
    Decode ``image/png`` outputs from an executed notebook in cell order.
    Usually the last images are the final comparison figures.
    """
    import matplotlib.image as mpimg
    import numpy as np

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list = []
    for cell in nb.get("cells", []):
        for o in cell.get("outputs") or []:
            data = o.get("data") or {}
            b64 = data.get("image/png")
            if not b64:
                continue
            if isinstance(b64, list):
                b64 = "".join(b64)
            try:
                raw = base64.b64decode(b64)
                im = mpimg.imread(BytesIO(raw))
            except (OSError, ValueError, MemoryError):
                continue
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=-1)
            elif im.shape[-1] == 4:
                im = im[..., :3]
            out.append(im)
    return out


def load_recon_compare_manifest(work_dir: Path) -> dict | None:
    """Optional ``recon_compare.json`` with ``baseline`` / ``reconstruction`` paths relative to work_dir."""
    p = work_dir / "recon_compare.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def load_image_path(work_dir: Path, rel: str):
    """Load image as float RGB array for matplotlib (0–1), or None."""
    import matplotlib.image as mpimg
    import numpy as np

    fp = (work_dir / rel).resolve()
    if not fp.is_file():
        return None
    try:
        im = mpimg.imread(fp)
    except OSError:
        return None
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    elif im.shape[-1] == 4:
        im = im[..., :3]
    return im
