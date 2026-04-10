#!/usr/bin/env python3
"""
Build a side-by-side montage: **shared baseline** | **method output** for each notebook run.

Images are resolved in order:

1. Optional per-slug ``recon_compare.json`` in the run folder (paths relative to that slug dir)::

     {"baseline": "zf.png", "reconstruction": "out.png"}

2. Otherwise PNG/JPEG files in the slug directory (preference order for baseline:
   names containing ``zf``, ``zero``, ``baseline``; for method: ``recon``, ``pred``, ``final``, ``refined``).

3. Otherwise decode ``image/png`` outputs from ``executed.ipynb`` (use
   ``--baseline-img-index`` / ``--method-img-index``; default **first** image for
   baseline slug, **last** image for each method — often the final summary figure).

Examples::

  python scripts/notebook_pipeline/visualize_reconstructions.py --run results/notebook_runs/20260409_185846 \\
      --baseline-slug week_5_notebook

  python scripts/notebook_pipeline/visualize_reconstructions.py --run results/notebook_runs/20260409_185846 \\
      --baseline-image path/to/shared_zf.png --method-img-index -1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import notebook_batch_utils as nbu


def _load_summary(run_root: Path) -> list[dict]:
    p = run_root / "combined_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("combined_summary.json must be a list")
    return data


def _slug_dir(run_root: Path, row: dict) -> Path:
    slug = row.get("slug") or "unknown"
    return (run_root / slug).resolve()


def _images_from_glob(work: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG"):
        out.extend(sorted(work.glob(pat)))
    return sorted(set(out), key=lambda p: p.name.lower())


def _pick_baseline_disk(work: Path) -> Path | None:
    files = _images_from_glob(work)
    if not files:
        return None
    low = [(p.name.lower(), p) for p in files]
    for key in ("zf", "zero", "baseline", "filled", "ifft"):
        for name, p in low:
            if key in name:
                return p
    return files[0]


def _pick_method_disk(work: Path) -> Path | None:
    files = _images_from_glob(work)
    if not files:
        return None
    low = [(p.name.lower(), p) for p in files]
    for key in ("recon", "pred", "final", "refined", "output", "result", "sr", "cnn"):
        for name, p in low:
            if key in name:
                return p
    return files[-1]


def _array_from_index(imgs: list, idx: int) -> np.ndarray | None:
    if not imgs:
        return None
    if idx < 0:
        idx = len(imgs) + idx
    if idx < 0 or idx >= len(imgs):
        return None
    im = imgs[idx]
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    elif im.shape[-1] == 4:
        im = im[..., :3]
    return im


def _load_baseline_array(
    run_root: Path,
    rows: list[dict],
    baseline_slug: str | None,
    baseline_image: Path | None,
    baseline_img_index: int,
) -> tuple[np.ndarray | None, str]:
    if baseline_image is not None:
        bp = baseline_image.expanduser().resolve()
        if bp.is_file():
            im = nbu.load_image_path(bp.parent, bp.name)
            return im, str(bp)
        return None, ""

    if not baseline_slug:
        return None, ""

    row = next((r for r in rows if r.get("slug") == baseline_slug), None)
    if row is None:
        return None, ""
    wd = _slug_dir(run_root, row)
    man = nbu.load_recon_compare_manifest(wd)
    if man and isinstance(man.get("baseline"), str):
        im = nbu.load_image_path(wd, man["baseline"])
        if im is not None:
            return im, f"{wd}/{man['baseline']}"

    p = _pick_baseline_disk(wd)
    if p is not None:
        im = nbu.load_image_path(wd, p.name)
        if im is not None:
            return im, str(p)

    exe = wd / "executed.ipynb"
    if exe.is_file():
        embedded = nbu.extract_png_images_from_notebook(exe)
        im = _array_from_index(embedded, baseline_img_index)
        if im is not None:
            return im, f"{exe} (embedded #{baseline_img_index})"
    return None, ""


def _method_array(
    run_root: Path,
    row: dict,
    method_img_index: int,
) -> tuple[np.ndarray | None, str]:
    wd = _slug_dir(run_root, row)
    man = nbu.load_recon_compare_manifest(wd)
    if man and isinstance(man.get("reconstruction"), str):
        im = nbu.load_image_path(wd, man["reconstruction"])
        if im is not None:
            return im, f"{wd}/{man['reconstruction']}"

    p = _pick_method_disk(wd)
    if p is not None:
        im = nbu.load_image_path(wd, p.name)
        if im is not None:
            return im, str(p)

    exe = wd / "executed.ipynb"
    if exe.is_file():
        embedded = nbu.extract_png_images_from_notebook(exe)
        im = _array_from_index(embedded, method_img_index)
        if im is not None:
            return im, f"{exe} (embedded #{method_img_index})"
    return None, ""


def render_reconstruction_montage(
    run_root: Path,
    baseline_slug: str | None = None,
    baseline_image: Path | None = None,
    baseline_img_index: int = 0,
    method_img_index: int = -1,
    only_ok: bool = False,
) -> Path:
    run_root = run_root.resolve()
    rows = _load_summary(run_root)
    if only_ok:
        rows = [r for r in rows if r.get("exit_code") == 0]

    base_im, base_src = _load_baseline_array(
        run_root, rows, baseline_slug, baseline_image, baseline_img_index
    )
    out_dir = run_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reconstruction_montage.png"

    n = len(rows)
    if n == 0:
        raise ValueError("No rows to plot")

    fig, axes = plt.subplots(n, 2, figsize=(9, 2.8 * n))
    if n == 1:
        axes = np.array([axes])
    if base_im is not None:
        sup = f"Shared baseline: {base_src[:140]}"
    else:
        sup = (
            "No baseline image — pass --baseline-slug or --baseline-image, "
            "or add recon_compare.json with a baseline path in each slug folder."
        )
    fig.suptitle(sup, fontsize=8, y=1.0)

    for i, row in enumerate(rows):
        label = nbu.methodology_label_from_row(row)
        m_im, _m_src = _method_array(run_root, row, method_img_index)

        ax0, ax1 = axes[i, 0], axes[i, 1]
        if base_im is not None:
            ax0.imshow(np.clip(base_im, 0, 1))
        else:
            ax0.text(0.5, 0.5, "(no baseline)", ha="center", va="center", fontsize=10, transform=ax0.transAxes)
        ax0.set_axis_off()
        if i == 0:
            ax0.set_title("Baseline (shared)", fontsize=9)

        if m_im is not None:
            ax1.imshow(np.clip(m_im, 0, 1))
        else:
            ax1.text(0.5, 0.5, "(no image)", ha="center", va="center", fontsize=9, transform=ax1.transAxes)
        ax1.set_axis_off()
        ax1.set_title(label[:80] + ("…" if len(label) > 80 else ""), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Montage: baseline | reconstruction per method")
    ap.add_argument("--run", type=Path, required=True, help="run directory with combined_summary.json")
    ap.add_argument(
        "--baseline-slug",
        default="week_5_notebook",
        help="Slug folder for shared baseline (default: week_5_notebook).",
    )
    ap.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not load a shared baseline (right column only shows each method's image).",
    )
    ap.add_argument("--baseline-image", type=Path, default=None, help="Explicit baseline image (overrides --baseline-slug)")
    ap.add_argument(
        "--baseline-img-index",
        type=int,
        default=0,
        help="When baseline comes from executed.ipynb images: which embedded image (0=first)",
    )
    ap.add_argument(
        "--method-img-index",
        type=int,
        default=-1,
        help="When method image comes from executed.ipynb: which image (-1=last)",
    )
    ap.add_argument("--only-ok", action="store_true", help="Only notebook rows with exit_code 0")
    args = ap.parse_args(argv)

    run_root = args.run.expanduser().resolve()
    bs = None if args.no_baseline else args.baseline_slug
    try:
        out = render_reconstruction_montage(
            run_root,
            baseline_slug=bs,
            baseline_image=args.baseline_image,
            baseline_img_index=args.baseline_img_index,
            method_img_index=args.method_img_index,
            only_ok=args.only_ok,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
