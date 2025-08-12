# app/merger.py
import os
import re
import csv
from typing import List, Tuple, Dict, Optional

from PIL import Image


# =========================
# Utilities for folder-based merge (legacy / existing)
# =========================

_TILE_RE = re.compile(r".*_(\d+)_(\d+)\.[A-Za-z0-9]+$")  # ..._<y>_<x>.<ext>


def _scan_tiles(tiles_dir: str) -> List[str]:
    """Return a sorted list of tile file paths under tiles_dir."""
    paths: List[str] = []
    for root, _, files in os.walk(tiles_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            # accept images only
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
                paths.append(fp)
    # keep deterministic order
    paths.sort()
    return paths


def _extract_xy_from_name(path: str) -> Optional[Tuple[int, int]]:
    """
    Extract (x, y) from filename suffix '_<y>_<x>.<ext>'.
    Returns (x, y) or None if not matched.
    """
    m = _TILE_RE.match(os.path.basename(path))
    if not m:
        return None
    y = int(m.group(1))
    x = int(m.group(2))
    return (x, y)


def _estimate_canvas_size(files: List[str]) -> Tuple[int, int, str]:
    """
    Estimate canvas size from a set of tiles named with '_<y>_<x>'.
    Returns (W, H, mode_hint).
    """
    if not files:
        raise ValueError("No tile files found.")
    # open first to get size/mode
    with Image.open(files[0]) as im0:
        tw, th = im0.size
        mode = im0.mode

    max_x2 = 0
    max_y2 = 0
    for fp in files:
        xy = _extract_xy_from_name(fp)
        if xy is None:
            # fall back: pack tiles in grid by index (not ideal)
            continue
        x, y = xy
        max_x2 = max(max_x2, x + tw)
        max_y2 = max(max_y2, y + th)

    if max_x2 == 0 or max_y2 == 0:
        # fallback if names not matched
        cols = int(len(files) ** 0.5 + 0.999)
        rows = (len(files) + cols - 1) // cols
        max_x2 = cols * tw
        max_y2 = rows * th

    return (max_x2, max_y2, mode)


def merge_tiles(tiles_dir: str, output_path: str) -> Tuple[int, int]:
    """
    Merge tiles from a folder where filenames end with '_<y>_<x>.<ext>'.
    Last write wins in overlapping areas.
    Returns (W, H) of the merged image.
    """
    files = _scan_tiles(tiles_dir)
    if not files:
        raise ValueError("No tiles found in the selected folder.")

    W, H, mode_hint = _estimate_canvas_size(files)

    # choose target mode
    target_mode = "RGB"
    if mode_hint in ("L", "I;16", "I;16B", "I;16L", "I", "F"):
        target_mode = "L"
    elif mode_hint in ("RGBA", "LA"):
        target_mode = "RGBA"

    bg = 0 if target_mode in ("L",) else (0, 0, 0, 0) if target_mode == "RGBA" else (0, 0, 0)
    canvas = Image.new(target_mode, (W, H), bg)

    for fp in files:
        with Image.open(fp) as im:
            if im.mode != target_mode:
                im = im.convert(target_mode)
            xy = _extract_xy_from_name(fp)
            if xy is None:
                # if not matched, skip or place sequentially (we skip to be strict)
                continue
            x, y = xy
            canvas.paste(im, (x, y))

    canvas.save(output_path)
    return canvas.size
# =========================
# END: folder-based merge
# =========================


# =========================
# NEW: Merge from manifest.csv
# =========================

_REQUIRED_COLS = ["x0", "y0", "w", "h", "t1_path"]
# optional: scene_id,tile_x,tile_y,t2_path,label_path,fold


def _read_manifest(manifest_csv: str) -> List[Dict[str, str]]:
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    rows: List[Dict[str, str]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [c.strip() for c in reader.fieldnames or []]
        missing = [c for c in _REQUIRED_COLS if c not in header]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def _infer_target_mode(first_image_path: str) -> str:
    with Image.open(first_image_path) as im:
        m = im.mode
        if m in ("L", "I;16", "I;16B", "I;16L", "I", "F"):
            return "L"
        if m in ("RGBA", "LA"):
            return "RGBA"
        return "RGB"


def merge_tiles_from_manifest(
    manifest_csv: str,
    output_path: str,
    fold_filter: Optional[str] = None,
    column: str = "t1_path",
    background: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, int]:
    """
    Merge tiles described in manifest.csv.
    - Uses x0,y0,w,h and the image path column (default: t1_path).
    - If fold_filter is provided, only rows with fold==fold_filter are used.
    - Overlaps are pasted in file order; last one wins.
    Returns (W, H).
    """
    rows = _read_manifest(manifest_csv)

    # optional fold filtering
    if fold_filter:
        rows = [r for r in rows if (r.get("fold", "").lower() == fold_filter.lower())]
        if not rows:
            raise ValueError(f"No rows found for fold='{fold_filter}' in manifest.")

    # compute canvas size
    max_x2 = 0
    max_y2 = 0
    first_img_path = None
    for r in rows:
        try:
            x0 = int(float(r["x0"]))
            y0 = int(float(r["y0"]))
            w = int(float(r["w"]))
            h = int(float(r["h"]))
        except Exception:
            raise ValueError("Manifest x0/y0/w/h must be numeric.")
        max_x2 = max(max_x2, x0 + w)
        max_y2 = max(max_y2, y0 + h)
        if not first_img_path:
            candidate = (r.get(column) or "").strip()
            if candidate:
                first_img_path = candidate

    if not first_img_path:
        raise ValueError(f"No valid image paths found in manifest column '{column}'.")

    target_mode = _infer_target_mode(first_img_path)
    if background is None:
        background = 0 if target_mode == "L" else (0, 0, 0, 0) if target_mode == "RGBA" else (0, 0, 0)

    canvas = Image.new(target_mode, (max_x2, max_y2), background)

    for r in rows:
        img_path = (r.get(column) or "").strip()
        if not img_path or not os.path.isfile(img_path):
            # skip silently; could also raise warning
            continue
        try:
            x0 = int(float(r["x0"]))
            y0 = int(float(r["y0"]))
            w = int(float(r["w"]))
            h = int(float(r["h"]))
        except Exception:
            # skip bad row
            continue

        with Image.open(img_path) as im:
            if im.mode != target_mode:
                im = im.convert(target_mode)
            # ensure size agrees (w,h) if present; if not, just paste
            if (w, h) != im.size:
                # resize to declared tile size in manifest to be safe
                im = im.resize((w, h), Image.BILINEAR)
            canvas.paste(im, (x0, y0))

    canvas.save(output_path)
    return canvas.size
# =========================
# END: merge from manifest
# =========================
