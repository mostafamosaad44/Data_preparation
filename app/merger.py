# app/merger.py
import os
import re
from typing import Tuple, List, Dict
from PIL import Image

# match anything that ends with _<y>_<x>.<ext>
_TILE_RE = re.compile(r"_(\d+)_(\d+)\.(\w+)$", re.IGNORECASE)

def _scan_tiles(tiles_dir: str) -> List[Dict]:
    files = []
    for name in os.listdir(tiles_dir):
        m = _TILE_RE.search(name)
        if not m:
            continue
        y = int(m.group(1))
        x = int(m.group(2))
        ext = m.group(3).lower()
        path = os.path.join(tiles_dir, name)
        files.append({"path": path, "x": x, "y": y, "ext": ext, "name": name})
    if not files:
        raise ValueError("No tiles found matching '*_Y_X.<ext>' at folder end (e.g., name_00064_00256.png).")
    return files

def _estimate_canvas_size(files: List[Dict]):
    mode = None
    max_right = 0
    max_bottom = 0
    for f in files:
        with Image.open(f["path"]) as im:
            if mode is None:
                mode = im.mode
            w, h = im.size
            right = f["x"] + w
            bottom = f["y"] + h
            max_right = max(max_right, right)
            max_bottom = max(max_bottom, bottom)
    return max_right, max_bottom, (mode or "RGB")

def merge_tiles(tiles_dir: str, output_path: str) -> Tuple[int, int]:
    files = _scan_tiles(tiles_dir)
    W, H, mode = _estimate_canvas_size(files)
    base = Image.new(mode, (W, H))
    for f in files:
        with Image.open(f["path"]) as im:
            base.paste(im, (f["x"], f["y"]))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base.save(output_path)
    return (W, H)
