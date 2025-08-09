# app/splitter.py
import os
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

try:
    from tifffile import TiffFile
    HAS_TIFFILE = True
except Exception:
    HAS_TIFFILE = False


def _load_image_any(input_path: str) -> np.ndarray:
    try:
        img = Image.open(input_path)
        arr = np.array(img)
        if arr.ndim in (2, 3):
            return arr
    except Exception:
        pass
    if HAS_TIFFILE:
        with TiffFile(input_path) as tif:
            arr = tif.asarray()
        if arr.ndim in (2, 3):
            return arr
    raise ValueError("Unsupported image format or could not load image.")


def _apply_band_selection(arr: np.ndarray, selected_bands: Optional[List[int]]) -> np.ndarray:
    if arr.ndim == 2 or not selected_bands:
        return arr
    H, W, C = arr.shape
    for b in selected_bands:
        if b < 0 or b >= C:
            raise ValueError(f"Selected band index {b} out of range [0..{C-1}]")
    return arr[:, :, selected_bands]


def _normalize_to_uint8(arr: np.ndarray, mode: str = "minmax") -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if mode not in ("minmax", "clip"):
        mode = "minmax"
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        if mode == "minmax":
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
        else:
            a = np.clip(a, 0, 255)
        return a.astype(np.uint8)
    H, W, C = arr.shape
    out = np.empty((H, W, C), dtype=np.uint8)
    for c in range(C):
        a = arr[..., c].astype(np.float32)
        if mode == "minmax":
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
        else:
            a = np.clip(a, 0, 255)
        out[..., c] = a.astype(np.uint8)
    return out


def _ensure_format_compat(arr: np.ndarray, extension: str, policy: str = "auto"):
    info: Dict[str, Any] = {}
    ext = extension.lower() if extension.startswith(".") else "." + extension
    if arr.ndim == 3:
        C = arr.shape[2]
        if ext in (".jpg", ".jpeg", ".webp"):
            if C not in (1, 3):
                if policy == "auto":
                    info["note"] = f"Incompatible channels ({C}) for {ext}; using first 3 channels."
                    arr = arr[:, :, :3]
                else:
                    raise ValueError(f"Incompatible channels ({C}) for {ext}. Use .png/.tif or policy='auto'.")
        if ext in (".png", ".tif", ".tiff") and C > 4:
            if policy == "auto":
                info["note"] = f"Channels >4 ({C}) may fail in Pillow; using first 4."
                arr = arr[:, :, :4]
            else:
                raise ValueError(f"Too many channels ({C}) for saver with {ext}.")
    return arr, info


def split_large_image(
    input_path: str,
    output_dir: str,
    tile_size: int,
    extension: str = ".png",
    selected_bands: Optional[List[int]] = None,
    normalize_mode: str = "minmax",
    policy: str = "auto",
    name_pattern: str = "{base}_tile_{y}_{x}"  # NEW
) -> Dict[str, Any]:
    """
    Split an image into tiles with a custom filename pattern.

    name_pattern tokens:
      {base}  : input file name without extension
      {y},{x} : top-left pixel offsets
      {row},{col} : aliases for y,x
      {i}     : running index (0,1,2,...) — supports formatting e.g. {i:06d}
      {ext}   : optional; if present, will be replaced by extension without dot
    """
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = extension if extension.startswith(".") else "." + extension

    arr = _load_image_any(input_path)
    orig_shape = tuple(arr.shape)

    arr = _apply_band_selection(arr, selected_bands)
    arr = _normalize_to_uint8(arr, mode=normalize_mode)
    arr, info = _ensure_format_compat(arr, ext, policy=policy)

    H, W = arr.shape[:2]
    is_color = (arr.ndim == 3)

    tiles = 0
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            tile = arr[y:y2, x:x2] if not is_color else arr[y:y2, x:x2, :]

            # build filename from pattern
            # allow {i} with formatting; we do two passes: format i first, then the rest
            filename = name_pattern
            # handle {i} formatting safely
            try:
                filename = filename.format(i=tiles)
            except Exception:
                # if user didn't include {i} or used wrong spec, ignore
                pass
            # fill other tokens
            filename = filename.format(
                base=base, y=y, x=x, row=y, col=x, ext=ext.lstrip(".")
            )
            if not filename.lower().endswith(ext):
                filename = f"{filename}{ext}"

            out_path = os.path.join(output_dir, filename)
            Image.fromarray(tile).save(out_path)
            tiles += 1

    result: Dict[str, Any] = {
        "tiles": tiles,
        "shape": orig_shape,
        "dtype": str(np.array(arr).dtype),
    }
    if "note" in info:
        result["note"] = info["note"]
    return result
