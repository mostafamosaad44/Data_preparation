import os
import warnings
from typing import List, Optional, Dict, Any, Tuple, Callable

import numpy as np
from PIL import Image

# اسمح بفتح الصور العملاقة (وكتم تحذير DecompressionBomb)
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

try:
    import tifffile as _TT
    HAS_TIFFILE = True
    # كتم تحذيرات tifffile (زي: parsing GDAL_NODATA ...)
    warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")
except Exception:
    HAS_TIFFILE = False


# ---------------- I/O helpers ----------------
def _load_image_any(input_path: str) -> np.ndarray:
    """Load image with Pillow first; fallback to tifffile for BigTIFF/multi-page."""
    try:
        with Image.open(input_path) as im:
            arr = np.array(im)
        if arr.ndim in (2, 3):
            return arr
    except Exception:
        pass

    if HAS_TIFFILE and input_path.lower().endswith((".tif", ".tiff")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _TT.TiffFile(input_path) as tif:
                arr = tif.asarray()
        if arr.ndim in (2, 3):
            return arr

    raise ValueError(f"Unsupported image format or could not load image: {input_path}")


def _apply_band_selection(arr: np.ndarray, selected_bands: Optional[List[int]]) -> np.ndarray:
    if arr.ndim == 2 or not selected_bands:
        return arr
    H, W, C = arr.shape
    for b in selected_bands:
        if b < 0 or b >= C:
            raise ValueError(f"Selected band index {b} out of range [0..{C-1}]")
    return arr[:, :, selected_bands]


def _normalize_to_uint8(arr: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Normalize to uint8. mode: 'minmax' (per-channel) or 'clip' (0..255)."""
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


def _ensure_format_compat(arr: np.ndarray, extension: str, policy: str = "auto") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ensure the array channels are compatible with the output format.
    - JPEG/WEBP: 1 or 3 channels only.
    - PNG/TIFF: عادة بيدعموا لحد 4 قنوات. أكتر => نقصّ لـ4 عند policy='auto'.
    """
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
        elif ext in (".png", ".tif", ".tiff"):
            if C > 4:
                if policy == "auto":
                    info["note"] = f"Channels >4 ({C}) may fail in Pillow; using first 4."
                    arr = arr[:, :, :4]
                else:
                    raise ValueError(f"Too many channels ({C}) for saver with {ext}.")
    return arr, info


def _save_kwargs_for_ext(ext: str) -> Dict[str, Any]:
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in (".jpg", ".jpeg"):
        return {"quality": 95, "optimize": True}
    if ext in (".tif", ".tiff"):
        return {"compression": "tiff_lzw"}
    return {}


# ---------- Internal: stream a crop safely ----------
def _extract_crop_as_uint8(
    im: Image.Image,
    box: Tuple[int, int, int, int],
    selected_bands: Optional[List[int]],
    normalize_mode: str,
) -> np.ndarray:
    crop = im.crop(box)

    # حاول نطبّق اختيار القنوات عبر PIL
    try:
        if selected_bands is not None:
            parts = crop.split()
            if len(parts) >= max(selected_bands) + 1:
                parts_sel = tuple(parts[i] for i in selected_bands)
                if len(parts_sel) == 1:
                    crop = parts_sel[0]
                elif len(parts_sel) == 3:
                    crop = Image.merge("RGB", parts_sel[:3])
                elif len(parts_sel) == 4:
                    crop = Image.merge("RGBA", parts_sel[:4])
                else:
                    crop = Image.merge("RGBA", parts_sel[:4])
    except Exception:
        pass

    arr = np.array(crop)
    if arr.ndim == 3 and selected_bands is not None and arr.shape[2] >= max(selected_bands) + 1:
        arr = arr[:, :, selected_bands]
    arr = _normalize_to_uint8(arr, mode=normalize_mode)
    return arr


# ---------------- Public splitter (streaming) ----------------
def split_large_image(
    input_path: str,
    output_dir: str,
    tile_size: int,
    extension: str = ".png",
    selected_bands: Optional[List[int]] = None,
    normalize_mode: str = "minmax",
    policy: str = "auto",
    name_pattern: str = "{base}_tile_{y}_{x}",
    overlap_pct: float = 0.0,
    # Optional T2 (aligned grid)
    t2_path: Optional[str] = None,
    # Base names & patterns
    t1_base: Optional[str] = None,
    t2_base: Optional[str] = None,
    name_pattern_t2: Optional[str] = None,
    # Optional manifest
    write_manifest: bool = True,
    scene_id: Optional[str] = None,
    label_path: str = "",
    fold: str = "train",
    write_parquet: bool = False,
    # progress callback: fn(i:int, total:int) -> bool (return True to cancel)
    progress: Optional[Callable[[int, int], bool]] = None,
) -> Dict[str, Any]:
    """
    Split image into tiles (crop-by-crop) with optional overlap.
    - يكتب التايلز في مجلدين: output_dir/T1 و output_dir/T2 (لو T2 موجود).
    - يكتب manifest.csv مستقل لكل واحد.
    - اسماء T2 بتستخدم base مختلف عن T1 (حسب t2_base أو base+"_T2").
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap_pct < 0 or overlap_pct >= 100:
        raise ValueError("overlap_pct must be in [0, <100).")

    os.makedirs(output_dir, exist_ok=True)

    base_in = os.path.splitext(os.path.basename(input_path))[0]
    base1 = (t1_base or "").strip() or base_in
    base2 = (t2_base or "").strip() or (base1 + "_T2")

    ext = extension if extension.startswith(".") else "." + extension
    save_kwargs = _save_kwargs_for_ext(ext)

    # --- افتح T1 بدون تحميل كامل ---
    try:
        im1 = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Could not open image: {input_path}\n{e}")

    W, H = im1.size
    # dtype تقريبي من كروب صغير
    try:
        probe = np.array(im1.crop((0, 0, min(W, 32), min(H, 32))))
        dtype_str = str(probe.dtype)
    except Exception:
        dtype_str = "unknown"

    # أنشئ مجلدات T1/T2
    t1_dir = os.path.join(output_dir, "T1")
    os.makedirs(t1_dir, exist_ok=True)

    im2 = None
    t2_used = False
    t2_dir = None
    if t2_path and os.path.isfile(t2_path):
        try:
            im2 = Image.open(t2_path)
            if im2.size == im1.size:
                t2_dir = os.path.join(output_dir, "T2")
                os.makedirs(t2_dir, exist_ok=True)
                t2_used = True
            else:
                im2.close()
                im2 = None
        except Exception:
            im2 = None
            t2_used = False

    # --- build coords with overlap ---
    overlap_px = int(round(tile_size * float(overlap_pct) / 100.0))
    step = tile_size if overlap_px <= 0 else max(1, tile_size - overlap_px)
    coords = [(y, x, min(y + tile_size, H), min(x + tile_size, W))
              for y in range(0, H, step)
              for x in range(0, W, step)]
    total = len(coords)

    # --- manifests ---
    rows_t1: List[List[Any]] = []
    rows_t2: List[List[Any]] = []
    scene = (scene_id or base1)
    fold = (fold or "train").lower()
    label_src = label_path or ""

    pat1 = name_pattern or "{base}_tile_{y}_{x}"
    pat2 = (name_pattern_t2 if name_pattern_t2 not in (None, "") else pat1)

    tiles = 0
    info_note: Optional[str] = None

    try:
        for i, (y, x, y2, x2) in enumerate(coords):
            if progress and progress(i, total):
                break

            # --- T1 ---
            try:
                tile_arr = _extract_crop_as_uint8(im1, (x, y, x2, y2), selected_bands, normalize_mode)
                tile_arr, info = _ensure_format_compat(tile_arr, ext, policy=policy)
                if not info_note and "note" in info:
                    info_note = info["note"]
            except MemoryError as me:
                raise MemoryError(f"Out of memory while processing tile ({y},{x}). Try smaller tile_size.\n{me}")
            except Exception as e:
                raise RuntimeError(f"Failed to generate tile at ({y},{x}): {e}")

            fname1 = pat1
            try:
                fname1 = fname1.format(i=i)
            except Exception:
                pass
            fname1 = fname1.format(base=base1, y=y, x=x, row=y, col=x, ext=ext.lstrip("."))
            if not fname1.lower().endswith(ext):
                fname1 = f"{fname1}{ext}"

            t1_path = os.path.join(t1_dir, fname1)
            Image.fromarray(tile_arr).save(t1_path, **save_kwargs)

            # manifest T1
            tile_x = x // step
            tile_y = y // step
            w = int(x2 - x)
            h = int(y2 - y)
            if write_manifest:
                rows_t1.append([scene, tile_x, tile_y, x, y, w, h, t1_path, label_src, fold])

            # --- T2 (إن وُجد) ---
            if t2_used and im2 is not None:
                try:
                    tile2 = _extract_crop_as_uint8(im2, (x, y, x2, y2), selected_bands, normalize_mode)
                    tile2, _ = _ensure_format_compat(tile2, ext, policy=policy)
                    fname2 = pat2
                    try:
                        fname2 = fname2.format(i=i)
                    except Exception:
                        pass
                    fname2 = fname2.format(base=base2, y=y, x=x, row=y, col=x, ext=ext.lstrip("."))
                    if not fname2.lower().endswith(ext):
                        fname2 = f"{fname2}{ext}"

                    t2_tile_path = os.path.join(t2_dir, fname2)
                    Image.fromarray(tile2).save(t2_tile_path, **save_kwargs)

                    if write_manifest:
                        rows_t2.append([scene, tile_x, tile_y, x, y, w, h, t2_tile_path, label_src, fold])
                except Exception as e:
                    if not info_note:
                        info_note = f"T2 tile failed at ({y},{x}): {e}"

            tiles += 1
    finally:
        try:
            im1.close()
        except Exception:
            pass
        try:
            if im2:
                im2.close()
        except Exception:
            pass

    # ---- write manifests (واحد لكل فولدر) ----
    t1_manifest = None
    t2_manifest = None
    if write_manifest and rows_t1:
        t1_manifest = os.path.join(t1_dir, "manifest.csv")
        import csv as _csv
        with open(t1_manifest, "w", newline="", encoding="utf-8") as f:
            wcsv = _csv.writer(f)
            # خليه generic: نفس أعمدة الـ GUI (تستخدم t1_path كاسم عمود افتراضي)
            wcsv.writerow(["scene_id", "tile_x", "tile_y", "x0", "y0", "w", "h", "t1_path", "label_path", "fold"])
            wcsv.writerows(rows_t1)

        if write_parquet:
            try:
                import pandas as pd
                cols = ["scene_id","tile_x","tile_y","x0","y0","w","h","t1_path","label_path","fold"]
                pd.DataFrame(rows_t1, columns=cols).to_parquet(
                    os.path.join(t1_dir, "manifest.parquet"), index=False
                )
            except Exception:
                pass

    if write_manifest and t2_used and rows_t2:
        t2_manifest = os.path.join(t2_dir, "manifest.csv")
        import csv as _csv
        with open(t2_manifest, "w", newline="", encoding="utf-8") as f:
            wcsv = _csv.writer(f)
            # للوضوح نكتب t2_path؛ والـ GUI بيتعامل مع أي اسم عمود (t1_path/t2_path/path)
            wcsv.writerow(["scene_id", "tile_x", "tile_y", "x0", "y0", "w", "h", "t2_path", "label_path", "fold"])
            wcsv.writerows(rows_t2)

        if write_parquet:
            try:
                import pandas as pd
                cols = ["scene_id","tile_x","tile_y","x0","y0","w","h","t2_path","label_path","fold"]
                pd.DataFrame(rows_t2, columns=cols).to_parquet(
                    os.path.join(t2_dir, "manifest.parquet"), index=False
                )
            except Exception:
                pass

    result: Dict[str, Any] = {
        "tiles": tiles,
        "shape": (H, W),
        "dtype": dtype_str,
        "t1_dir": t1_dir,
        "t2_dir": t2_dir if t2_used else "",
        "t1_manifest": t1_manifest or "",
        "t2_manifest": t2_manifest or "",
        "t2_used": bool(t2_used),
    }
    if info_note:
        result["note"] = info_note
    return result
