# app/splitter.py
import os
import warnings
from typing import List, Optional, Dict, Any, Tuple, Callable, Iterable, Union

import numpy as np
from PIL import Image

# -------- أمان وتحذيرات --------
# اسمح بفتح الصور العملاقة، وكتم تحذير DecompressionBomb
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

try:
    import tifffile as _TT
    HAS_TIFFILE = True
    # كتم تحذيرات tifffile (زي: parsing GDAL_NODATA ...)
    warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")
except Exception:
    HAS_TIFFILE = False


# -------- مساعدات عامة --------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _normalize_extension(ext: str) -> str:
    if not ext:
        raise ValueError("Output extension is empty.")
    e = ext.lower()
    if not e.startswith("."):
        e = "." + e
    if e not in VALID_EXTS:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {sorted(VALID_EXTS)}")
    return e


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise OSError(f"Could not create output directory '{path}': {e}") from e


def _validate_selected_bands(sel: Optional[Iterable[Union[int, np.integer]]]) -> Optional[List[int]]:
    if sel is None:
        return None
    out: List[int] = []
    for v in sel:
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
        else:
            raise ValueError(f"Selected band index '{v}' is not an integer.")
    if len(out) != len(set(out)):
        raise ValueError("Selected bands contain duplicates.")
    if any(b < 0 for b in out):
        raise ValueError("Selected band indices must be >= 0.")
    return out


# ---------------- I/O helpers (تحميل كامل عند الحاجة فقط) ----------------
def _load_image_any(input_path: str) -> np.ndarray:
    """
    يحاول تحميل الصورة بالكامل (استعمال داخلي/اختياري).
    Pillow أولاً؛ لو TIFF كبير نجرّب tifffile.
    """
    # Pillow
    try:
        with Image.open(input_path) as im:
            arr = np.array(im)
        if arr.ndim in (2, 3):
            return arr
    except Exception:
        pass

    # tifffile (لو TIFF)
    if HAS_TIFFILE and input_path.lower().endswith((".tif", ".tiff")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _TT.TiffFile(input_path) as tif:
                arr = tif.asarray()
        if arr.ndim in (2, 3):
            return arr

    raise ValueError(f"Unsupported image format or could not load image: {input_path}")


# ---------------- معالجة قنوات وتطبيع ----------------
def _apply_band_selection(arr: np.ndarray, selected_bands: Optional[List[int]]) -> np.ndarray:
    if arr.ndim == 2 or not selected_bands:
        return arr
    H, W, C = arr.shape
    for b in selected_bands:
        if b < 0 or b >= C:
            raise ValueError(f"Selected band index {b} out of range [0..{C-1}]")
    return arr[:, :, selected_bands]


def _normalize_to_uint8(arr: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    Normalize to uint8. mode: 'minmax' (per-channel) أو 'clip' (0..255).
    """
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
    يتأكد من توافق عدد القنوات مع الصيغة.
    - JPEG/WEBP: 1 أو 3 قنوات فقط.
    - PNG/TIFF: عادة حتى 4 قنوات. أكتر => نقصّ لـ4 لو policy='auto'.
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
    """
    إعدادات حفظ مناسبة لكل صيغة.
    """
    ext = _normalize_extension(ext)
    if ext in (".jpg", ".jpeg"):
        return {"quality": 95, "optimize": True}
    if ext in (".tif", ".tiff"):
        # الأفضل وجود imagecodecs؛ وإلا Pillow بيشتغل برضه
        return {"compression": "tiff_lzw"}
    return {}


# ---------- Internal: stream a crop safely ----------
def _extract_crop_as_uint8(
    im: Image.Image,
    box: Tuple[int, int, int, int],
    selected_bands: Optional[List[int]],
    normalize_mode: str,
) -> np.ndarray:
    """
    يقرأ كروب صغير من PIL ويطبّق اختيار القنوات + التطبيع إلى uint8.
    """
    # بعض الصيغ (P, I;16, F) بنسهلها للعرض
    if im.mode in ("P",):
        im = im.convert("RGB")

    crop = im.crop(box)

    # اختيار القنوات عبر PIL إن أمكن
    try:
        if selected_bands is not None and len(selected_bands) > 0:
            parts = crop.split()  # tuple of single-band images
            if len(parts) >= max(selected_bands) + 1:
                parts_sel = tuple(parts[i] for i in selected_bands)
                if len(parts_sel) == 1:
                    crop = parts_sel[0]
                elif len(parts_sel) == 3:
                    crop = Image.merge("RGB", parts_sel[:3])
                elif len(parts_sel) == 4:
                    crop = Image.merge("RGBA", parts_sel[:4])
                else:
                    # >4 قنوات للعرض/الحفظ: ناخد أول 4 مؤقتًا
                    crop = Image.merge("RGBA", parts_sel[:4])
    except Exception:
        # fallback للنومبا
        pass

    # حول لnumpy وطبّق minmax (لو مش uint8)
    arr = np.array(crop)
    if arr.ndim == 3 and selected_bands is not None and arr.shape[2] >= max(selected_bands) + 1:
        # في حالة PIL رجّع 5+ قنوات (نادر)، نطبّق سيلكشن هنا
        arr = arr[:, :, selected_bands]
    arr = _normalize_to_uint8(arr, mode=normalize_mode)
    return arr


# ---------------- Public splitter (streaming، من غير تحميل كامل) ----------------
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
    # Optional manifest
    write_manifest: bool = False,
    scene_id: Optional[str] = None,
    label_path: str = "",
    fold: str = "train",
    write_parquet: bool = False,
    # Optional progress callback: fn(i:int, total:int) -> bool (return True to cancel)
    progress: Optional[Callable[[int, int], bool]] = None,
) -> Dict[str, Any]:
    """
    يقطع الصورة لتايلات (crop-by-crop) مع دعم overlap%، وT2، وmanifest.

    ملاحظات:
      • التطبيع يتم لكل تايل على حدة لتقليل الذاكرة.
      • في حال قنوات كثيرة >4، بنحاول نلائم الصيغة تلقائيًا لما policy='auto'.
    """
    # ----- Validation قوية لكل المُدخلات -----
    if not input_path or not os.path.isfile(input_path):
        raise ValueError("Input image path is invalid or not found.")
    if t2_path and not os.path.isfile(t2_path):
        raise ValueError("T2 path is set but not found on disk.")
    if tile_size is None or int(tile_size) <= 0:
        raise ValueError("tile_size must be a positive integer.")
    try:
        tile_size = int(tile_size)
    except Exception:
        raise ValueError("tile_size must be an integer.")
    try:
        overlap_pct = float(overlap_pct)
    except Exception:
        raise ValueError("overlap_pct must be a number.")
    if overlap_pct < 0 or overlap_pct >= 100:
        raise ValueError("overlap_pct must be in [0, <100).")
    extension = _normalize_extension(extension)
    if normalize_mode not in ("minmax", "clip"):
        raise ValueError("normalize_mode must be 'minmax' or 'clip'.")
    if policy not in ("auto", "strict"):
        raise ValueError("policy must be 'auto' or 'strict'.")
    selected_bands = _validate_selected_bands(selected_bands)
    _ensure_dir(output_dir)

    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = extension
    save_kwargs = _save_kwargs_for_ext(ext)

    # --- افتح T1 (بدون تحميل كامل) ---
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

    # --- optional T2 ---
    im2 = None
    t2_used = False
    t2_dir = None
    if t2_path:
        try:
            im2 = Image.open(t2_path)
            if im2.size != im1.size:
                # mismatch => تجاهل
                try:
                    im2.close()
                except Exception:
                    pass
                im2 = None
            else:
                t2_dir = os.path.join(output_dir, "T2")
                _ensure_dir(t2_dir)
                t2_used = True
        except Exception as e:
            # هنكمّل T1 فقط
            im2 = None
            t2_used = False

    # --- build coords with overlap ---
    overlap_px = int(round(tile_size * overlap_pct / 100.0))
    step = tile_size if overlap_px <= 0 else max(1, tile_size - overlap_px)
    coords = [(y, x, min(y + tile_size, H), min(x + tile_size, W))
              for y in range(0, H, step)
              for x in range(0, W, step)]
    total = len(coords)

    # --- manifest rows (optional) ---
    rows: List[List[Any]] = []
    scene = (scene_id or base)
    fold = (fold or "train").lower()
    label_src = label_path or ""

    # --- tiling (streaming) ---
    tiles = 0
    info_note: Optional[str] = None

    try:
        for i, (y, x, y2, x2) in enumerate(coords):
            if progress and progress(i, total):
                # cancelled by caller
                break

            # T1 tile
            try:
                tile_arr = _extract_crop_as_uint8(im1, (x, y, x2, y2), selected_bands, normalize_mode)
                # تأكد من التوافق مع الفورمات
                tile_arr, info = _ensure_format_compat(tile_arr, ext, policy=policy)
                if not info_note and "note" in info:
                    info_note = info["note"]
            except MemoryError as me:
                raise MemoryError(
                    f"Out of memory while processing tile at (y={y}, x={x}). "
                    f"Try a smaller tile_size or lower overlap."
                ) from me
            except Exception as e:
                raise RuntimeError(f"Failed to generate tile at (y={y}, x={x}): {e}") from e

            # اسم الملف (two-pass formatting)
            filename = name_pattern
            try:
                filename = filename.format(i=i)
            except Exception:
                # لو المستخدم حاطط فورمات غريبة لـ {i}، نتجاهلها
                pass
            try:
                filename = filename.format(base=base, y=y, x=x, row=y, col=x, ext=ext.lstrip("."))
            except KeyError as ke:
                raise ValueError(f"Unknown token in name_pattern: {ke}. Allowed: "
                                 "{base},{y},{x},{row},{col},{i},{ext}") from ke
            if not filename.lower().endswith(ext):
                filename = f"{filename}{ext}"

            t1_path = os.path.join(output_dir, filename)
            try:
                Image.fromarray(tile_arr).save(t1_path, **save_kwargs)
            except Exception as e:
                raise OSError(f"Failed to save tile to '{t1_path}': {e}") from e

            # T2 tile (لو موجود)
            t2_tile_path = ""
            if t2_used and im2 is not None:
                try:
                    tile2 = _extract_crop_as_uint8(im2, (x, y, x2, y2), selected_bands, normalize_mode)
                    tile2, _ = _ensure_format_compat(tile2, ext, policy=policy)
                    t2_tile_path = os.path.join(t2_dir, filename)
                    Image.fromarray(tile2).save(t2_tile_path, **save_kwargs)
                except Exception as e:
                    # ما نوقفش العملية كلها؛ نضيف ملاحظة ونكمل
                    t2_tile_path = ""
                    if not info_note:
                        info_note = f"T2 tile failed at (y={y}, x={x}): {e}"

            if write_manifest:
                tile_x = x // step
                tile_y = y // step
                w = int(x2 - x)
                h = int(y2 - y)
                rows.append([scene, tile_x, tile_y, x, y, w, h, t1_path, t2_tile_path, label_src, fold])

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

    manifest_path = None
    if write_manifest and rows:
        manifest_path = os.path.join(output_dir, "manifest.csv")
        import csv as _csv
        try:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                wcsv = _csv.writer(f)
                wcsv.writerow(["scene_id", "tile_x", "tile_y", "x0", "y0", "w", "h",
                               "t1_path", "t2_path", "label_path", "fold"])
                wcsv.writerows(rows)
        except Exception as e:
            raise OSError(f"Failed to write manifest CSV: {e}") from e

        if write_parquet:
            try:
                import pandas as pd
                cols = ["scene_id", "tile_x", "tile_y", "x0", "y0", "w", "h",
                        "t1_path", "t2_path", "label_path", "fold"]
                parquet_path = os.path.join(output_dir, "manifest.parquet")
                pd.DataFrame(rows, columns=cols).to_parquet(parquet_path, index=False)
            except Exception:
                # لو مفيش pandas/pyarrow، أو أي خطأ—نتجاهل بهدوء (الـCSV موجود)
                pass

    result: Dict[str, Any] = {
        "tiles": tiles,
        "shape": (H, W),  # القنوات غير ثابتة عبر PIL لبعض الصيغ
        "dtype": dtype_str,
        "t2_used": bool(t2_used),
    }
    if info_note:
        result["note"] = info_note
    if manifest_path:
        result["manifest"] = manifest_path

    return result


__all__ = [
    "_load_image_any",
    "_apply_band_selection",
    "_normalize_to_uint8",
    "_ensure_format_compat",
    "_save_kwargs_for_ext",
    "split_large_image",
]
