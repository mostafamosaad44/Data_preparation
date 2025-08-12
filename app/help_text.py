# app/help_text.py
# -*- coding: utf-8 -*-

"""
Help content for the Data Preparation app.
Feel free to edit this file — the GUI reads the text as-is.
You can use the variables {APP_TITLE} and {APP_VERSION}; they will be substituted.
"""

HELP_TEXT_EN = r"""
{APP_TITLE} — {APP_VERSION}

Quick overview
--------------
This tool helps you prepare very large imagery for deep learning:
• Split an image into tiles with an optional overlap (as a percentage).
• Optionally split a second image (T2) on the exact same grid.
• Write a separate manifest for **T1** and for **T2** (each in its own folder).
• Merge tiles back either from filenames or from a manifest.csv.

Core inputs
-----------
1) T1 Input: the main image to be tiled.
2) T2 Input (optional): must have the exact same dimensions as T1 (otherwise it will be skipped).
3) Output Folder: we create subfolders **T1/** and **T2/** (if T2 exists).
4) Tile Size: tile width/height in pixels (e.g., 256, 512, 1024).
5) Overlap (%): percentage of tile size (0.. <100).
6) Output Format: file extension (.png / .jpg / .tif / .tiff / .jpeg).
7) Base (T1/T2): filename prefix. If empty:
   - Base(T1) = the T1 filename without extension
   - Base(T2) = Base(T1) + "_T2"
8) Filename Pattern: controls how tile names are generated. Supported tokens:
   {{base}}  {{y}}  {{x}}  {{row}}  {{col}}  {{i}}  {{ext}}
   Examples:
     {{base}}_tile_{{y}}_{{x}}
     {{base}}_{{y:05d}}_{{x:05d}}
   Note: merge-from-folder expects both {{y}} and {{x}} in the name.

Band selection
--------------
If your image has multiple bands, choose which bands to export.
When saving as JPEG/WEBP you must have 1 (grayscale) or 3 (RGB) channels.
PNG/TIFF safely support up to 4 channels.

Output layout
-------------
Output/
  ├─ T1/
  │   ├─ manifest.csv
  │   └─ <T1 tiles>
  └─ T2/  (if T2 is provided)
      ├─ manifest.csv
      └─ <T2 tiles>

Manifest columns
----------------
- For T1 (manifest.csv inside T1/):
  scene_id, tile_x, tile_y, x0, y0, w, h, t1_path, label_path, fold
- For T2 (manifest.csv inside T2/):
  scene_id, tile_x, tile_y, x0, y0, w, h, t2_path, label_path, fold

• tile_x, tile_y: grid indices (step = tile_size - overlap_px)  
• x0, y0: top-left pixel of each tile in the original image  
• w, h: on-disk size of the tile (width, height)

Preview
-------
“Preview First Tile” shows the first tile (0..tile_size) after band selection and normalization.

Merge
-----
• From Folder: requires filenames that end with _<y>_<x>.<ext>  
• From manifest.csv (recommended and more accurate):
  Reads x0, y0, w, h and the image path from the column (t1_path / t2_path / path).
  Prefer manifest-based merge because it does not rely on a strict naming scheme.

Performance tips
----------------
• For very large images, use a smaller tile_size to reduce memory usage.  
• TIFF with LZW compression offers a good size/speed trade-off.  
• Installing the **imagecodecs** package speeds up I/O, especially for TIFF/PNG.

Common errors & fixes
---------------------
• “Overlap too large”: decrease the overlap percentage or increase tile_size.  
• “T2 dimensions do not match”: T2 must have the same H×W as T1.  
• “Out of memory”: try a smaller tile_size.  
• “Unsupported image format”: convert to a standard format (TIFF/PNG/JPEG).

FAQ
---
Q: How do I choose the Base?  
A: Base is the filename prefix for tiles. Example: Base(T1)=2012_test ⇒  
   2012_test_tile_0_0.png … while Base(T2)=2012_test_T2 ⇒  
   2012_test_T2_tile_0_0.png

Q: Which merge method should I use?  
A: The manifest-based merge is more accurate because it uses exact coordinates.


Note: Merge using Manifest give results better than merge from Folder

Good luck ♥
"""

def get_help_text(app_title: str, app_version: str, lang: str = "en") -> str:
    """Return help text with {APP_TITLE}/{APP_VERSION} substituted.
    We keep other double-braced tokens like {{base}} intact.
    """
    text = HELP_TEXT_EN
    return text.replace("{APP_TITLE}", app_title).replace("{APP_VERSION}", app_version)
