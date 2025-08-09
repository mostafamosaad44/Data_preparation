WHAT THIS APP DOES
• Split very large images into tiles, and merge tiles back into a single image.
• Works with most formats supported by Pillow: TIFF/TIF, JPEG/JPG, PNG, BMP, GIF, WEBP.
• Supports grayscale and multi-band images (e.g., 3-band RGB or 4-band RGB+NIR).

SPLITTER (Left Menu > Splitter)
1) Choose an input image and an output folder.
2) Tile Size: tile width/height in pixels (e.g., 256 / 512 / 1024).
3) Output Format: .png / .jpg / .tif / .tiff / .jpeg.
4) Filename Pattern controls tile names. Tokens:
   {base} -> input filename without extension
   {y}, {x} (or {row}, {col}) -> tile top-left offsets
   {i} -> running index (supports formatting like {i:06d})
   {ext} -> extension without dot (optional)
   Example patterns:
     {base}_tile_{y}_{x}
     {base}_{y:05d}_{x:05d}
   NOTE: For the Merge tool to work, include {y} and {x} in the pattern
         so filenames end with _<y>_<x>.<ext>.

BANDS
• When you pick an input image, the app lists available bands (channels).
• Select which bands you want to export (default: all are selected).
• If saving as JPEG and channels > 3, the app keeps the first 3 (policy='auto').

PREVIEW
• "Preview First Tile" shows the first tile (0:tile_size, 0:tile_size) with your band selection.

MERGE TOOL
• Pick the folder of tiles and choose an output filename.
• "Scan & Estimate Size" reads positions from filenames ending with _<y>_<x>.<ext>.
• "Merge Tiles" stitches them back to a single image.
• Prefer .tif for very large outputs.

TIPS
• Recommended tile sizes for ML: 256 / 512 / 1024.
• Install 'imagecodecs' for faster TIFF/PNG decoding.
• Extremely large merges may require lots of RAM; use .tif.