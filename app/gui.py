# app/gui.py
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np
from PIL import Image, ImageTk

from app.splitter import split_large_image, _load_image_any
from app.merger import merge_tiles

APP_TITLE = "Data Prepration"
APP_VERSION = "v1.0"
AUTHOR = "Built by Mosatafa Mosaad"

SETTINGS_FILE = "splitter_settings.json"

# Output formats shown to user
EXT_OPTIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# File dialog filter
INPUT_FILTER = [
    ("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"),
    ("All files", "*.*"),
]


class ImagePrepApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x640")
        self.minsize(900, 600)

        # ---- State (with sensible defaults) ----
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.tile_size = tk.IntVar(value=512)
        self.extension = tk.StringVar(value=".png")
        self.name_pattern = tk.StringVar(value="{base}_tile_{y}_{x}")
        self.selected_bands_vars = []   # populated per-image

        self.tiles_dir = tk.StringVar()
        self.merge_out_path = tk.StringVar()

        # ---- Layout: Sidebar + Content ----
        self._build_layout()
        self._create_pages()
        self._wire_sidebar()
        self._load_settings()

        # default page
        self._show_page("split")

        # save settings on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- Layout ----------------
    def _build_layout(self):
        # Header
        header = tk.Frame(self, bg="#f5f5f7", height=52)
        header.pack(side="top", fill="x")
        tk.Label(header, text=f"{APP_TITLE} — {APP_VERSION}",
                 font=("Segoe UI", 13, "bold"), bg="#f5f5f7").pack(side="left", padx=12, pady=10)
        tk.Label(header, text=AUTHOR, fg="#666", bg="#f5f5f7").pack(side="right", padx=12)

        # Body split: sidebar + content
        body = tk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        # Sidebar
        self.sidebar = tk.Frame(body, width=180, bg="#23262d")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.btn_split = tk.Button(self.sidebar, text="Splitter", fg="white", bg="#2e3138",
                                   relief="flat", command=lambda: self._show_page("split"))
        self.btn_merge = tk.Button(self.sidebar, text="Merge Tool", fg="white", bg="#2e3138",
                                   relief="flat", command=lambda: self._show_page("merge"))
        self.btn_preview = tk.Button(self.sidebar, text="Preview", fg="white", bg="#2e3138",
                                     relief="flat", command=lambda: self._show_page("preview"))
        self.btn_help = tk.Button(self.sidebar, text="Help", fg="white", bg="#2e3138",
                                  relief="flat", command=lambda: self._show_page("help"))

        for b in (self.btn_split, self.btn_merge, self.btn_preview, self.btn_help):
            b.pack(fill="x", padx=10, pady=8, ipady=6)

        # Content area (stacked pages)
        self.content = tk.Frame(body, bg="white")
        self.content.pack(side="left", fill="both", expand=True)

        self.pages = {}  # name -> frame

    def _wire_sidebar(self):
        # highlight current button (simple)
        def highlight(name):
            for btn, key in [
                (self.btn_split, "split"),
                (self.btn_merge, "merge"),
                (self.btn_preview, "preview"),
                (self.btn_help, "help"),
            ]:
                btn.configure(bg="#5a5f6b" if key == name else "#2e3138")

        self._highlight = highlight

    def _show_page(self, name: str):
        for k, f in self.pages.items():
            f.place_forget()
        self.pages[name].place(x=0, y=0, relwidth=1, relheight=1)
        self._highlight(name)

    # --------------- Pages (Frames) ---------------
    def _create_pages(self):
        self.pages["split"] = self._build_split_page(self.content)
        self.pages["merge"] = self._build_merge_page(self.content)
        self.pages["preview"] = self._build_preview_page(self.content)
        self.pages["help"] = self._build_help_page(self.content)

    # ---------------- Split Page ----------------
    def _build_split_page(self, parent):
        page = tk.Frame(parent, bg="white")

        row_y = 24
        gap = 40

        tk.Label(page, text="Input Image:", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.input_path, width=90).place(x=140, y=row_y)
        tk.Button(page, text="Browse...", command=self._pick_input).place(x=840, y=row_y-2)
        row_y += gap

        tk.Label(page, text="Output Folder:", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.output_dir, width=90).place(x=140, y=row_y)
        tk.Button(page, text="Browse...", command=self._pick_output).place(x=840, y=row_y-2)
        row_y += gap

        tk.Label(page, text="Tile Size (px):", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.tile_size, width=10).place(x=140, y=row_y)
        row_y += gap

        tk.Label(page, text="Output Format:", bg="white").place(x=20, y=row_y)
        ttk.Combobox(page, textvariable=self.extension, values=EXT_OPTIONS,
                     state="readonly", width=12).place(x=140, y=row_y)
        row_y += gap

        tk.Label(page, text="Filename Pattern:", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.name_pattern, width=60).place(x=140, y=row_y)
        tk.Label(page, text="Tokens: {base} {y} {x} {row} {col} {i} {ext}",
                 fg="#666", bg="white").place(x=140, y=row_y+24)
        row_y += gap + 24

        # Bands checklist
        tk.Label(page, text="Select Bands:", bg="white").place(x=20, y=row_y)
        self.band_box = tk.Frame(page, bd=1, relief="groove", bg="white")
        self.band_box.place(x=20, y=row_y+28, width=360, height=320)

        # Actions
        tk.Button(page, text="Preview First Tile", bg="#1976D2", fg="white",
                  command=self._preview_from_split).place(x=420, y=row_y+28, width=200, height=34)
        tk.Button(page, text="Split Image", bg="#2E7D32", fg="white",
                  command=self._do_split).place(x=420, y=row_y+74, width=200, height=34)

        self.split_status = tk.Label(page, text="", fg="#666", bg="white")
        self.split_status.place(x=420, y=row_y+120)

        return page

    # ---------------- Merge Page ----------------
    def _build_merge_page(self, parent):
        page = tk.Frame(parent, bg="white")

        row_y = 24
        gap = 44

        tk.Label(page, text="Tiles Folder:", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.tiles_dir, width=90).place(x=140, y=row_y)
        tk.Button(page, text="Browse...", command=self._pick_tiles_dir).place(x=840, y=row_y-2)
        row_y += gap

        tk.Label(page, text="Output File:", bg="white").place(x=20, y=row_y)
        tk.Entry(page, textvariable=self.merge_out_path, width=90).place(x=140, y=row_y)
        tk.Button(page, text="Save As...", command=self._pick_merge_out).place(x=840, y=row_y-2)
        row_y += gap

        tk.Button(page, text="Scan & Estimate Size", command=self._estimate_merge).place(x=20, y=row_y)
        self.merge_estimate_lbl = tk.Label(page, text="", fg="#666", bg="white")
        self.merge_estimate_lbl.place(x=200, y=row_y+2)
        row_y += gap

        tk.Button(page, text="Merge Tiles", bg="#6A1B9A", fg="white",
                  command=self._do_merge).place(x=20, y=row_y, width=200, height=36)
        self.merge_status_lbl = tk.Label(page, text="", fg="#666", bg="white")
        self.merge_status_lbl.place(x=20, y=row_y+46)

        return page

    # ---------------- Preview Page ----------------
    def _build_preview_page(self, parent):
        page = tk.Frame(parent, bg="white")

        tk.Label(page, text="Quick Preview (uses Splitter settings)", bg="white",
                 font=("Segoe UI", 11, "bold")).place(x=20, y=20)

        tk.Button(page, text="Preview First Tile", bg="#1976D2", fg="white",
                  command=self._preview_from_split).place(x=20, y=60, width=220, height=34)

        self.preview_info = tk.Label(page, text="", fg="#666", bg="white")
        self.preview_info.place(x=20, y=110)

        return page

    # ---------------- Help Page ----------------
    def _build_help_page(self, parent):
        page = tk.Frame(parent, bg="white")

        tk.Label(page, text=f"{APP_TITLE} — {APP_VERSION}", font=("Segoe UI", 14, "bold"),
                 bg="white").pack(anchor="w", padx=16, pady=(16, 6))
        tk.Label(page, text=AUTHOR, fg="#666", bg="white").pack(anchor="w", padx=16, pady=(0, 6))

        box = scrolledtext.ScrolledText(page, wrap="word", width=120, height=28)
        box.pack(fill="both", expand=True, padx=16, pady=10)

        help_text = """
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

"""
        box.insert("1.0", help_text)
        box.configure(state="disabled")

        return page

    # --------------- Pickers ---------------
    def _pick_input(self):
        path = filedialog.askopenfilename(filetypes=INPUT_FILTER)
        if path:
            self.input_path.set(path)
            self._populate_bands(path)
            self._save_settings()

    def _pick_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir.set(folder)
            self._save_settings()

    def _pick_tiles_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.tiles_dir.set(folder)

    def _pick_merge_out(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif;*.tiff"), ("PNG", "*.png"),
                       ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if path:
            self.merge_out_path.set(path)

    # --------------- Bands ---------------
    def _populate_bands(self, file_path: str):
        # Clear old
        for w in self.band_box.winfo_children():
            w.destroy()
        self.selected_bands_vars.clear()

        try:
            arr = _load_image_any(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read image:\n{e}")
            return

        if arr.ndim == 2:
            var = tk.IntVar(value=1)
            cb = tk.Checkbutton(self.band_box, text="Band 0", variable=var, bg="white")
            cb.pack(anchor="w", padx=6, pady=2)
            self.selected_bands_vars.append(var)
        else:
            bands = arr.shape[2]
            for b in range(bands):
                var = tk.IntVar(value=1)  # select all by default
                cb = tk.Checkbutton(self.band_box, text=f"Band {b}", variable=var, bg="white")
                cb.pack(anchor="w", padx=6, pady=2)
                self.selected_bands_vars.append(var)

    def _selected_bands(self):
        # return None if none picked -> splitter treats as "all"
        picked = [i for i, v in enumerate(self.selected_bands_vars) if v.get() == 1]
        return picked or None

    # --------------- Preview ---------------
    def _preview_from_split(self):
        if not self.input_path.get():
            messagebox.showinfo("Preview", "Please select an input image first (Splitter page).")
            return

        try:
            tile = int(self.tile_size.get())
            if tile <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid size", "Tile size must be a positive integer.")
            return

        try:
            arr = _load_image_any(self.input_path.get())
            sel = self._selected_bands()
            if arr.ndim == 3 and sel is not None:
                arr = arr[:, :, sel]

            arr = self._to_uint8_minmax(arr)
            y2 = min(tile, arr.shape[0])
            x2 = min(tile, arr.shape[1])
            tile_arr = arr[0:y2, 0:x2] if arr.ndim == 2 else arr[0:y2, 0:x2, :]
            self._show_preview(tile_arr)
            self.preview_info.config(text="Preview opened (first tile).")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def _show_preview(self, tile_arr: np.ndarray):
        img = Image.fromarray(tile_arr)
        max_side = 512
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        top = tk.Toplevel(self)
        top.title("Preview")
        top.resizable(False, False)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)

    @staticmethod
    def _to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        if arr.ndim == 2:
            a = arr.astype(np.float32)
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
            return a.astype(np.uint8)
        h, w, c = arr.shape
        out = np.empty((h, w, c), dtype=np.uint8)
        for i in range(c):
            a = arr[..., i].astype(np.float32)
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
            out[..., i] = a.astype(np.uint8)
        return out

    # --------------- Split action ---------------
    def _do_split(self):
        if not self.input_path.get() or not os.path.isfile(self.input_path.get()):
            messagebox.showerror("Error", "Please select a valid input image.")
            return
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        try:
            tile = int(self.tile_size.get())
            if tile <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Tile size must be a positive integer.")
            return

        selected_bands = self._selected_bands()

        try:
            result = split_large_image(
                input_path=self.input_path.get(),
                output_dir=self.output_dir.get(),
                tile_size=tile,
                extension=self.extension.get(),
                selected_bands=selected_bands,
                normalize_mode="minmax",
                policy="auto",
                name_pattern=self.name_pattern.get()
            )
            msg = (
                f"✅ Splitting complete!\n"
                f"Tiles created: {result['tiles']}\n"
                f"Original shape: {result['shape']}\n"
                f"Data type: {result['dtype']}"
            )
            if "note" in result:
                msg += f"\n\nNote: {result['note']}"
            messagebox.showinfo("Success", msg)
            self.split_status.config(text="Done.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Splitting failed:\n{e}")
            self.split_status.config(text="Failed.", fg="red")

    # --------------- Merge actions ---------------
    def _estimate_merge(self):
        try:
            from app.merger import _scan_tiles, _estimate_canvas_size
            files = _scan_tiles(self.tiles_dir.get())
            W, H, mode = _estimate_canvas_size(files)
            self.merge_estimate_lbl.config(text=f"Estimated: {W} x {H} px, mode: {mode}")
        except Exception as e:
            messagebox.showerror("Estimate error", str(e))

    def _do_merge(self):
        if not self.tiles_dir.get():
            messagebox.showerror("Error", "Please select a tiles folder.")
            return
        if not self.merge_out_path.get():
            messagebox.showerror("Error", "Please choose an output file.")
            return
        try:
            W, H = merge_tiles(self.tiles_dir.get(), self.merge_out_path.get())
            self.merge_status_lbl.config(text=f"✅ Merged image saved ({W} x {H}).", fg="green")
            messagebox.showinfo("Success", f"Merged image saved.\nSize: {W} x {H}")
        except Exception as e:
            self.merge_status_lbl.config(text=f"❌ Merge failed: {e}", fg="red")
            messagebox.showerror("Merge error", str(e))

    # --------------- Settings (save/load) ---------------
    def _save_settings(self):
        data = {
            "input_path": self.input_path.get(),
            "output_dir": self.output_dir.get(),
            "tile_size": self.tile_size.get(),
            "extension": self.extension.get(),
            "name_pattern": self.name_pattern.get(),
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.input_path.set(data.get("input_path") or "")
            self.output_dir.set(data.get("output_dir") or "")
            self.tile_size.set(int(data.get("tile_size", 512)))
            self.extension.set(data.get("extension") or ".png")
            self.name_pattern.set(data.get("name_pattern") or "{base}_tile_{y}_{x}")
            if self.input_path.get():
                self._populate_bands(self.input_path.get())
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()


def run_app():
    app = ImagePrepApp()
    app.mainloop()
