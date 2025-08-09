# app/gui.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from app.splitter import split_large_image, _load_image_any
from app.merger import merge_tiles


EXT_OPTIONS = [".png", ".jpg", ".tif"]


class ImageSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dtata Preparation")
        self.root.geometry("840x620")

        self._build_notebook()

        # ----- Split state -----
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.tile_size = tk.IntVar(value=512)
        self.extension = tk.StringVar(value=".png")
        self.name_pattern = tk.StringVar(value="{base}_tile_{y}_{x}")
        self.selected_bands_vars = []
        self.band_checkbuttons = []

        # ----- Merge state -----
        self.tiles_dir = tk.StringVar()
        self.merge_out_path = tk.StringVar()
        self.merge_estimate_lbl = None

        # tabs
        self._build_split_tab()
        self._build_merge_tab()

    # ---------------- Notebook ----------------
    def _build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.tab_split = ttk.Frame(self.nb)
        self.tab_merge = ttk.Frame(self.nb)
        self.nb.add(self.tab_split, text="Split")
        self.nb.add(self.tab_merge, text="Merge")
        self.nb.pack(fill="both", expand=True)

    # ---------------- Split Tab ----------------
    def _build_split_tab(self):
        tab = self.tab_split

        # Input
        tk.Label(tab, text="Input Image:").place(x=20, y=20)
        tk.Entry(tab, textvariable=self.input_path, width=78).place(x=140, y=20)
        tk.Button(tab, text="Browse...", command=self._browse_input).place(x=740, y=18)

        # Output dir
        tk.Label(tab, text="Output Directory:").place(x=20, y=60)
        tk.Entry(tab, textvariable=self.output_dir, width=78).place(x=140, y=60)
        tk.Button(tab, text="Browse...", command=self._browse_output).place(x=740, y=58)

        # Tile size
        tk.Label(tab, text="Tile Size (px):").place(x=20, y=100)
        tk.Entry(tab, textvariable=self.tile_size, width=10).place(x=140, y=100)

        # Output format
        tk.Label(tab, text="Output Format:").place(x=20, y=140)
        ttk.Combobox(tab, textvariable=self.extension, values=EXT_OPTIONS,
                     state="readonly", width=12).place(x=140, y=140)

        # Filename pattern
        tk.Label(tab, text="Filename Pattern:").place(x=20, y=180)
        tk.Entry(tab, textvariable=self.name_pattern, width=62).place(x=140, y=180)
        tk.Label(tab, text="Tokens: {base} {y} {x} {row} {col} {i} {ext}",
                 fg="gray").place(x=140, y=205)

        # Bands
        tk.Label(tab, text="Select Bands to Export:").place(x=20, y=240)
        self.bands_frame = tk.Frame(tab, bd=1, relief="groove")
        self.bands_frame.place(x=20, y=270, width=360, height=300)

        # Actions
        tk.Button(tab, text="Preview First Tile",
                  command=self._preview_first_tile, bg="#1976D2", fg="white").place(x=420, y=270, width=200, height=34)
        tk.Button(tab, text="Split Image",
                  command=self._split_image, bg="#2E7D32", fg="white").place(x=420, y=320, width=200, height=34)

        # Status
        self.status_lbl = tk.Label(tab, text="", fg="gray")
        self.status_lbl.place(x=420, y=380)

    # Pickers
    def _browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"),
                       ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)
            self._load_band_checklist(path)

    def _browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir.set(folder)

    # Bands
    def _load_band_checklist(self, file_path: str):
        for w in self.bands_frame.winfo_children():
            w.destroy()
        self.selected_bands_vars.clear()
        self.band_checkbuttons.clear()

        try:
            arr = _load_image_any(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read image:\n{e}")
            return

        if arr.ndim == 2:
            var = tk.IntVar(value=1)
            cb = tk.Checkbutton(self.bands_frame, text="Band 0", variable=var)
            cb.pack(anchor="w", padx=6, pady=2)
            self.selected_bands_vars.append(var)
            self.band_checkbuttons.append(cb)
        else:
            bands = arr.shape[2]
            for b in range(bands):
                var = tk.IntVar(value=1)  # default select all
                cb = tk.Checkbutton(self.bands_frame, text=f"Band {b}", variable=var)
                cb.pack(anchor="w", padx=6, pady=2)
                self.selected_bands_vars.append(var)
                self.band_checkbuttons.append(cb)

    def _selected_bands(self):
        return [i for i, v in enumerate(self.selected_bands_vars) if v.get() == 1] or None

    # Preview
    def _preview_first_tile(self):
        if not self.input_path.get():
            messagebox.showinfo("Preview", "Please select an input image first.")
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
            self._show_preview_window(tile_arr)
            self.status_lbl.config(text="Preview opened.", fg="gray")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def _show_preview_window(self, tile_arr: np.ndarray):
        img = Image.fromarray(tile_arr)
        max_side = 512
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        top = tk.Toplevel(self.root)
        top.title("Preview")
        top.resizable(False, False)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo  # keep ref
        lbl.pack(padx=8, pady=8)

    @staticmethod
    def _to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        if arr.ndim == 2:
            a = arr.astype(np.float32)
            mn, mx = float(a.min()), float(a.max())
            if mx > mn:
                a = (a - mn) / (mx - mn) * 255.0
            else:
                a[:] = 0
            return a.astype(np.uint8)
        h, w, c = arr.shape
        out = np.empty((h, w, c), dtype=np.uint8)
        for i in range(c):
            a = arr[..., i].astype(np.float32)
            mn, mx = float(a.min()), float(a.max())
            if mx > mn:
                a = (a - mn) / (mx - mn) * 255.0
            else:
                a[:] = 0
            out[..., i] = a.astype(np.uint8)
        return out

    # Split
    def _split_image(self):
        if not self.input_path.get() or not os.path.isfile(self.input_path.get()):
            messagebox.showerror("Error", "Please select a valid input image.")
            return
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
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
            self.status_lbl.config(text="Done.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Splitting failed:\n{e}")
            self.status_lbl.config(text="Failed.", fg="red")

    # ---------------- Merge Tab ----------------
    def _build_merge_tab(self):
        tab = self.tab_merge

        tk.Label(tab, text="Tiles Folder:").place(x=20, y=30)
        tk.Entry(tab, textvariable=self.tiles_dir, width=78).place(x=140, y=30)
        tk.Button(tab, text="Browse...", command=self._browse_tiles_dir).place(x=740, y=28)

        tk.Label(tab, text="Output File:").place(x=20, y=70)
        tk.Entry(tab, textvariable=self.merge_out_path, width=78).place(x=140, y=70)
        tk.Button(tab, text="Save As...", command=self._choose_merge_output).place(x=740, y=68)

        tk.Button(tab, text="Scan & Estimate Size", command=self._estimate_merge_size).place(x=20, y=120)
        self.merge_estimate_lbl = tk.Label(tab, text="", fg="gray")
        self.merge_estimate_lbl.place(x=200, y=123)

        tk.Button(tab, text="Merge Tiles", command=self._do_merge,
                  bg="#6A1B9A", fg="white").place(x=20, y=170, width=200, height=36)

        self.merge_status_lbl = tk.Label(tab, text="", fg="gray")
        self.merge_status_lbl.place(x=20, y=220)

    def _browse_tiles_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.tiles_dir.set(folder)

    def _choose_merge_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif;*.tiff"), ("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if path:
            self.merge_out_path.set(path)

    def _estimate_merge_size(self):
        try:
            # reuse scanner/estimator from merger
            from app.merger import _scan_tiles, _estimate_canvas_size
            files = _scan_tiles(self.tiles_dir.get())
            W, H, mode = _estimate_canvas_size(files)
            self.merge_estimate_lbl.config(text=f"Estimated size: {W} x {H} px, mode: {mode}")
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


def run_app():
    root = tk.Tk()
    app = ImageSplitterGUI(root)
    root.mainloop()
