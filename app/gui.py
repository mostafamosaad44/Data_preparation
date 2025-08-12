# app/gui.py
import os
import json
import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np
from PIL import Image, ImageTk

from app.splitter import _load_image_any
from app.merger import (
    _scan_tiles,
    _estimate_canvas_size,
)

APP_TITLE = "Data Preparation"
APP_VERSION = "v2.2"
AUTHOR = "Built by Mosatafa Mosaad"

SETTINGS_FILE = "splitter_settings.json"

EXT_OPTIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
INPUT_FILTER = [
    ("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"),
    ("All files", "*.*"),
]


# ------------------------- Tooltip helper -------------------------
class Tooltip:
    def __init__(self, widget, text, delay=600, wraplength=360):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._id = None
        self._tip = None
        self.widget.bind("<Enter>", self._schedule)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
            background="#ffffe0",
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=8, ipady=6)

    def _hide(self, _event=None):
        self._unschedule()
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ------------------------- Progress dialog (with Cancel) -------------------------
class ProgressDialog:
    def __init__(self, master: tk.Tk, title="Working...", determinate=True):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.top.resizable(False, False)
        self.top.transient(master)
        self.top.grab_set()
        self._total = 0
        self._value = 0
        self.cancelled = False

        frm = ttk.Frame(self.top, padding=12)
        frm.pack(fill="both", expand=True)

        self.msg = ttk.Label(frm, text="Please wait...")
        self.msg.pack(fill="x", pady=(0, 8))

        mode = "determinate" if determinate else "indeterminate"
        self.bar = ttk.Progressbar(frm, length=380, mode=mode, maximum=100)
        self.bar.pack(fill="x")

        bottom = ttk.Frame(frm)
        bottom.pack(fill="x", pady=(10, 0))

        self.counter = ttk.Label(bottom, text="")
        self.counter.pack(side="left")

        self.btn_cancel = ttk.Button(bottom, text="Cancel", command=self._on_cancel)
        self.btn_cancel.pack(side="right")

        if not determinate:
            self.bar.start(8)

        # center
        self.top.update_idletasks()
        w = self.top.winfo_width()
        h = self.top.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - w) // 2
        y = master.winfo_rooty() + (master.winfo_height() - h) // 2
        self.top.geometry(f"+{max(0,x)}+{max(0,y)}")

    def _on_cancel(self):
        self.cancelled = True
        self.set_message("Cancelling… please wait")

    def set_message(self, text: str):
        self.master.after(0, lambda: self.msg.config(text=text))

    def set_total(self, n: int):
        def _apply():
            self._total = max(1, int(n))
            self._value = 0
            self.bar.configure(mode="determinate", maximum=self._total, value=0)
            self.counter.config(text="0 / {}".format(self._total))
        self.master.after(0, _apply)

    def step(self, inc: int = 1):
        def _apply():
            self._value += inc
            self.bar.configure(value=self._value)
            self.counter.config(text="{} / {}".format(min(self._value, self._total), self._total))
        self.master.after(0, _apply)

    def set_indeterminate(self):
        def _apply():
            self.bar.configure(mode="indeterminate", maximum=100, value=0)
            self.bar.start(8)
            self.counter.config(text="")
        self.master.after(0, _apply)

    def close(self):
        def _apply():
            try:
                self.top.grab_release()
            except Exception:
                pass
            self.top.destroy()
        self.master.after(0, _apply)


# ------------------------- Main App -------------------------
class ImagePrepApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master: tk.Tk = master
        self.master.title(f"{APP_TITLE} — {APP_VERSION}")
        self.master.geometry("1100x780")
        self.master.minsize(1040, 720)

        self._init_style()

        # ---- State ----
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.tile_size = tk.IntVar(value=512)
        self.overlap_pct = tk.DoubleVar(value=0.0)  # [0, <100)
        self.extension = tk.StringVar(value=".png")
        self.name_pattern = tk.StringVar(value="{base}_tile_{y}_{x}")
        self.selected_bands_vars = []   # populated per-image

        # Rich manifest
        self.scene_id = tk.StringVar(value="")
        self.fold = tk.StringVar(value="train")
        self.dual_time = tk.BooleanVar(value=False)  # OFF => T1-only
        self.input_path_t2 = tk.StringVar()
        self.label_path = tk.StringVar()
        self.write_parquet = tk.BooleanVar(value=False)

        # Merge
        self.tiles_dir = tk.StringVar()
        self.merge_out_path = tk.StringVar()
        self.merge_fold_var = tk.StringVar(value="")  # "", "train", "val", "test"

        # Theme
        self.theme_var = tk.StringVar(value="System")  # Light / Dark / System

        # Layout
        self._build_root_layout()
        self._create_pages()
        self._load_settings()

        # default page
        self._show_page("split")

        # save on close
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI Style ----------------
    def _init_style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self._apply_theme_colors("System")

    def _apply_theme_colors(self, mode: str):
        if mode == "Dark":
            bg = "#1f232a"
            fg = "#e6e6e6"
            subbg = "#272c34"
            acc = "#3a82f7"
        elif mode == "Light":
            bg = "#ffffff"
            fg = "#111"
            subbg = "#f3f4f6"
            acc = "#0b5ed7"
        else:  # System
            bg = "#f8f9fb"
            fg = "#111"
            subbg = "#ffffff"
            acc = "#0b5ed7"

        self.master.configure(bg=bg)
        for elem in ("TFrame", "TLabelframe", "TLabelframe.Label", "TLabel", "TNotebook", "TScrollbar"):
            self.style.configure(elem, background=subbg if "Label" in elem else bg, foreground=fg)
        self.style.configure("TButton", background=subbg, foreground=fg)
        self.style.map("TButton", background=[("active", acc)])

        self.style.configure("Status.TLabel", foreground="#666" if mode != "Dark" else "#aab")
        self.style.configure("Accent.TButton", padding=6)
        try:
            self.style.configure("Horizontal.TProgressbar", troughcolor=subbg, background=acc)
        except Exception:
            pass

    # ---------------- Root Layout ----------------
    def _build_root_layout(self):
        header = ttk.Frame(self.master)
        header.pack(side="top", fill="x", padx=10, pady=(10, 6))

        left = ttk.Frame(header)
        left.pack(side="left")
        ttk.Label(left, text=APP_TITLE, font=("Segoe UI Semibold", 14)).pack(side="left")
        ttk.Label(left, text=f"{APP_VERSION} — {AUTHOR}", style="Status.TLabel").pack(side="left", padx=(10, 0))

        right = ttk.Frame(header)
        right.pack(side="right")
        ttk.Label(right, text="Theme:").pack(side="left", padx=(0, 6))
        theme_combo = ttk.Combobox(right, textvariable=self.theme_var,
                                   values=["System", "Light", "Dark"], state="readonly", width=10)
        theme_combo.pack(side="left")
        theme_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_theme_colors(self.theme_var.get()))
        Tooltip(theme_combo, "Switch basic theme colors.\n(System, Light, or Dark).")

        # Body
        body = ttk.Frame(self.master)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=6)

        # Side menu
        self.sidebar = ttk.Frame(body, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.btn_split = ttk.Button(self.sidebar, text="Splitter", command=lambda: self._show_page("split"))
        self.btn_merge = ttk.Button(self.sidebar, text="Merge Tool", command=lambda: self._show_page("merge"))
        self.btn_preview = ttk.Button(self.sidebar, text="Preview", command=lambda: self._show_page("preview"))
        self.btn_help = ttk.Button(self.sidebar, text="Help", command=lambda: self._show_page("help"))
        for b in (self.btn_split, self.btn_merge, self.btn_preview, self.btn_help):
            b.pack(fill="x", pady=6)

        # Content
        self.content = ttk.Frame(body)
        self.content.pack(side="left", fill="both", expand=True)
        self.pages = {}

        # Status bar
        status_row = ttk.Frame(self.master)
        status_row.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        self.status = ttk.Label(status_row, text="Ready", style="Status.TLabel", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

    def _show_page(self, name: str):
        for _, f in self.pages.items():
            f.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.status.config(text=f"Ready — {name.title()}")

    # ---------------- Pages ----------------
    def _create_pages(self):
        self.pages["split"] = self._build_split_page(self.content)
        self.pages["merge"] = self._build_merge_page(self.content)
        self.pages["preview"] = self._build_preview_page(self.content)
        self.pages["help"] = self._build_help_page(self.content)

    # ---------------- Split Page ----------------
    def _build_split_page(self, parent):
        page = ttk.Frame(parent)

        # Group: I/O
        io_frame = ttk.LabelFrame(page, text="Input / Output")
        io_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(io_frame, text="Input Image (T1):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        in_entry = ttk.Entry(io_frame, textvariable=self.input_path, width=80)
        in_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        in_btn = ttk.Button(io_frame, text="Browse...", command=self._pick_input)
        in_btn.grid(row=0, column=2, padx=6, pady=6)
        Tooltip(in_btn, "Choose the main (T1) image to split.")

        ttk.Label(io_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        out_entry = ttk.Entry(io_frame, textvariable=self.output_dir, width=80)
        out_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        out_btn = ttk.Button(io_frame, text="Browse...", command=self._pick_output)
        out_btn.grid(row=1, column=2, padx=6, pady=6)
        Tooltip(out_btn, "Choose where tiles and manifest will be saved.")
        io_frame.columnconfigure(1, weight=1)

        # Group: Tiling
        tiling = ttk.LabelFrame(page, text="Tiling")
        tiling.pack(fill="x", padx=10, pady=10)

        ttk.Label(tiling, text="Tile Size (px):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tile_entry = ttk.Entry(tiling, textvariable=self.tile_size, width=10)
        tile_entry.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        Tooltip(tile_entry, "Tile width and height in pixels (e.g., 256 / 512 / 1024).")

        ttk.Label(tiling, text="Overlap (%):").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        ov_entry = ttk.Entry(tiling, textvariable=self.overlap_pct, width=10)
        ov_entry.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        Tooltip(ov_entry, "Percentage of tile size to overlap between adjacent tiles.\n"
                          "Set 0 for no overlap. Typical values: 10–50%.")

        ttk.Label(tiling, text="Output Format:").grid(row=0, column=4, sticky="w", padx=6, pady=6)
        fmt_combo = ttk.Combobox(tiling, textvariable=self.extension, values=EXT_OPTIONS, state="readonly", width=10)
        fmt_combo.grid(row=0, column=5, sticky="w", padx=6, pady=6)
        Tooltip(fmt_combo, "File format for tiles (.png / .jpg / .tif / .tiff / .jpeg).")

        ttk.Label(tiling, text="Filename Pattern:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        pat_entry = ttk.Entry(tiling, textvariable=self.name_pattern, width=60)
        pat_entry.grid(row=1, column=1, columnspan=5, sticky="we", padx=6, pady=6)
        tip = ("Tokens:\n"
               "• {base}: original filename without extension\n"
               "• {y}/{x} (or {row}/{col}): top-left offsets\n"
               "• {i}: running index (supports {i:06d})\n"
               "• {ext}: extension without dot\n"
               "Examples: {base}_tile_{y}_{x}   |   {base}_{y:05d}_{x:05d}\n"
               "NOTE: folder-based Merge needs {y} and {x} in filenames.")
        Tooltip(pat_entry, tip)
        tiling.columnconfigure(1, weight=1)

        # Bands
        bands = ttk.LabelFrame(page, text="Bands (Channels)")
        bands.pack(fill="x", padx=10, pady=10)
        self.band_box = ttk.Frame(bands)
        self.band_box.pack(fill="x", padx=8, pady=8)

        # Metadata
        meta = ttk.LabelFrame(page, text="Training Metadata")
        meta.pack(fill="x", padx=10, pady=10)

        ttk.Label(meta, text="Scene ID:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        scene_entry = ttk.Entry(meta, textvariable=self.scene_id, width=28)
        scene_entry.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        Tooltip(scene_entry, "Unique identifier for the scene (defaults to base filename if empty).")

        ttk.Label(meta, text="Fold:").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        fold_combo = ttk.Combobox(meta, textvariable=self.fold, values=["train", "val", "test"], state="readonly", width=12)
        fold_combo.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        Tooltip(fold_combo, "Predefined split for ML datasets. Keeps scenes independent across folds.")

        dual_chk = ttk.Checkbutton(meta, text="Dual-time (T1/T2)", variable=self.dual_time)
        dual_chk.grid(row=1, column=0, sticky="w", padx=6, pady=6)
        Tooltip(dual_chk, "Uncheck for T1-only.\nCheck to split a second image (T2) on the exact same grid.")

        ttk.Label(meta, text="T2 Image:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        t2_entry = ttk.Entry(meta, textvariable=self.input_path_t2, width=60)
        t2_entry.grid(row=2, column=1, columnspan=2, sticky="we", padx=6, pady=6)
        t2_btn = ttk.Button(meta, text="Browse...", command=lambda: self._pick_input_generic(self.input_path_t2))
        t2_btn.grid(row=2, column=3, padx=6, pady=6)
        Tooltip(t2_btn, "Pick the second-time image (T2).\nDimensions must match T1; otherwise T2 is skipped.")

        ttk.Label(meta, text="Label (optional):").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        lbl_entry = ttk.Entry(meta, textvariable=self.label_path, width=60)
        lbl_entry.grid(row=3, column=1, columnspan=2, sticky="we", padx=6, pady=6)
        lbl_btn = ttk.Button(meta, text="Browse...", command=lambda: self._pick_input_generic(self.label_path))
        lbl_btn.grid(row=3, column=3, padx=6, pady=6)
        Tooltip(lbl_btn, "Optional per-scene label or annotations path. Stored as-is in manifest.")

        pq_chk = ttk.Checkbutton(meta, text="Write Parquet alongside CSV", variable=self.write_parquet)
        pq_chk.grid(row=4, column=0, sticky="w", padx=6, pady=6)
        Tooltip(pq_chk, "If pandas/pyarrow are installed, a manifest.parquet will be saved next to manifest.csv.")

        meta.columnconfigure(1, weight=1)
        meta.columnconfigure(2, weight=1)

        # Actions
        actions = ttk.Frame(page)
        actions.pack(fill="x", padx=10, pady=(0, 10))

        prev_btn = ttk.Button(actions, text="Preview First Tile", command=self._preview_from_split)
        prev_btn.pack(side="left", padx=(0, 8))
        Tooltip(prev_btn, "Shows the first tile (0:tile, 0:tile) using current band selection.")

        split_btn = ttk.Button(actions, text="Split Image", command=self._do_split_threaded)
        split_btn.pack(side="left")
        Tooltip(split_btn, "Start splitting with progress bar and write a rich manifest.\n"
                           "If Dual-time is enabled, T2 tiles are saved into an 'T2/' subfolder.")

        return page

    # ---------------- Merge Page ----------------
    def _build_merge_page(self, parent):
        page = ttk.Frame(parent)

        src = ttk.LabelFrame(page, text="Source")
        src.pack(fill="x", padx=10, pady=10)

        ttk.Label(src, text="Tiles Folder:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tiles_entry = ttk.Entry(src, textvariable=self.tiles_dir, width=80)
        tiles_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        tiles_btn = ttk.Button(src, text="Browse...", command=self._pick_tiles_dir)
        tiles_btn.grid(row=0, column=2, padx=6, pady=6)
        Tooltip(tiles_btn, "Select a folder of tiles whose names end with _<y>_<x>.<ext>.")

        ttk.Label(src, text="Output File:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        out_entry = ttk.Entry(src, textvariable=self.merge_out_path, width=80)
        out_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        out_btn = ttk.Button(src, text="Save As...", command=self._pick_merge_out)
        out_btn.grid(row=1, column=2, padx=6, pady=6)
        Tooltip(out_btn, "Where to save the merged image (use .tif for very large sizes).")
        src.columnconfigure(1, weight=1)

        ops = ttk.LabelFrame(page, text="Operations")
        ops.pack(fill="x", padx=10, pady=10)

        est_btn = ttk.Button(ops, text="Scan & Estimate Size", command=self._estimate_merge)
        est_btn.grid(row=0, column=0, padx=6, pady=6)
        Tooltip(est_btn, "Scan the folder and estimate merged canvas size and mode.")

        self.merge_estimate_lbl = ttk.Label(ops, text="")
        self.merge_estimate_lbl.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        merge_btn = ttk.Button(ops, text="Merge from Folder", command=self._do_merge_threaded)
        merge_btn.grid(row=1, column=0, padx=6, pady=6)
        Tooltip(merge_btn, "Merge tiles from folder (requires names ending in _<y>_<x>.<ext>).")

        m_manifest_btn = ttk.Button(ops, text="Merge from manifest.csv", command=self._merge_from_manifest_threaded)
        m_manifest_btn.grid(row=1, column=1, padx=6, pady=6)
        Tooltip(m_manifest_btn, "Merge by reading positions from manifest.csv (x0, y0, w, h, t1_path).")

        ttk.Label(ops, text="(optional) Fold filter:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        fold_combo = ttk.Combobox(ops, textvariable=self.merge_fold_var, values=["", "train", "val", "test"],
                                  state="readonly", width=10)
        fold_combo.grid(row=1, column=3, padx=6, pady=6, sticky="w")
        Tooltip(fold_combo, "If set, only rows with this fold are merged from the manifest.")

        return page

    # ---------------- Preview Page ----------------
    def _build_preview_page(self, parent):
        page = ttk.Frame(parent)

        info = ttk.Label(page, text="Quick Preview (uses Splitter settings)",
                         font=("Segoe UI Semibold", 11))
        info.pack(anchor="w", padx=10, pady=(10, 6))

        btn = ttk.Button(page, text="Preview First Tile", command=self._preview_from_split)
        btn.pack(anchor="w", padx=10, pady=6)
        Tooltip(btn, "Open a preview window with the first tile after band selection and normalization.")

        self.preview_info = ttk.Label(page, text="", style="Status.TLabel")
        self.preview_info.pack(anchor="w", padx=10, pady=(6, 10))

        return page

    # ---------------- Help Page ----------------
    def _build_help_page(self, parent):
        page = ttk.Frame(parent)

        ttk.Label(page, text=f"{APP_TITLE} — {APP_VERSION}", font=("Segoe UI Semibold", 14)).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        ttk.Label(page, text=AUTHOR, style="Status.TLabel").pack(anchor="w", padx=10, pady=(0, 8))

        box = scrolledtext.ScrolledText(page, wrap="word", width=120, height=30, font=("Segoe UI", 10))
        box.pack(fill="both", expand=True, padx=10, pady=10)

        help_text = """
OVERVIEW
This tool prepares large imagery for ML:
• Split huge images into tiles with optional overlap (%, not px).
• Optionally split a second-time image (T2) on the same grid.
• Write a rich manifest for training.
• Merge tiles back, either from folder naming or from manifest.csv.

SPLITTER
1) Input Image (T1) + Output Folder.
2) Tile Size (px). Recommended: 256 / 512 / 1024.
3) Overlap (%): 0.. <100. Typical: 10–50.
4) Output Format: .png / .jpg / .tif / .tiff / .jpeg.
5) Filename Pattern tokens:
   {base}  {y}  {x}  {row}  {col}  {i}  {ext}
   Examples:
     {base}_tile_{y}_{x}
     {base}_{y:05d}_{x:05d}
   NOTE: folder-based Merge needs {y} and {x} in filenames.
6) Bands: select channels to export (defaults to all).
7) Training Metadata:
   • Scene ID (defaults to base filename if blank)
   • Fold: train / val / test
   • Dual-time (T1/T2): tiles for T2 saved under "T2/" if dimensions match
   • Label (optional)
   • Parquet: saves manifest.parquet if pandas/pyarrow available.

MANIFEST COLUMNS
scene_id, tile_x, tile_y, x0, y0, w, h, t1_path, t2_path, label_path, fold
• tile_x, tile_y: grid indices (step = tile_size - overlap_px).
• x0, y0: top-left pixel offsets; w, h: on-disk tile size.

MERGE
• From Folder: requires filenames ending with _<y>_<x>.<ext>. Last tile wins in overlap.
• From manifest.csv: uses x0, y0, w, h, t1_path (or t2_path).
  Optional Fold filter to merge only train/val/test rows.

TIPS
• Install 'imagecodecs' for faster TIFF/PNG I/O.
• Use .tif for extremely large merged outputs.
• Keep naming stable for reproducible experiments.
"""
        box.insert("1.0", help_text.strip())
        box.configure(state="disabled")

        return page

    # ---------------- Pickers ----------------
    def _pick_input(self):
        path = filedialog.askopenfilename(filetypes=INPUT_FILTER)
        if path:
            self.input_path.set(path)
            self._populate_bands(path)
            self._save_settings()
            self.status.config(text=f"Selected T1: {path}")

    def _pick_input_generic(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=INPUT_FILTER)
        if path:
            var.set(path)
            self._save_settings()

    def _pick_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir.set(folder)
            self._save_settings()
            self.status.config(text=f"Output folder: {folder}")

    def _pick_tiles_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.tiles_dir.set(folder)
            self.status.config(text=f"Tiles folder: {folder}")

    def _pick_merge_out(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif;*.tiff"), ("PNG", "*.png"),
                       ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if path:
            self.merge_out_path.set(path)
            self.status.config(text=f"Merged output: {path}")

    # ---------------- Bands ----------------
    def _populate_bands(self, file_path: str):
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
            cb = ttk.Checkbutton(self.band_box, text="Band 0", variable=var)
            cb.pack(anchor="w", padx=6, pady=2)
            self.selected_bands_vars.append(var)
        else:
            bands = arr.shape[2]
            for b in range(bands):
                var = tk.IntVar(value=1)
                cb = ttk.Checkbutton(self.band_box, text=f"Band {b}", variable=var)
                cb.pack(anchor="w", padx=6, pady=2)
                self.selected_bands_vars.append(var)

    def _selected_bands(self):
        picked = [i for i, v in enumerate(self.selected_bands_vars) if v.get() == 1]
        return picked or None

    # ---------------- Preview ----------------
    def _preview_from_split(self):
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
            self._show_preview(tile_arr)
            self.preview_info.config(text="Preview opened (first tile).")
            self.status.config(text="Preview shown.")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))
            self.status.config(text="Preview failed.")

    def _show_preview(self, tile_arr: np.ndarray):
        img = Image.fromarray(tile_arr)
        max_side = 512
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        top = tk.Toplevel(self.master)
        top.title("Preview")
        top.resizable(False, False)
        photo = ImageTk.PhotoImage(img)
        lbl = ttk.Label(top, image=photo)
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

    # ---------- filename pattern helper ----------
    @staticmethod
    def _format_name_from_pattern(pattern: str, base: str, y: int, x: int, i: int, ext: str) -> str:
        name = pattern
        try:
            name = name.format(i=i)
        except Exception:
            pass
        name = name.format(base=base, y=y, x=x, row=y, col=x, ext=ext.lstrip("."))
        if not name.lower().endswith(ext if ext.startswith(".") else "." + ext):
            name = f"{name}{ext if ext.startswith('.') else '.' + ext}"
        return name

    # ---------------- Split (threaded with determinate progress & cancel) ----------------
    def _do_split_threaded(self):
        t = threading.Thread(target=self._do_split_worker, daemon=True)
        t.start()

    def _do_split_worker(self):
        # Validation
        if not self.input_path.get() or not os.path.isfile(self.input_path.get()):
            return self._msg_error("Please select a valid input image (T1).")
        if not self.output_dir.get():
            return self._msg_error("Please select an output folder.")

        try:
            tile = int(self.tile_size.get())
            if tile <= 0:
                raise ValueError
        except Exception:
            return self._msg_error("Tile size must be a positive integer.")

        try:
            pct = float(self.overlap_pct.get())
        except Exception:
            return self._msg_error("Overlap (%) must be a number.")
        if pct < 0 or pct >= 100:
            return self._msg_error("Overlap (%) must be in [0, <100).")

        overlap_px = int(round(tile * pct / 100.0))
        if overlap_px >= tile and pct > 0:
            return self._msg_error("Overlap too large for this tile size.")

        # Load arrays
        in_path = self.input_path.get()
        out_dir = self.output_dir.get()
        os.makedirs(out_dir, exist_ok=True)

        selected_bands = self._selected_bands()

        try:
            arr = _load_image_any(in_path)
            if arr.ndim == 3 and selected_bands is not None:
                arr = arr[:, :, selected_bands]
            arr = self._to_uint8_minmax(arr)
        except Exception as e:
            return self._msg_error(f"Could not read T1:\n{e}")

        H, W = arr.shape[:2]
        is_color = (arr.ndim == 3)
        base = os.path.splitext(os.path.basename(in_path))[0]
        ext = self.extension.get() if self.extension.get().startswith(".") else "." + self.extension.get()

        # Optional T2
        dual = bool(self.dual_time.get())
        t2_enabled = dual and os.path.isfile(self.input_path_t2.get().strip())
        arr2 = None
        t2_dir = None
        if t2_enabled:
            try:
                arr2 = _load_image_any(self.input_path_t2.get().strip())
                if arr2.ndim == 3 and selected_bands is not None:
                    arr2 = arr2[:, :, selected_bands]
                arr2 = self._to_uint8_minmax(arr2)
                if arr2.shape[:2] != (H, W):
                    self._msg_warn("T2 dimensions do not match T1. Skipping T2.")
                    t2_enabled = False
                    arr2 = None
                else:
                    t2_dir = os.path.join(out_dir, "T2")
                    os.makedirs(t2_dir, exist_ok=True)
            except Exception as e:
                self._msg_warn(f"Could not read T2. Skipping T2.\n{e}")
                t2_enabled = False
                arr2 = None

        # Build coords
        step = tile if overlap_px == 0 else max(1, tile - overlap_px)
        coords = []
        for y in range(0, H, step):
            for x in range(0, W, step):
                if y >= H or x >= W:
                    continue
                y2, x2 = min(y + tile, H), min(x + tile, W)
                coords.append((y, x, y2, x2))
        total = len(coords)

        # Progress dialog
        pd = ProgressDialog(self.master, title="Splitting tiles...", determinate=True)
        pd.set_message("Splitting tiles...")
        pd.set_total(total)

        # Manifest rows
        rows = []
        label_src = self.label_path.get().strip()
        scene_id = self.scene_id.get().strip() or base
        fold = (self.fold.get() or "train").lower()

        # Do work
        tiles = 0
        try:
            for i, (y, x, y2, x2) in enumerate(coords):
                if pd.cancelled:
                    break

                tile_arr = arr[y:y2, x:x2] if not is_color else arr[y:y2, x:x2, :]
                fname = self._format_name_from_pattern(self.name_pattern.get(), base, y, x, i, ext)
                t1_path = os.path.join(out_dir, fname)
                Image.fromarray(tile_arr).save(t1_path)

                t2_path = ""
                if t2_enabled and arr2 is not None:
                    tile2 = arr2[y:y2, x:x2] if not is_color else arr2[y:y2, x:x2, :]
                    t2_path = os.path.join(t2_dir, fname)
                    Image.fromarray(tile2).save(t2_path)

                tile_x = x // step
                tile_y = y // step
                w = int(x2 - x)
                h = int(y2 - y)
                rows.append([scene_id, tile_x, tile_y, x, y, w, h, t1_path, t2_path, label_src, fold])

                tiles += 1
                if i % 8 == 0:
                    pd.set_message(f"Splitting tiles... ({i+1}/{total})")
                pd.step(1)

            # Write manifest (even if cancelled, نكتب للي اتعمل)
            if rows:
                man_path = os.path.join(out_dir, "manifest.csv")
                with open(man_path, "w", newline="", encoding="utf-8") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["scene_id","tile_x","tile_y","x0","y0","w","h","t1_path","t2_path","label_path","fold"])
                    wcsv.writerows(rows)
            else:
                man_path = "(none)"

            pd.close()
            if pd.cancelled:
                self._msg_warn(f"Operation cancelled.\nTiles written: {tiles}\nManifest: {man_path}")
                self.status.config(text="Split cancelled.")
            else:
                self._msg_info("Success",
                               f"✅ Splitting complete!\nTiles: {tiles}\nManifest: {man_path}"
                               + (f"\nT2 tiles saved under: {t2_dir}" if t2_enabled else ""))
                self.status.config(text="Split done.")
        except Exception as e:
            pd.close()
            self._msg_error(f"Splitting failed:\n{e}")
            self.status.config(text="Split failed.")

    # ---------------- Merge estimate (folder-based) ----------------
    def _estimate_merge(self):
        try:
            if not self.tiles_dir.get():
                messagebox.showinfo("Estimate", "Please select a tiles folder first.")
                return
            files = _scan_tiles(self.tiles_dir.get())
            if not files:
                raise ValueError("No tiles found in the selected folder.")
            W, H, mode = _estimate_canvas_size(files)
            self.merge_estimate_lbl.config(text=f"Estimated: {W} x {H} px, mode: {mode}")
            self.status.config(text="Merge estimate ready.")
        except Exception as e:
            messagebox.showerror("Estimate error", str(e))
            self.status.config(text="Merge estimate failed.")

    # ---------------- Merge (threaded with determinate progress & cancel) ----------------
    def _do_merge_threaded(self):
        t = threading.Thread(target=self._do_merge_worker, daemon=True)
        t.start()

    def _infer_mode_from_image(self, path: str) -> str:
        with Image.open(path) as im:
            m = im.mode
            if m in ("L", "I;16", "I;16B", "I;16L", "I", "F"):
                return "L"
            if m in ("RGBA", "LA"):
                return "RGBA"
            return "RGB"

    def _do_merge_worker(self):
        if not self.tiles_dir.get():
            return self._msg_error("Please select a tiles folder.")
        if not self.merge_out_path.get():
            return self._msg_error("Please choose an output file.")

        # prepare file list
        try:
            files = _scan_tiles(self.tiles_dir.get())
            if not files:
                return self._msg_error("No tiles found in the selected folder.")
            W, H, _ = _estimate_canvas_size(files)
            target_mode = self._infer_mode_from_image(files[0])
        except Exception as e:
            return self._msg_error(f"Scan failed:\n{e}")

        bg = 0 if target_mode == "L" else (0, 0, 0, 0) if target_mode == "RGBA" else (0, 0, 0)
        canvas = Image.new(target_mode, (W, H), bg)

        pd = ProgressDialog(self.master, title="Merging from folder...", determinate=True)
        pd.set_message("Merging tiles (folder)...")
        pd.set_total(len(files))

        pasted = 0
        try:
            # filenames must end with _<y>_<x>.<ext>
            import re
            _TILE_RE = re.compile(r".*_(\d+)_(\d+)\.[A-Za-z0-9]+$")
            for i, fp in enumerate(files):
                if pd.cancelled:
                    break
                with Image.open(fp) as im:
                    if im.mode != target_mode:
                        im = im.convert(target_mode)
                    m = _TILE_RE.match(os.path.basename(fp))
                    if not m:
                        # skip non-matching
                        pd.step(1)
                        continue
                    y = int(m.group(1))
                    x = int(m.group(2))
                    canvas.paste(im, (x, y))
                pasted += 1
                if i % 16 == 0:
                    pd.set_message(f"Merging tiles... ({i+1}/{len(files)})")
                pd.step(1)

            pd.close()
            if pd.cancelled:
                self._msg_warn("Merge cancelled. No output saved.")
                self.status.config(text="Merge cancelled.")
            else:
                canvas.save(self.merge_out_path.get())
                self._msg_info("Success", f"Merged image saved.\nSize: {W} x {H}")
                self.status.config(text="Merge from folder done.")
                self.merge_estimate_lbl.config(text=f"Merged: {W} x {H}")
        except Exception as e:
            pd.close()
            self._msg_error(f"Merge failed:\n{e}")
            self.status.config(text="Merge failed.")

    def _merge_from_manifest_threaded(self):
        t = threading.Thread(target=self._merge_from_manifest_worker, daemon=True)
        t.start()

    def _merge_from_manifest_worker(self):
        manifest_path = filedialog.askopenfilename(
            title="Select manifest.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not manifest_path:
            return

        if not self.merge_out_path.get():
            out = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF", "*.tif;*.tiff"), ("PNG", "*.png"),
                           ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            if not out:
                return
            self.merge_out_path.set(out)

        # read manifest rows (+ optional fold filter)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = [c.strip() for c in (reader.fieldnames or [])]
                for req in ["x0", "y0", "w", "h", "t1_path"]:
                    if req not in header:
                        return self._msg_error(f"Manifest missing column: {req}")
                rows = [r for r in reader]
        except Exception as e:
            return self._msg_error(f"Could not read manifest:\n{e}")

        fold_filter = self.merge_fold_var.get().strip()
        if fold_filter:
            rows = [r for r in rows if (r.get("fold", "").lower() == fold_filter.lower())]

        if not rows:
            return self._msg_error("No rows to merge (after filtering).")

        # canvas size + target mode
        try:
            max_x2 = 0
            max_y2 = 0
            first_img = None
            for r in rows:
                x0 = int(float(r["x0"])); y0 = int(float(r["y0"]))
                w = int(float(r["w"]));   h = int(float(r["h"]))
                max_x2 = max(max_x2, x0 + w)
                max_y2 = max(max_y2, y0 + h)
                if not first_img and (r.get("t1_path") or "").strip():
                    first_img = r["t1_path"].strip()
            if not first_img:
                return self._msg_error("No valid t1_path in manifest.")
            target_mode = self._infer_mode_from_image(first_img)
        except Exception as e:
            return self._msg_error(f"Manifest parse error:\n{e}")

        bg = 0 if target_mode == "L" else (0, 0, 0, 0) if target_mode == "RGBA" else (0, 0, 0)
        canvas = Image.new(target_mode, (max_x2, max_y2), bg)

        pd = ProgressDialog(self.master, title="Merging from manifest...", determinate=True)
        pd.set_message("Merging tiles (manifest)...")
        pd.set_total(len(rows))

        pasted = 0
        try:
            for i, r in enumerate(rows):
                if pd.cancelled:
                    break

                img_path = (r.get("t1_path") or "").strip()
                if not img_path or not os.path.isfile(img_path):
                    pd.step(1)
                    continue

                try:
                    x0 = int(float(r["x0"])); y0 = int(float(r["y0"]))
                    w = int(float(r["w"]));   h = int(float(r["h"]))
                except Exception:
                    pd.step(1)
                    continue

                with Image.open(img_path) as im:
                    if im.mode != target_mode:
                        im = im.convert(target_mode)
                    if im.size != (w, h):
                        im = im.resize((w, h), Image.BILINEAR)
                    canvas.paste(im, (x0, y0))

                pasted += 1
                if i % 32 == 0:
                    pd.set_message(f"Merging tiles... ({i+1}/{len(rows)})")
                pd.step(1)

            pd.close()
            if pd.cancelled:
                self._msg_warn("Merge (manifest) cancelled. No output saved.")
                self.status.config(text="Merge from manifest cancelled.")
            else:
                canvas.save(self.merge_out_path.get())
                self._msg_info("Success", f"Merged image saved.\nSize: {max_x2} x {max_y2}")
                self.status.config(text="Merge from manifest done.")
                self.merge_estimate_lbl.config(text=f"Merged: {max_x2} x {max_y2}")
        except Exception as e:
            pd.close()
            self._msg_error(f"Merge (manifest) failed:\n{e}")
            self.status.config(text="Merge from manifest failed.")

    # ---------------- Messages (thread-safe helpers) ----------------
    def _msg_info(self, title, text):
        self.master.after(0, lambda: messagebox.showinfo(title, text))

    def _msg_error(self, text):
        self.master.after(0, lambda: messagebox.showerror("Error", text))

    def _msg_warn(self, text):
        self.master.after(0, lambda: messagebox.showwarning("Warning", text))

    # ---------------- Settings ----------------
    def _save_settings(self):
        data = {
            # splitter
            "input_path": self.input_path.get(),
            "output_dir": self.output_dir.get(),
            "tile_size": self.tile_size.get(),
            "overlap_pct": float(self.overlap_pct.get()),
            "extension": self.extension.get(),
            "name_pattern": self.name_pattern.get(),
            # manifest-rich
            "scene_id": self.scene_id.get(),
            "fold": self.fold.get(),
            "dual_time": bool(self.dual_time.get()),
            "input_path_t2": self.input_path_t2.get(),
            "label_path": self.label_path.get(),
            "write_parquet": bool(self.write_parquet.get()),
            # merge
            "tiles_dir": self.tiles_dir.get(),
            "merge_out_path": self.merge_out_path.get(),
            "merge_fold": self.merge_fold_var.get(),
            # theme
            "theme": self.theme_var.get(),
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
            self.overlap_pct.set(float(data.get("overlap_pct", 0.0)))
            self.extension.set(data.get("extension") or ".png")
            self.name_pattern.set(data.get("name_pattern") or "{base}_tile_{y}_{x}")
            self.scene_id.set(data.get("scene_id", ""))
            self.fold.set(data.get("fold", "train"))
            self.dual_time.set(bool(data.get("dual_time", False)))
            self.input_path_t2.set(data.get("input_path_t2", ""))
            self.label_path.set(data.get("label_path", ""))
            self.write_parquet.set(bool(data.get("write_parquet", False)))
            self.tiles_dir.set(data.get("tiles_dir", ""))
            self.merge_out_path.set(data.get("merge_out_path", ""))
            self.merge_fold_var.set(data.get("merge_fold", ""))
            self.theme_var.set(data.get("theme", "System"))
            self._apply_theme_colors(self.theme_var.get())

            if self.input_path.get():
                self._populate_bands(self.input_path.get())
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.master.destroy()


def run_app():
    root = tk.Tk()
    app = ImagePrepApp(root)
    root.mainloop()
