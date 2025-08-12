# app/gui.py
# -*- coding: utf-8 -*-

import os
import json
import csv
import math
import threading
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np
from PIL import Image, ImageTk

# Safety: allow very large images; silence DecompressionBomb & tifffile user warnings
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

# split/merge helpers
from app.splitter import split_large_image
from app.merger import _scan_tiles, _estimate_canvas_size

# help loader (fallback to static string if module missing)
try:
    from app.help_text import get_help_text
except Exception:
    def get_help_text(app_title: str, app_version: str, lang: str = "en") -> str:
        return f"{app_title} — {app_version}\n\nHelp file not found (app/help_text.py)."

APP_TITLE = "Data Preparation"
APP_VERSION = "v2.4"
AUTHOR = "Built by Mosatafa Mosaad"

SETTINGS_FILE = "splitter_settings.json"

EXT_OPTIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
INPUT_FILTER = [
    ("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp"),
    ("All files", "*.*"),
]


# ------------------------- Tooltip helper -------------------------
class Tooltip:
    """Small tooltip popover for any Tk widget."""

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
    """Modal progress dialog with Cancel button and thread-safe updates."""

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

        # center on parent
        self.top.update_idletasks()
        w = self.top.winfo_width()
        h = self.top.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - w) // 2
        y = master.winfo_rooty() + (master.winfo_height() - h) // 2
        self.top.geometry(f"+{max(0, x)}+{max(0, y)}")

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


# ------------------------- Quick probe (H,W,C) -------------------------
def _probe_image_info_fast(path: str):
    """Return (H, W, C) by reading metadata only (no full pixel load)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            try:
                c = len(im.getbands()) if im.getbands() else 1
            except Exception:
                c = 1
            return int(h), int(w), int(c)
    except Exception as e:
        raise ValueError(f"Could not read image metadata: {e}")


def _save_kwargs_for_ext(ext: str) -> dict:
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in (".jpg", ".jpeg"):
        return {"quality": 95, "optimize": True}
    if ext in (".tif", ".tiff"):
        return {"compression": "tiff_lzw"}
    return {}


# ------------------------- Main App -------------------------
class ImagePrepApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master: tk.Tk = master
        self.master.title(f"{APP_TITLE} — {APP_VERSION}")
        self.master.geometry("1120x800")
        self.master.minsize(1040, 720)

        self._init_style()

        # ---- State ----
        self.input_path = tk.StringVar()
        self.input_path_t2 = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.tile_size = tk.IntVar(value=512)
        self.overlap_pct = tk.DoubleVar(value=0.0)
        self.extension = tk.StringVar(value=".png")

        # Optional base names for T1/T2
        self.t1_base = tk.StringVar(value="")
        self.t2_base = tk.StringVar(value="")

        self.name_pattern = tk.StringVar(value="{base}_tile_{y}_{x}")
        self.selected_bands_vars = []

        # Training metadata
        self.scene_id = tk.StringVar(value="")
        self.fold = tk.StringVar(value="train")
        self.label_path = tk.StringVar()
        self.write_parquet = tk.BooleanVar(value=False)

        # Merge
        self.tiles_dir = tk.StringVar()
        self.merge_out_path = tk.StringVar()
        self.merge_fold_var = tk.StringVar(value="")

        # Theme
        self.theme_var = tk.StringVar(value="System")

        # Layout
        self._build_root_layout()
        self._create_pages()
        self._load_settings()

        self._show_page("split")
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
            bg = "#1f232a"; fg = "#e6e6e6"; subbg = "#272c34"; acc = "#3a82f7"
        elif mode == "Light":
            bg = "#ffffff"; fg = "#111"; subbg = "#f3f4f6"; acc = "#0b5ed7"
        else:
            bg = "#f8f9fb"; fg = "#111"; subbg = "#ffffff"; acc = "#0b5ed7"

        self.master.configure(bg=bg)
        for elem in ("TFrame", "TLabelframe", "TLabelframe.Label", "TLabel", "TNotebook", "TScrollbar"):
            self.style.configure(elem, background=subbg if "Label" in elem else bg, foreground=fg)
        self.style.configure("TButton", background=subbg, foreground=fg)
        self.style.map("TButton", background=[("active", acc)])
        self.style.configure("Status.TLabel", foreground="#666" if mode != "Dark" else "#aab")

    # ---------------- Root Layout ----------------
    def _build_root_layout(self):
        header = ttk.Frame(self.master)
        header.pack(side="top", fill="x", padx=10, pady=(10, 6))

        left = ttk.Frame(header); left.pack(side="left")
        ttk.Label(left, text=APP_TITLE, font=("Segoe UI Semibold", 14)).pack(side="left")
        ttk.Label(left, text=f"{APP_VERSION} — {AUTHOR}", style="Status.TLabel").pack(side="left", padx=(10, 0))

        right = ttk.Frame(header); right.pack(side="right")
        ttk.Label(right, text="Theme:").pack(side="left", padx=(0, 6))
        theme_combo = ttk.Combobox(right, textvariable=self.theme_var,
                                   values=["System", "Light", "Dark"], state="readonly", width=10)
        theme_combo.pack(side="left")
        theme_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_theme_colors(self.theme_var.get()))
        Tooltip(theme_combo, "Switch theme (System / Light / Dark).")

        body = ttk.Frame(self.master); body.pack(side="top", fill="both", expand=True, padx=10, pady=6)

        # Sidebar
        self.sidebar = ttk.Frame(body, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ttk.Button(self.sidebar, text="Splitter", command=lambda: self._show_page("split")).pack(fill="x", pady=6)
        ttk.Button(self.sidebar, text="Merge Tool", command=lambda: self._show_page("merge")).pack(fill="x", pady=6)
        ttk.Button(self.sidebar, text="Preview", command=lambda: self._show_page("preview")).pack(fill="x", pady=6)
        ttk.Button(self.sidebar, text="Help", command=lambda: self._show_page("help")).pack(fill="x", pady=6)

        # Content
        self.content = ttk.Frame(body); self.content.pack(side="left", fill="both", expand=True)
        self.pages = {}

        # Status bar
        status_row = ttk.Frame(self.master); status_row.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
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

        # I/O (T1, then T2 underneath)
        io_frame = ttk.LabelFrame(page, text="Input / Output")
        io_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(io_frame, text="Input Image (T1):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        t1_entry = ttk.Entry(io_frame, textvariable=self.input_path, width=80)
        t1_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        b1 = ttk.Button(io_frame, text="Browse...", command=self._pick_input)
        b1.grid(row=0, column=2, padx=6, pady=6)
        Tooltip(b1, "Choose the main (T1) image to split.")

        ttk.Label(io_frame, text="Input Image (T2):").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        t2_entry = ttk.Entry(io_frame, textvariable=self.input_path_t2, width=80)
        t2_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        b2 = ttk.Button(io_frame, text="Browse...", command=lambda: self._pick_input_generic(self.input_path_t2))
        b2.grid(row=1, column=2, padx=6, pady=6)
        Tooltip(b2, "Optional second-time image (T2). Must match T1 size.")

        ttk.Label(io_frame, text="Output Folder:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        out_entry = ttk.Entry(io_frame, textvariable=self.output_dir, width=80)
        out_entry.grid(row=2, column=1, sticky="we", padx=6, pady=6)
        b3 = ttk.Button(io_frame, text="Browse...", command=self._pick_output)
        b3.grid(row=2, column=2, padx=6, pady=6)
        Tooltip(b3, "Where T1/ and T2/ folders will be created.")
        io_frame.columnconfigure(1, weight=1)

        # Tiling
        tiling = ttk.LabelFrame(page, text="Tiling")
        tiling.pack(fill="x", padx=10, pady=10)

        ttk.Label(tiling, text="Tile Size (px):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tile_entry = ttk.Entry(tiling, textvariable=self.tile_size, width=10)
        tile_entry.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        Tooltip(tile_entry, "Tile width/height in pixels (e.g., 256 / 512 / 1024).")

        ttk.Label(tiling, text="Overlap (%):").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        ov_entry = ttk.Entry(tiling, textvariable=self.overlap_pct, width=10)
        ov_entry.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        Tooltip(ov_entry, "Percentage of tile size to overlap between tiles. Set 0 for no overlap.")

        ttk.Label(tiling, text="Output Format:").grid(row=0, column=4, sticky="w", padx=6, pady=6)
        fmt_combo = ttk.Combobox(tiling, textvariable=self.extension, values=EXT_OPTIONS, state="readonly", width=10)
        fmt_combo.grid(row=0, column=5, sticky="w", padx=6, pady=6)
        Tooltip(fmt_combo, "File format for tiles (.png / .jpg / .tif / .tiff / .jpeg).")

        # Base names for T1/T2
        ttk.Label(tiling, text="Base (T1):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        t1base_entry = ttk.Entry(tiling, textvariable=self.t1_base, width=28)
        t1base_entry.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        Tooltip(t1base_entry, "If empty, the T1 filename (without extension) will be used.")

        ttk.Label(tiling, text="Base (T2):").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        t2base_entry = ttk.Entry(tiling, textvariable=self.t2_base, width=28)
        t2base_entry.grid(row=1, column=3, sticky="w", padx=6, pady=6)
        Tooltip(t2base_entry, "If empty, Base(T1) + '_T2' will be used.")

        ttk.Label(tiling, text="Filename Pattern:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        pat_entry = ttk.Entry(tiling, textvariable=self.name_pattern, width=60)
        pat_entry.grid(row=2, column=1, columnspan=5, sticky="we", padx=6, pady=6)
        Tooltip(
            pat_entry,
            "Tokens: {base} {y} {x} {row} {col} {i} {ext}\n"
            "Example: {base}_tile_{y}_{x}\n"
            "Tip: merge-from-folder expects {y} and {x} in the name."
        )
        tiling.columnconfigure(1, weight=1)
        tiling.columnconfigure(3, weight=1)

        # Bands
        bands = ttk.LabelFrame(page, text="Bands (Channels)")
        bands.pack(fill="x", padx=10, pady=10)
        self.band_box = ttk.Frame(bands)
        self.band_box.pack(fill="x", padx=8, pady=8)

        # Training metadata
        meta = ttk.LabelFrame(page, text="Training Metadata")
        meta.pack(fill="x", padx=10, pady=10)

        ttk.Label(meta, text="Scene ID:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        scene_entry = ttk.Entry(meta, textvariable=self.scene_id, width=28)
        scene_entry.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        Tooltip(scene_entry, "Unique identifier for the scene. Defaults to T1 base name if empty.")

        ttk.Label(meta, text="Fold:").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        fold_combo = ttk.Combobox(meta, textvariable=self.fold, values=["train", "val", "test"],
                                  state="readonly", width=12)
        fold_combo.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        Tooltip(fold_combo, "Predefined split for ML datasets.")

        ttk.Label(meta, text="Label (optional):").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        lbl_entry = ttk.Entry(meta, textvariable=self.label_path, width=60)
        lbl_entry.grid(row=1, column=1, columnspan=2, sticky="we", padx=6, pady=6)
        Tooltip(lbl_entry, "Optional label/annotation path to store in manifest.")

        pq_chk = ttk.Checkbutton(meta, text="Write Parquet alongside CSV", variable=self.write_parquet)
        pq_chk.grid(row=1, column=3, sticky="w", padx=6, pady=6)
        Tooltip(pq_chk, "If pandas/pyarrow are installed, also write manifest.parquet.")

        meta.columnconfigure(1, weight=1)
        meta.columnconfigure(2, weight=1)

        # Actions
        actions = ttk.Frame(page)
        actions.pack(fill="x", padx=10, pady=(0, 10))
        prev_btn = ttk.Button(actions, text="Preview First Tile", command=self._preview_from_split)
        prev_btn.pack(side="left", padx=(0, 8))
        Tooltip(prev_btn, "Open a preview window of the first tile (0..tile_size).")

        split_btn = ttk.Button(actions, text="Split Image", command=self._do_split_threaded)
        split_btn.pack(side="left", padx=(0, 8))
        Tooltip(split_btn, "Start splitting with progress and per-folder manifests.")

        reset_btn = ttk.Button(actions, text="Reset", command=self._reset_all)
        reset_btn.pack(side="left")
        Tooltip(reset_btn, "Reset all fields to defaults and clear saved settings.")

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
        Tooltip(tiles_btn, "Select a folder of tiles. Names should end with _<y>_<x>.<ext>.")

        ttk.Label(src, text="Output File:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        out_entry = ttk.Entry(src, textvariable=self.merge_out_path, width=80)
        out_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        out_btn = ttk.Button(src, text="Save As...", command=self._pick_merge_out)
        out_btn.grid(row=1, column=2, padx=6, pady=6)
        Tooltip(out_btn, "Save merged image (use .tif for very large sizes).")
        src.columnconfigure(1, weight=1)

        ops = ttk.LabelFrame(page, text="Operations")
        ops.pack(fill="x", padx=10, pady=10)

        est_btn = ttk.Button(ops, text="Scan & Estimate Size", command=self._estimate_merge)
        est_btn.grid(row=0, column=0, padx=6, pady=6)
        Tooltip(est_btn, "Scan folder and estimate merged size and mode.")

        self.merge_estimate_lbl = ttk.Label(ops, text="")
        self.merge_estimate_lbl.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        merge_btn = ttk.Button(ops, text="Merge from Folder", command=self._do_merge_threaded)
        merge_btn.grid(row=1, column=0, padx=6, pady=6)
        Tooltip(merge_btn, "Merge based on tile filenames (expects _<y>_<x> in name).")

        m_manifest_btn = ttk.Button(ops, text="Merge from manifest.csv", command=self._merge_from_manifest_threaded)
        m_manifest_btn.grid(row=1, column=1, padx=6, pady=6)
        Tooltip(m_manifest_btn, "Merge by reading positions from manifest.csv (recommended).")

        ttk.Label(ops, text="(optional) Fold filter:").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        fold_combo = ttk.Combobox(ops, textvariable=self.merge_fold_var, values=["", "train", "val", "test"],
                                  state="readonly", width=10)
        fold_combo.grid(row=1, column=3, padx=6, pady=6, sticky="w")
        Tooltip(fold_combo, "If set, only rows with this fold are merged from the manifest.")

        return page

    # ---------------- Preview Page ----------------
    def _build_preview_page(self, parent):
        page = ttk.Frame(parent)
        info = ttk.Label(page, text="Quick Preview (uses Splitter settings)", font=("Segoe UI Semibold", 11))
        info.pack(anchor="w", padx=10, pady=(10, 6))

        btn = ttk.Button(page, text="Preview First Tile", command=self._preview_from_split)
        btn.pack(anchor="w", padx=10, pady=6)
        Tooltip(btn, "Open a preview for the first tile after band selection and normalization.")

        self.preview_info = ttk.Label(page, text="", style="Status.TLabel")
        self.preview_info.pack(anchor="w", padx=10, pady=(6, 10))
        return page

    # ---------------- Help Page ----------------
    def _build_help_page(self, parent):
        page = ttk.Frame(parent)

        ttk.Label(page, text=f"{APP_TITLE} — {APP_VERSION}", font=("Segoe UI Semibold", 14))\
            .pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(page, text=AUTHOR, style="Status.TLabel").pack(anchor="w", padx=10, pady=(0, 8))

        # Controls
        btns = ttk.Frame(page); btns.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(btns, text="Reload Help", command=lambda: self._reload_help_text(self._help_textbox))\
            .pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Open help file", command=self._open_help_source)\
            .pack(side="left")

        # Text box
        box = scrolledtext.ScrolledText(page, wrap="word", width=120, height=30, font=("Segoe UI", 10))
        box.pack(fill="both", expand=True, padx=10, pady=10)

        try:
            help_text = get_help_text(APP_TITLE, APP_VERSION, lang="en")
        except Exception as e:
            help_text = f"Failed to load help text:\n{e}"
        box.insert("1.0", help_text)
        box.configure(state="disabled")

        # keep a reference for reloading
        self._help_textbox = box
        return page

    # ---------------- Pickers ----------------
    def _pick_input(self):
        try:
            path = filedialog.askopenfilename(filetypes=INPUT_FILTER)
            if not path:
                return
            try:
                H, W, C = _probe_image_info_fast(path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read image metadata:\n{e}")
                return
            self.input_path.set(path)
            self._populate_bands_from_count(C)
            self._save_settings()
            self.status.config(text=f"Selected T1: {path} (H={H}, W={W}, C={C})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed selecting input:\n{e}")

    def _pick_input_generic(self, var: tk.StringVar):
        try:
            path = filedialog.askopenfilename(filetypes=INPUT_FILTER)
            if path:
                var.set(path)
                self._save_settings()
        except Exception as e:
            messagebox.showerror("Error", f"Failed selecting file:\n{e}")

    def _pick_output(self):
        try:
            folder = filedialog.askdirectory()
            if folder:
                self.output_dir.set(folder)
                self._save_settings()
                self.status.config(text=f"Output folder: {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed selecting output folder:\n{e}")

    def _pick_tiles_dir(self):
        try:
            folder = filedialog.askdirectory()
            if folder:
                self.tiles_dir.set(folder)
                self.status.config(text=f"Tiles folder: {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed selecting tiles folder:\n{e}")

    def _pick_merge_out(self):
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF", "*.tif;*.tiff"), ("PNG", "*.png"),
                           ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )
            if path:
                self.merge_out_path.set(path)
                self.status.config(text=f"Merged output: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed choosing output file:\n{e}")

    # ---------------- Bands ----------------
    def _populate_bands_from_count(self, bands_count: int):
        """Populate band checkboxes based on channel count (without loading full image)."""
        for w in self.band_box.winfo_children():
            w.destroy()
        self.selected_bands_vars.clear()
        try:
            if bands_count <= 1:
                var = tk.IntVar(value=1)
                ttk.Checkbutton(self.band_box, text="Band 0", variable=var).pack(anchor="w", padx=6, pady=2)
                self.selected_bands_vars.append(var)
            else:
                for b in range(bands_count):
                    var = tk.IntVar(value=1)
                    ttk.Checkbutton(self.band_box, text=f"Band {b}", variable=var).pack(anchor="w", padx=6, pady=2)
                    self.selected_bands_vars.append(var)
        except Exception as e:
            messagebox.showerror("Error", f"Failed populating bands:\n{e}")

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
            with Image.open(self.input_path.get()) as im:
                crop = im.crop((0, 0, min(im.width, tile), min(im.height, tile)))
                try:
                    parts = crop.split()
                    sel = self._selected_bands()
                    if sel is not None and len(parts) >= max(sel) + 1:
                        parts_sel = tuple(parts[i] for i in sel)
                        if len(parts_sel) == 1:
                            crop = parts_sel[0]
                        elif len(parts_sel) == 3:
                            crop = Image.merge("RGB", parts_sel[:3])
                        elif len(parts_sel) == 4:
                            crop = Image.merge("RGBA", parts_sel[:4])
                except Exception:
                    arr = np.array(crop)
                    sel = self._selected_bands()
                    if sel is not None and arr.ndim == 3 and arr.shape[2] >= max(sel) + 1:
                        arr = arr[:, :, sel]
                    # min-max normalization
                    if arr.dtype != np.uint8:
                        a = arr.astype(np.float32)
                        if a.ndim == 2:
                            mn, mx = float(a.min()), float(a.max())
                            a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
                        else:
                            for c in range(a.shape[2]):
                                mn, mx = float(a[..., c].min()), float(a[..., c].max())
                                a[..., c] = (a[..., c] - mn) / (mx - mn) * 255.0 if mx > mn else 0
                        arr = a.astype(np.uint8)
                    crop = Image.fromarray(arr)

                max_side = 512
                scale = min(max_side / max(crop.size), 1.0)
                if scale < 1.0:
                    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.BILINEAR)

                top = tk.Toplevel(self.master)
                top.title("Preview")
                top.resizable(False, False)
                photo = ImageTk.PhotoImage(crop)
                lbl = ttk.Label(top, image=photo)
                lbl.image = photo
                lbl.pack(padx=8, pady=8)

            self.preview_info.config(text="Preview opened (first tile).")
            self.status.config(text="Preview shown.")
        except Exception as e:
            messagebox.showerror("Preview error", f"Could not preview:\n{e}")
            self.status.config(text="Preview failed.")

    # ---------------- Reset ----------------
    def _reset_all(self):
        """Reset all fields to defaults and clear saved settings file."""
        self.input_path.set("")
        self.input_path_t2.set("")
        self.output_dir.set("")
        self.tile_size.set(512)
        self.overlap_pct.set(0.0)
        self.extension.set(".png")
        self.t1_base.set("")
        self.t2_base.set("")
        self.name_pattern.set("{base}_tile_{y}_{x}")
        self.scene_id.set("")
        self.fold.set("train")
        self.label_path.set("")
        self.write_parquet.set(False)
        for w in self.band_box.winfo_children():
            w.destroy()
        self.selected_bands_vars.clear()
        self.status.config(text="Reset to defaults.")
        try:
            if os.path.exists(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
        except Exception:
            pass

    # ---------------- Split (threaded) ----------------
    def _do_split_threaded(self):
        t = threading.Thread(target=self._do_split_worker, daemon=True)
        t.start()

    def _do_split_worker(self):
        # Validation
        in_path = (self.input_path.get() or "").strip()
        out_dir = (self.output_dir.get() or "").strip()
        if not in_path or not os.path.isfile(in_path):
            return self._msg_error("Please select a valid input image (T1).")
        if not out_dir:
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

        # Estimate total tiles for progress bar
        try:
            H, W, _ = _probe_image_info_fast(in_path)
            overlap_px = int(round(tile * pct / 100.0))
            step = tile if overlap_px <= 0 else max(1, tile - overlap_px)
            total = math.ceil(H / step) * math.ceil(W / step)
        except Exception:
            total = 0

        pd = ProgressDialog(self.master, title="Splitting tiles...", determinate=True)
        pd.set_message("Splitting tiles...")
        if total > 0:
            pd.set_total(total)
        else:
            pd.set_indeterminate()

        try:
            result = split_large_image(
                input_path=in_path,
                output_dir=out_dir,
                tile_size=tile,
                extension=self.extension.get(),
                selected_bands=self._selected_bands(),
                normalize_mode="minmax",
                policy="auto",
                name_pattern=self.name_pattern.get(),
                overlap_pct=pct,
                t2_path=(self.input_path_t2.get() or "").strip() or None,
                # custom base names & T2 pattern:
                t1_base=(self.t1_base.get() or "").strip() or None,
                t2_base=(self.t2_base.get() or "").strip() or None,
                name_pattern_t2=None,  # reuse same pattern with different base
                # manifest per folder:
                write_manifest=True,
                scene_id=(self.scene_id.get() or "").strip() or None,
                label_path=(self.label_path.get() or "").strip(),
                fold=(self.fold.get() or "train"),
                write_parquet=bool(self.write_parquet.get()),
                # progress callback: return True to cancel
                progress=lambda i, tot: (pd.step(1), pd.set_message(f"Splitting tiles... ({i+1}/{tot})") or pd.cancelled)[-1],
            )
            pd.close()

            msg = [
                "✅ Splitting complete!",
                f"Tiles total (T1+T2): {result.get('tiles', 0)}",
                f"T1 dir: {result.get('t1_dir','')}",
                f"T1 manifest: {result.get('t1_manifest','(none)')}",
            ]
            if result.get("t2_used"):
                msg.append(f"T2 dir: {result.get('t2_dir','')}")
                msg.append(f"T2 manifest: {result.get('t2_manifest','(none)')}")
            if result.get("note"):
                msg.append(f"Note: {result['note']}")
            self._msg_info("Success", "\n".join(msg))
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

    # ---------------- Merge (threaded) ----------------
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

        try:
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
                        pd.step(1)
                        continue
                    y = int(m.group(1))
                    x = int(m.group(2))
                    canvas.paste(im, (x, y))
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

        # Read manifest rows (+ optional fold filter)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = [c.strip() for c in (reader.fieldnames or [])]

                # Path column can be t1_path, t2_path, or generic path
                img_col = None
                for cand in ("t1_path", "t2_path", "path"):
                    if cand in header:
                        img_col = cand
                        break
                if img_col is None:
                    return self._msg_error("Manifest must contain a path column (t1_path / t2_path / path).")

                for req in ["x0", "y0", "w", "h"]:
                    if req not in header:
                        return self._msg_error(f"Manifest missing column: {req}")

                rows = [r for r in reader]
        except Exception as e:
            return self._msg_error(f"Could not read manifest:\n{e}")

        fold_filter = (self.merge_fold_var.get() or "").strip()
        if fold_filter:
            rows = [r for r in rows if (r.get("fold", "").lower() == fold_filter.lower())]

        if not rows:
            return self._msg_error("No rows to merge (after filtering).")

        # Canvas size + target mode
        try:
            max_x2 = 0
            max_y2 = 0
            first_img = None
            for r in rows:
                x0 = int(float(r["x0"])); y0 = int(float(r["y0"]))
                w = int(float(r["w"]));   h = int(float(r["h"]))
                max_x2 = max(max_x2, x0 + w)
                max_y2 = max(max_y2, y0 + h)
                if not first_img and (r.get(img_col) or "").strip():
                    first_img = r[img_col].strip()
            if not first_img:
                return self._msg_error("No valid path in manifest.")
            target_mode = self._infer_mode_from_image(first_img)
        except Exception as e:
            return self._msg_error(f"Manifest parse error:\n{e}")

        bg = 0 if target_mode == "L" else (0, 0, 0, 0) if target_mode == "RGBA" else (0, 0, 0)
        canvas = Image.new(target_mode, (max_x2, max_y2), bg)

        pd = ProgressDialog(self.master, title="Merging from manifest...", determinate=True)
        pd.set_message("Merging tiles (manifest)...")
        pd.set_total(len(rows))

        try:
            for i, r in enumerate(rows):
                if pd.cancelled:
                    break

                img_path = (r.get(img_col) or "").strip()
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

    # ---------------- Help helpers ----------------
    def _reload_help_text(self, box_widget=None):
        """Reload help text from app/help_text.py without restarting the app."""
        try:
            import importlib
            from app import help_text as _help_mod
            importlib.reload(_help_mod)
            text = _help_mod.get_help_text(APP_TITLE, APP_VERSION, lang="en")
        except Exception as e:
            text = f"Failed to reload help text:\n{e}"

        box = box_widget or getattr(self, "_help_textbox", None)
        if box is not None:
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text)
            box.configure(state="disabled")
            self.status.config(text="Help reloaded.")

    def _open_help_source(self):
        """Open the help source file in the default OS editor."""
        try:
            import pathlib, sys, subprocess, os as _os
            help_path = pathlib.Path(__file__).parent / "help_text.py"
            if sys.platform.startswith("win"):
                _os.startfile(help_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(help_path)])
            else:
                subprocess.Popen(["xdg-open", str(help_path)])
        except Exception as e:
            messagebox.showinfo("Help file", f"Help file path:\n{help_path}\n\nError opening:\n{e}")

    # ---------------- Messages & Settings ----------------
    def _msg_info(self, title, text):
        self.master.after(0, lambda: messagebox.showinfo(title, text))

    def _msg_error(self, text):
        self.master.after(0, lambda: messagebox.showerror("Error", text))

    def _msg_warn(self, text):
        self.master.after(0, lambda: messagebox.showwarning("Warning", text))

    def _save_settings(self):
        data = {
            "input_path": self.input_path.get(),
            "input_path_t2": self.input_path_t2.get(),
            "output_dir": self.output_dir.get(),
            "tile_size": self.tile_size.get(),
            "overlap_pct": float(self.overlap_pct.get()),
            "extension": self.extension.get(),
            "t1_base": self.t1_base.get(),
            "t2_base": self.t2_base.get(),
            "name_pattern": self.name_pattern.get(),
            "scene_id": self.scene_id.get(),
            "fold": self.fold.get(),
            "label_path": self.label_path.get(),
            "write_parquet": bool(self.write_parquet.get()),
            "tiles_dir": self.tiles_dir.get(),
            "merge_out_path": self.merge_out_path.get(),
            "merge_fold": self.merge_fold_var.get(),
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
            self.input_path_t2.set(data.get("input_path_t2") or "")
            self.output_dir.set(data.get("output_dir") or "")
            self.tile_size.set(int(data.get("tile_size", 512)))
            self.overlap_pct.set(float(data.get("overlap_pct", 0.0)))
            self.extension.set(data.get("extension") or ".png")
            self.t1_base.set(data.get("t1_base", ""))
            self.t2_base.set(data.get("t2_base", ""))
            self.name_pattern.set(data.get("name_pattern") or "{base}_tile_{y}_{x}")
            self.scene_id.set(data.get("scene_id", ""))
            self.fold.set(data.get("fold", "train"))
            self.label_path.set(data.get("label_path", ""))
            self.write_parquet.set(bool(data.get("write_parquet", False)))
            self.tiles_dir.set(data.get("tiles_dir", ""))
            self.merge_out_path.set(data.get("merge_out_path", ""))
            self.merge_fold_var.set(data.get("merge_fold", ""))
            self.theme_var.set(data.get("theme", "System"))
            self._apply_theme_colors(self.theme_var.get())

            if self.input_path.get():
                try:
                    _, _, C = _probe_image_info_fast(self.input_path.get())
                    self._populate_bands_from_count(C)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_close(self):
        # To delete settings on exit instead, uncomment the next 3 lines.
        # try:
        #     if os.path.exists(SETTINGS_FILE): os.remove(SETTINGS_FILE)
        # except Exception: pass
        self._save_settings()
        self.master.destroy()


def run_app():
    root = tk.Tk()
    app = ImagePrepApp(root)
    root.mainloop()
