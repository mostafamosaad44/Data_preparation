import os
import threading
import customtkinter as ctk
from tkinter import filedialog
from app.splitter import split_large_image

def run_app():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    class LargeImageSplitterApp(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("Large Image Splitter")
            self.geometry("600x400")

            self.input_path = None
            self.output_dir = None
            self.extension = ".png"

            self.create_widgets()

        def create_widgets(self):
            ctk.CTkLabel(self, text="Large Image Splitter Tool", font=("Arial", 20, "bold")).pack(pady=20)

            ctk.CTkButton(self, text="Select Large Image (.tif)", command=self.select_input_file).pack(pady=10)
            self.label_input = ctk.CTkLabel(self, text="No file selected", wraplength=500)
            self.label_input.pack(pady=5)

            ctk.CTkButton(self, text="Select Output Folder", command=self.select_output_folder).pack(pady=10)
            self.label_output = ctk.CTkLabel(self, text="No folder selected", wraplength=500)
            self.label_output.pack(pady=5)

            self.entry_tile_size = ctk.CTkEntry(self, placeholder_text="Enter tile size (e.g., 256)")
            self.entry_tile_size.pack(pady=10)

            self.btn_split = ctk.CTkButton(self, text="Split Image", command=self.run_split_thread)
            self.btn_split.pack(pady=20)

            self.label_status = ctk.CTkLabel(self, text="")
            self.label_status.pack(pady=10)

        def select_input_file(self):
            path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff")])
            if path:
                self.input_path = path
                self.label_input.configure(text=f"Selected: {os.path.basename(path)}")

        def select_output_folder(self):
            folder = filedialog.askdirectory()
            if folder:
                self.output_dir = folder
                self.label_output.configure(text=f"Output folder: {folder}")

        def run_split_thread(self):
            threading.Thread(target=self.split_image, daemon=True).start()

        def split_image(self):
            if not self.input_path or not self.output_dir:
                self.label_status.configure(text="Please select input and output paths.")
                return

            try:
                tile_size = int(self.entry_tile_size.get())
            except ValueError:
                self.label_status.configure(text="Tile size must be a number.")
                return

            self.label_status.configure(text="Processing...")

            try:
                total = split_large_image(self.input_path, self.output_dir, tile_size, self.extension)
                self.label_status.configure(text=f"✅ Done! {total} tiles saved.")
            except Exception as e:
                self.label_status.configure(text=f"❌ Error: {str(e)}")

    app = LargeImageSplitterApp()
    app.mainloop()
