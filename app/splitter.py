import os
from tifffile import TiffFile
from PIL import Image
import numpy as np

def split_large_image(input_path, output_dir, tile_size, extension=".png"):
    with TiffFile(input_path) as tif:
        page = tif.pages[0]
        image_data = page.asarray()
        height, width = image_data.shape[:2]
        is_color = len(image_data.shape) == 3

        total_tiles = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = image_data[y:y + tile_size, x:x + tile_size] if not is_color \
                    else image_data[y:y + tile_size, x:x + tile_size, :]

                tile_img = Image.fromarray(tile)
                tile_name = f"tile_{y}_{x}{extension}"
                tile_path = os.path.join(output_dir, tile_name)
                tile_img.save(tile_path)
                total_tiles += 1

    return total_tiles
