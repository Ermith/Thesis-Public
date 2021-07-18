from .matrix_manipulation import cutout, scale_down
from .images import save_image
import numpy as np


def get_window(image, row, col, reduction, cut):
    """Cuts out a window from an image and scales it down to to (cut, cut) shape.
    Image is correctly centered for the pixel we want to generate."""
    window_size = cut * reduction
    window = cutout(image,
    row - 2 - window_size // 2,
    col - window_size // 2,
    window_size, window_size)
    #save_image(window, f"contexts_raw/{row}_{col}.png")
    window = scale_down(window, reduction)

    return window


def parse(image, cut, reduction, file_name):
    """Parses an image into layer specified by the reduction. Saves the result into file."""

    rows = image.shape[0]
    cols = image.shape[1]

    half_width = 160
    row_start = half_width + 2
    row_end = rows - half_width
    row_range = row_end - row_start
    col_start = half_width
    col_end = cols - half_width
    col_range = col_end - col_start
    
    tokens = [None] * (row_range * col_range)
    indices = [[x] for x in range(len(tokens))]

    
    def func(_index):
        i = _index[0]
        row = row_start + i // col_range
        col = col_start + i % col_range
        tokens[i] = get_window(image, row, col, reduction, cut).flatten()
        
    np.apply_along_axis(func, 1, indices)
    np.save(file_name, tokens, allow_pickle=True)