from libraries.matrix_manipulation import cutout, scale_down
import numpy as np


def get_window(image, row, col, reduction, cut):
    window_size = cut * reduction
    window = cutout(image,
    row - 2 - window_size // 2,
    col - window_size // 2,
    window_size, window_size)

    window = scale_down(window, reduction)

    return window


def parse(heights, cut, reduction, file_name):
    tokens = []

    rows = heights.shape[0]
    cols = heights.shape[1]

    half_width = 160

    for row in range(half_width + 2, rows - half_width):
        for col in range(half_width, cols - half_width):
            window = get_window(heights, row, col, reduction, cut)
            tokens.append(window.flatten())

    np.save(file_name, tokens, allow_pickle=True)