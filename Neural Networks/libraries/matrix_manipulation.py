import numpy as np

def rescale(matrix):
    minimum = np.min(matrix)
    maximum = np.max(matrix)
    length = maximum - minimum

    if (length == 0):
        return matrix * 0

    return (matrix - minimum) / length   


def rescale2(matrix):
    minimum = np.min(matrix)
    maximum = np.max(matrix)
    scalar = max(abs(maximum), abs(minimum))
    if (scalar == 0):
        return matrix

    return matrix / scalar


def derivatives(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    r_ders = np.zeros((rows, cols))
    c_ders = np.zeros((rows, cols))

    for r in range(rows - 1):
        for c in range(cols):
            r_ders[r, c] = matrix[r + 1, c] - matrix[r, c]

    for r in range(rows):
        for c in range(cols - 1):
            c_ders[r, c] = matrix[r, c + 1] - matrix[r, c]

    return (r_ders, c_ders)


def cutout(matrix, row, col, r, c):
    return matrix[row : row + r, col : col + c]

def scale_down(matrix, reduction):
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    new_rows = rows // reduction
    new_cols = cols // reduction
    new_matrix = np.reshape(matrix, (new_rows, reduction, new_cols, reduction)).mean(-1).mean(1)

    return new_matrix
