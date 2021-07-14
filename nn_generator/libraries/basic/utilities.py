import numpy as np
import math

def noisify_exp(matrix, val):
    """Applies random noise with exponential distribution to a matrix."""
    l = abs(np.max(matrix) - np.min(matrix))
    return matrix + np.random.exponential(l / val, matrix.shape)

def noisify_cauchy(matrix, rng):
    """Applies random noise with cauchy distribution to a matrix."""
    random_matrix = np.random.standard_cauchy(matrix.shape) * rng
    matrix += random_matrix

    return matrix

def unary_encoding(num, encoding_length):
    arr = np.zeros(encoding_length)
    _int = int(num)
    _max = min(_int, encoding_length)
    for i in range(_max):
        arr[i] = 1

    if (_max < encoding_length):
        arr[_max] = num - _int
    return arr

def unary_encoding_reversed(num, encoding):
    arr = np.ones(encoding)
    _int = int(num)
    _max = min(_int, encoding)
    for i in range(_max):
        arr[i] = 0
    if (_max < encoding):
        arr[_max] = 1 - (num - _int)
    return arr

def unary_log_encoding_reversed(num, encoding):
    if num != 0:
        x = -math.log2(abs(num))
    else:
        x = encoding
    
    encoded = unary_encoding_reversed(x, encoding)

    if num < 0:
        encoded *= -1

    return encoded

def unary_log_encoding_array_reversed(arr, encoding):
    ret = []

    for i in range(len(arr)):
        ret.extend(unary_log_encoding_reversed(arr[i], encoding))

    return np.asarray(ret)

def unary_log_encoding(num, encoding_length):
    if num != 0:
        x = -math.log2(abs(num))
    else:
        x = encoding_length
    
    encoded = unary_encoding(x, encoding_length)

    if num < 0:
        encoded *= -1

    return encoded

def unary_linear_encoding(num, encoding_length):
    x = abs(num * encoding_length)
    encoded = unary_encoding(x, encoding_length)
    if num < 0:
        encoded *= -1

    return encoded

def unary_log_encoding_array(arr, encoding_length):
    ret = []

    for i in range(len(arr)):
        ret.extend(unary_log_encoding(arr[i], encoding_length))

    return np.asarray(ret)

def unary_linear_encoding_array(arr, encoding_length):
    ret = []

    for i in range(len(arr)):
        ret.extend(unary_linear_encoding(arr[i], encoding_length))

    return np.asarray(ret)


def unary_log_decoding_reversed(arr):
    k = 0
    _sum = 0
    for i in arr:
        sgn = 1
        if (i < 0): sgn = -1
        _sum += sgn * (-2 ** (-abs(i) - k) + 2**-k)
        k += 1

    return _sum


def unary_log_decoding(arr):
    s = np.sum(arr)
    sum = np.sum(abs(arr))

    result = float(2)**(-abs(sum)) * np.sign(s)
    return result

def unary_linear_decoding(arr):
    s = np.sum(arr)
    return abs(s) / len(arr)

def unary_log_decoding_array(arr, encoding_length):
    ret = []
    for i in range(0, len(arr), encoding_length):
        part = arr[i : i + encoding_length]
        decoded = unary_log_decoding(part)
        ret.append(decoded)

    return np.asarray(ret)

def relativization(arr):
    """Changes array elements to be relative to some absolute value."""
    absolute = np.average(arr)
    arr -= absolute
    return absolute