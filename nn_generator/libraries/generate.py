from .basic.matrix_manipulation import rescale, cutout, derivatives
from .basic.images import save_image, init_cmaps, save_all, show_image
from .basic.parsing import get_window
from .basic.utilities import noisify_exp, unary_log_encoding_array, unary_log_decoding, relativization, unary_linear_encoding,  unary_log_encoding_array_reversed, unary_log_decoding_reversed
from tensorflow.keras.models import load_model
from copy import deepcopy
import sys
import os
import numpy as np
import cv2


def check_path(path):
    if not os.path.exists(path):
        print(f"'{path}' does not exist.")
        sys.exit()

def load_data(model_file, data_folder, result_folder, data_name):
    image_name = f"{data_folder}/{data_name}.pgm"
    check_path(model_file)
    check_path(image_name)
    model = load_model(model_file)
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = rescale(image)

    save_image(image, f"{result_folder}/{data_name}.png")

    return (model, image)

def generate_map(
    generation_data,
    results_folder,
    config_file = "nn_generator/examples/generation_config.txt",
    random_modifier = 10,
):
    """Contains all logic for generation of maps using given RNNs.
    generation_data: str
        Folder containing user input.
    results_folder: str
        Folder to store the results.
    config_file: str
        Text file specifying used networks.
    """

    cut = 5
    encoding = 8
    # Universal padding
    padding = 64 * 5 // 2 + 2
    
    models = {}
    with open(config_file) as f:
        for line in f:
            tokens = line.strip().split("=")
            models[tokens[0]] = tokens[1]

    colors = init_cmaps()

    if not os.path.exists(results_folder):
    	os.makedirs(results_folder)


    def generate(model, padding, get_context, rows, cols, generated):

        for r in range(padding, rows + padding):
            for c in range(padding, cols + padding):
                context = get_context(r, c)
                recurrent = get_window(generated, r, c, 1, cut).flatten()[:-3]
                absolute = relativization(recurrent)
                absolute_encoded = unary_linear_encoding(absolute, encoding)
                recurrent = unary_log_encoding_array_reversed(recurrent, encoding)
                recurrent = np.concatenate(( recurrent, absolute_encoded ))
                _input = np.concatenate(( context, recurrent )).reshape(( 1, len(context) + len(recurrent) ))
                output = np.asarray(model(_input))
                output_decoded = unary_log_decoding_reversed(np.asarray(output).flatten())
                o = float(output_decoded) + absolute
                generated[r, c] = o 

        return generated

    def generate_other(model, get_context, rows, cols, generated, encode, _round):
        for r in range(padding, rows + padding):
            for c in range(padding, cols + padding):
                context = get_context(r, c)
                recurrent = get_window(generated, r, c, 1, cut).flatten()[:-3]
                if encode:
                    recurrent = unary_log_encoding_array(recurrent, encoding)
                _input = np.concatenate(( context, recurrent))
                _input = np.reshape(_input, (1, len(_input)))
                output = model(_input)
                if encode:
                    output = unary_log_decoding(np.asarray(output[0]))
                if _round:
                    output = round(float(output))
                generated[r, c] = float(output)

        return generated



    # 64 -> 16
    #======================================

    # HEIGHTS

    heights_model, heights = load_data(models["heights_64-16"], generation_data, results_folder, "heights")
    rows = int(np.ceil(heights.shape[0] / 16))
    cols = int(np.ceil(heights.shape[1] / 16))

    #heights = noisify_exp(heights, random_modifier)
    heights = rescale(heights)
    #heights = 1 - heights
    heights = np.pad(heights, padding, 'edge')


    def height_context_16x(row, col):
        r = (row - padding) * 16 + padding
        c = (col - padding) * 16 + padding
        window = get_window(heights, r, c, 64, cut)
        #show_image(window)
        absolute = relativization(window.flatten())
        absolute = unary_linear_encoding(absolute, encoding)
        window = unary_log_encoding_array_reversed(window.flatten(), encoding)

        return np.concatenate(( window, absolute ))

    initial = cv2.resize(heights, (rows, cols))
    show_image(initial)
    initial = np.pad(initial, padding, 'edge')
    generated_heights_16x = generate(heights_model, padding, height_context_16x, rows, cols, initial)
    generated_heights_16x = cutout(generated_heights_16x, padding, padding, rows, cols)
    height_diff_rows_16x, height_diff_cols_16x = derivatives(generated_heights_16x)
    height_diff_rows_16x = np.pad(height_diff_rows_16x, padding, 'reflect')
    height_diff_cols_16x = np.pad(height_diff_cols_16x, padding, 'reflect')


    show_image(generated_heights_16x)
    #sys.exit()

    save_image(
        generated_heights_16x,
        f"{results_folder}/generated_heights_16x.png")

    # ROADS

    road_model, roads = load_data(models["roads_64-16"], generation_data, results_folder, "roads")

    def road_context(r, c):
        heights1_rows = get_window(height_diff_rows_16x, r, c, 16, cut).flatten()
        heights1_cols = get_window(height_diff_cols_16x, r, c, 16, cut).flatten()
        roads1 = get_window(
            roads,
            (r - padding) * 16 + padding,
            (c - padding) * 16 + padding,
            64, cut).flatten()
        heights0_rows = get_window(height_diff_rows_16x, r, c, 4, cut).flatten()
        heights0_cols = get_window(height_diff_cols_16x, r, c, 4, cut).flatten()

        heights1_rows = unary_log_encoding_array(heights1_rows, encoding)
        heights1_cols = unary_log_encoding_array(heights1_cols, encoding)
        roads1 = unary_log_encoding_array(roads1, encoding)
        heights0_rows = unary_log_encoding_array(heights0_rows, encoding)
        heights0_cols = unary_log_encoding_array(heights0_cols, encoding)

        return np.concatenate(( heights1_rows, heights1_cols, roads1, heights0_rows, heights0_cols))

    initial = cv2.resize(roads, (rows, cols))
    initial = np.pad(initial, padding, 'reflect')
    roads = np.pad(roads, padding, 'reflect')
    generated_roads_16x = generate_other(road_model, road_context, rows, cols, initial, True, False)
    save_image(
        cutout(generated_roads_16x, padding, padding, rows, cols),
        f"{results_folder}/generated_roads_16x.png")

    # RIVERS
    river_model, rivers = load_data(models["rivers_64-16"], generation_data, results_folder, "rivers")

    def river_context(r, c):
        heights1_rows = get_window(height_diff_rows_16x, r, c, 16, cut).flatten()
        heights1_cols = get_window(height_diff_cols_16x, r, c, 16, cut).flatten()
        rivers1 = get_window(
            rivers, 
            (r - padding) * 16 + padding,
            (c - padding) * 16 + padding,
            64, cut).flatten()
        heights0_rows = get_window(height_diff_rows_16x, r, c, 4, cut).flatten()
        heights0_cols = get_window(height_diff_cols_16x, r, c, 4, cut).flatten()
        return unary_log_encoding_array(
                np.concatenate((
                    heights1_rows,
                    heights1_cols,
                    rivers1,
                    heights0_rows,
                    heights0_cols)),
                encoding)


    initial = cv2.resize(rivers, (rows, cols))
    initial = np.pad(initial, padding, 'reflect')
    rivers = np.pad(rivers, padding, 'reflect')
    generated_rivers_16x = generate_other(river_model, river_context, rows, cols, initial, True, False)

    save_image(
        cutout(generated_rivers_16x, padding, padding, rows, cols),
        f"{results_folder}/generated_rivers_16x.png")



    # BUILDINGS
    building_model, buildings = load_data(models["buildings_64-16"], generation_data, results_folder, "buildings")
    generated_heights_16x = np.pad(generated_heights_16x, padding, 'reflect')

    def building_context(r, c):
        heights1 = get_window(generated_heights_16x, r, c, 16, cut).flatten()
        rivers1 = get_window(generated_rivers_16x, r, c, 16, cut).flatten()
        roads1 = get_window(generated_roads_16x, r, c, 16, cut).flatten()
        buildings1 = get_window(
            buildings,
            (r - padding) * 16 + padding,
            (c - padding) * 16 + padding,
            64, cut).flatten()
        heights0 = get_window(generated_heights_16x, r, c, 4, cut).flatten()
        rivers0 = get_window(generated_rivers_16x, r, c, 4, cut).flatten()
        roads0 = get_window(generated_roads_16x, r, c, 4, cut).flatten()

        absolute1 = relativization(heights1)
        absolute1 = unary_linear_encoding(absolute1, encoding)
        heights1 = unary_log_encoding_array(heights1, encoding)
        rivers1 = unary_log_encoding_array(rivers1, encoding)
        roads1 = unary_log_encoding_array(roads1, encoding)
        buildings1 = unary_log_encoding_array(buildings1, encoding)
        absolute0 = relativization(heights0)
        absolute0 = unary_linear_encoding(absolute0, encoding)
        heights0 = unary_log_encoding_array(heights0, encoding)
        rivers0 = unary_log_encoding_array(rivers0, encoding)
        roads0 = unary_log_encoding_array(roads0, encoding)

        return np.concatenate((
            heights1,
            absolute1,
            rivers1,
            roads1,
            buildings1,
            heights0,
            absolute0,
            rivers0,
            roads0,
        ))


    initial = cv2.resize(buildings, (rows, cols))
    initial = np.pad(initial, padding, 'reflect')
    buildings = np.pad(buildings, padding)
    generated_buildings_16x = generate_other(building_model, building_context, rows, cols, initial, True, False)

    save_image(
        cutout(generated_buildings_16x, padding, padding, rows, cols),
        f"{results_folder}/generated_buildings_16x.png")

    # (64, 16) -> 4
    #======================================

    generated_heights_16x = cutout(generated_heights_16x, padding, padding, rows, cols)
    height_diff_rows_16x = cutout(height_diff_rows_16x, padding, padding, rows, cols)
    height_diff_cols_16x = cutout(height_diff_cols_16x, padding, padding, rows, cols)
    generated_roads_16x = cutout(generated_roads_16x, padding, padding, rows, cols)
    generated_rivers_16x = cutout(generated_rivers_16x, padding, padding, rows, cols)
    generated_buildings_16x = cutout(generated_buildings_16x, padding, padding, rows, cols)

    rows *= 4
    cols *= 4

    generated_heights_16x = cv2.resize(generated_heights_16x, (rows, cols))
    height_diff_rows_16x = cv2.resize(height_diff_rows_16x, (rows, cols))
    height_diff_cols_16x = cv2.resize(height_diff_cols_16x, (rows, cols))
    generated_roads_16x = cv2.resize(generated_roads_16x, (rows, cols))
    generated_rivers_16x = cv2.resize(generated_rivers_16x, (rows, cols))
    generated_buildings_16x = cv2.resize(generated_buildings_16x, (rows, cols))

    generated_heights_16x = np.pad(generated_heights_16x, padding, 'edge')
    height_diff_rows_16x = np.pad(height_diff_rows_16x, padding, 'reflect')
    height_diff_cols_16x = np.pad(height_diff_cols_16x, padding, 'reflect')
    generated_roads_16x = np.pad(generated_roads_16x, padding, 'reflect')
    generated_rivers_16x = np.pad(generated_rivers_16x, padding, 'reflect')
    generated_buildings_16x = np.pad(generated_buildings_16x, padding, 'reflect')
    
    generated_heights_16x = noisify_exp(generated_heights_16x, random_modifier)

    # HEIGHTS

    heights_model = load_model(models["heights_64-16-4"])

    def height_context_4x(r, c):
        heights2 = get_window(generated_heights_16x, r, c, 16, cut).flatten()
        heights1 = get_window(generated_heights_16x, r, c, 4, cut).flatten()
        absolute2 = relativization(heights2)
        absolute1 = relativization(heights1)
        heights2 = unary_log_encoding_array_reversed(heights2, encoding)
        heights1 = unary_log_encoding_array_reversed(heights1, encoding)
        absolute2 = unary_linear_encoding(absolute2, encoding)
        absolute1 = unary_linear_encoding(absolute1, encoding)
        return np.concatenate(( heights2, absolute2, heights1, absolute1 ))

    generated_heights_4x = generate(heights_model, padding, height_context_4x, rows, cols, deepcopy(generated_heights_16x))
    generated_heights_4x = cutout(generated_heights_4x, padding, padding, rows, cols)
    height_diff_rows_4x, height_diff_cols_4x = derivatives(generated_heights_16x)
    height_diff_rows_4x = np.pad(height_diff_rows_4x, padding, 'reflect')
    height_diff_cols_4x = np.pad(height_diff_cols_4x, padding, 'reflect')

    save_image(
        generated_heights_4x,
        f"{results_folder}/generated_heights_4x.png")


    # Roads

    road_model = load_model(models["roads_64-16-4"])

    def road_context_4x(r, c):
        heights2_rows = get_window(height_diff_rows_4x, r, c, 16, cut).flatten()
        heights2_cols = get_window(height_diff_cols_4x, r, c, 16, cut).flatten()
        roads2 = get_window(generated_roads_16x, r, c, 16, cut).flatten()

        heights1_rows = get_window(height_diff_rows_4x, r, c, 4, cut).flatten()
        heights1_cols = get_window(height_diff_cols_4x, r, c, 4, cut).flatten()
        roads1 = get_window(generated_roads_16x, r, c, 4, cut).flatten()

        heights0_rows = get_window(height_diff_rows_4x, r, c, 1, cut).flatten()
        heights0_cols = get_window(height_diff_cols_4x, r, c, 1, cut).flatten()

        return unary_log_encoding_array(
            np.concatenate((
                heights2_rows,
                heights2_cols,
                roads2,
                heights1_rows,
                heights1_cols,
                roads1,
                heights0_rows,
                heights0_cols
            )),
            encoding)

    generated_roads_4x = generate_other(road_model, road_context_4x, rows, cols, deepcopy(generated_roads_16x), True, False)
    save_image(
        cutout(generated_roads_4x, padding, padding, rows, cols),
        f"{results_folder}/generated_roads_4x.png")


    # RIVERS

    river_model = load_model(models["rivers_64-16-4"])

    def river_context_4x(r, c):
        heights2_rows = get_window(height_diff_rows_4x, r, c, 16, cut).flatten()
        heights2_cols = get_window(height_diff_cols_4x, r, c, 16, cut).flatten()
        rivers2 = get_window(generated_rivers_16x, r, c, 16, cut).flatten()

        heights1_rows = get_window(height_diff_rows_4x, r, c, 4, cut).flatten()
        heights1_cols = get_window(height_diff_cols_4x, r, c, 4, cut).flatten()
        rivers1 = get_window(generated_rivers_16x, r, c, 4, cut).flatten()

        heights0_rows = get_window(height_diff_rows_4x, r, c, 1, cut).flatten()
        heights0_cols = get_window(height_diff_cols_4x, r, c, 1, cut).flatten()

        return unary_log_encoding_array(
            np.concatenate((
                heights2_rows,
                heights2_cols,
                rivers2,
                heights1_rows,
                heights1_cols,
                rivers1,
                heights0_rows,
                heights0_cols
            )),
            encoding)

    generated_rivers_4x = generate_other(river_model, river_context_4x, rows, cols, deepcopy(generated_rivers_16x), True, False)
    save_image(
        cutout(generated_rivers_4x, padding, padding, rows, cols),
        f"{results_folder}/generated_rivers_4x.png")


    # BUILDINGS

    building_model = load_model(models["buildings_64-16-4"])
    generated_heights_4x = np.pad(generated_heights_4x, padding, 'reflect')

    def building_context_4x(r, c):
        heights2 = get_window(generated_heights_4x, r, c, 16, cut).flatten()
        rivers2 = get_window(generated_rivers_4x, r, c, 16, cut).flatten()
        roads2 = get_window(generated_roads_4x, r, c, 16, cut).flatten()
        buildings2 = get_window(generated_buildings_16x, r, c, 16, cut).flatten()
        heights1 = get_window(generated_heights_4x, r, c, 4, cut).flatten()
        rivers1 = get_window(generated_rivers_4x, r, c, 4, cut).flatten()
        roads1 = get_window(generated_roads_4x, r, c, 4, cut).flatten()
        buildings1 = get_window(generated_buildings_16x, r, c, 4, cut).flatten()
        heights0 = get_window(generated_heights_4x, r, c, 1, cut).flatten()
        rivers0 = get_window(generated_rivers_4x, r, c, 1, cut).flatten()
        roads0 = get_window(generated_roads_4x, r, c, 1, cut).flatten()

        absolute2 = relativization(heights2)
        absolute2 = unary_linear_encoding(absolute2, encoding)
        heights2 = unary_log_encoding_array(heights2, encoding)
        rivers2 = unary_log_encoding_array(rivers2, encoding)
        roads2 = unary_log_encoding_array(roads2, encoding)
        buildings2 = unary_log_encoding_array(buildings2, encoding)

        absolute1 = relativization(heights1)
        absolute1 = unary_linear_encoding(absolute1, encoding)
        heights1 = unary_log_encoding_array(heights1, encoding)
        rivers1 = unary_log_encoding_array(rivers1, encoding)
        roads1 = unary_log_encoding_array(roads1, encoding)
        buildings1 = unary_log_encoding_array(buildings1, encoding)

        absolute0 = relativization(heights0)
        absolute0 = unary_linear_encoding(absolute0, encoding)
        heights0 = unary_log_encoding_array(heights0, encoding)
        rivers0 = unary_log_encoding_array(rivers0, encoding)
        roads0 = unary_log_encoding_array(roads0, encoding)

        return np.concatenate((
            heights2,
            absolute2,
            rivers2,
            roads2,
            buildings2,
            heights1,
            absolute1,
            rivers1,
            roads1,
            buildings1,
            heights0,
            absolute0,
            rivers0,
            roads0,
        ))

    generated_buildings_4x = generate_other(building_model, building_context_4x, rows, cols, deepcopy(generated_buildings_16x),  True, False)
    save_image(
        cutout(generated_buildings_4x, padding, padding, rows, cols),
        f"{results_folder}/generated_buildings_4x.png")


    # (16, 4) -> 1
    #======================================
    generated_heights_4x = cutout(generated_heights_4x, padding, padding, rows, cols)
    height_diff_rows_4x = cutout(height_diff_rows_4x, padding, padding, rows, cols)
    height_diff_cols_4x = cutout(height_diff_cols_4x, padding, padding, rows, cols)
    generated_roads_4x = cutout(generated_roads_4x, padding, padding, rows, cols)
    generated_rivers_4x = cutout(generated_rivers_4x, padding, padding, rows, cols)
    generated_buildings_4x = cutout(generated_buildings_4x, padding, padding, rows, cols)

    rows *= 4
    cols *= 4

    generated_heights_4x = cv2.resize(generated_heights_4x, (rows, cols))
    height_diff_rows_4x = cv2.resize(height_diff_rows_4x, (rows, cols))
    height_diff_cols_4x = cv2.resize(height_diff_cols_4x, (rows, cols))
    generated_roads_4x = cv2.resize(generated_roads_4x, (rows, cols))
    generated_rivers_4x = cv2.resize(generated_rivers_4x, (rows, cols))
    generated_buildings_4x = cv2.resize(generated_buildings_4x, (rows, cols))

    generated_heights_4x = np.pad(generated_heights_4x, padding, 'edge')
    height_diff_rows_4x = np.pad(height_diff_rows_4x, padding, 'reflect')
    height_diff_cols_4x = np.pad(height_diff_cols_4x, padding, 'reflect')
    generated_roads_4x = np.pad(generated_roads_4x, padding, 'reflect')
    generated_rivers_4x = np.pad(generated_rivers_4x, padding, 'reflect')
    generated_buildings_4x = np.pad(generated_buildings_4x, padding, 'reflect')
    
    generated_heights_4x = noisify_exp(generated_heights_4x, random_modifier)

    # HEIGHTS

    heights_model = load_model(models["heights_16-4-1"])


    def height_context_1x(r, c):
        heights2 = get_window(generated_heights_4x, r, c, 16, cut).flatten()
        heights1 = get_window(generated_heights_4x, r, c, 4, cut).flatten()
        absolute2 = relativization(heights2)
        absolute1 = relativization(heights1)
        heights2 = unary_log_encoding_array_reversed(heights2, encoding)
        heights1 = unary_log_encoding_array_reversed(heights1, encoding)
        absolute2 = unary_linear_encoding(absolute2, encoding)
        absolute1 = unary_linear_encoding(absolute1, encoding)
        return np.concatenate(( heights2, absolute2, heights1, absolute1 ))

    generated_heights_1x = generate(heights_model, padding, height_context_1x, rows, cols, deepcopy(generated_heights_4x))
    generated_heights_1x = cutout(generated_heights_1x, padding, padding, rows, cols)
    height_diff_rows_1x, height_diff_cols_1x = derivatives(generated_heights_1x)
    height_diff_rows_1x = np.pad(height_diff_rows_1x, padding, 'reflect')
    height_diff_cols_1x = np.pad(height_diff_cols_1x, padding, 'reflect')

    save_image(
        generated_heights_1x,
        f"{results_folder}/generated_heights_1x.png")

    road_model = load_model(models["roads_16-4-blurry"])

    def road_context_1x(r, c):
        heights2_rows = get_window(height_diff_rows_1x, r, c, 16, cut).flatten()
        heights2_cols = get_window(height_diff_cols_1x, r, c, 16, cut).flatten()
        roads2 = get_window(generated_roads_4x, r, c, 16, cut).flatten()

        heights1_rows = get_window(height_diff_rows_1x, r, c, 4, cut).flatten()
        heights1_cols = get_window(height_diff_cols_1x, r, c, 4, cut).flatten()
        roads1 = get_window(generated_roads_4x, r, c, 4, cut).flatten()

        heights0_rows = get_window(height_diff_rows_1x, r, c, 1, cut).flatten()
        heights0_cols = get_window(height_diff_cols_1x, r, c, 1, cut).flatten()

        return unary_log_encoding_array(
            np.concatenate((
                heights2_rows,
                heights2_cols,
                roads2,
                heights1_rows,
                heights1_cols,
                roads1,
                heights0_rows,
                heights0_cols
                )),
            encoding)

    generated_roads_1x = generate_other(road_model, road_context_1x, rows, cols, deepcopy(generated_roads_4x), True, False)
    generated_roads_1x = cutout(generated_roads_1x, padding, padding, rows, cols)

    save_image(
        generated_roads_1x,
        f"{results_folder}/generated_roads_blurry_1x.png")

    # Sharpen
    generated_roads_1x = np.pad(generated_roads_1x, padding, 'reflect')
    road_sharp_model = load_model(models["roads_sharp"])
    def road_sharp_context(row, col):
        window = get_window(generated_roads_1x, row, col, 1, cut)
        return unary_log_encoding_array(window.flatten(), encoding)

    generated_roads_1x = generate_other(road_sharp_model, road_sharp_context, rows, cols, deepcopy(generated_roads_1x), False,False)
    save_image(
        cutout(generated_roads_1x, padding, padding, rows, cols),
        f"{results_folder}/generated_roads_1x.png")


    # RIVERS

    river_model = load_model(models["rivers_16-4-blurry"])

    def river_context_1x(r, c):
        heights2_rows = get_window(height_diff_rows_1x, r, c, 16, cut).flatten()
        heights2_cols = get_window(height_diff_cols_1x, r, c, 16, cut).flatten()
        rivers2 = get_window(generated_rivers_4x, r, c, 16, cut).flatten()

        heights1_rows = get_window(height_diff_rows_1x, r, c, 4, cut).flatten()
        heights1_cols = get_window(height_diff_cols_1x, r, c, 4, cut).flatten()
        rivers1 = get_window(generated_rivers_4x, r, c, 4, cut).flatten()

        heights0_rows = get_window(height_diff_rows_1x, r, c, 1, cut).flatten()
        heights0_cols = get_window(height_diff_cols_1x, r, c, 1, cut).flatten()

        return unary_log_encoding_array(
            np.concatenate((
                heights2_rows,
                heights2_cols,
                rivers2,
                heights1_rows,
                heights1_cols,
                rivers1,
                heights0_rows,
                heights0_cols
            )),
            encoding)

    generated_rivers_1x = generate_other(river_model, river_context_1x, rows, cols, deepcopy(generated_rivers_4x), True, False)
    generated_rivers_1x = cutout(generated_rivers_1x, padding, padding, rows, cols)
    save_image(
        generated_rivers_1x,
        f"{results_folder}/generated_rivers_blurry_1x.png")

    # Sharpen
    generated_rivers_1x = np.pad(generated_rivers_1x, padding, 'reflect')
    river_sharp_model = load_model(models["rivers_sharp"])
    def river_sharp_context(row, col):
        window = get_window(generated_rivers_1x, row, col, 1, cut)
        return unary_log_encoding_array(window.flatten(), encoding)

    generated_rivers_1x = generate_other(river_sharp_model, river_sharp_context, rows, cols, deepcopy(generated_rivers_1x), False, False)

    save_image(
        cutout(generated_rivers_1x, padding, padding, rows, cols),
        f"{results_folder}/generated_rivers_1x.png")


    # BUILDING

    building_model = load_model(models["buildings_16-4-blurry"])
    generated_heights_1x = np.pad(generated_heights_1x, padding, 'reflect')

    def building_context_1x(r, c):
        heights2 = get_window(generated_heights_1x, r, c, 16, cut).flatten()
        rivers2 = get_window(generated_rivers_1x, r, c, 16, cut).flatten()
        roads2 = get_window(generated_roads_1x, r, c, 16, cut).flatten()
        buildings2 = get_window(generated_buildings_4x, r, c, 16, cut).flatten()

        heights1 = get_window(generated_heights_1x, r, c, 4, cut).flatten()
        rivers1 = get_window(generated_rivers_1x, r, c, 4, cut).flatten()
        roads1 = get_window(generated_roads_1x, r, c, 4, cut).flatten()
        buildings1 = get_window(generated_buildings_4x, r, c, 4, cut).flatten()

        heights0 = get_window(generated_heights_1x, r, c, 1, cut).flatten()
        rivers0 = get_window(generated_rivers_1x, r, c, 1, cut).flatten()
        roads0 = get_window(generated_roads_1x, r, c, 1, cut).flatten()

        absolute2 = relativization(heights2)
        absolute2 = unary_linear_encoding(absolute2, encoding)
        heights2 = unary_log_encoding_array(heights2, encoding)
        rivers2 = unary_log_encoding_array(rivers2, encoding)
        roads2 = unary_log_encoding_array(roads2, encoding)
        buildings2 = unary_log_encoding_array(buildings2, encoding)

        absolute1 = relativization(heights1)
        absolute1 = unary_linear_encoding(absolute1, encoding)
        heights1 = unary_log_encoding_array(heights1, encoding)
        rivers1 = unary_log_encoding_array(rivers1, encoding)
        roads1 = unary_log_encoding_array(roads1, encoding)
        buildings1 = unary_log_encoding_array(buildings1, encoding)

        absolute0 = relativization(heights0)
        absolute0 = unary_linear_encoding(absolute0, encoding)
        heights0 = unary_log_encoding_array(heights0, encoding)
        rivers0 = unary_log_encoding_array(rivers0, encoding)
        roads0 = unary_log_encoding_array(roads0, encoding)

        return np.concatenate((
            heights2,
            absolute2,
            rivers2,
            roads2,
            buildings2,
            heights1,
            absolute1,
            rivers1,
            roads1,
            buildings1,
            heights0,
            absolute0,
            rivers0,
            roads0,
        ))

    generated_buildings_1x = generate_other(building_model, building_context_1x, rows, cols, deepcopy(generated_buildings_4x), True, False)

    # Sharpen

    generated_buildings_1x = cutout(generated_buildings_1x, padding, padding, rows, cols)
    save_image(
        generated_buildings_1x,
        f"{results_folder}/generated_buildings_blurry_1x.png")


    generated_buildings_1x = np.pad(generated_buildings_1x, padding, 'reflect')
    building_sharp_model = load_model(models["buildings_sharp"])
    def building_sharp_context(row, col):
        window = get_window(generated_buildings_1x, row, col, 1, cut)
        return unary_log_encoding_array(window.flatten(), encoding)

    generated_buildings_1x = generate_other(building_sharp_model, building_sharp_context, rows, cols, deepcopy(generated_buildings_1x), False, False)

    save_image(
        cutout(generated_buildings_1x, padding, padding, rows, cols),
        f"{results_folder}/generated_buildings_1x.png")

    save_all([
        cutout(generated_heights_1x, padding, padding, rows, cols),
        cutout(generated_rivers_1x, padding, padding, rows, cols),
        cutout(generated_roads_1x, padding, padding, rows, cols),
        cutout(generated_buildings_1x, padding, padding, rows, cols)
        ],
        colors,
        f"{results_folder}/generated_all.png"
    )