from libraries.matrix_manipulation import rescale, rescale2
from libraries.utilities import unary_log_encoding_array, noisify_exp, relativization, unary_linear_encoding, unary_log_encoding_array_reversed
import os
import numpy as np

def process_training_data(mode, reduction0, reduction1, reduction2=None, training_folder="training_data", size=150000):
    encoding = 8

    if reduction2 == None:
        output_folder_name = f"{training_folder}/{mode}s_{reduction1}-{reduction0}_{size}"
    else:
        output_folder_name = f"{training_folder}/{mode}s_{reduction2}-{reduction1}-{reduction0}_{size}"


    def make_dir_checked(dir):
        if not os.path.exists(dir):
    	    os.makedirs(dir)

    def heights_file(reduction):
        return f"{training_folder}/height_layers/normalized_layer_{reduction}x.npy"

    def heights_diff_file(reduction):
        rows = f"{training_folder}/height_layers/differences_rows_layer_{reduction}x.npy"
        columns = f"{training_folder}/height_layers/differences_columns_layer_{reduction}x.npy"

        return (rows, columns)

    def mode_file(mode, reduction):
        return f"{training_folder}/{mode}_layers/layer_{reduction}x.npy"

    def mode_blurred_file(mode, reduction):
        return f"{training_folder}/{mode}_layers/layer_blurry_{reduction}x.npy"



    make_dir_checked(output_folder_name)

    if mode == "height":
        layer0 = np.load(heights_file(reduction0))
        count = layer0.shape[0]
        randomizer = np.random.choice(count, size=size, replace=False)
        func = lambda t : unary_log_encoding_array_reversed(t, encoding)
        noisify = lambda t : noisify_exp(t, 20)

        def context_func(arr):
            absolute = relativization(arr)
            a = unary_log_encoding_array_reversed(arr, encoding)
            b = unary_linear_encoding(absolute, encoding)
            return np.concatenate(( a, b ))

        absolutes = []

        def recurrent_func(arr):
            absolute = relativization(arr)
            absolutes.append(absolute)
            a = unary_log_encoding_array_reversed(arr, encoding)
            b = unary_linear_encoding(absolute, encoding)
            return np.concatenate(( a, b ))

        layer0 = layer0[randomizer]
        layer0 = np.apply_along_axis(noisify, 1, layer0)
        layer0 = rescale(layer0)
        outputs = layer0[:, -3]
        recurrents = layer0[:, :-3]
        recurrents = np.apply_along_axis(recurrent_func, 1, recurrents)
        outputs -= np.asarray(absolutes)
        outputs = np.reshape(outputs, (size, 1))
        outputs = np.apply_along_axis(func, 1, outputs)

        layer1 = np.load(heights_file(reduction1))
        layer1 = layer1[randomizer]
        layer1 = np.apply_along_axis(noisify, 1, layer1)
        layer1 = rescale(layer1)
        layer1 = np.apply_along_axis(context_func, 1, layer1)

        inputs = np.concatenate((layer1, recurrents), axis=1)

        if (reduction2 != None):
            layer2 = np.load(heights_file(reduction2))
            layer2 = layer2[randomizer]
            layer2 = np.apply_along_axis(noisify, 1, layer2)
            layer2 = rescale(layer2)
            layer2 = np.apply_along_axis(context_func, 1, layer2)

            inputs = np.concatenate((layer2, inputs), axis=1)

        np.save(f"{output_folder_name}/inputs", inputs)
        np.save(f"{output_folder_name}/outputs", outputs)

    if mode == "road" or mode == "river":
        # Reduction1
        #======================

        # Paths1
        paths1 = np.load(mode_file(mode, reduction1))
        # Thin out empty entries
        nonzero_indices = np.argwhere(np.any(paths1 != 0, axis=1)).flatten()
        randomizer = np.random.choice(nonzero_indices, size=size, replace=False)

        paths1 = paths1[randomizer]
        func = lambda t: unary_log_encoding_array(t, encoding)
        paths1 = np.apply_along_axis(func, 1, paths1)

        # Paths0
        file0 = mode_file(mode, reduction0) if reduction0 != 1 else mode_blurred_file(mode, reduction0)
        paths0 = np.load(file0)[randomizer]
        outputs = paths0[:, -3]
        recurrents = paths0[:, :-3]

        outputs = np.reshape(outputs, (len(outputs), 1))
        outputs = np.apply_along_axis(func, 1, outputs)
        recurrents = np.apply_along_axis(func, 1, recurrents)

        # Height differences
        def diff_func(arr):
            arr = rescale2(arr)
            arr = unary_log_encoding_array(arr, 8)
            return arr
    
        row_file, col_file = heights_diff_file(reduction0)
        height0_diff_rows = np.load(row_file)
        height0_diff_rows = height0_diff_rows[randomizer]
        height0_diff_cols = np.load(col_file)[randomizer]
        height0_diff_rows = np.apply_along_axis(diff_func, 1, height0_diff_rows)
        height0_diff_cols = np.apply_along_axis(diff_func, 1, height0_diff_cols)
    
        row_file, col_file = heights_diff_file(reduction1)
        height1_diff_rows = np.load(row_file)[randomizer]
        height1_diff_cols = np.load(col_file)[randomizer]
        height1_diff_rows = np.apply_along_axis(diff_func, 1, height1_diff_rows)
        height1_diff_cols = np.apply_along_axis(diff_func, 1, height1_diff_cols)
    
        # Put it together
        inputs = np.concatenate((
            height1_diff_rows,
            height1_diff_cols,
            paths1,
            height0_diff_rows,
            height0_diff_cols,
            recurrents),
            axis=1
        )

        # Reduction 2
        #==================
        if (reduction2 != None):
            paths2 = np.load(mode_file(mode, reduction2))[randomizer]
            paths2 = np.apply_along_axis(func, 1, paths2)
            row_file, col_file = heights_diff_file(reduction2)
            height2_diff_rows = np.load(row_file)[randomizer]
            height2_diff_cols = np.load(col_file)[randomizer]
            height2_diff_rows = np.apply_along_axis(diff_func, 1, height2_diff_rows)
            height2_diff_cols = np.apply_along_axis(diff_func, 1, height2_diff_cols)

            inputs = np.concatenate((height2_diff_rows, height2_diff_cols, paths2, inputs), axis=1)

        if (reduction0 == 1):
            make_dir_checked(f"{output_folder_name}/blurry")
            np.save(f"{output_folder_name}/blurry/inputs", inputs)
            np.save(f"{output_folder_name}/blurry/outputs", outputs)
        else:
            np.save(f"{output_folder_name}/inputs", inputs)
            np.save(f"{output_folder_name}/outputs", outputs)


        # Sharp
        #========================
    
        if (reduction0 == 1):
            blurry = np.load(mode_blurred_file(mode, reduction0))[randomizer]
            blurry = noisify_exp(blurry, 20)
            blurry = np.apply_along_axis(func, 1, blurry)

            sharp = np.load(mode_file(mode, reduction0))[randomizer]
            outputs = sharp[:, -3]
            recurrents = sharp[:, :-3]

            inputs = np.concatenate((blurry, recurrents), axis=1)

            make_dir_checked(f"{output_folder_name}/sharp")
            np.save(f"{output_folder_name}/sharp/inputs", inputs)
            np.save(f"{output_folder_name}/sharp/outputs", outputs)

    if mode == "building":
        # Reduction1
        #=======================

        buildings1 = np.load(mode_file("building", reduction1))
        # Thin out empty entries
        nonzero_indices = np.argwhere(np.any(buildings1 != 0, axis=1)).flatten()
        randomizer = np.random.choice(nonzero_indices, size=size, replace=False)

        buildings1 = buildings1[randomizer]
        func = lambda t: unary_log_encoding_array(t, encoding)
        noisify = lambda t: noisify_exp(t, 20)
        buildings1 = np.apply_along_axis(func, 1, buildings1)

        file0 =  mode_file(mode, reduction0) if reduction0 != 1 else mode_blurred_file(mode, reduction0)
        buildings0 = np.load(mode_file("building", reduction0))[randomizer]
        outputs = buildings0[:, -3]
        recurrents = buildings0[:, :-3]

        # Roads
        roads0 = np.load(mode_file("road", reduction0))[randomizer]
        roads1 = np.load(mode_file("road", reduction1))[randomizer]
        roads1 = np.apply_along_axis(func, 1, roads1)

        # Rivers
        rivers0 = np.load(mode_file("river", reduction0))[randomizer]
        rivers1 = np.load(mode_file("river", reduction1))[randomizer]
        rivers1 = np.apply_along_axis(func, 1, rivers1)

        # Heights
        heights0 = np.load(heights_file(reduction0))[randomizer]
        heights1 = np.load(heights_file(reduction1))[randomizer]

        def context_func_b(arr):
            absolute = relativization(arr)
            a = unary_log_encoding_array(arr, encoding)
            b = unary_linear_encoding(absolute, encoding)
            return np.concatenate(( a, b ))


        heights0 = np.apply_along_axis(noisify, 1, heights0)
        heights0 = rescale(heights0)
        heights0 = np.apply_along_axis(context_func_b, 1, heights0)

        heights1 = np.apply_along_axis(noisify, 1, heights1)
        heights1 = rescale(heights1)
        heights1 = np.apply_along_axis(context_func_b, 1, heights1)

        roads0 = np.apply_along_axis(func, 1, roads0)
        rivers0 = np.apply_along_axis(func, 1, rivers0)
        buildings0 = np.apply_along_axis(func, 1, buildings0)
        recurrents = np.apply_along_axis(func, 1, recurrents)

        outputs = np.reshape(outputs, (len(outputs), 1))
        outputs = np.apply_along_axis(func, 1, outputs)

        # Put it together
        inputs = np.concatenate((
            heights1,
            rivers1,
            roads1,
            buildings1,
            heights0,
            rivers0,
            roads0,
            recurrents),
            axis=1
        )
    
        # Reduction2
        #==================
        if reduction2 != None:
            heights2 = np.load(heights_file(reduction2))[randomizer]
            heights2 = rescale(heights2)
            rivers2 = np.load(mode_file("river", reduction2))[randomizer]
            rivers2 = np.apply_along_axis(func, 1, rivers2)
            roads2 = np.load(mode_file("road", reduction2))[randomizer]
            roads2 = np.apply_along_axis(func, 1, roads2)
            buildings2 = np.load(mode_file("building", reduction2))[randomizer]
            buildings2 = np.apply_along_axis(func, 1, buildings2)

            heights2 = np.apply_along_axis(noisify, 1, heights2)
            heights2 = rescale(heights2)
            heights2 = np.apply_along_axis(context_func_b, 1, heights2)

            inputs = np.concatenate((
                heights2,
                rivers2,
                roads2,
                buildings2,
                inputs),
                axis=1
            )

        if (reduction0 == 1):
            make_dir_checked(f"{output_folder_name}/blurry")
            np.save(f"{output_folder_name}/blurry/inputs", inputs)
            np.save(f"{output_folder_name}/blurry/outputs", outputs)
        else:
            np.save(f"{output_folder_name}/inputs", inputs)
            np.save(f"{output_folder_name}/outputs", outputs)

        # Sharp
        #========================
    
        if (reduction0 == 1):
            blurry = np.load(mode_blurred_file(mode, reduction0))[randomizer]
            blurry = noisify_exp(blurry, 20)
            blurry = np.apply_along_axis(func, 1, blurry)

            sharp = np.load(mode_file(mode, reduction0))[randomizer]
            outputs = sharp[:, -3]
            recurrents = sharp[:, :-3]

            inputs = np.concatenate((blurry, recurrents), axis=1)

            make_dir_checked(f"{output_folder_name}/sharp")
            np.save(f"{output_folder_name}/sharp/inputs", inputs)
            np.save(f"{output_folder_name}/sharp/outputs", outputs)
            