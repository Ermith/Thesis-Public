import numpy as np


def normalize_heights(folder, reductions=[1, 4, 16, 64]):
    """Performs standard score normalization on heights data.
    https://en.wikipedia.org/wiki/Standard_score
    """
    heights_folder = f"{folder}/heights_layers"

    for reduction in reductions:
        layer = np.load(f"{heights_folder}/layer_{reduction}x.npy")
        layer = (layer - np.mean(layer)) / np.std(layer)
        np.save(f"{heights_folder}/normalized_layer_{reduction}x.npy", layer, allow_pickle=True)