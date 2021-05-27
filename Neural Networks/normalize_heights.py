import numpy as np


def normalize_heights():
    folder_name = "training_data/height_layers"

    means = []
    deviations = []

    for i in range(4):
        layer = np.load(f"{folder_name}/layer_{4**i}x.npy")
        means.append(np.mean(layer))
        deviations.append(np.std(layer))

    for i in range(4):
        layer = np.load(f"{folder_name}/layer_{4**i}x.npy")
        layer = (layer - means[i]) / deviations[i]
        np.save(f"{folder_name}/normalized_layer_{4**i}x.npy", layer, allow_pickle=True)