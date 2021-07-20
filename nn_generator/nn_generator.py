"""This file contains main functions for learning and generation using pixel RNNs."""

from cv2 import data
from .libraries.parse import parse_image
from .libraries.process_training_data import process_training_data
from .libraries.normalize_heights import normalize_heights
from .libraries.train import train
from .libraries.generate import generate_map
import os
from typing import List

def parse(
    mode,
    image,
    reductions = [1, 4, 16, 64],
    output_folder = "training_data"
    ):
    """This function parses an image into layers specified by reductions.
    mode: str
        What are we parsing? "heights", "roads", "rivers" or "buildings" ?
    image: str
        Image to be parsed.
    reductions: [int]
        Specifies the layers.
    output_folder: str
        Creates layer subfolders. Path is created if doesn't exist.
    """

    print(f"Parsing {mode} . . .")

    folder = f"{output_folder}/{mode}_layers"

    parse_image(image, folder, reductions=reductions)

    if (mode == "heights"):
        normalize_heights(output_folder, reductions)
        parse_image(image, folder, reductions=reductions, compute_differences=True)
    else:
        parse_image(image, folder, reductions=[reductions[0]], blur=True)

def construct_training_datasets(
    mode,
    reduction0 = 1,
    reduction1 = 4,
    reduction2 = 16,
    reduction3 = 64,
    size = 150000,
    folder = "training_data"
    ):
    """After parsing images, constructs training datasets from the folder containing parsed images into layers.
    mode : str
        What are we parsing? "heights", "roads", "rivers" or "buildings" ?
    reduction[0-3] : int
        Layers used for datasets.
    size : int
        Number of training examples in constructed datasets.
    folder : str
        Folder containing parsed images. Results will be also saved into this folder.
    """
    print(f"Constructing {mode} datasets . . .")

    if not os.path.exists(folder):
        print(f"{folder} does not exist. Did you parse the data?")
        return

    # Do 3 variants
    process_training_data(mode, reduction0=reduction2, reduction1=reduction3, reduction2=None, size=size, training_folder=folder)
    process_training_data(mode, reduction0=reduction1, reduction1=reduction2, reduction2=reduction3, size=size, training_folder=folder)
    process_training_data(mode, reduction0=reduction0, reduction1=reduction1, reduction2=reduction2, size=size, training_folder=folder)

def train_networks(
    mode,
    structures_folder,
    output_folder,
    reduction0 = 1,
    reduction1 = 4,
    reduction2 = 16,
    reduction3 = 64,
    size = 150000,
    epochs = (1, 1, 1, 1),
    data_folder = "training_data"
    ):
    """Trains networks based on given parameters.
    mode: str
        What are we parsing? "heights", "roads", "rivers" or "buildings" ?
    structures_folder: str
        Folder containing text files specifying structures of the networks to be trained.
    output_folder: str
        Where to save the networks.
        Path is created if it does not exist.
    reduction[0-3]: int
        Layers used.
    size: int
        Size of training dataset to be loaded.
        This exact dataset needs to be prepared beforehand.
    epochs: (int, int, int, int)
        Number of epochs for each network.
        For heights, only 3 is needed.
        For the rest 4 is needed, due to the blurry and sharp layer.
    data_folder: str
        Folder containing training data.
    """
    print(f"Training {mode} . . .")
    if mode != "heights" and len(epochs) < 4:
        print(f"At least 4 numbers needed in 'epochs' argument for {mode}.")
        print(f"See help(nn_generator.train_networks)")
        return

    if mode == "heights" and len(epochs) < 3:
        print(f"At least 3 numbers needed in 'epochs' argument for {mode}.")
        print(f"See help(nn_generator.train_networks)")
        return

    def structure(num):
        return f"{structures_folder}/{mode}{num}.txt"
    
    def check_data(data):
        if not os.path.exists(data):
            print(f"{data} does not exist.")
            return False
        
        return True

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data2 = f"{data_folder}/{mode}_{reduction3}-{reduction2}_{size}"
    data1 = f"{data_folder}/{mode}_{reduction3}-{reduction2}-{reduction1}_{size}"

    if not check_data(data2) or not check_data(data1):
        return

    train(structure(2), data2, f"{output_folder}/{mode}_{reduction3}-{reduction2}", epochs=epochs[0])
    train(structure(1), data1, f"{output_folder}/{mode}_{reduction3}-{reduction2}-{reduction1}", epochs=epochs[1])

    if (mode == "heights"):
        data0 = f"{data_folder}/heights_{reduction2}-{reduction1}-{reduction0}_{size}"
        if not check_data(data0): return
        train(structure(0), data0, f"{output_folder}/heights_{reduction2}-{reduction1}-{reduction0}", epochs=epochs[2])
    else:
        data_blurry = f"{data_folder}/{mode}_{reduction2}-{reduction1}-{reduction0}_{size}/blurry"
        data_focus = f"{data_folder}/{mode}_{reduction2}-{reduction1}-{reduction0}_{size}/sharp"

        if not check_data(data_blurry) or not check_data(data_focus):
            return

        train(structure("0_blurry"), data_blurry, f"{output_folder}/{mode}_{reduction2}-{reduction1}-blurry", epochs=epochs[2])
        train(structure("0_sharp"), data_focus, f"{output_folder}/{mode}_blurry-sharp", epochs=epochs[3])

def learn(
    training_data,
    training_folder = "training_data",
    dataset_sizes = {"heights" : 150000, "roads" : 150000, "rivers" : 150000, "buildings" : 150000},
    epochs = {
        "heights" : (10,10,10),
        "roads" : (6, 6, 6, 6),
        "rivers" : (6, 6, 6, 6),
        "buildings" : (6, 6, 6, 6)
    },
    network_structures = "nn_generator/examples/model_structures",
    network_folder = "models"
    ):
    """Parse the images, construct datasets and create and train neural networks for heights, roads, rivers and buildings.
    After executingthis function, the networks are ready to generate images.

    training_data: str
        Name of the file containing training input.
        Expects 4 PGM images named: heights.pgm, roads.pgm, rivers.pgm and buildings.pgm
    training_folder: str
        Folder to store all parsed data and constructed training datasets.
    dataset_sizes: dict[str, int]
        Number of training examples used to train each network.
        See the default value.
    epochs: dict[str, (int, int, int, int)]
        Number of epochs for each mode, for each network.
        In the case of heights, only 3 numbers are needed.
        In other cases 4, due to blurry and sharp layers.
        See the default value.
    network_structures: str
        Folder containing text files specifying structures for the networks.
    network_folder: str
        Folder, where networks will be saved.
        The path is created if it does not exist.
    """

    heights_image = f"{training_data}/heights.pgm"
    roads_image = f"{training_data}/roads.pgm"
    rivers_image = f"{training_data}/rivers.pgm"
    buildings_image = f"{training_data}/buildings.pgm"
    
    print()
    print("PARSING IMAGES")
    print("===================")
    parse("heights", heights_image, output_folder=training_folder)
    parse("roads", roads_image, output_folder=training_folder)
    parse("rivers", rivers_image, output_folder=training_folder)
    parse("buildings", buildings_image, output_folder=training_folder)

    print()
    print("CONSTRUCTING TRAINING DATA")
    print("==============================")
    construct_training_datasets("heights", folder=training_folder, size=dataset_sizes["heights"])
    construct_training_datasets("roads", folder=training_folder, size=dataset_sizes["roads"])
    construct_training_datasets("rivers", folder=training_folder, size=dataset_sizes["rivers"])
    construct_training_datasets("buildings", folder=training_folder, size=dataset_sizes["buildings"])

    print()
    print("TRAINING THE NETWORKS")
    print("=========================")
    train_networks(
        "heights",
        structures_folder=network_structures,
        output_folder=network_folder,
        size=dataset_sizes["heights"],
        epochs=epochs["heights"])
    
    train_networks(
        "roads",
        structures_folder=network_structures,
        output_folder=network_folder,
        size=dataset_sizes["roads"],
        epochs=epochs["roads"])
    
    train_networks(
        "rivers",
        structures_folder=network_structures,
        output_folder=network_folder,
        size=dataset_sizes["rivers"],
        epochs=epochs["rivers"])
    
    train_networks(
        "buildings",
        structures_folder=network_structures,
        output_folder=network_folder,
        size=dataset_sizes["buildings"],
        epochs=epochs["buildings"])

def generate(
    generation_data,
    output_folder,
    config_file = "nn_generator/examples/generation_config.txt",
    random_modifier = 5
    ):
    """Uses neural networks to generate images.
    generation_data: str
        Name of a folder conaining user input for generation of all images.
    output_folder: str
        Where should the output be stored.
        Creates path if it does not exist.
    config_file: str
        Name of a text file containing locations of neural networks used for the generation.
    random_modifier: float
        SMALLER the number, BIGGER the randomness applied to the process. '0' turns the randomness off.
    """
    print("Configuration file:\n")
    print("======================\n")

    with open(config_file) as f:
        for line in f:
            print(line)

    return generate_map(generation_data, output_folder, config_file, random_modifier)