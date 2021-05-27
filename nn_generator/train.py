from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Sequential
from getopt import getopt
import numpy as np
import sys

def train(model_file, training_folder, output_name, epochs=10, save_interval=1):
    def parse_line(line):
        tags = line.split()
        dictionary = {}

        for tag in tags:
            pair = tag.split("=")
            dictionary[pair[0]] = pair[1]
            
        return dictionary

    def parse_model(model_file):
        with open(model_file) as f:
            layers = []
            line = f.readline().strip()

            while (line != "COMPILE"):
                layers.append(parse_line(line))
                line = f.readline().strip()

            compile_params = parse_line(f.readline().strip())
    
        model = Sequential()

        for params in layers:
            layer = get_layer(params)
            model.add(layer)

        optimizer = compile_params["optimizer"]
        loss = compile_params["loss"]
        model.compile(optimizer=optimizer, loss=loss)

        return model

        

    def get_layer(layer_params):
        layer_type = layer_params["type"]
        if (layer_type == "Input"):
            size = int(layer_params["size"])
            return Input(size)
    
        if (layer_type == "Dense"):
            size = int(layer_params["size"])
            activation = layer_params.get("activation")
            return Dense(size, activation)

        if (layer_type == "Dropout"):
            value = float(layer_params["value"])
            return Dropout(value)



    # Build the model
    #==================
    model = parse_model(model_file)
    model.summary()


    # Load training data
    #========================
    inputs = np.load(f"{training_folder}/inputs.npy")
    outputs = np.load(f"{training_folder}/outputs.npy")


    # Train the model
    #====================
    for epoch in range(epochs):
        model.fit(inputs, outputs)

        if (epoch % save_interval == 0):
            model.save(f"{output_name}_{epoch}.h5")