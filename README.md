# Remixing OSM maps using neural nets

## Dependencies

* **.NET Core 3.0 :** Download with Visual Studio: https://visualstudio.microsoft.com/downloads/
* **System.Drawing.Common :** Download in Visual Studio in OSM\_Parser project under NuGet Packages function. 
* **Python 3+ :** Download from https://www.python.org/downloads/
* **Tensorflow 2.2.0 :** Use `python -m pip install tensorflow=2.2.0` in the command line or navigate to https://www.tensorflow.org/install. It needs to be this specific version.
* **NVCuda 10.1+ :** Follow the guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
* **cuDNN 7.6 :** Follow the guide: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html It needs to be this specific version.
* **openCV**
* **numpy**
* **pyplot**
* **Jupyter Notebook**
* **QGIS 2.18.28 (optional) :** Navigate to https://qgis.org/downloads/ and download the correct version. Then through "manage plugins" feature, download OSMDownloader plugin.

**Install python the packages:**
```
python -m pip install numpy
python -m pip install opencv-python
python -m pip install matplotlib
python -m pip install jupyterlab
python -m pip install tensorflow=2.2.0
```

## Getting the data
1. Go to https://portal.opentopography.org/datasets and select the region you want to export by hand.
2. Under "Global Data" select Global Multi-Resolution Topography (GMRT) Data Synthesis option.
3. You can adjust the longitude and latitude values precisely at this point.
4. Select ESRI ArcASCII format and hight resolution. Download the data and extract them.
5. Navigate to https://www.openstreetmap.org/ and use the export feature to download the same coordinates as before. If your area is too large, open QGIS and use OSMDownloader instead.

The selected area needs to be beg enough for sufficient amount of training examples. Area of Czech republic is recommended:
* lower left corner: 13.500000000, 49.28252211106
* upper right corner: 16.165283203, 50.3701685954

## Preparing the data
1. Open `OSM_Parser/OSM_Parser.sln` in visual studio.
2. Install the System.Drawing.Common library via NuGet packages function.
3. Build the solution in release mode.
4. Copy your osm and asc files into the release directory.
5. Parse the files using the program:
```
./OSM_Parser.exe file.asc file.osm
```
This will result in 4 images: `heights.pgm`, `roads.pgm`, `rivers.pgm`, and `buildings.pgm`. If osm file is not specified, only heights file will be created.

## Training and running the generator
Place the files in a folder of your choosing and navigate to `nn_generator` folder in the command line and execute jupyter notebook:
```
cd nn_generator
jupyter notebook
```
Browser window shold now open.  

1. *Click* `Example.ipynb`  
2. *Click* "Run cell" on the cell you would like to run  

The details are specified in the Example file.

## Generator Description
To access the library, a simple import is needed:
```
import nn_generator
```
Two functions are present to allow for running of the entire algorithm:
```
nn_generator.learn("training_data_folder")
nn_generator.generate("input_folder", "output_folder")
```
It also provides lower level functions for more control and experimentation with the generation. Every function has documentation and is displayed by the help function:
```
help(nn_generator.function)
```
For further examples see `Example.ipynb`.

The internal structure of the library is devised as follows:

* `nn_generator.py` script puts everything together and contains definition of all functions intended for the user to use.
* `__init__.py` runs when the library is imported. It ensures that only intended functions are seen by the user.
* `libraries/parse.py` parses an input image into layers. It is used to parse training images into Layers 64x, 16x, 4x and 1x. It also computes blurry layer and layers of altitude differences.
* `libraries/normalize_heights.py` uses standard score normalization on the height layers.
* `libraries/process_training_data.py` takes the given layers and processes them into training data for a single network.
* `libraries/train.py` is parametrized with a name of a file that specifies network structure. It builds the network based on this file, trains it based on given parameters and saves it into given location.
* `libraries/generate.py` is provided with configuration file "generation\_config.txt", which specifies which networks are used for the generation. This script is given a name of a folder containing user input and generates the maps into the specified output folder.
* `libraries/basic` folder contains libraries with helpful utility functions.
* `examples` folder contains default configuration files and examples of generation and training input. These inputs were used for testing the algorithm in the thesis.

It is important to note that these computations are very demanding and time consuming. Especially data used for training the network for generation of buildings. This network takes data from all the previous layers as an input.