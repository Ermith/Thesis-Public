# How to use the software

## Following dependencies are needed for the algorithm used in this thesis. These are the steps to get them on a windows based system:

* **.NET Core 3.0 :** Download with Visual Studio: https://visualstudio.microsoft.com/downloads/
* **System.Drawing.Common :** Download in Visual Studio in OSM\_Parser project under NuGet Packages function. 
* **Python 3+ :** Download from https://www.python.org/downloads/
* **Tensorflow 2.2.0 :** Use `python -m pip install tensorflow=2.2.0` in the command line or navigate to https://www.tensorflow.org/install. It needs to be this specific version.
* **NVCuda 10.1+ :** Follow the guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
* **cuDNN 7.6 :** Follow the guide: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html It needs to be this specific version.
* **openCV :** `python -m pip install opencv-python`
* **numpy :** `python -m pip install numpy`
* **pyplot :** `python -m pip install matplotlib`
* **Jupyter Notebook :** `python -m pip install jupyterlab`
* **QGIS 2.18.28 (optional) :** Navigate to https://qgis.org/downloads/ and download the correct version. Then through "manage plugins" feature, download OSMDownloader plugin. 

## The complete steps for the whole process of downloading and parsing the data, training the networks and generating the maps are described as follows:
* Go to https://portal.opentopography.org/datasets and select the region you want to export by hand.
* Under "Global Data" select Global Multi-Resolution Topography (GMRT) Data Synthesis option.
* You can adjust the longitude and latitude values precisely at this point.
* Select ESRI ArcASCII format and hight resolution. Download the data and extract them.
* Navigate to https://www.openstreetmap.org/ and use the export feature to download the same coordinates as before. If your area is too large, open QGIS and use OSMDownloader instead.
* Open OSM/OSM\_Parser/OSM\_Parser.sln in visual studio.
* Install the System.Drawing.Common library via NuGet packages function.
* Build the solution in release mode.
* Copy your osm and asc files into the release directory.
* Parse the files using the program: `./OSM_Parser.exe file.asc file.osm`. This will result in 4 images: heights.pgm, roads.pgm, rivers.pgm, and buildings.pgm. If osm file is not specified, only heights file will be created.
* Place the files in a folder of your choosing and navigate to "Neural Networks" folder in the command line.
* Execute `jupyter notebook` in the command line and open "Example.ipynb" in the browser window that is now opened. This file contains rest of the instructions on how to parse the data, train the networks and generate new maps.


## To summarize the python scripts, "Neural Networks" folder contains 5 main scripts. Each of them contains a single function that does its functionality.
* **parse.py** parses the input image into layers. It is used to parse training images into Layers 64x, 16x, 4x and 1x. It also computes blurry layer and layers of altitude differences.
* **normalize\_heights.py** uses standard score normalization on the height layers.
* **process\_training\_data.py** takes the given layers and processes them into training data for a single network.
* **train.py** is parametrized with a name of a file that specifies network structure. It builds the network based on this file, trains it based on given parameters and saves it into given location.
* **generate.py** is provided with configuration file "generation\_config.txt", which specifies which networks are used for the generation. This script is given a name of a folder containing user input and generates the maps into the specified output folder.
* **libraries** folder contains libraries with helpful functions.
* **model structures** folder contains files describing structures of each network.
* **generation data** folder contains example files for the generation.
* **Example.ipynb** is, as mentioned above, example python notebook file containing script on how to perform the algorithm from parsing to generation.

It is important to note that these computations are very demanding and time consuming. Especially data used for training the network for generation of buildings. This network takes data from all the previous layers as an input.
The software comes with prepared unparsed PGM images in **raw\_data** folder. It also comes with pre-trained networks in **models** folder. This is for demonstration purposes.