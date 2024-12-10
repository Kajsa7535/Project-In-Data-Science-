# Railway Propagation Prediction of the Swedish Railway Network

## Authors

Achini Eranga Nanayakkar, Anna Hellman & Kajsa Odén Uhrenius

Project in Data Science 1DL507

## Overview

A project to predict delay propogation in the Swedish railway network using a diffusion like model with some alteration, including individual delay prediction and directed delay prediction.

## File Structure

```
├── data # Contains network data csv files
├── data_extraction # Data extraction files for creating small
│   ├── create_test_network_U_Fln_Cst_Ga.py # Creates the test network create_test_network_U_Fln_Cst_Ga.csv
├── images # Generated images from the prediction
├── network # Network files and classes
├── prediction.py # Main file for running the prediction
└── README.md
```

## Dependencies

- numpy
- pandas
- matplotlib
- PyGraphviz
- networkx

## Usage

Run the prediction.py file

It has 1 mandatory argument and 5 optional.

### Mandatory argument:

- The name of the file of the test network, must be in a folder named data
  - Ex: The file in in data/test_network.csv. Then I only need to run "prediction.py test_network"

### Optional argumnets:

- Network start time: The start time of the prediction you want to make.
  - Ex: "prediction.py test_network --network_start_time "2019-03-27 16:39" ". The format only needs to be in format "YYYY-MM-DD HH:MM"
  - Default value is the first time a delay can be found in the network
- Numer of time steps to predict: This is how many time steps you want the prediction to do
  - Ex: "prediction.py test_network --time_steps 10"
  - Default value is 10
- Time step size: This is the size of each time step (in minutes)
  - Ex: "prediction.py test_network --time_step_size 3"
  - Default valuee is 1
- Visualization: This is a flag that decides if you want to generate maps of the visualization or not. These images will be created at each timestep and for both the actual delay and the predicted.
  - Ex: "prediction.py test_network --visualize True"
  - Default value is False
- Directed delay: This is a flag that decised what model you want to use. If true, the model with directed delay will be used (the delay will mostly only spread forward and not in all directions). If false, the model will use the model that spreads in all directions.
  - Ex: "prediction.py test_network --directed_delay False"
  - Default value is True

