# GAEA: Graph Augmentation for Equitable Access via Reinforcement Learning

This readme outlines the research project for the Graph Augmentation for Equitable Access(GAEA) problem and submission to the KDD conference.

### Prerequisites

The following non-standard dependencies exist for this repository: 

```
tensorflow (1.13-1.15)
keras
geopandas
networkx
igraph
scipy
networkx
numpy
ujson
bs4
pandas
shapley
geopandas
fiona
haversine
geographiclib.geodesic
```

Hardware: We ran on a Quadro GV100 with 32GB RAM. 

Dataset: 
1. Dataset merging public census, school, and transportation datasets for the city of Chicago is provided under data/{demographics | network | schools}
2. For Facebook100 dataset download the data as described in in http://sociograph.blogspot.com/2011/03/facebook100-data-and-parser-for-it.html and place the unziped data under data/facebook100

## Running: Synthetic

Edit repository path and the output path for the project in paths_inc.py .

The run_experiments.py generates all synthetic results for: 

1. Original graph 
2. Baseline method
3. Proposed method

On each of the outputted graphs, we run monte carlo weighted walk simulations and estimate the distribution of expected rewards of walkers. On this distribution, we evaluate our main two criteria:

1. Expected Utility
2. Gini Index of Expected Utility

## Running: Graph editing on Chicago school network
python run_experiments.py --exp edit --graph chicago

## Running: Graph editing on Facebook100 schools
python run_experiments.py --exp edit --graph chicago

## Running: Graph editing on synthetic network
python run_experiments.py --exp edit --graph fb --school Caltech36
Other school network we tried are: Mich67 and Reed98 

## Running: Facility Placement
python run_experiments.py --exp facility_placement --graph chicago
