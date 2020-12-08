# GAEA: Graph Augmentation for Equitable Access via Reinforcement Learning

This repository contains the data and code for [Salesforce Research](https://einstein.ai) paper: [GAEA: Graph Augmentation for Equitable Access via Reinforcement Learning
](https://arxiv.org/abs/2012.03900)

## Citation
If you use this code, data or our results in your research, please cite as appropriate:

```
@misc{ramachandran2020gaea,
      title={GAEA: Graph Augmentation for Equitable Access via Reinforcement Learning}, 
      author={Govardana Sachithanandam Ramachandran and Ivan Brugere and Lav R. Varshney and Caiming Xiong},
      year={2020},
      eprint={2012.03900},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Prerequisites

### Software
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

### Hardware
We ran on a Quadro GV100 with 32GB RAM. 

### Dataset 
1. Dataset merging public census, school, and transportation datasets for the city of Chicago is provided under data/{demographics | network | schools}
2. For Facebook100 dataset download the data as described in http://sociograph.blogspot.com/2011/03/facebook100-data-and-parser-for-it.html and place the unziped data under data/facebook100


## Experiments

Edit repository path and the output path for the project in paths_inc.py .

The run_experiments.py generates all results for: 

1. Original graph 
2. Baseline method
3. Proposed method

On each of the outputted graphs, we run monte carlo weighted walk simulations and estimate the distribution of expected rewards of walkers. On this distribution, we evaluate our main two criteria:

1. Expected Utility
2. Gini Index of Expected Utility

### Graph editing on Chicago school network
```python run_experiments.py --exp edit --graph chicago```

### Graph editing on Facebook100 schools
```python run_experiments.py --exp edit --graph fb --school Caltech36```

Other school network we tried are: Mich67 and Reed98 

### Graph editing on synthetic network
```python run_experiments.py --exp edit --graph synthetic```

### Facility Placement
```python run_experiments.py --exp facility_placement --graph chicago```

