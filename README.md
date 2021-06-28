# Convolutional Hypercomplex Embeddings for Link Prediction

## Installation

First clone the repository:
```
https://github.com/dice-group/dice-group-Convolutional-Hypercomplex-Embeddings-for-Link-Prediction.git
```
Then obtain the required libraries:
```
conda env create -f environment.yml
conda activate hypercomplex
```
The code is compatible with Python 3.6.4

## Reproducing reported results
- ```unzip KGs.zip```
- Download pretrained models (2.1 GB) via [Google Drive](https://drive.google.com/file/d/1ueR6nQwiZ7ZiV6toR7zoZnSl1fHE-kLZ/view?usp=sharing)
- ```unzip PretrainedModels.zip```  
- Reproduce reported link prediction results ``` python reproduce_link_prediction_results.py```
- Reproduce reported link prediction results based on only tail entity rankings``` python reproduce_link_prediction_results_based_on_tail_entity_rankings.py```

## Acknowledgement 
We based our implementation on the open source implementation of [TuckER](https://github.com/ibalazevic/TuckER). We would like to thank for the readable codebase.