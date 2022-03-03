# [Convolutional Hypercomplex Embeddings for Link Prediction](https://proceedings.mlr.press/v157/demir21a.html)
This open-source project contains the Pytorch implementation of our approaches (QMult, OMult, ConvQ, and ConvO), training and evaluation scripts. 
We added our models into [Knowledge Graph Embeddings at Scale](https://github.com/dice-group/DAIKIRI-Embedding) open-source project to ease the deployment and the distributed computing.
Therein, we provided pre-trained models on many knowledge graphs.
## Installation

First clone the repository:
```
https://github.com/dice-group/Convolutional-Hypercomplex-Embeddings-for-Link-Prediction.git
```
Then obtain the required libraries:
```
conda env create -f environment.yml
conda activate hypercomplex
```
The code is compatible with Python 3.6.4

## Reproduce link prediction results
- Download datasets via [HOBBIT](https://hobbitdata.informatik.uni-leipzig.de/KGE/ConvHyper/KGs.zip)
- ```unzip KGs.zip```
- Download pretrained models (2.1 GB) via [HOBBIT](https://hobbitdata.informatik.uni-leipzig.de/KGE/ConvHyper/PretrainedModels.zip)
- ```unzip PretrainedModels.zip```  
- Reproduce reported link prediction results ``` python reproduce_link_prediction_results.py```

## Link Prediction Results
In the below, we provide a brief overview of the link prediction results.
#### WN18RR
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|--------:|
| QMult   |.438   |.393    |.449    |.537   | 
| OMult   |.449   |.406    |.467    |.539| 
| ConvQ   |.457   |.424    |.470    |.525| 
| ConvO   |.458   |.427    |.473    |.521| 

#### FB15K-237
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|--------:|
| QMult   |.346   |.252    |.383    |.535   | 
| OMult   |.347   |.253    |.383    |.534| 
| ConvQ   |.343   |.251    |.376    |.528| 
| ConvO   |.366   |.271    |.403    |.543| 

#### YAGO3-10
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|--------:|
| QMult   |.555   |.475    |.602    |.698   | 
| OMult   |.543   |.461    |.592    |.692| 
| ConvQ   |.539   |.459    |.587    |.687| 
| ConvO   |.489   |.395    |.546    |.664| 

## Link Prediction Results via applying Semantic Constraint on Pretrained Models
This is ongoing work. Currently, we are investigating the idea of constraining predictions based on semantic information provided in the input KG.
#### WN18RR
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|--------: |
| QMult   |.473   |.427    |.491    |.565      | 
| OMult   |.484   |.444    |.504    |.563      | 
| ConvQ   |.473   |.442    |.487    |.535      | 
| ConvO   |.471   |.440    |.484    |.529      | 

#### FB15K-237
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|---------:|
| QMult   |.382   |.285    |.421    |.576      | 
| OMult   |.381   |.284    |.418    |.575      | 
| ConvQ   |.375   |.280    |.409    |.568      | 
| ConvO   |.398   |.301    |.437    |.592      | 

#### YAGO3-10
|         |   MRR | Hits@1 | Hits@3 | Hits@10  |
|---------|------:|-------:|-------:|--------: |
| QMult   |.576   |.490    |.631    |.728      | 
| OMult   |.563   |.480    |.612    |.716      | 
| ConvQ   |.545   |.463    |.594    |.694      | 
| ConvO   |.497   |.403    |.553    |.672      | 


## How to cite
```
@InProceedings{pmlr-v157-demir21a,
  title = 	 {Convolutional Hypercomplex Embeddings for Link Prediction},
  author =       {Demir, Caglar and Moussallem, Diego and Heindorf, Stefan and Ngonga Ngomo, Axel-Cyrille},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {656--671},
  year = 	 {2021},
  editor = 	 {Balasubramanian, Vineeth N. and Tsang, Ivor},
  volume = 	 {157},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--19 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v157/demir21a/demir21a.pdf},
  url = 	 {https://proceedings.mlr.press/v157/demir21a.html},
  abstract = 	 {Knowledge graph embedding research has mainly focused on the two smallest normed division algebras, $\mathbb{R}$ and $\mathbb{C}$. Recent results suggest that trilinear products of quaternion-valued embeddings can be a more effective means to tackle link prediction. In addition, models based on convolutions on real-valued embeddings often yield state-of-the-art results for link prediction. In this paper, we investigate a composition of convolution operations with hypercomplex multiplications. We propose the four approaches QMult, OMult, ConvQ and ConvO  to tackle the link prediction problem. QMult and OMult can be considered as quaternion and octonion extensions of previous state-of-the-art approaches, including DistMult and ComplEx. ConvQ and ConvO build upon QMult and OMult by including convolution operations in a way inspired by the residual learning framework. We evaluated our approaches on seven link prediction datasets including WN18RR, FB15K-237 and YAGO3-10. Experimental results suggest that the benefits of learning hypercomplex-valued vector representations become more apparent as the size and complexity of the knowledge graph grows. ConvO outperforms state-of-the-art approaches on FB15K-237 in MRR, Hit@1 and Hit@3, while QMult, OMult, ConvQ and ConvO outperform state-of-the-approaches on YAGO3-10 in all metrics. Results also suggest that link prediction performances can be further improved via prediction averaging. To foster reproducible research, we provide an open-source implementation of approaches, including training and evaluation scripts as well as pretrained models.}
}
```

For any further questions or suggestions, please contact:  ```caglar.demir@upb.de``` or  ```caglardemir8@gmail.com```