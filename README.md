# PowerGraph Explainability Analysis

This repository contains the code to implement the explainability methods presented in the paper 
They explain power outage in power grids and detect cascading failure. 


With the InMemoryDatasets Class, we generate the GNN datasets for the UK, IEEE24, IEEE39, IEEE118 power grids. We use **InMemoryDataset** class of Pytorch Geometric.

## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

Load every additional packages:

```
pip install -r requirements.txt
```

## Prerequisites and data structure

To reproduce the results presented in the paper, download the following compressed data from [here](https://figshare.com/articles/dataset/PowerGraph/22820534) (~1.8GB, when uncompressed):

```bash
wget -O data.tar.gz "https://figshare.com/ndownloader/files/40571123"
tar -xf data.tar.gz
```

Each dataset folder contains the following files:

- `blist.mat`: branch list also called edge order or edge index
- `of_bi.mat`: binary classification labels
- `of_reg.mat`: regression labels
- `of_mc.mat`: multi-class labels
- `Bf.mat`: node feature matrix
- `Ef.mat`: edge feature matrix
- `exp.mat`: groundtruth explanation

## Dataset description

| Dataset    |     Name     | Description                    |
| ---------- | :----------: | ------------------------------ |
| IEEE-24    |   `ieee24`   | IEEE-24 (Powergrid dataset)    |
| IEEE-39    |   `ieee39`   | IEEE-39 (Powergrid dataset)    |
| IEEE-118   |  `ieee118`   | IEEE-118 (Powergrid dataset)   |
| UK         |     `uk`     | UK (Powergrid dataset)         |

We have created a graph dataset that models cascading failure events, which are the main cause of blackouts in power grids. To generate a comprehensive dataset for different power grids, we used a physics-based cascading failure model called Cascades. 
This model simulates how failures propagate in the IEEE24, IEEE39, IEEE118 and UK power grids. The output of the model is the final demand not being served (DNS). Our dataset consists of a large set of power grid states, representing the operating conditions before an outage, and is linked to an initial triggering outage (one or more failed elements). Each power grid state is represented as a graph, with a graph-level label assigned based on the results of the physics-based model. The dataset is designed for various graph-level tasks, such as multi-class classification, binary classification, and regression. Bus and branches are the elements of a power grid, buses include loads and generators which represent the nodes of the graph, while branches include transmission lines and transformers which represent the edges of the graph. 
We provide three features per node: net active power, net apparent power and voltage magnitude. While the features per edge are four: active power flow, reactive power flow, line reactance and line rating.


## GNN Explainability Benchmarking for Graph Classification Tasks

| Non-generative Explainer | Paper                                                                               |
| :----------------------- | :---------------------------------------------------------------------------------- |
| Occlusion                | Visualizing and understanding convolutional networks                                |
| SA                       | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM                 | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients     | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer             | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX                | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer            | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

| Generative Explainer | Paper                                                                             |
| :------------------- | :-------------------------------------------------------------------------------- |
| RCExplainer          | Reinforced Causal Explainer for Graph Neural Networks                             |
| GSAT                 | Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism |
| DiffExplainer        | D4Explainer (not published)                                                       |

**Run explainability methods**

```bash
python3 code/main.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

We have the main arguments to control namely

**--model_name**: transformer / gin / gat

**--datatype**: multiclass

**--dataset_name**: uk_mc / ieee24_mc / ieee39_mc

**--explainer_name**: random / sa / ig / gradcam / occlusion / basic_gnnexplainer / gnnexplainer / subgraphx / pgmexplainer / rcexplainer / gsat / diffexplainer 

Default args:

**--explained_target**: 0 - Only Category A powergrids are explained (DNS>0 and Cascading failure)

## License

This work is licensed under a CC BY 4.0 license.
