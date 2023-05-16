# Benchmarking GNN datasets for PowerGrids

We generated the InMemoryDatasets in UK, IEEE24, IEEE39 formats. We use **InMemoryDataset** class of Pytorch Geometric for the dataloader part.
_Description of the contribution of this GitHub repo..._

## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

Load every additional packages:

```
pip install -r requirements.txt
```

## Dataset desciption

| Dataset    |     Name     | Description                    |
| ---------- | :----------: | ------------------------------ |
| BA-2motifs | `ba_2motifs` | Random BA graph with 2 motifs. |
| IEEE-24    |   `ieee24`   | IEEE-24 (Powergrid dataset)    |
| IEEE-39    |   `ieee39`   | IEEE-39 (Powergrid dataset)    |
| IEEE-118   |  `ieee118`   | IEEE-118 (Powergrid dataset)   |
| UK         |     `uk`     | UK (Powergrid dataset)         |

_Describe more the datasets and how they were obtained_

## GNN Benchmarking

To test the datasets with different GNN architectures: GCN, GINe, GAT and Transformer, run,

```
python code/train_gnn.py
```

We have the main arguments to control namely
**--model_name**: transformer / gin / gat
**--datatype**: binary / multiclass / regression
**--dataset_name**: uk / ieee24 / ieee39

_Explain why GCN is not used --> no edge attributes_

Make sure you have the dataset as per format. Models will be saved as per format (make sure you have the model folder)

├── code
├── dataset
│ ├── processed
│ ├── raw
| | ├── \*.mat
├──model
| ├──ieee24
| ├──ieee39
| ├──uk

Remove the for loop in train_gnn.py if running for a specific **--hidden_dim** and **num_layers**.

The models will be saved in **model** directory

## GNN Explainability Benchmarking

_Add citation/link to papers_

**Graph Classification Tasks**

| Explainer            | Paper                                                                               |
| :------------------- | :---------------------------------------------------------------------------------- |
| Occlusion            | Visualizing and understanding convolutional networks                                |
| SA                   | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM             | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer         | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX            | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer        | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

**Run explainability methods**

```bash
python3 code/main.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

We have the main arguments to control namely

**--model_name**: transformer / gin / gat

**--datatype**: multiclass

**--dataset_name**: uk_mc / ieee24_mc / ieee39_mc

**--explainer_name**: random / sa / ig / gradcam / occlusion / basic_gnnexplainer / gnnexplainer / subgraphx / pgmexplainer / pgexplainer / graphcfe

Default args:

**--explained_target**: 0 - Only Category A powergrids are explained (DNS>0 and Cascading failure)
