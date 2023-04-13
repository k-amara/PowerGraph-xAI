# Benchmarking GNN datasets for PowerGrids

We generated the Inmemory datasets in UK, IEEE24, IEEE39 formats. We use **InMemoryDataset** class of Pytorch Geometric for the dataloader part.  

To test the datasets with different GNN architectures: GCN, GINe, GAT and Transformer, run,

    python code/train_gnn.py

We have the main arguments to control namely 
**--model_name**: transformer / gin / gat / gcn
**--datatype**: binary / multiclass / regression
**--dataset_name**: uk / ieee24 / ieee39


Make sure you have the dataset as per format. Models will be saved as per format (make sure you have the model folder)

├── code
├── dataset
│   ├── processed
│   ├── raw 
|   |   ├── *.mat
├──model
|   ├──ieee24
|   ├──ieee39
|   ├──uk


Remove the for loop in train_gnn.py if running for a specific **--hidden_dim** and **num_layers**.

The models will be saved in **model** directory