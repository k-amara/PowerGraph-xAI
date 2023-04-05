# Benchmarking GNN datasets for PowerGrids

We generated the datasets in UK, IEEE24, IEEE39, SwissGrid and IEEE118 formats. We use **InMemoryDataset** class of Pytorch Geometric for the dataloader part.  

To run the file,

    python code/train_gnn.py

We have the main arguments to control namely 
**--model_name**: transformer / gin / gat / gcn
**--datatype**: binary / multiclass / regression
**--dataset_name**: uk / ieee24 / ieee39

Other parameters like batch_size can be controlled in the train_gnn file. 
