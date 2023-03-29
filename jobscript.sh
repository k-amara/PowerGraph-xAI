#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=all_datasets
#SBATCH --output=logs/uk_multiclass_gat
#SBATCH --error=error/uk_multiclass_gat
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:8G

# /cluster/scratch/alakshmanan/miniconda3/envs/gnn/bin/python /cluster/home/alakshmanan/ra_work/all_inmem_class.py
# /cluster/scratch/alakshmanan/miniconda3/envs/gnn/bin/python /cluster/home/alakshmanan/ra_work/InMemoryPowerGridsDatasets.py
/cluster/scratch/alakshmanan/mininconda3/envs/gnn/bin/python /cluster/home/alakshmanan/ra_work/code/train_gnn.py --dataset_name uk --datatype multiclass --model_name gat
