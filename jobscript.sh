#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=all_datasets
#SBATCH --output=logs/ieee24_multiclass_gat
#SBATCH --error=error/ieee24_multiclass_gat
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:8G

/cluster/scratch/alakshmanan/mininconda3/envs/gnn/bin/python /cluster/home/alakshmanan/ra_work/code/train_gnn.py --dataset_name ieee24 --datatype multiclass --model_name gat
