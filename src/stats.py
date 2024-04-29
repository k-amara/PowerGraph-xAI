import os
import torch
import numpy as np
from gendata import get_dataset
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_graph_size_args,
)
import yaml


parser, args = arg_parse()
args = get_graph_size_args(args)
print('args:', args)
# Get the absolute path to the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Load the config file
config_path = os.path.join(parent_dir, "config", "dataset.yaml")
# read the configuration file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# loop through the config and add any values to the parser as arguments
for key, value in config[args.dataset_name].items():
    setattr(args, key, value)

args_group = create_args_group(parser, args)


fix_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_params = args_group["dataset_params"]
dataset = get_dataset(
        dataset_root=args.data_save_dir,
        **dataset_params,
    )

num_top_edges = []
for data in dataset:
    if (data.edge_mask is not None) & (data.y==0):
        true_explanation = data.edge_mask.cpu().numpy()
        n = len(np.where(true_explanation == 1)[0])
        num_top_edges.append(n)

print('num_top_edges:', num_top_edges)
print('number of instances with gt explanation:', len(num_top_edges))
print('avg_num_top_edges:', np.mean(num_top_edges))
print('max_num_top_edges:', np.max(num_top_edges))