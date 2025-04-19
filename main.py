import os
import argparse
import torch
import numpy as np
import pandas as pd
from util import ConfigParser
from train import train
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()

def flatten_namespace(nested_namespace):
    flat_namespace = SimpleNamespace()
    def add_attributes(ns):
        for key, value in vars(ns).items():
            if isinstance(value, SimpleNamespace):
                add_attributes(value)
            else:
                setattr(flat_namespace, key, value)
    
    add_attributes(nested_namespace)
    return flat_namespace

def main():
    args = parse_args()
    args = ConfigParser.parse_yaml(args.config)
    args = flatten_namespace(args)
    
    files = ["pancreas_human"]
    results = pd.DataFrame()
    save_path = args.save_path

    for dataset in files:
        print(f"Training on {dataset}")
        args.dataset = dataset
        args.save_path = make_dir(save_path, dataset)

        res = train(args)
        # results =results.append(res)
        results = pd.concat([results, pd.DataFrame([res])], ignore_index=True)
        results.to_csv(args.results_path + f"/{dataset}_results.csv", header=True)

if __name__ == "__main__":
    main()



