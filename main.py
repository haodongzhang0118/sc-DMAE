import os
import argparse
import torch
import numpy as np
import pandas as pd
from train import train

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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='sc-DMAE')
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='/data/sc_data/all_data/',
                        help='Path to the data directory')
    parser.add_argument('--results_path', type=str, default='./res/',
                        help='Path to save results')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/',
                        help='Path to save checkpoint')
    parser.add_argument('--dataset', type=str, default='10X_PBMC',
                        help='Dataset name to use')
    parser.add_argument('--data_dim', type=int, default=1000,
                        help='Dimension of input data')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of classes/clusters')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Dimension of hidden space')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--masked_weights', type=float, default=0.75,
                        help='Masked weights')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--CNN', type=bool, default=False,
                        help='Using CNN as the backbone')
    parser.add_argument('--UWL', type=bool, default=False,
                        help='Using UWL choose weight of each loss')
    
    return parser.parse_args()

def main():
    args = parse_args()
    files = []
    results = pd.DataFrame()
    save_path = args.checkpoint_path

    for dataset in files:
        print(f"Training on {dataset}")
        args.dataset = dataset
        args.checkpoint_path = make_dir(save_path, dataset)

        res = train(args)
        results = results.append(res)
        results.to_csv(args.results_path + f"/{dataset}_results.csv", header=True)



