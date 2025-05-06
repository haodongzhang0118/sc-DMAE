import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from util import ConfigParser
from train import train
from types import SimpleNamespace

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    
    h5_files = glob.glob(os.path.join(args.data_path, "*.h5"))
    files = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
    print(f"Found {len(files)} datasets under {args.data_path}")
    print(f"They are: {files}")
    
    results = pd.DataFrame()
    save_path = args.save_path

    if args.UWL:
      print("Using UWL")
    else:
      print("Not using UWL")

    for dataset in files:
        if dataset != "hrvatin":
            continue
        args.dataset = dataset
        args.save_path = make_dir(save_path, dataset)

        results_dataset_dir = os.path.join(args.results_path, dataset)
        os.makedirs(results_dataset_dir, exist_ok=True)
        
        all_iterations_results = []
        avg_metrics_by_epoch = {}
        seeds = [83, 21, 64, 97, 11, 49, 70, 39, 0, 58, 16, 75, 27, 32, 3, 95, 45, 90, 8, 66]
        iterations = args.iterations
        for iteration in range(iterations):
            seed = seeds[iteration]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            print(f"Iteration {iteration+1}/{iterations} for dataset {dataset}")
            
            res_list = train(args, iteration)
            
            for res in res_list:
                res['iteration'] = iteration + 1
            
            all_iterations_results.extend(res_list)
            
            for res in res_list:
                epoch = res['epoch']
                if epoch not in avg_metrics_by_epoch:
                    avg_metrics_by_epoch[epoch] = {
                        'nmi_sum': 0, 'ari_sum': 0, 'acc_sum': 0, 'sil_sum': 0, 'count': 0
                    }
                
                avg_metrics_by_epoch[epoch]['nmi_sum'] += res['nmi']
                avg_metrics_by_epoch[epoch]['ari_sum'] += res['ari']
                avg_metrics_by_epoch[epoch]['acc_sum'] += res['acc']
                avg_metrics_by_epoch[epoch]['sil_sum'] += res['sil']
                avg_metrics_by_epoch[epoch]['count'] += 1
            print(f"Iteration {iteration+1}/{iterations} Completed!")
            print()
        
        all_results_df = pd.DataFrame(all_iterations_results)
        all_results_df.to_csv(args.results_path + f"/{dataset}/all_iterations_results.csv", index=False)
        
        avg_metrics = []
        for epoch, metrics in avg_metrics_by_epoch.items():
            count = metrics['count']
            avg_metrics.append({
                'epoch': epoch,
                'avg_nmi': metrics['nmi_sum'] / count,
                'avg_ari': metrics['ari_sum'] / count,
                'avg_acc': metrics['acc_sum'] / count,
                'avg_sil': metrics['sil_sum'] / count,
                'iterations': count,
            })
        
        avg_metrics_df = pd.DataFrame(avg_metrics)
        avg_metrics_df.to_csv(args.results_path + f"/{dataset}/average_metrics.csv", index=False)
        
        print(f"Completed all {iterations} iterations for dataset {dataset}")
        print(f"Results saved to: {args.results_path}/{dataset}/all_iterations_results.csv")
        print(f"Average metrics saved to: {args.results_path}/{dataset}/average_metrics.csv")
        print()

if __name__ == "__main__":
    main()

