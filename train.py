import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import random
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from datasets import Loader, apply_noise
from model import AutoEncoder
from evaluate import evaluate
from util import AverageMeter



def inference(net, data_loader_test):
    net.eval()
    feature_vector = []
    labels_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            feature_vector.extend(net.feature(x.cuda()).detach().cpu().numpy())
            labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
    Find the optimal Leiden resolution that produces the desired number of clusters
    
    Args:
        adata: AnnData object containing cell embeddings
        fixed_clus_count: Target number of clusters to find
        increment: Step size for resolution values
        
    Returns:
        resolution: Optimal resolution value
    '''
    # List to store differences between found and target cluster counts
    dis = []
    # Generate resolution values from 0.01 to 2.5, sorted high to low
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)
    i = 0
    res_new = []
    
    # Try each resolution value
    for res in resolutions:
        # Apply Leiden clustering with current resolution
        sc.tl.leiden(adata, random_state=0, resolution=res)
        # Count how many unique clusters were found
        count_unique_leiden = len(pd.DataFrame(
            adata.obs['leiden']).leiden.unique())
        # Store difference between found and target cluster count
        dis.append(abs(count_unique_leiden-fixed_clus_count))
        res_new.append(res)
        # If exact match found, stop searching
        if count_unique_leiden == fixed_clus_count:
            break
            
    # Select resolution with cluster count closest to target
    reso = resolutions[np.argmin(dis)]

    return reso

def train(args):
    pass

