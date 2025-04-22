import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from datasets import Loader, apply_noise
from model import Autoencoder
from evaluate import evaluate
from util import AverageMeter


def inference(net, data_loader_test):
    net.eval()
    feature_vector = []
    labels_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            feature_vector.extend(net.inference(x.cuda()).detach().cpu().numpy())
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

def train(args, iteration):
    data_load = Loader(args, dataset_name=args.dataset, drop_last=True)
    data_loader = data_load.train_loader
    data_loader_test = data_load.test_loader
    X_shape = args.data_dim
    results = []
    init_lr = args.learning_rate
    max_epochs = args.epochs
    mask_prob = [0.4] * X_shape
    
    model = Autoencoder(num_genes=X_shape, 
                        hidden_size=args.hidden_size, 
                        dropout=args.dropout, 
                        masked_weights=args.masked_weights, 
                        CNN=args.CNN,
                        UWL=args.UWL).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    iteration_dir = os.path.join(args.save_path, f"{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)
    log_file = os.path.join(iteration_dir, "training_log.csv")
    
    # Initialize the log with headers
    with open(log_file, 'w') as f:
        f.write("epoch,total_loss,reconstruction_loss,weight_r,mask_loss,weight_m,latent_loss,weight_l\n")
    
    print(f"Starting training for {max_epochs} epochs")
    epoch_pbar = tqdm(range(max_epochs), desc="Training Progress", 
                     bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in epoch_pbar:
        model.train()
        meter = AverageMeter()
        
        for i, (x, y) in enumerate(data_loader):
            x = x.cuda()
            x_corrputed, mask, corrupted_X_comp = apply_noise(x, mask_prob)
            optimizer.zero_grad()
            loss = model.compute_loss(
                x=x_corrputed,
                y=x,
                comp_x=corrupted_X_comp,
                mask=mask
            )
            loss["total_loss"].backward()
            optimizer.step()
            model._update_teacher()
            meter.update(loss)
            
            if (i + 1) % 5 == 0 or (i + 1) == len(data_loader):
                epoch_pbar.set_postfix({
                    "loss": f"{meter.avg:.4f}",
                    "r_loss": f"{meter.reconstruction_loss_avg:.4f}",
                    "m_loss": f"{meter.mask_loss_avg:.4f}",
                    "l_loss": f"{meter.latent_loss_avg:.4f}",
                    "batch": f"{i+1}/{len(data_loader)}"
                })
        
        epoch_pbar.set_description(f"Epoch {epoch+1}/{max_epochs} - Loss: {meter.avg:.4f}")
        
        # Log metrics to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{meter.avg:.6f},{meter.reconstruction_loss_avg:.6f},"
                    f"{meter.weight_r_avg:.6f},{meter.mask_loss_avg:.6f},"
                    f"{meter.weight_m_avg:.6f},{meter.latent_loss_avg:.6f},"
                    f"{meter.weight_l_avg:.6f}\n")
        
        if (epoch+1) % 20 == 0 or epoch == max_epochs - 1:
            print(f"\nSaving checkpoint and evaluating at epoch {epoch+1}...")
            torch.save({
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict()
                }, args.save_path + f"/{iteration}/checkpoint_{epoch+1}.pth"
            )
            latent, true_label = inference(model, data_loader_test)
            if latent.shape[0] < 10000:
                clustering_model = KMeans(n_clusters=args.n_classes)
                clustering_model.fit(latent)
                pred_label = clustering_model.labels_
            else:
                adata = sc.AnnData(latent)
                sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
                # sc.tl.umap(adata)
                reso = res_search_fixed_clus(adata, args.n_classes)
                sc.tl.leiden(adata, resolution=reso)
                pred = adata.obs['leiden'].to_list()
                pred_label = [int(x) for x in pred]
            
            nmi, ari, acc = evaluate(true_label, pred_label)
            ss = silhouette_score(latent, pred_label)
            res = {}
            res["epoch"] = epoch
            res["nmi"] = nmi
            res["ari"] = ari
            res["acc"] = acc
            res["sil"] = ss
            results.append(res)
            print(f"\tEvaluation: [nmi: {nmi:.4f}] [ari: {ari:.4f}] [acc: {acc:.4f}] [sil: {ss:.4f}]")
            np.save(args.save_path +f"/{iteration}/embedding_"+str(epoch)+".npy", 
                    latent)
            pd.DataFrame({"True": true_label, 
                        "Pred": pred_label}).to_csv(args.save_path +f"/{iteration}/types_"+str(epoch)+".txt")
            
    return results

