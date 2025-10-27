#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# s3r_core.py

import os
import sys
import time
import math
import gc
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
import optuna

# -------------------- device helpers --------------------
NUM_CUDA = torch.cuda.device_count()
USE_MPS = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

def pick_device_for_trial(trial):
    if NUM_CUDA > 0:
        gpu_id = trial.number % NUM_CUDA
        print(f"Trial {trial.number} -> cuda:{gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    if USE_MPS:
        print(f"Trial {trial.number} -> mps")
        return torch.device("mps")
    print(f"Trial {trial.number} -> cpu")
    return torch.device("cpu")

# default device for tensors created outside trials
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    _DEFAULT_DEVICE = torch.device("cuda")
else:
    _DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- utilities --------------------
def torch_r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_knockoff_threshold(W, q=0.1, offset=1):
    W = np.array(W)
    order = np.argsort(np.abs(W))[::-1]
    W_sorted = W[order]
    p = len(W)
    num_neg = 0
    T_candidate = None
    for j in range(p):
        t_j = abs(W_sorted[j])
        if W_sorted[j] < 0:
            num_neg += 1
        num_selected = j + 1
        ratio = (offset + num_neg) / max(1, num_selected)
        if ratio <= q:
            T_candidate = t_j
    if T_candidate is None:
        T_candidate = 1e9
    return T_candidate

def plot_loss(loss):
    plt.figure(figsize=(8, 4))
    x = list(range(len(loss)))
    plt.plot(x, loss, label='total loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.ylim(0, max(loss))
    plt.legend()
    plt.grid(True)
    plt.show()

def filter_and_trim_matrix(restored_matrix, threshold=0.001, retain_ratio=0.97):
    filtered_means = []
    filtered_matrix = restored_matrix.copy()
    for index, row in restored_matrix.iterrows():
        filtered_values = row[np.abs(row) > threshold]
        mean_value = np.mean(np.abs(filtered_values)) if len(filtered_values) > 0 else 0
        filtered_means.append(mean_value)
    filtered_means = np.array(filtered_means)
    total_mean_sum = np.sum(filtered_means)
    sorted_indices = np.argsort(-filtered_means)
    cumulative_sum = np.cumsum(filtered_means[sorted_indices])
    cutoff_index = np.where(cumulative_sum <= retain_ratio * total_mean_sum)[0][-1] + 1
    retained_indices = sorted_indices[:cutoff_index]
    for i, row in enumerate(restored_matrix.iterrows()):
        _, data_row = row
        if i in retained_indices:
            filtered_matrix.iloc[i, np.abs(data_row) < threshold] = 0
        else:
            filtered_matrix.iloc[i, :] = 0
    return filtered_matrix

# -------------------- losses --------------------
def custom_loss_v4_mask(X, Y, DeltaW, L, lambda_smooth, alpha, beta):
    if Y.dim() == 1:
        Yp = Y.unsqueeze(1)
    elif Y.dim() == 2 and Y.shape[1] == 1:
        Yp = Y
    elif Y.dim() == 2 and Y.shape[0] == 1:
        Yp = Y.t()
    else:
        raise ValueError(f"Y shape not supported: {tuple(Y.shape)}")
    mask = (Yp != 0).float()
    DeltaW_eff = DeltaW * mask
    n = X.shape[0]
    adjusted_pred = torch.sum(DeltaW_eff * X, dim=1, keepdim=True)
    residuals = Y - adjusted_pred
    data_loss = torch.mean(residuals ** 2)
    product = torch.mm(L, DeltaW_eff)
    smoothness_loss = lambda_smooth * torch.norm(product, p=1, dim=0).sum()
    l1_loss = alpha * torch.sum(torch.abs(DeltaW_eff))
    sqrt_n = torch.sqrt(torch.tensor(n).float())
    l2_loss = beta * (sqrt_n * torch.sum(torch.sqrt(torch.sum(DeltaW_eff ** 2, dim=0))))
    total_loss = data_loss + smoothness_loss + l1_loss + l2_loss
    return total_loss

def custom_loss_tv_mask(X, Y, DeltaW, H, lambda_tv, alpha, beta):
    if Y.dim() == 1:
        Yp = Y.unsqueeze(1)
    else:
        Yp = Y
    mask = (Yp != 0).float()
    DeltaW_eff = DeltaW * mask
    n = X.shape[0]
    adjusted_pred = torch.sum(DeltaW_eff * X, dim=1, keepdim=True)
    residuals = Y - adjusted_pred
    data_loss = torch.mean(residuals ** 2)
    product = torch.mm(H, DeltaW_eff)
    tv_loss = lambda_tv * torch.sum(torch.abs(product))
    l1_loss = alpha * torch.sum(torch.abs(DeltaW_eff))
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=DeltaW.device))
    l2_loss = beta * (sqrt_n * torch.sum(torch.sqrt(torch.sum(DeltaW_eff ** 2, dim=0))))
    total_loss = data_loss + tv_loss + l1_loss + l2_loss
    return total_loss

def custom_loss_adj(X, Y, DeltaW, L, lambda_smooth, alpha, beta, var_weights, smooth_weights=None):
    n = X.shape[0]
    adjusted_pred = torch.sum(DeltaW * X, dim=1, keepdim=True)
    residuals = Y - adjusted_pred
    data_loss = torch.mean(residuals ** 2)
    product = torch.mm(L, DeltaW)
    if smooth_weights is not None:
        smoothness_loss = lambda_smooth * torch.sum(smooth_weights * torch.norm(product, p=1, dim=0))
    else:
        smoothness_loss = lambda_smooth * (torch.norm(product, p=1, dim=0).sum())
    l1_loss = alpha * torch.sum(var_weights * torch.abs(DeltaW))
    sqrt_n = torch.sqrt(torch.tensor(n).float())
    l2_loss = beta * (sqrt_n * torch.sum(var_weights * torch.sqrt((torch.sum(DeltaW ** 2, dim=0)))))
    total_loss = data_loss + smoothness_loss + l1_loss + l2_loss
    return total_loss

# -------------------- graph helpers --------------------
def calculate_E_matrix(coords, k_neighbors=7):
    n = coords.shape[0]
    DD = pairwise_distances(coords, metric='euclidean')
    neighbors = np.argsort(DD, axis=1)[:, 1:(k_neighbors + 1)]
    distances = np.sort(DD, axis=1)[:, 1:(k_neighbors + 1)]
    Weight_matrix = np.zeros((n, n))
    for i in range(n):
        for idx, j in enumerate(neighbors[i]):
            dist = distances[i][idx]
            Weight_matrix[i, j] = np.exp(-dist)
        Weight_matrix[i, i] = -np.sum(np.exp(-distances[i]))
    E = Weight_matrix.copy()
    E[E > 0] = 1
    E[E < 0] = -1
    E[-1, :] = 0
    return E

def z_score_normalization(d):
    d = (d - d.min()) / (d.max() - d.min())
    return d

def complete_feature_indices(selected_indices, max_gap=4):
    selected_indices = np.sort(selected_indices)
    completed_indices = set(selected_indices)
    for i in range(len(selected_indices) - 1):
        current_idx = selected_indices[i]
        next_idx = selected_indices[i + 1]
        if next_idx - current_idx <= max_gap:
            for j in range(current_idx + 1, next_idx):
                completed_indices.add(j)
    return np.array(sorted(completed_indices))

def find_neighbors_from_laplacian(test_idx, train_index_split, L_norm, delta_beta_train):
    neighbors = torch.where(L_norm[test_idx] == -1)[0]
    if isinstance(train_index_split, torch.Tensor):
        train_index_split = train_index_split.to(L_norm.device)
    else:
        train_index_split = torch.tensor(train_index_split).to(L_norm.device)
    valid_neighbors = [n.item() for n in neighbors if n in train_index_split]
    if len(valid_neighbors) == 0:
        smaller_neighbors = train_index_split[train_index_split < test_idx]
        larger_neighbors = train_index_split[train_index_split > test_idx]
        delta_beta_sum = 0
        count = 0
        if len(smaller_neighbors) > 0:
            closest_smaller = smaller_neighbors[-1].item()
            smaller_idx_in_delta_beta = torch.where(train_index_split == closest_smaller)[0][0].item()
            delta_beta_sum += delta_beta_train[smaller_idx_in_delta_beta]
            count += 1
        if len(larger_neighbors) > 0:
            closest_larger = larger_neighbors[0].item()
            larger_idx_in_delta_beta = torch.where(train_index_split == closest_larger)[0][0].item()
            delta_beta_sum += delta_beta_train[larger_idx_in_delta_beta]
            count += 1
        delta_beta_avg = delta_beta_sum / count
        return delta_beta_avg.unsqueeze(0)
    train_indices_in_delta_beta = torch.tensor([torch.where(train_index_split == n)[0][0].item() for n in valid_neighbors])
    neighbor_delta_beta = delta_beta_train[train_indices_in_delta_beta]
    delta_beta_avg = torch.mean(neighbor_delta_beta, dim=0, keepdim=True)
    return delta_beta_avg

# -------------------- run_s3r (post-optimization) --------------------
def run_s3r(
    train_result: dict,
    X_path: str,
    Y_path: str,
    coords_path: str,
    save_dir: str,
    loss_mode: str = "L",
    device_name: str = "cuda",
    post_lr: float = 3e-4,
    post_iters: int = 100000,
    early_start_iter: int = 5000,
    early_window: int = 1000,
    early_eps: float = 1e-14,
    fig_size=(15, 12),
    lambda_smooth: float = None,
    alpha: float = None,
    beta: float = None,
):
    import os
    import torch
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(save_dir, exist_ok=True)

    # ====== 1️⃣ Resolve hyperparameters ======
    if lambda_smooth is None:
        lambda_smooth = train_result["picked_params"]["lambda_smooth"]
    if alpha is None:
        alpha = train_result["picked_params"]["alpha"]
    if beta is None:
        beta = train_result["picked_params"]["beta"]

    print(f"λ={lambda_smooth:.5f}, α={alpha:.5f}, β={beta:.5f}")
    
    # ========= 2️⃣ Read data =========
    data_df = pd.read_csv(X_path, index_col=0, header=0).T
    target_df = pd.read_csv(Y_path, index_col=0, header=0)
    coords_df = pd.read_csv(coords_path, index_col=0, header=0)

    coords_array = coords_df[['row', 'col']].to_numpy()
    E_matrix = calculate_E_matrix(coords_array, k_neighbors=7)
    L_norm_np = E_matrix.T
    A = np.where(L_norm_np == 1, 1, 0)
    edges_np = np.column_stack(np.where(np.triu(A, 1) == 1))

    gene_names = data_df.columns.tolist()
    sample_names = data_df.index.tolist()
    chosen_base = target_df.T.index[0]
    n_samples, n_features = data_df.shape

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    data = torch.tensor(data_df.values, dtype=torch.float32, device=device)
    target = torch.tensor(target_df.values, dtype=torch.float32, device=device)
    L_norm = torch.tensor(L_norm_np, dtype=torch.float32, device=device)

    colors = ["blue", "white", "red"]
    cm = LinearSegmentedColormap.from_list("my_cmap", colors, N=n_samples)

    def optimize_once(loss_fn, struct_tensor, apply_mask, tag_suffix, csv_suffix):
        DeltaW2 = torch.randn(n_samples, n_features, device=device, requires_grad=True)
        mask = (target != 0).float().view(-1, 1)
        if apply_mask:
            with torch.no_grad():
                DeltaW2.data *= mask

        optimizer = torch.optim.Adam([DeltaW2], lr=post_lr)
        all_loss = []
        for i in range(post_iters):
            optimizer.zero_grad()
            loss = loss_fn(data, target, DeltaW2, struct_tensor, lambda_smooth, alpha, beta)
            loss.backward()
            optimizer.step()
            if apply_mask:
                with torch.no_grad():
                    DeltaW2.data *= mask
            all_loss.append(loss.item())
            if i > early_start_iter:
                m1 = np.mean(all_loss[-early_window:-1])
                m2 = np.mean(all_loss[-2 * early_window:-early_window - 1])
                if abs(m1 - m2) < early_eps:
                    print(f"early stop at {i}")
                    break

        delta_beta = DeltaW2.detach().cpu().numpy().T
        plt.figure(figsize=fig_size)
        sns.heatmap(delta_beta, cmap=cm, center=0)
        png_path = os.path.join(save_dir, f"{chosen_base}_{tag_suffix}.png")
        plt.savefig(png_path, format="png", bbox_inches="tight")
        plt.close()

        csv_path = os.path.join(save_dir, f"{chosen_base}_{csv_suffix}.csv")
        pd.DataFrame(delta_beta, index=gene_names, columns=sample_names).to_csv(csv_path)
        return {"png": png_path, "csv": csv_path}

    if loss_mode.upper() == "L":
        print(">>> Running L mode ...")
        res_mask = optimize_once(custom_loss_v4_mask, L_norm, True, "##noknockoff_BT", "##noknockoff_BT_LNORM")
        res_nomask = optimize_once(custom_loss_v4_mask, L_norm, False, "##noknockoff_BT_nomask", "##noknockoff_BT_LNORM_nomask")
        return {"mode": "L", "mask": res_mask, "nomask": res_nomask}

    print(">>> Running H mode ...")
    m_edges = edges_np.shape[0]
    H_tv = torch.zeros((m_edges, n_samples), dtype=torch.float32, device=device)
    if m_edges > 0:
        rows = torch.arange(m_edges, device=device)
        H_tv[rows, torch.as_tensor(edges_np[:, 0], device=device)] = 1.0
        H_tv[rows, torch.as_tensor(edges_np[:, 1], device=device)] = -1.0
    res_htv = optimize_once(custom_loss_tv_mask, H_tv, True, "##noknockoff_BT", "##noknockoff_BT_HTV")
    return {"mode": "H", "tv": res_htv}

# -------------------- train_s3r (hyperparameter search) --------------------
def train_s3r(
    directory: str,
    NNN: int = 1,
    loss_mode: str = "L",
    n_splits: int = 5,
    random_seed: int = 42,
    X_path: str | None = None,
    Y_path: str | None = None,
    coords_path: str | None = None,
    n_trials: int = 5,
    n_jobs: int = 4,
    learning_rate: float = 1e-3,
    num_iterations: int = 30000,
    convergence_threshold: float = 1e-4,
    convergence_threshold2: float = 2e-3,
    max_convergence_count: int = 20,
    knn_k: int = 7,
    topk_features: int = 50,
    verbose: bool = True,
):


    x_file = X_path
    y_file = Y_path
    coords_file = coords_path
    if verbose:
        print(x_file); print(y_file); print(coords_file)

    coords = pd.read_csv(coords_file, index_col=0, header=0)
    coords_array = coords[['row', 'col']].to_numpy()
    E_matrix = calculate_E_matrix(coords_array, k_neighbors=knn_k)
    if verbose:
        print("E matrix shape:", E_matrix.shape)
    L_norm_np = E_matrix.T
    if verbose:
        print(f"L_norm shape: {L_norm_np.shape}\n")

    data = pd.read_csv(x_file, index_col=0, header=0).T
    sample_names = data.index.values
    gene_names = data.columns.values
    n_samples, n_features = data.shape
    if verbose:
        print(f"Data shape: {data.shape}\n")

    target = pd.read_csv(y_file, index_col=0, header=0)
    chosen_base = target.T.index.tolist()[0]
    if verbose:
        print(f"Target shape: {target.shape}\n")

    A = np.where(L_norm_np == 1, 1, 0)
    edges_np = np.column_stack(np.where(np.triu(A, 1) == 1))

    data_t = torch.from_numpy(data.values).type(torch.float32).to(_DEFAULT_DEVICE)
    target_t = torch.from_numpy(target.values).type(torch.float32).to(_DEFAULT_DEVICE)
    L_norm_t = torch.from_numpy(L_norm_np).type(torch.float32).to(_DEFAULT_DEVICE)
    spatial_coords = torch.from_numpy(coords.values).type(torch.float32).to(_DEFAULT_DEVICE)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits_data, splits_y, splits_H, splits_coords = [], [], [], []
    train_indices_list, test_indices_list = [], []
    for _, (train_indices, test_indices) in enumerate(kf.split(np.arange(n_samples))):
        train_indices_list.append(train_indices)
        test_indices_list.append(test_indices)
        train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
        test_index_tensor = torch.tensor(test_indices, dtype=torch.long)
        data_train, data_test = data_t[train_index_tensor], data_t[test_index_tensor]
        y_train, y_test = target_t[train_index_tensor], target_t[test_index_tensor]
        H_train = L_norm_t[train_index_tensor][:, train_index_tensor]
        H_test = L_norm_t[test_index_tensor][:, test_index_tensor]
        coords_train, coords_test = spatial_coords[train_indices], spatial_coords[test_indices]
        splits_data.append((data_train, data_test))
        splits_y.append((y_train, y_test))
        splits_H.append((H_train, H_test))
        splits_coords.append((coords_train, coords_test))

    best_deltas_per_fold = None
    best_mse = float('inf')
    feature_count = np.zeros(n_features, dtype=int)

    def objective_L(trial):
        nonlocal best_deltas_per_fold, best_mse, feature_count
        device = pick_device_for_trial(trial)
        L_norm_local = L_norm_t.clone().to(device)
        data_local = data_t.clone().to(device)
        target_local = target_t.clone().to(device)

        if best_deltas_per_fold is None:
            best_deltas_per_fold = [None] * len(splits_data)

        lambda_smooth = trial.suggest_float('lambda_smooth', 1e-3, 1, log=True)
        alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
        beta = trial.suggest_float('beta', 1e-3, 1, log=True)

        MSE_score_test = torch.zeros(1).to(device)
        current_deltas_per_fold = [None] * len(splits_data)

        for m, (data_split, y_split, H_split, train_index_split, test_index_split) in enumerate(
            zip(splits_data, splits_y, splits_H, train_indices_list, test_indices_list)
        ):
            data_train = data_split[0].to(device)
            y_train = y_split[0].to(device)
            H_train = H_split[0].to(device)
            data_test = data_split[1].to(device)
            y_test = y_split[1].to(device)

            train_index_split = torch.tensor(train_index_split, dtype=torch.long, device=device)
            n_train, p = data_train.shape
            y_tmp = y_train.unsqueeze(1) if y_train.dim() == 1 else y_train
            mask_train = (y_tmp != 0).float()

            if best_deltas_per_fold[m] is not None:
                DeltaW = best_deltas_per_fold[m].clone().detach().to(device)
                DeltaW.requires_grad_(True)
                with torch.no_grad():
                    DeltaW.data *= mask_train
            else:
                DeltaW = torch.randn(n_train, p, device=device, requires_grad=True)
                with torch.no_grad():
                    DeltaW.data *= mask_train

            optimizer = torch.optim.Adam([DeltaW], lr=learning_rate)
            all_loss = []
            convergence_count = 0

            for i in range(num_iterations):
                if convergence_count < max_convergence_count:
                    optimizer.zero_grad()
                    loss = custom_loss_v4_mask(data_train, y_train, DeltaW, H_train, lambda_smooth, alpha, beta)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        DeltaW.data *= mask_train
                    all_loss.append(loss.item())

                    if i > 100:
                        if abs(np.mean(all_loss[-40:-1]) - np.mean(all_loss[-80:-41])) < convergence_threshold:
                            convergence_count += 1
                        else:
                            convergence_count = 0
                        if convergence_count >= max_convergence_count:
                            print(i); print("stop early __ good")
                            break
                    if i > 3000 and abs(np.mean(all_loss[-500:-1]) - np.mean(all_loss[-1000:-501])) < convergence_threshold2:
                        print("Oscillating Convergence"); print(i)
                        break

            delta_beta = DeltaW.detach()
            sort = torch.sum(torch.abs(delta_beta), dim=0)
            sort_indices = torch.sort(sort, descending=True).indices
            ss = sort_indices[:topk_features].cpu().numpy()
            feature_count[ss] += 1

            test_index_split = torch.tensor(test_index_split, dtype=torch.long, device=device)
            delta_beta_test = []
            for test_idx in test_index_split:
                delta_beta_avg = find_neighbors_from_laplacian(test_idx, train_index_split, L_norm_local.clone(), delta_beta)
                delta_beta_test.append(delta_beta_avg)
            delta_beta_test = torch.cat(delta_beta_test, dim=0)
            y_hat_test = (data_test * delta_beta_test).sum(dim=1)
            MSE_test = torch.mean((y_test.squeeze() - y_hat_test) ** 2)
            MSE_score_test += MSE_test

            current_deltas_per_fold[m] = delta_beta.clone().cpu()

        avg_MSE_score_test = MSE_score_test.item() / len(splits_data)
        best_deltas_per_fold = current_deltas_per_fold
        if avg_MSE_score_test < best_mse:
            best_mse = avg_MSE_score_test
            print(f"update best MSE: {best_mse}")
        return avg_MSE_score_test

    def objective_H(trial):
        nonlocal best_deltas_per_fold, best_mse, feature_count
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            dev = torch.device("cpu")
        else:
            gpu_id = trial.number % gpu_count
            dev = torch.device(f"cuda:{gpu_id}")
        print(f"Trial {trial.number} is using GPU {dev}")

        L_norm_local = L_norm_t.clone().to(dev)
        data_local = data_t.clone().to(dev)
        target_local = target_t.clone().to(dev)

        if best_deltas_per_fold is None:
            best_deltas_per_fold = [None] * len(splits_data)

        lambda_smooth = trial.suggest_float('lambda_smooth', 1e-3, 1, log=True)
        alpha = trial.suggest_float('alpha', 1e-3, 1, log=True)
        beta = trial.suggest_float('beta', 1e-3, 1, log=True)

        MSE_score_test = torch.zeros(1).to(dev)
        current_deltas_per_fold = [None] * len(splits_data)

        for m, (data_split, y_split, H_split, train_index_split, test_index_split) in enumerate(
            zip(splits_data, splits_y, splits_H, train_indices_list, test_indices_list)
        ):
            data_train = data_split[0].to(dev)
            y_train = y_split[0].to(dev)
            H_train = H_split[0].to(dev)
            data_test = data_split[1].to(dev)
            y_test = y_split[1].to(dev)

            X_mean = data_train.mean(dim=0, keepdim=True)
            X_std  = data_train.std(dim=0, keepdim=True).clamp_min(1e-8)
            data_train = (data_train - X_mean) / X_std
            data_test  = (data_test  - X_mean) / X_std

            if isinstance(train_index_split, torch.Tensor):
                train_idx_np = train_index_split.detach().cpu().numpy()
            else:
                train_idx_np = np.asarray(train_index_split)
            n_train = train_idx_np.shape[0]
            assert n_train == data_train.shape[0]

            train_mask_vec = np.zeros(n_samples, dtype=bool)
            train_mask_vec[train_idx_np] = True
            keep_edge = train_mask_vec[edges_np[:, 0]] & train_mask_vec[edges_np[:, 1]]
            edges_train = edges_np[keep_edge]
            global_to_local = {int(g): i for i, g in enumerate(train_idx_np)}
            m_tr = edges_train.shape[0]
            H_train_tv = torch.zeros((m_tr, n_train), dtype=torch.float32, device=dev)
            if m_tr > 0:
                rows = np.arange(m_tr)
                cols_i = [global_to_local[int(g)] for g in edges_train[:, 0]]
                cols_j = [global_to_local[int(g)] for g in edges_train[:, 1]]
                H_train_tv[rows, cols_i] = 1.0
                H_train_tv[rows, cols_j] = -1.0

            print("H_train_tv:", H_train_tv.shape, "| data_train:", data_train.shape, "| DeltaW target shape:", (n_train, data_train.shape[1]))

            train_index_split = torch.tensor(train_index_split, dtype=torch.long, device=dev)
            test_index_split  = torch.tensor(test_index_split,  dtype=torch.long, device=dev)

            y_tmp = y_train.unsqueeze(1) if y_train.dim() == 1 else y_train
            mask_train = (y_tmp != 0).float()
            DeltaW = (best_deltas_per_fold[m].clone().detach().to(dev) if best_deltas_per_fold[m] is not None
                      else torch.randn(n_train, data_train.shape[1], device=dev, requires_grad=True))
            if DeltaW.requires_grad is False:
                DeltaW.requires_grad_(True)
            with torch.no_grad():
                DeltaW.data *= mask_train

            optimizer = torch.optim.Adam([DeltaW], lr=learning_rate)
            all_loss = []
            convergence_count = 0
            for i in range(num_iterations):
                if convergence_count < max_convergence_count:
                    optimizer.zero_grad()
                    loss = custom_loss_tv_mask(data_train, y_train, DeltaW, H_train_tv, lambda_smooth, alpha, beta)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        DeltaW.data *= mask_train
                    all_loss.append(loss.item())

                    if i > 100:
                        if abs(np.mean(all_loss[-40:-1]) - np.mean(all_loss[-80:-41])) < convergence_threshold:
                            convergence_count += 1
                        else:
                            convergence_count = 0
                        if convergence_count >= max_convergence_count:
                            print(i); print("stop early __ good")
                            break
                    if i > 3000 and abs(np.mean(all_loss[-500:-1]) - np.mean(all_loss[-1000:-501])) < convergence_threshold2:
                        print("Oscillating Convergence"); print(i)
                        break

            delta_beta = DeltaW.detach()
            sort = torch.sum(torch.abs(delta_beta), dim=0)
            sort_indices = torch.sort(sort, descending=True).indices
            ss = sort_indices[:topk_features].cpu().numpy()
            feature_count[ss] += 1

            delta_beta_test = []
            for test_idx in test_index_split:
                delta_beta_avg = find_neighbors_from_laplacian(test_idx, train_index_split, L_norm_local.clone(), delta_beta)
                delta_beta_test.append(delta_beta_avg)
            delta_beta_test = torch.cat(delta_beta_test, dim=0)
            y_hat_test = (data_test * delta_beta_test).sum(dim=1)
            MSE_test = torch.mean((y_test.squeeze() - y_hat_test) ** 2)
            MSE_score_test += MSE_test

            current_deltas_per_fold[m] = delta_beta.clone().cpu()

        avg_MSE_score_test = MSE_score_test.item() / len(splits_data)
        best_deltas_per_fold = current_deltas_per_fold
        if avg_MSE_score_test < best_mse:
            best_mse = avg_MSE_score_test
            print(f"update best MSE: {best_mse}")
        return avg_MSE_score_test

    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    if loss_mode.upper() == "H":
        study.optimize(objective_H, n_trials=n_trials, n_jobs=n_jobs)
    else:
        study.optimize(objective_L, n_trials=n_trials, n_jobs=n_jobs)
    end_time = time.time()
    run_time_minutes = (end_time - start_time) / 60

    trial = study.best_trial
    params = trial.params
    lambda_smooth = params['lambda_smooth']
    alpha = params['alpha']
    beta = params['beta']

    return {
        "study": study,
        "best_trial": trial,
        "best_params": trial.params,
        "best_mse": trial.value,
        "feature_count": feature_count,
        "run_time_minutes": run_time_minutes,
        "files": {"X": x_file, "Y": y_file, "coords": coords_file},
        "picked_params": {"lambda_smooth": lambda_smooth, "alpha": alpha, "beta": beta},
    }
