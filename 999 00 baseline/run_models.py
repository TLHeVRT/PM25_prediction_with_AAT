import os

import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
torch.set_num_threads(1)

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from tslearn.metrics import SoftDTWLossPyTorch
from mult_model import BaselineLSTM, BaselineTransformer, DataSet
import sys



class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# Dataset
class PredictionDatasetWithFuture(Dataset):
    """Dataset with fraud filtering."""

    def __init__(self, met_data, pol_data, mask_data, cache_file,
                 T_short=144, pred_len=48,
                 R_stations=32, num_iterations=500):
        self.T_short = T_short
        self.pred_len = pred_len
        self.R = R_stations
        self.num_iterations = num_iterations

        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data
        self.N_stations = met_data.shape[0]

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        cache_data = torch.load(cache_file)
        self.valid_starts_per_station = cache_data['valid_starts']
        self.eligible_stations = cache_data['eligible_stations']

        print(f"Loaded cache: {cache_file} (Eligible stations: {len(self.eligible_stations)}/{self.N_stations})")

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, index):
        perm = torch.randperm(len(self.eligible_stations))[:self.R]
        station_ids = self.eligible_stations[perm]

        station_ids_list = station_ids.tolist()
        starts = []
        for sid in station_ids_list:
            vs = self.valid_starts_per_station[sid]
            idx = random.randint(0, len(vs) - 1)
            starts.append(vs[idx].item())

        device = self.x_data.device
        starts_tensor = torch.tensor(starts, dtype=torch.long, device=device)
        station_ids_dev = station_ids.to(device)

        sid_idx = station_ids_dev.unsqueeze(1)
        short_idx = starts_tensor.unsqueeze(1) + torch.arange(self.T_short, device=device)
        target_idx = (starts_tensor + self.T_short).unsqueeze(1) + torch.arange(self.pred_len, device=device)

        short_segs = self.x_data[sid_idx, short_idx, :]
        short_masks = self.mask_data[sid_idx, short_idx]
        targets = self.x_data[sid_idx, target_idx, :]
        future_mets = targets[:, :, :-1]

        return short_segs, short_masks, future_mets, targets, station_ids


# Loss & Visualization
def pm25_loss(preds, targets, pol_mean, pol_std, gamma=0.25, k_ratio=0.1):
    """Combined loss with Dynamic Sequence Scaling."""
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = pm25_targets[valid_mask]

    # Return 0 if no valid data
    if preds_filtered.shape[0] == 0:
        zero_val = preds.sum() * 0.0
        return zero_val, zero_val.detach(), zero_val.detach(), zero_val.detach()

    # 1. Base Loss
    abs_diff_base = torch.abs(preds_filtered - targets_filtered)
    loss_base_seq = abs_diff_base.mean(dim=1)  # Shape: (N,)

    # 2. Trend Loss
    diff_preds = preds_filtered[:, 1:] - preds_filtered[:, :-1]
    diff_targets = targets_filtered[:, 1:] - targets_filtered[:, :-1]
    loss_trend_seq = torch.abs(diff_preds - diff_targets).mean(dim=1)  # Shape: (N,)

    # 3. Soft-DTW Loss
    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds_filtered.unsqueeze(2)
    targets_3d = targets_filtered.unsqueeze(2)

    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    with torch.no_grad():
        dtw_yy = sdtw_criterion(targets_3d, targets_3d)

    loss_divergence_seq = (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0)  # Shape: (N,)

    # 4. Top-K Loss
    seq_len = targets_filtered.shape[1]
    k_num = max(1, int(seq_len * k_ratio))

    topk_target_vals, topk_indices = torch.topk(targets_filtered, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds_filtered, 1, topk_indices)

    abs_diff_topk = torch.abs(topk_pred_vals - topk_target_vals)
    loss_topk_seq = abs_diff_topk.mean(dim=1)  # Shape: (N,)

    # 5. Total Weighted Loss
    alpha = 1.0  # Base 权重
    beta = 0.5  # Trend 权重
    gamma_w = 0.05  # DTW 权重
    delta_w = 0.3  # Top-K 权重

    total_loss_seq = (alpha * loss_base_seq) + \
                     (beta * loss_trend_seq) + \
                     (gamma_w * loss_divergence_seq) + \
                     (delta_w * loss_topk_seq)

    # 6. Dynamic Scaling
    limit = 2.0
    scale_factors = torch.where(
        total_loss_seq > limit,
        limit / total_loss_seq.detach(),
        torch.ones_like(total_loss_seq)
    )

    final_loss = (total_loss_seq * scale_factors).mean()

    scaled_base = (loss_base_seq * scale_factors).mean().detach()
    scaled_dtw = (loss_divergence_seq * scale_factors).mean().detach()
    scaled_topk = (loss_topk_seq * scale_factors).mean().detach()
    return final_loss, scaled_base, scaled_dtw, scaled_topk


def plot_results(epoch, preds, targets, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    n_plots = min(10, preds.shape[0])
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    x = np.arange(preds.shape[1])

    for i in range(n_plots):
        ax = axes[i]
        ax.plot(x, targets[i], '.-', color='dodgerblue', label='True', markersize=5)
        ax.plot(x, preds[i], '.-', color='coral', label='Predicted', markersize=5)
        ax.set_title(f"Validation Sample {i + 1} (PM2.5)")
        ax.set_xlabel("Future Time Steps")
        ax.set_ylabel("Normalized PM2.5")
        ax.legend(loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/epoch_{epoch}.png", dpi=150)
    plt.close()


def run_experiment(run_id, model_type):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    # Hyperparameters
    T_short = 144
    pred_len = 48
    R_stations = 32
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    T_0 = 100

    exp_dir = f"exp_{model_type}_run_{run_id}"
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(exp_dir, "log.txt"))
    sys.stdout = logger
    print(f"Starting experiment: {model_type.upper()} in {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std

    total_time_steps = met_data.shape[1]
    split_idx = int(total_time_steps * (3 / 4))

    train_met = met_data[:, :split_idx, :].to(device)
    train_pol = pol_data[:, :split_idx].to(device)
    train_mask = mask_data[:, :split_idx].to(device)

    val_met = met_data[:, split_idx:, :].to(device)
    val_pol = pol_data[:, split_idx:].to(device)
    val_mask = mask_data[:, split_idx:].to(device)

    train_cache_file = f"dataset_cache_train_T{T_short}_P{pred_len}.pt"
    val_cache_file = f"dataset_cache_val_T{T_short}_P{pred_len}.pt"

    train_pred_ds = PredictionDatasetWithFuture(
        train_met, train_pol, train_mask,
        cache_file=train_cache_file,
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=1500
    )

    val_pred_ds = PredictionDatasetWithFuture(
        val_met, val_pol, val_mask,
        cache_file=val_cache_file,
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=400
    )

    train_pred_loader = DataLoader(train_pred_ds, batch_size=1, shuffle=True, num_workers=0)
    val_pred_loader = DataLoader(val_pred_ds, batch_size=1, shuffle=True, num_workers=0)

    # Initialize model
    if model_type == 'lstm':
        predictor = BaselineLSTM(
            in_channels=in_channels,
            met_channels=in_channels - 1,
            hidden_size=256,
            num_layers=6
        ).to(device)
    elif model_type == 'transformer':
        predictor = BaselineTransformer(
            in_channels=in_channels,
            d_model=256,
            max_len=T_short + pred_len,
            num_layers = 6,
            dropout = 0.2
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    all_params = list(predictor.parameters())
    optimizer = optim.AdamW(all_params, lr=max_lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    # Resume training
    start_epoch = 0
    ema_alpha = 0.25
    combined_loss_history = []
    best_ema_combined_loss = float('inf')
    best_model_pattern = os.path.join(exp_dir, "best_model_ema_combined_loss_*.pth")
    for best_model_file in glob.glob(best_model_pattern):
        match = re.search(r"best_model_ema_combined_loss_([0-9]+(?:\.[0-9]+)?)\.pth", os.path.basename(best_model_file))
        if match:
            best_ema_combined_loss = min(best_ema_combined_loss, float(match.group(1)))

    weight_pattern = os.path.join(exp_dir, f"{model_type}_model_*.pth")
    weight_files = glob.glob(weight_pattern)
    if weight_files:
        epochs_found = [
            int(re.search(rf"{model_type}_model_(\d+)\.pth", os.path.basename(f)).group(1))
            for f in weight_files if re.search(rf"{model_type}_model_(\d+)\.pth", os.path.basename(f))
        ]
        if epochs_found:
            max_epoch = max(epochs_found)
            start_epoch = max_epoch
            ckpt_path = os.path.join(exp_dir, f"{model_type}_model_{max_epoch}.pth")
            ckpt = torch.load(ckpt_path, map_location=device)

            predictor.load_state_dict(ckpt['predictor'])
            combined_loss_history = ckpt.get('combined_loss_history', [])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                print(f"Resumed from epoch {max_epoch}.")

    # Training loop
    for epoch in range(start_epoch, epochs):
        predictor.train()

        prediction_loss_total = 0.0
        train_base_total, train_dtw_total, train_topk_total = 0.0, 0.0, 0.0

        num_train_batches = max(len(train_pred_loader), 1)

        for batch_idx, (short_segs, short_masks, future_mets, targets, station_ids) in enumerate(train_pred_loader):
            # LR Warmup
            if epoch == 0:
                warmup_start_lr = 1e-5
                current_lr = warmup_start_lr + (max_lr - warmup_start_lr) * (batch_idx / num_train_batches)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            short_segs = short_segs.squeeze(0).to(device)
            short_masks = short_masks.squeeze(0).to(device)
            future_mets = future_mets.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)

            optimizer.zero_grad()
            preds = predictor(short_segs, future_met=future_mets, mask=short_masks)

            loss, l_base, l_dtw, l_topk = pm25_loss(preds, targets, pol_mean=pol_mean, pol_std=pol_std)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            prediction_loss_total += loss.item()
            train_base_total += l_base.item()
            train_dtw_total += l_dtw.item()
            train_topk_total += l_topk.item()

        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max_lr
        else:
            scheduler.step()

        num_train_batches = max(len(train_pred_loader), 1)
        avg_prediction_loss = prediction_loss_total / num_train_batches
        avg_t_base = train_base_total / num_train_batches
        avg_t_dtw = train_dtw_total / num_train_batches
        avg_t_topk = train_topk_total / num_train_batches

        # Validation
        predictor.eval()
        val_loss_total = 0.0
        val_base_total, val_dtw_total, val_topk_total = 0.0, 0.0, 0.0
        draw_flag = ((epoch + 1) % 5 == 0)
        saved_preds, saved_targets = [], []

        with torch.no_grad():
            for short_segs, short_masks, future_mets, targets, station_ids in val_pred_loader:
                short_segs = short_segs.squeeze(0).to(device)
                short_masks = short_masks.squeeze(0).to(device)
                future_mets = future_mets.squeeze(0).to(device)
                targets = targets.squeeze(0).to(device)

                preds = predictor(short_segs, future_met=future_mets, mask=short_masks)

                loss_v, lv_base, lv_dtw, lv_topk = pm25_loss(preds, targets, pol_mean=pol_mean, pol_std=pol_std)
                val_loss_total += loss_v.item()
                val_base_total += lv_base.item()
                val_dtw_total += lv_dtw.item()
                val_topk_total += lv_topk.item()

                if draw_flag and len(saved_preds) < 10:
                    saved_preds.extend(preds.cpu().numpy())
                    saved_targets.extend(targets[:, :, -1].cpu().numpy())

        num_val_batches = max(len(val_pred_loader), 1)
        avg_val_loss = val_loss_total / num_val_batches
        avg_v_base = val_base_total / num_val_batches
        avg_v_dtw = val_dtw_total / num_val_batches
        avg_v_topk = val_topk_total / num_val_batches
        combined_loss_history.append(avg_val_loss)
        ema_combined_loss = combined_loss_history[0]
        for combined_loss in combined_loss_history[1:]:
            ema_combined_loss = ema_alpha * combined_loss + (1.0 - ema_alpha) * ema_combined_loss

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {avg_prediction_loss:.4f} (Base: {avg_t_base:.4f}, DTW: {avg_t_dtw:.4f}, TopK: {avg_t_topk:.4f}) | "
              f"Val Loss:   {avg_val_loss:.4f} (Base: {avg_v_base:.4f}, DTW: {avg_v_dtw:.4f}, TopK: {avg_v_topk:.4f}) | "
              f"EMA Combined Loss: {ema_combined_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save
        if ema_combined_loss < best_ema_combined_loss:
            previous_best_files = glob.glob(best_model_pattern)
            best_model_path = os.path.join(exp_dir, f"best_model_ema_combined_loss_{ema_combined_loss:.12f}.pth")
            torch.save({
                'predictor': predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'combined_loss_history': combined_loss_history,
                'ema_combined_loss': ema_combined_loss,
            }, best_model_path)
            for previous_best_file in previous_best_files:
                if os.path.abspath(previous_best_file) != os.path.abspath(best_model_path):
                    os.remove(previous_best_file)
            best_ema_combined_loss = ema_combined_loss

        if draw_flag:
            if len(saved_preds) >= 10:
                plot_out_dir = os.path.join(exp_dir, f"plot_{model_type}")
                plot_results(epoch + 1, np.array(saved_preds[:10]), np.array(saved_targets[:10]), out_dir=plot_out_dir)

            save_path = os.path.join(exp_dir, f"{model_type}_model_{epoch + 1}.pth")
            torch.save({
                'predictor': predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'combined_loss_history': combined_loss_history,
                'ema_combined_loss': ema_combined_loss,
            }, save_path)

    logger.close()


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Enable TF32 for Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    print("Starting 5 Transformer processes...")
    tf_processes = []
    for i in range(1, 6):
        p_tf = mp.Process(target=run_experiment, args=(i, 'transformer'))
        p_tf.start()
        tf_processes.append(p_tf)
        print(f"Started process {p_tf.pid} for experiment {i}")

    for p in tf_processes:
        p.join()

    print("All tasks completed.")


if __name__ == '__main__':
    main()
