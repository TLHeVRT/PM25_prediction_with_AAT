import os

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
from mult_model import StaticProfileEncoder, ShortTermPredictorWithFuture, DataSet
import sys
import torch.multiprocessing as mp


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


def build_static_features(train_met, train_pol, train_mask, pol_mean, pol_std, valid_stations,
                          geo_csv_path="station_features.csv"):
    print("Building static profile features from historical data...")
    N, T, C_met = train_met.shape
    hours_per_year = 365 * 24

    if T >= hours_per_year:
        met_y = train_met[:, -hours_per_year:, :]
        pol_y = train_pol[:, -hours_per_year:]
        mask_y = train_mask[:, -hours_per_year:]
    else:
        met_y = train_met
        pol_y = train_pol
        mask_y = train_mask

    # Restore raw PM2.5
    if isinstance(pol_mean, torch.Tensor):
        p_mean = pol_mean.to(pol_y.device)
        p_std = pol_std.to(pol_y.device)
    else:
        p_mean, p_std = pol_mean, pol_std

    pol_y_raw = pol_y * p_std + p_mean

    valid_mask_global = (mask_y > 0) & (pol_y_raw <= 1000)
    valid_mask_float = valid_mask_global.float()

    chunk_size = met_y.shape[1] // 12
    pm25_means, pm25_stds, met_means = [], [], []

    for i in range(12):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 11 else met_y.shape[1]

        p_c = pol_y[:, start:end]
        met_c = met_y[:, start:end, :]

        m_c = valid_mask_float[:, start:end]

        sum_p = (p_c * m_c).sum(dim=1)
        cnt_p = m_c.sum(dim=1).clamp(min=1)
        mean_p = sum_p / cnt_p

        var_p = (((p_c - mean_p.unsqueeze(1)) * m_c) ** 2).sum(dim=1) / cnt_p
        std_p = torch.sqrt(var_p.clamp(min=1e-8))

        pm25_means.append(mean_p)
        pm25_stds.append(std_p)

        met_means.append(met_c.mean(dim=1))

    pm25_means = torch.stack(pm25_means, dim=1)
    pm25_stds = torch.stack(pm25_stds, dim=1)
    met_means = torch.cat(met_means, dim=1)

    corrs = []
    for c in range(C_met):
        mc = met_y[:, :, c]
        valid = valid_mask_global
        corr_c = torch.zeros(N)
        for n in range(N):
            v = valid[n]
            if v.sum() > 2:
                x = mc[n, v]
                y = pol_y[n, v]
                vx = x - x.mean()
                vy = y - y.mean()

                denom_sq = (vx ** 2).sum() * (vy ** 2).sum()
                denom = torch.sqrt(denom_sq.clamp(min=1e-12))

                if denom > 1e-8:
                    corr_c[n] = (vx * vy).sum() / denom
        corrs.append(corr_c)
    corrs = torch.stack(corrs, dim=1)

    bc_features = torch.cat([pm25_means, pm25_stds, met_means, corrs], dim=1)
    mean_bc = bc_features.mean(dim=0, keepdim=True)
    std_bc = bc_features.std(dim=0, keepdim=True).clamp(min=1e-8)
    bc_features = (bc_features - mean_bc) / std_bc

    if os.path.exists(geo_csv_path):
        df_geo = pd.read_csv(geo_csv_path)
        a_features_np = df_geo.iloc[:, 1:].astype(float).values
        a_features = torch.tensor(a_features_np, dtype=torch.float32)

        if valid_stations is not None:
            a_features = a_features[valid_stations]
    else:
        a_features = torch.zeros(N, 13)

    final_static_features = torch.cat([a_features, bc_features], dim=1)
    return final_static_features


class PredictionDatasetWithFuture(Dataset):
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
            raise FileNotFoundError(
                f"\n[错误] 找不到缓存文件: {cache_file}\n"
                f"请先运行 'python generate_cache.py' 生成缓存数据！"
            )

        cache_data = torch.load(cache_file)
        self.valid_starts_per_station = cache_data['valid_starts']
        self.eligible_stations = cache_data['eligible_stations']

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


def pm25_loss(preds, targets, pol_mean, pol_std, gamma=0.25, k_ratio=0.1):
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = pm25_targets[valid_mask]

    if preds_filtered.shape[0] == 0:
        zero_val = preds.sum() * 0.0
        return zero_val, zero_val.detach(), zero_val.detach(), zero_val.detach()

    abs_diff_base = torch.abs(preds_filtered - targets_filtered)
    loss_base_seq = abs_diff_base.mean(dim=1)

    diff_preds = preds_filtered[:, 1:] - preds_filtered[:, :-1]
    diff_targets = targets_filtered[:, 1:] - targets_filtered[:, :-1]
    loss_trend_seq = torch.abs(diff_preds - diff_targets).mean(dim=1)

    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds_filtered.unsqueeze(2)
    targets_3d = targets_filtered.unsqueeze(2)

    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    with torch.no_grad():
        dtw_yy = sdtw_criterion(targets_3d, targets_3d)

    loss_divergence_seq = (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0)

    seq_len = targets_filtered.shape[1]
    k_num = max(1, int(seq_len * k_ratio))
    topk_target_vals, topk_indices = torch.topk(targets_filtered, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds_filtered, 1, topk_indices)

    abs_diff_topk = torch.abs(topk_pred_vals - topk_target_vals)
    loss_topk_seq = abs_diff_topk.mean(dim=1)

    alpha = 1.0
    beta = 0.5
    gamma_w = 0.05
    delta_w = 0.3

    total_loss_seq = (alpha * loss_base_seq) + \
                     (beta * loss_trend_seq) + \
                     (gamma_w * loss_divergence_seq) + \
                     (delta_w * loss_topk_seq)

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


def run_experiment(run_id):
    T_short = 144
    pred_len = 48
    R_stations = 32
    d_profile = 256
    d_model = 256
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    T_0 = 100

    exp_dir = f"exp_run_{run_id}"
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(exp_dir, "log.txt"))
    sys.stdout = logger

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std

    total_time_steps = met_data.shape[1]
    split_idx = int(total_time_steps * (3 / 4))

    train_met_cpu = met_data[:, :split_idx, :]
    train_pol_cpu = pol_data[:, :split_idx]
    train_mask_cpu = mask_data[:, :split_idx]

    # Compute on CPU to save VRAM
    static_features_global = build_static_features(
        train_met_cpu, train_pol_cpu, train_mask_cpu,
        pol_mean=pol_mean,
        pol_std=pol_std,
        valid_stations=data_set.valid_stations,
        geo_csv_path="station_features.csv"
    ).to(device)

    train_met = train_met_cpu.to(device)
    train_pol = train_pol_cpu.to(device)
    train_mask = train_mask_cpu.to(device)

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
        R_stations=R_stations, num_iterations=600
    )

    train_pred_loader = DataLoader(
        train_pred_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    val_pred_loader = DataLoader(
        val_pred_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    num_static_features = static_features_global.shape[1]
    encoder = StaticProfileEncoder(
        in_features=num_static_features,
        d_profile=d_profile,
        dropout=0.2
    ).to(device)

    predictor = ShortTermPredictorWithFuture(
        seq_len_short=T_short,
        pred_len=pred_len,
        in_channels=in_channels,
        met_channels=in_channels - 1,
        d_profile=d_profile,
        d_model=d_model,
        n_heads=8,
        e_layers=6,
        dropout=0.2,
    ).to(device)

    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.AdamW(all_params, lr=max_lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"[*] Total trainable parameters: {total_params:,}")

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    start_epoch = 0
    ema_alpha = 0.25
    combined_loss_history = []
    best_ema_combined_loss = float('inf')
    best_model_pattern = os.path.join(exp_dir, "best_model_ema_combined_loss_*.pth")
    for best_model_file in glob.glob(best_model_pattern):
        match = re.search(r"best_model_ema_combined_loss_([0-9]+(?:\.[0-9]+)?)\.pth", os.path.basename(best_model_file))
        if match:
            best_ema_combined_loss = min(best_ema_combined_loss, float(match.group(1)))

    weight_pattern = os.path.join(exp_dir, "dual_model_with_future_*.pth")
    weight_files = glob.glob(weight_pattern)
    if weight_files:
        epochs_found = [int(re.search(r"dual_model_with_future_(\d+)\.pth", os.path.basename(f)).group(1))
                        for f in weight_files if
                        re.search(r"dual_model_with_future_(\d+)\.pth", os.path.basename(f))]
        if epochs_found:
            max_epoch = max(epochs_found)
            start_epoch = max_epoch
            ckpt_path = os.path.join(exp_dir, f"dual_model_with_future_{max_epoch}.pth")
            ckpt = torch.load(ckpt_path, map_location=device)

            encoder.load_state_dict(ckpt['encoder'])
            predictor.load_state_dict(ckpt['predictor'])
            combined_loss_history = ckpt.get('combined_loss_history', [])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])

    for epoch in range(start_epoch, epochs):
        encoder.train()
        predictor.train()

        prediction_loss_total = 0.0
        train_base_total, train_dtw_total, train_topk_total = 0.0, 0.0, 0.0

        for short_segs, short_masks, future_mets, targets, station_ids in train_pred_loader:
            short_segs = short_segs.squeeze(0).to(device)
            short_masks = short_masks.squeeze(0).to(device)
            future_mets = future_mets.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            station_ids = station_ids.squeeze(0).to(device)

            optimizer.zero_grad()

            batch_static_feats = static_features_global[station_ids]
            profiles = encoder(batch_static_feats)

            preds = predictor(short_segs, profiles, future_met=future_mets, mask=short_masks)

            loss, l_base, l_dtw, l_topk = pm25_loss(preds, targets, pol_mean=pol_mean, pol_std=pol_std)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            prediction_loss_total += loss.item()
            train_base_total += l_base.item()
            train_dtw_total += l_dtw.item()
            train_topk_total += l_topk.item()

        scheduler.step()

        num_train_batches = max(len(train_pred_loader), 1)
        avg_prediction_loss = prediction_loss_total / num_train_batches
        avg_t_base = train_base_total / num_train_batches
        avg_t_dtw = train_dtw_total / num_train_batches
        avg_t_topk = train_topk_total / num_train_batches

        encoder.eval()
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
                station_ids = station_ids.squeeze(0).to(device)

                batch_static_feats = static_features_global[station_ids]
                profiles = encoder(batch_static_feats)

                preds = predictor(short_segs, profiles, future_met=future_mets, mask=short_masks)

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

        if ema_combined_loss < best_ema_combined_loss:
            previous_best_files = glob.glob(best_model_pattern)
            best_model_path = os.path.join(exp_dir, f"best_model_ema_combined_loss_{ema_combined_loss:.12f}.pth")
            torch.save({
                'encoder': encoder.state_dict(),
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
                plot_out_dir = os.path.join(exp_dir, "plot_with_future")
                plot_results(epoch + 1, np.array(saved_preds[:10]), np.array(saved_targets[:10]), out_dir=plot_out_dir)

            save_path = os.path.join(exp_dir, f"dual_model_with_future_{epoch + 1}.pth")
            torch.save({
                'encoder': encoder.state_dict(),
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

    processes = []
    for i in range(1, 6):
        p = mp.Process(target=run_experiment, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
