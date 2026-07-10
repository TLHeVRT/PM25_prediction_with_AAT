import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
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
from mult_model import CNN1DEncoder


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
        print(f"Geo feature file '{geo_csv_path}' not found. Using zero placeholder.")
        a_features = torch.zeros(N, 13)

    final_static_features = torch.cat([a_features, bc_features], dim=1)
    return final_static_features


@torch.no_grad()
def build_or_load_memory_bank(dataset, encoder, device, cache_path, batch_size=1024):
    if os.path.exists(cache_path):
        print(f"Loaded memory bank from cache: {cache_path}")
        bank_data = torch.load(cache_path, map_location=device)
        return bank_data['vectors'], bank_data['stids'], bank_data['starts']

    print(f"Building memory bank (Total Stations: {dataset.N_stations})...")
    encoder.eval()
    encoder.to(device)

    all_vecs, all_stids, all_starts = [], [], []
    samples = []

    for sid in dataset.eligible_stations.tolist():
        starts = dataset.valid_starts_per_station[sid]
        for st in starts:
            samples.append((sid, st.item()))

    x_data = dataset.x_data.to(device)
    for i in tqdm(range(0, len(samples), batch_size), desc="Building Bank"):
        batch = samples[i:i + batch_size]
        sids_t = torch.tensor([s[0] for s in batch], device=device)
        sts_t = torch.tensor([s[1] for s in batch], device=device)

        short_idx = sts_t.unsqueeze(1) + torch.arange(dataset.T_short, device=device)
        inputs = x_data[sids_t.unsqueeze(1), short_idx, :]

        vecs = encoder(inputs)

        all_vecs.append(vecs)
        all_stids.append(sids_t)
        all_starts.append(sts_t)

    bank_vectors = torch.cat(all_vecs, dim=0)
    bank_stids = torch.cat(all_stids, dim=0)
    bank_starts = torch.cat(all_starts, dim=0)

    torch.save({
        'vectors': bank_vectors,
        'stids': bank_stids,
        'starts': bank_starts
    }, cache_path)

    print(f"Memory bank built. Records: {bank_vectors.shape[0]}. Saved to: {cache_path}")
    return bank_vectors, bank_stids, bank_starts


def retrieve_top10_sequences(curr_vecs, starts_batch, bank_vectors, bank_norms, bank_stids, bank_starts, x_data, T_long, T_pred):
    B = curr_vecs.shape[0]
    T_total = T_long + T_pred
    device = curr_vecs.device
    hours_per_year = 365 * 24

    sample_size = min(1000000, bank_vectors.shape[0])
    rand_idx = torch.randint(0, bank_vectors.shape[0], (sample_size,), device=device)

    sub_bank_vecs = bank_vectors[rand_idx]
    sub_bank_norms = bank_norms[rand_idx]
    sub_bank_stids = bank_stids[rand_idx]
    sub_bank_starts = bank_starts[rand_idx]

    chunk_size = 32
    all_topk_idx = []
    k_num = min(10, sample_size)

    for i in range(0, B, chunk_size):
        end_idx = min(i + chunk_size, B)
        q_chunk = curr_vecs[i:end_idx]
        s_chunk = starts_batch[i:end_idx]

        dot_chunk = torch.matmul(q_chunk, sub_bank_vecs.T)
        dists_chunk = sub_bank_norms.unsqueeze(0) - 2 * dot_chunk

        # Standard: aux sequence strictly before main sequence
        valid_standard = sub_bank_starts.unsqueeze(0) <= (s_chunk.unsqueeze(1) - T_total)

        # Special for year 1
        is_year1 = (s_chunk + T_total) <= hours_per_year

        # Restrict year 1 aux sequences to year 2
        valid_year2 = (sub_bank_starts.unsqueeze(0) >= hours_per_year) & \
                      (sub_bank_starts.unsqueeze(0) < 2 * hours_per_year) & \
                      (sub_bank_starts.unsqueeze(0) >= (s_chunk.unsqueeze(1) + T_total))

        # Dynamic mask
        valid_mask = torch.where(is_year1.unsqueeze(1), valid_year2, valid_standard)

        # Mask invalid sequences
        mask_chunk = ~valid_mask
        dists_chunk.masked_fill_(mask_chunk, float('inf'))

        _, topk_idx_chunk = torch.topk(dists_chunk, k=k_num, dim=1, largest=False)
        all_topk_idx.append(topk_idx_chunk)

        del dot_chunk, dists_chunk, mask_chunk, valid_standard, valid_year2, valid_mask

    topk_idx = torch.cat(all_topk_idx, dim=0)  # shape: [B, 10]

    t10_stids = sub_bank_stids[topk_idx]
    t10_starts = sub_bank_starts[topk_idx]

    batch_idx = torch.arange(T_total, device=device).view(1, 1, -1)

    target_starts = t10_starts.unsqueeze(2) + batch_idx
    target_sids = t10_stids.unsqueeze(2).expand(-1, -1, T_total)

    matched_seqs_batch = x_data[target_sids, target_starts, :]

    return matched_seqs_batch, t10_stids


class PredictionDatasetWithFuture(Dataset):
    def __init__(self, met_data, pol_data, mask_data, cache_file,
                 T_short=144, pred_len=48,
                 R_stations=32, num_iterations=500, train_start_offset=0):
        self.T_short = T_short
        self.pred_len = pred_len
        self.R = R_stations
        self.num_iterations = num_iterations

        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data
        self.N_stations = met_data.shape[0]
        self.train_start_offset = train_start_offset

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

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
            valid_pool = vs[vs >= self.train_start_offset]
            idx = random.randint(0, len(valid_pool) - 1)
            starts.append(valid_pool[idx].item())

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

        return short_segs, short_masks, future_mets, targets, station_ids, starts_tensor


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

    seq_len = targets_filtered.shape[1]
    loss_divergence_seq = (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) / seq_len

    seq_len = targets_filtered.shape[1]
    k_num = max(1, int(seq_len * k_ratio))
    topk_target_vals, topk_indices = torch.topk(targets_filtered, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds_filtered, 1, topk_indices)

    abs_diff_topk = torch.abs(topk_pred_vals - topk_target_vals)
    loss_topk_seq = abs_diff_topk.mean(dim=1)

    alpha = 1.0
    beta = 0.5
    gamma_w = 2.4
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


def add_continuous_noise_mask(aux_seqs):
    B, K, T, C = aux_seqs.shape
    masked_seqs = aux_seqs.clone()
    mask_info = []

    stds = aux_seqs[:, :, :, -1].std(dim=-1).clamp(min=1e-5)
    do_mask = torch.rand((B, K)) < 0.5

    max_len = 48

    for b in range(B):
        for k in range(K):
            if do_mask[b, k]:
                actual_length = random.randint(12, max_len)

                start = random.randint(0, T - max_len)
                end = start + max_len

                std_val = stds[b, k]
                num_points = max(2, actual_length // 4)

                # 仅生成 actual_length 长度的连续噪声
                coarse_noise = torch.randn(1, 1, num_points, device=aux_seqs.device) * std_val
                continuous_noise = F.interpolate(coarse_noise, size=actual_length, mode='linear',
                                                 align_corners=True).squeeze()

                padded_noise = torch.zeros(max_len, device=aux_seqs.device)
                padded_noise[:actual_length] = continuous_noise

                masked_seqs[b, k, start:end, -1] += padded_noise

                # 将实际长度 actual_length 也记录下来
                mask_info.append((b, k, start, end, actual_length))

    return masked_seqs, mask_info

def run_experiment(run_id, bank_global):
    bank_vecs_global, bank_stids_global, bank_starts_global = bank_global
    bank_norms_global = (bank_vecs_global ** 2).sum(dim=1)

    T_short = 144
    pred_len = 48
    R_stations = 16
    d_model = 256
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    T_0 = 100
    d_profile = 256

    exp_dir = f"exp_run_d{d_profile}_id{run_id}"
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(exp_dir, "log.txt"))
    sys.stdout = logger
    print(f"Starting experiment in {exp_dir} (d_profile={d_profile})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 载入原始数据集
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
        R_stations=R_stations, num_iterations=1500,
        train_start_offset=0
    )

    val_pred_ds = PredictionDatasetWithFuture(
        val_met, val_pol, val_mask,
        cache_file=val_cache_file,
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=600,
        train_start_offset=0
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

    # Load pre-trained lightweight encoder
    print("Loading lightweight encoder...")
    light_encoder = CNN1DEncoder(in_channels=in_channels, d_model=128).to(device)
    light_encoder.load_state_dict(torch.load("best_encoder_only.pth", map_location=device))
    light_encoder.eval()
    for param in light_encoder.parameters():
        param.requires_grad = False

    global_x_data = torch.cat([met_data.to(device), pol_data.to(device).unsqueeze(-1)], dim=-1)

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
    predictor = torch.compile(predictor, mode="reduce-overhead")

    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.AdamW(all_params, lr=max_lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

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
                print(f"Resumed from epoch {max_epoch}.")

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, epochs):
        encoder.train()
        predictor.train()

        total_loss_sum = 0.0
        prediction_loss_total = 0.0
        aux_loss_total_sum = 0.0
        train_base_total, train_dtw_total, train_topk_total = 0.0, 0.0, 0.0
        num_warmup_steps = max(len(train_pred_loader) - 1, 1)
        for batch_idx, (short_segs, short_masks, future_mets, targets, station_ids, starts_batch) in enumerate(
                train_pred_loader):
            # LR warmup
            if epoch == 0:
                warmup_lr = 1e-5 + (max_lr - 1e-5) * (batch_idx / num_warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            short_segs = short_segs.squeeze(0).to(device)
            short_masks = short_masks.squeeze(0).to(device)
            future_mets = future_mets.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            station_ids = station_ids.squeeze(0).to(device)
            starts_batch = starts_batch.squeeze(0).to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                curr_vecs = light_encoder(short_segs)
                top10_matched_seqs, t10_stids = retrieve_top10_sequences(
                    curr_vecs, starts_batch,
                    bank_vecs_global, bank_norms_global, bank_stids_global, bank_starts_global,
                    global_x_data, T_short, pred_len
                )

            masked_matched_seqs, mask_info = add_continuous_noise_mask(top10_matched_seqs)

            batch_static_feats = static_features_global[station_ids]
            aux_static_feats = static_features_global[t10_stids]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                profiles = encoder(batch_static_feats)
                aux_profiles = encoder(aux_static_feats)
                preds, aux_reconstruct = predictor(
                    short_segs, profiles, future_met=future_mets, mask=short_masks,
                    matched_hist=masked_matched_seqs, aux_profiles=aux_profiles
                )
                loss, l_base, l_dtw, l_topk = pm25_loss(preds, targets, pol_mean=pol_mean, pol_std=pol_std)

                aux_loss_val = torch.tensor(0.0, device=device)
                if mask_info:
                    target_list = []
                    pred_list = []
                    length_list = []

                    for b, k, start, end, actual_length in mask_info:
                        target_list.append(top10_matched_seqs[b, k, start:end, :])
                        pred_list.append(aux_reconstruct[b, k, start:end])
                        length_list.append(actual_length)

                    if target_list:
                        target_batch = torch.stack(target_list, dim=0)
                        pred_batch = torch.stack(pred_list, dim=0) # [N, 48]

                        N_items = target_batch.shape[0]
                        max_len = target_batch.shape[1]
                        # Mask valid noise vs padding
                        lengths_tensor = torch.tensor(length_list, device=device).unsqueeze(1)
                        idx_tensor = torch.arange(max_len, device=device).unsqueeze(0)
                        valid_mask = idx_tensor < lengths_tensor

                        pm25_targets = target_batch[:, :, -1]
                        # Keep prediction in valid range, otherwise use target
                        pred_batch_masked = torch.where(valid_mask, pred_batch, pm25_targets)

                        a_loss, _, _, _ = pm25_loss(pred_batch_masked, target_batch, pol_mean=pol_mean, pol_std=pol_std)
                        B_current = short_segs.shape[0]
                        aux_loss_val = a_loss * 0.1 * (N_items / B_current)

                total_loss = loss + aux_loss_val

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss_sum += total_loss.item()
            aux_loss_total_sum += aux_loss_val.item()
            prediction_loss_total += loss.item()
            train_base_total += l_base.item()
            train_dtw_total += l_dtw.item()
            train_topk_total += l_topk.item()

        scheduler.step()

        num_train_batches = max(len(train_pred_loader), 1)
        avg_total_loss = total_loss_sum / num_train_batches
        avg_aux_loss = aux_loss_total_sum / num_train_batches
        avg_prediction_loss = prediction_loss_total / num_train_batches
        avg_t_base = train_base_total / num_train_batches
        avg_t_dtw = train_dtw_total / num_train_batches
        avg_t_topk = train_topk_total / num_train_batches

        # Validation
        encoder.eval()
        predictor.eval()
        val_loss_total = 0.0
        val_base_total, val_dtw_total, val_topk_total = 0.0, 0.0, 0.0
        draw_flag = ((epoch + 1) % 5 == 0)
        saved_preds, saved_targets = [], []

        with torch.no_grad():
            for short_segs, short_masks, future_mets, targets, station_ids, starts_batch in val_pred_loader:
                short_segs = short_segs.squeeze(0).to(device)
                short_masks = short_masks.squeeze(0).to(device)
                future_mets = future_mets.squeeze(0).to(device)
                targets = targets.squeeze(0).to(device)
                station_ids = station_ids.squeeze(0).to(device)
                starts_batch = starts_batch.squeeze(0).to(device)

                curr_vecs = light_encoder(short_segs)
                # Convert starts_batch to global index
                starts_batch_global = starts_batch + split_idx
                top10_matched_seqs , t10_stids= retrieve_top10_sequences(
                    curr_vecs, starts_batch_global,
                    bank_vecs_global, bank_norms_global, bank_stids_global, bank_starts_global,
                    global_x_data, T_short, pred_len
                )

                batch_static_feats = static_features_global[station_ids]
                aux_static_feats = static_features_global[t10_stids]

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    profiles = encoder(batch_static_feats)
                    aux_profiles = encoder(aux_static_feats)

                    preds, _ = predictor(
                        short_segs, profiles, future_met=future_mets, mask=short_masks,
                        matched_hist=top10_matched_seqs, aux_profiles=aux_profiles
                    )
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
              f"Total Loss: {avg_total_loss:.4f} | Aux Loss: {avg_aux_loss:.4f} | "
              f"Train Loss: {avg_prediction_loss:.4f} (Base: {avg_t_base:.4f}, DTW: {avg_t_dtw:.4f}, TopK: {avg_t_topk:.4f}) | "
              f"Val Loss:   {avg_val_loss:.4f} (Base: {avg_v_base:.4f}, DTW: {avg_v_dtw:.4f}, TopK: {avg_v_topk:.4f}) | "
              f"EMA Combined Loss: {ema_combined_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save
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

    print("Initializing shared GPU memory bank...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T_short = 144
    pred_len = 48

    # Temp dataset for Bank generation
    data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized.to(device)
    pol_data = data_set.pol_data_normalized.to(device)
    mask_data = data_set.pol_mask_matrix.to(device)

    total_time_steps = met_data.shape[1]
    split_idx = int(total_time_steps * (3 / 4))

    train_met = met_data[:, :split_idx, :]
    train_pol = pol_data[:, :split_idx]
    train_mask = mask_data[:, :split_idx]

    val_met = met_data[:, split_idx:, :]
    val_pol = pol_data[:, split_idx:]
    val_mask = mask_data[:, split_idx:]

    train_pred_ds = PredictionDatasetWithFuture(
        train_met, train_pol, train_mask,
        cache_file=f"dataset_cache_train_T{T_short}_P{pred_len}.pt",
        R_stations=1, num_iterations=1, train_start_offset=0
    )
    val_pred_ds = PredictionDatasetWithFuture(
        val_met, val_pol, val_mask,
        cache_file=f"dataset_cache_val_T{T_short}_P{pred_len}.pt",
        R_stations=1, num_iterations=1
    )

    light_encoder = CNN1DEncoder(in_channels=10, d_model=128).to(device)
    light_encoder.load_state_dict(torch.load("best_encoder_only.pth", map_location=device))

    bank_cache_train = f"memory_bank_train_T{T_short}.pt"
    bank_cache_val = f"memory_bank_val_T{T_short}.pt"

    print("Preparing Train Bank...")
    bank_tr = build_or_load_memory_bank(train_pred_ds, light_encoder, device, bank_cache_train, batch_size=8192)
    print("Preparing Val Bank...")
    bank_va = build_or_load_memory_bank(val_pred_ds, light_encoder, device, bank_cache_val, batch_size=8192)

    bank_vecs_tr, bank_stids_tr, bank_starts_tr = bank_tr
    bank_vecs_va, bank_stids_va, bank_starts_va = bank_va

    # Convert val start indices to global
    bank_starts_va_global = bank_starts_va + split_idx

    bank_vecs_global = torch.cat([bank_vecs_tr, bank_vecs_va], dim=0)
    bank_stids_global = torch.cat([bank_stids_tr, bank_stids_va], dim=0)
    bank_starts_global = torch.cat([bank_starts_tr, bank_starts_va_global], dim=0)

    bank_global = (bank_vecs_global, bank_stids_global, bank_starts_global)

    for tensor in bank_global:
        tensor.share_memory_()
    del train_pred_ds, val_pred_ds, train_met, val_met, data_set
    del bank_vecs_tr, bank_vecs_va, bank_tr, bank_va
    light_encoder.cpu()
    torch.cuda.empty_cache()

    print("Starting experiments...")

    total_experiments = 4
    batch_size = 2

    for batch_start in range(1, total_experiments + 1, batch_size):
        processes = []
        batch_end = min(batch_start + batch_size, total_experiments + 1)
        batch_num = (batch_start // batch_size) + 1

        print(f"Starting batch {batch_num} (experiments {batch_start} to {batch_end - 1})...")
        for i in range(batch_start, batch_end):
            p = mp.Process(target=run_experiment, args=(i, bank_global))
            p.start()
            processes.append(p)
            print(f"Started process {p.pid} for experiment {i}")

        for p in processes:
            p.join()

        print(f"Batch {batch_num} completed.")
        # Clear GPU cache
        torch.cuda.empty_cache()

    print("All experiments completed.")


if __name__ == '__main__':
    main()
