import os
import random
import csv
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tslearn.metrics import SoftDTWLossPyTorch
from mult_model import (
    ShortTermPredictorWithFuture,
    StaticProfileEncoder,
    get_base_skill_site_indices,
)
import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

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


def build_static_features(train_met, train_pol, train_mask, pol_mean, pol_std,
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
    site_groups = get_base_skill_site_indices()
    held_out_indices = set(
        site_groups["low"] + site_groups["mid"] + site_groups["high"]
    )
    fit_station_indices = [
        n for n in range(N) if n not in held_out_indices
    ]
    fit_bc_features = bc_features[fit_station_indices]
    mean_bc = fit_bc_features.mean(dim=0, keepdim=True)
    std_bc = fit_bc_features.std(dim=0, keepdim=True).clamp(min=1e-8)
    bc_features = (bc_features - mean_bc) / std_bc

    if os.path.exists(geo_csv_path):
        df_geo = pd.read_csv(geo_csv_path)
        a_features_np = df_geo.iloc[:, 1:].astype(float).values
        a_features = torch.tensor(a_features_np, dtype=torch.float32)
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


# Loss
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
        return (zero_val, zero_val.detach(), zero_val.detach(),
                zero_val.detach(), zero_val.detach())

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

    loss_divergence_seq = (
        (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) /
        targets_filtered.shape[1]
    )

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
    gamma_w = 2.4  # DTW 权重
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
    scaled_trend = (loss_trend_seq * scale_factors).mean().detach()
    scaled_dtw = (loss_divergence_seq * scale_factors).mean().detach()
    scaled_topk = (loss_topk_seq * scale_factors).mean().detach()
    return final_loss, scaled_base, scaled_trend, scaled_dtw, scaled_topk


def pm25_no_trend_loss_components(preds, targets, pol_mean, pol_std,
                                  gamma=0.25, k_ratio=0.1):
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds = preds[valid_mask]
    pm25_targets = pm25_targets[valid_mask]

    loss_base_seq = torch.abs(preds - pm25_targets).mean(dim=1)

    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds.unsqueeze(2)
    targets_3d = pm25_targets.unsqueeze(2)
    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    dtw_yy = sdtw_criterion(targets_3d, targets_3d)
    loss_dtw_seq = (
        (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) /
        pm25_targets.shape[1]
    )

    k_num = max(1, int(pm25_targets.shape[1] * k_ratio))
    topk_target_vals, topk_indices = torch.topk(pm25_targets, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds, 1, topk_indices)
    loss_topk_seq = torch.abs(topk_pred_vals - topk_target_vals).mean(dim=1)

    total_loss_seq = loss_base_seq + 2.4 * loss_dtw_seq + 0.3 * loss_topk_seq
    return loss_base_seq, loss_dtw_seq, loss_topk_seq, total_loss_seq


def pm25_validation_loss(preds, targets, pol_mean, pol_std, limit=2.0):
    loss_base_seq, loss_dtw_seq, loss_topk_seq, total_loss_seq = (
        pm25_no_trend_loss_components(
        preds, targets, pol_mean, pol_std
        )
    )
    scale_factors = torch.where(
        total_loss_seq > limit,
        limit / total_loss_seq,
        torch.ones_like(total_loss_seq)
    )
    scaled_base = (loss_base_seq * scale_factors).mean()
    scaled_dtw = (loss_dtw_seq * scale_factors).mean()
    scaled_topk = (loss_topk_seq * scale_factors).mean()
    scaled_total = (total_loss_seq * scale_factors).mean()
    return scaled_base, scaled_dtw, scaled_topk, scaled_total


def train_main_model(data_set, output_dir, cache_files, seed):
    T_short = 144
    pred_len = 48
    R_stations = 32
    d_profile = 256
    d_model = 256
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    warmup_start_lr = 1e-5
    warmup_epochs = 1
    scheduler_t_max = 100
    hours_per_year = 365 * 24

    exp_dir = output_dir
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(
        exp_dir, f"main_model_training_seed{seed}.log"
    ))
    sys.stdout = logger

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Process seed: {seed}")

    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std

    total_time_steps = met_data.shape[1]
    expected_time_steps = 4 * hours_per_year
    if total_time_steps != expected_time_steps:
        raise ValueError(
            f"期望最近四年共 {expected_time_steps} 个小时时间步，实际得到 {total_time_steps}。"
        )

    train_end = 2 * hours_per_year
    val_end = 3 * hours_per_year

    train_met_cpu = met_data[:, :train_end, :]
    train_pol_cpu = pol_data[:, :train_end]
    train_mask_cpu = mask_data[:, :train_end]

    # Compute on CPU to save VRAM
    static_features_global = build_static_features(
        train_met_cpu, train_pol_cpu, train_mask_cpu,
        pol_mean=pol_mean,
        pol_std=pol_std,
        geo_csv_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "station_features.csv",
        )
    ).to(device)

    train_met = train_met_cpu.to(device)
    train_pol = train_pol_cpu.to(device)
    train_mask = train_mask_cpu.to(device)

    val_met = met_data[:, train_end:val_end, :].to(device)
    val_pol = pol_data[:, train_end:val_end].to(device)
    val_mask = mask_data[:, train_end:val_end].to(device)

    train_cache_file = cache_files['train']
    val_cache_file = cache_files['val']

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

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed)
    train_pred_loader = DataLoader(
        train_pred_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=train_loader_generator,
    )
    val_pred_loader = DataLoader(
        val_pred_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    num_static_features = static_features_global.shape[1]
    encoder = StaticProfileEncoder(
        in_features=num_static_features,
        d_profile=d_profile,
        dropout=0.2
    ).to(device)

    predictor = ShortTermPredictorWithFuture(
        seq_history_len=T_short,
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=1e-6
    )

    ema_alpha = 0.25
    ema_no_trend_loss = None
    best_ema_no_trend_loss = float('inf')
    best_model_path = os.path.join(
        exp_dir, f"main_model_best_seed{seed}.pth"
    )
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    for epoch in range(epochs):
        encoder.train()
        predictor.train()

        prediction_loss_total = 0.0
        train_base_total, train_dtw_total, train_topk_total = 0.0, 0.0, 0.0

        for batch_index, (
                short_segs,
                short_masks,
                future_mets,
                targets,
                station_ids,
        ) in enumerate(train_pred_loader):
            if epoch < warmup_epochs:
                warmup_steps = warmup_epochs * len(train_pred_loader)
                completed_steps = epoch * len(train_pred_loader) + batch_index
                warmup_progress = completed_steps / max(warmup_steps - 1, 1)
                current_lr = warmup_start_lr + (
                    max_lr - warmup_start_lr
                ) * warmup_progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            short_segs = short_segs.squeeze(0).to(device)
            short_masks = short_masks.squeeze(0).to(device)
            future_mets = future_mets.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            station_ids = station_ids.squeeze(0).to(device)

            optimizer.zero_grad()

            batch_static_feats = static_features_global[station_ids]
            profiles = encoder(batch_static_feats)

            preds = predictor(short_segs, profiles, future_met=future_mets, mask=short_masks)

            loss, l_base, _, l_dtw, l_topk = pm25_loss(
                preds,
                targets,
                pol_mean=pol_mean,
                pol_std=pol_std,
            )
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

                lv_base, lv_dtw, lv_topk, loss_v = pm25_validation_loss(
                    preds,
                    targets,
                    pol_mean=pol_mean,
                    pol_std=pol_std,
                )
                val_loss_total += loss_v.item()
                val_base_total += lv_base.item()
                val_dtw_total += lv_dtw.item()
                val_topk_total += lv_topk.item()

        num_val_batches = max(len(val_pred_loader), 1)
        avg_val_loss = val_loss_total / num_val_batches
        avg_v_base = val_base_total / num_val_batches
        avg_v_dtw = val_dtw_total / num_val_batches
        avg_v_topk = val_topk_total / num_val_batches
        if ema_no_trend_loss is None:
            ema_no_trend_loss = avg_val_loss
        else:
            ema_no_trend_loss = (
                ema_alpha * avg_val_loss
                + (1.0 - ema_alpha) * ema_no_trend_loss
            )

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Mixed Loss (Trend, limit=2): {avg_prediction_loss:.4f} "
              f"(Base: {avg_t_base:.4f}, DTW: {avg_t_dtw:.4f}, TopK: {avg_t_topk:.4f}) | "
              f"Val Mixed Loss (NoTrend, limit=2): {avg_val_loss:.4f} "
              f"(Base: {avg_v_base:.4f}, DTW: {avg_v_dtw:.4f}, TopK: {avg_v_topk:.4f}) | "
              f"EMA NoTrend Loss: {ema_no_trend_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if ema_no_trend_loss < best_ema_no_trend_loss:
            torch.save({
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'seed': seed,
                'epoch': epoch + 1,
                'val_no_trend_loss': avg_val_loss,
                'ema_no_trend_loss': ema_no_trend_loss,
            }, best_model_path)
            best_ema_no_trend_loss = ema_no_trend_loss

    if not os.path.exists(best_model_path):
        raise RuntimeError("训练结束后未生成最佳模型权重。")

    best_checkpoint = torch.load(best_model_path, map_location=device)
    encoder.load_state_dict(best_checkpoint['encoder'])
    predictor.load_state_dict(best_checkpoint['predictor'])
    encoder.eval()
    predictor.eval()
    print(
        f"[*] Loaded best model from epoch {best_checkpoint['epoch']} "
        f"(EMA NoTrend Loss: {best_checkpoint['ema_no_trend_loss']:.6f})"
    )

    logger.close()
    return {
        'encoder': encoder,
        'predictor': predictor,
        'static_features': static_features_global,
        'device': device,
        'best_model_path': best_model_path,
        'pol_mean': pol_mean,
        'pol_std': pol_std,
        'T_short': T_short,
        'pred_len': pred_len,
        'seed': seed,
    }


def sample_cached_windows(cache_file, sample_fraction, random_seed):
    """Sample an exact fraction of all cached (station, start) test windows."""
    cache_data = torch.load(cache_file, map_location='cpu')
    valid_starts = cache_data['valid_starts']
    station_counts = np.asarray(
        [len(starts) for starts in valid_starts], dtype=np.int64
    )
    cumulative_counts = np.cumsum(station_counts)
    total_windows = int(cumulative_counts[-1]) if len(cumulative_counts) else 0

    sampled_count = int(total_windows * sample_fraction)
    rng = np.random.default_rng(random_seed)
    flat_indices = rng.choice(
        total_windows, size=sampled_count, replace=False
    ).astype(np.int64, copy=False)
    flat_indices.sort()

    station_ids = np.searchsorted(
        cumulative_counts, flat_indices, side='right'
    ).astype(np.int64, copy=False)
    previous_counts = np.zeros_like(flat_indices)
    has_previous = station_ids > 0
    previous_counts[has_previous] = cumulative_counts[
        station_ids[has_previous] - 1
    ]
    indices_within_station = flat_indices - previous_counts

    starts = np.empty(sampled_count, dtype=np.int64)
    unique_stations, first_positions, selected_counts = np.unique(
        station_ids, return_index=True, return_counts=True
    )
    for station_id, first, count in zip(
            unique_stations, first_positions, selected_counts
    ):
        selected_indices = torch.from_numpy(
            indices_within_station[first:first + count]
        )
        starts[first:first + count] = valid_starts[int(station_id)][
            selected_indices
        ].numpy()

    return station_ids, starts, total_windows


def evaluate_random_test_subset(
        data_set,
        test_cache_files,
        training_result,
        report_file,
        sample_fraction,
        random_seed=10001,
        batch_size=32,
):
    """Evaluate independent low/mid/high test subsets and write three rows."""
    device = training_result['device']
    encoder = training_result['encoder']
    predictor = training_result['predictor']
    static_features = training_result['static_features']
    T_short = training_result['T_short']
    pred_len = training_result['pred_len']
    pol_mean = training_result['pol_mean']
    pol_std = training_result['pol_std']

    test_start = 3 * 365 * 24
    test_met = data_set.met_data_normalized[:, test_start:, :].to(device)
    test_pol = data_set.pol_data_normalized[:, test_start:].to(device)
    test_mask = data_set.pol_mask_matrix[:, test_start:].to(device)
    x_data = torch.cat([test_met, test_pol.unsqueeze(-1)], dim=-1)

    encoder.eval()
    predictor.eval()
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    fieldnames = [
        'skill_group',
        'mean_base_loss',
        'mean_dtw_loss',
        'mean_topk_loss',
        'mean_weighted_total_loss',
        'sampled_count',
        'evaluated_count',
        'filtered_count',
        'available_test_windows',
        'sample_fraction',
        'random_seed',
    ]
    rows = []
    for skill_group in ("low", "mid", "high"):
        station_ids_np, starts_np, total_test_windows = sample_cached_windows(
            test_cache_files[skill_group], sample_fraction, random_seed
        )

        total_base_loss = 0.0
        total_dtw_loss = 0.0
        total_topk_loss = 0.0
        total_weighted_loss = 0.0
        evaluated_count = 0
        filtered_count = 0

        with torch.no_grad():
            for batch_start in range(0, len(station_ids_np), batch_size):
                batch_end = min(batch_start + batch_size, len(station_ids_np))
                batch_station_ids_np = station_ids_np[batch_start:batch_end]
                batch_starts_np = starts_np[batch_start:batch_end]

                station_ids = torch.from_numpy(
                    batch_station_ids_np
                ).to(device=device, dtype=torch.long)
                starts = torch.from_numpy(
                    batch_starts_np
                ).to(device=device, dtype=torch.long)

                sid_idx = station_ids.unsqueeze(1)
                short_idx = starts.unsqueeze(1) + torch.arange(
                    T_short, device=device
                )
                target_idx = (starts + T_short).unsqueeze(1) + torch.arange(
                    pred_len, device=device
                )

                short_segs = x_data[sid_idx, short_idx, :]
                short_masks = test_mask[sid_idx, short_idx]
                targets = x_data[sid_idx, target_idx, :]
                future_mets = targets[:, :, :-1]

                profiles = encoder(static_features[station_ids])
                preds = predictor(
                    short_segs,
                    profiles,
                    future_met=future_mets,
                    mask=short_masks,
                )
                (
                    base_losses,
                    dtw_losses,
                    topk_losses,
                    weighted_losses,
                ) = pm25_no_trend_loss_components(
                    preds, targets, pol_mean=pol_mean, pol_std=pol_std
                )

                pm25_targets = targets[:, :, -1]
                pm25_targets_raw = pm25_targets * pol_std + pol_mean
                valid_mask = (pm25_targets_raw <= 1000).all(dim=1)
                valid_count = int(valid_mask.sum().item())
                evaluated_count += valid_count
                filtered_count += (batch_end - batch_start) - valid_count
                total_base_loss += base_losses.sum().item()
                total_dtw_loss += dtw_losses.sum().item()
                total_topk_loss += topk_losses.sum().item()
                total_weighted_loss += weighted_losses.sum().item()

        if evaluated_count > 0:
            mean_base_loss = total_base_loss / evaluated_count
            mean_dtw_loss = total_dtw_loss / evaluated_count
            mean_topk_loss = total_topk_loss / evaluated_count
            mean_weighted_total_loss = (
                total_weighted_loss / evaluated_count
            )
        else:
            mean_base_loss = float('nan')
            mean_dtw_loss = float('nan')
            mean_topk_loss = float('nan')
            mean_weighted_total_loss = float('nan')

        rows.append({
            'skill_group': skill_group,
            'mean_base_loss': mean_base_loss,
            'mean_dtw_loss': mean_dtw_loss,
            'mean_topk_loss': mean_topk_loss,
            'mean_weighted_total_loss': mean_weighted_total_loss,
            'sampled_count': len(station_ids_np),
            'evaluated_count': evaluated_count,
            'filtered_count': filtered_count,
            'available_test_windows': total_test_windows,
            'sample_fraction': sample_fraction,
            'random_seed': random_seed,
        })

    with open(report_file, 'w', newline='', encoding='utf-8-sig') as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                'mean_base_loss': f"{row['mean_base_loss']:.4f}",
                'mean_dtw_loss': f"{row['mean_dtw_loss']:.4f}",
                'mean_topk_loss': f"{row['mean_topk_loss']:.4f}",
                'mean_weighted_total_loss': (
                    f"{row['mean_weighted_total_loss']:.4f}"
                ),
            })

    return {
        row['skill_group']: {**row, 'report_file': report_file}
        for row in rows
    }
