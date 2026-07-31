import os

import random
import csv
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tslearn.metrics import SoftDTWLossPyTorch
from mult_model import (
    HOURS_PER_YEAR,
    ShortTermPredictorWithFuture,
    StaticProfileEncoder,
)
import sys


class Logger(object):
    """Mirror stdout to both the terminal and one training log file."""

    def __init__(self, filename="log.txt"):
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
    """Build station profiles from the last historical year and geographic data."""
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

        met_mask = m_c.unsqueeze(-1)
        met_sum = (met_c * met_mask).sum(dim=1)
        met_count = met_mask.sum(dim=1).clamp(min=1)
        met_means.append(met_sum / met_count)

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

    else:
        print(f"Geo feature file '{geo_csv_path}' not found. Using zero placeholder.")
        a_features = torch.zeros(N, 13)

    final_static_features = torch.cat([a_features, bc_features], dim=1)
    return final_static_features


def _temporal_candidate_bounds(sorted_starts, query_start, total_length):
    """Find the slice of candidate starts allowed for one target window."""
    query_start_value = int(query_start.item())
    if query_start_value + total_length <= HOURS_PER_YEAR:
        # Baseline special case: year-1 queries draw non-overlapping windows
        # from year 2, including only starts before the end of year 2.
        lower_bound = max(
            HOURS_PER_YEAR,
            query_start_value + total_length,
        )
        upper_bound = 2 * HOURS_PER_YEAR
        left = torch.searchsorted(
            sorted_starts,
            sorted_starts.new_tensor(lower_bound),
            right=False,
        ).item()
        right = torch.searchsorted(
            sorted_starts,
            sorted_starts.new_tensor(upper_bound),
            right=False,
        ).item()
    else:
        # Baseline standard case: the complete auxiliary window must end no
        # later than the query start.
        cutoff = query_start_value - total_length
        left = 0
        right = torch.searchsorted(
            sorted_starts,
            sorted_starts.new_tensor(cutoff),
            right=True,
        ).item()
    return int(left), int(right)


def retrieve_random_same_station_sequences(
        station_ids,
        starts_batch,
        station_starts,
        x_data,
        T_long,
        T_pred,
        sample_count=10,
):
    """Randomly select temporally valid auxiliaries from each target station."""
    total_length = T_long + T_pred
    device = x_data.device
    selected_starts = []

    for station_id, query_start in zip(station_ids, starts_batch):
        station_id_value = int(station_id.item())
        candidate_starts = station_starts[station_id_value]
        left, right = _temporal_candidate_bounds(
            candidate_starts,
            query_start,
            total_length,
        )
        offsets = random.sample(range(right - left), sample_count)
        positions = torch.tensor(
            offsets,
            dtype=torch.long,
            device=candidate_starts.device,
        ) + left
        selected_starts.append(candidate_starts[positions])

    auxiliary_starts = torch.stack(selected_starts, dim=0).to(device)
    auxiliary_station_ids = station_ids.unsqueeze(1).expand(-1, sample_count)
    time_offsets = torch.arange(total_length, device=device).view(1, 1, -1)
    sequence_indices = auxiliary_starts.unsqueeze(2) + time_offsets
    sequence_station_ids = auxiliary_station_ids.unsqueeze(2).expand(
        -1, -1, total_length
    )
    matched_sequences = x_data[
        sequence_station_ids,
        sequence_indices,
        :,
    ]
    return matched_sequences, auxiliary_station_ids


class PredictionDatasetWithFuture(Dataset):
    """Sample stations and valid prediction windows from one temporal split."""

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

        return short_segs, short_masks, future_mets, targets, station_ids, starts_tensor


class RandomPredictionSubset(Dataset):
    """A fixed random subset of all cached station/window pairs."""

    def __init__(
            self,
            met_data,
            pol_data,
            mask_data,
            cache_file,
            fraction=0.1,
            seed=0,
            T_short=144,
            pred_len=48,
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        self.T_short = T_short
        self.pred_len = pred_len
        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data

        cache_data = torch.load(cache_file, map_location="cpu")
        valid_starts = cache_data["valid_starts"]
        lengths = torch.tensor([len(starts) for starts in valid_starts], dtype=torch.long)
        total_samples = int(lengths.sum().item())
        if total_samples == 0:
            raise RuntimeError(f"No valid samples in cache: {cache_file}")

        sample_count = max(1, int(round(total_samples * fraction)))
        generator = torch.Generator().manual_seed(seed)
        sampled_flat_indices = torch.randperm(total_samples, generator=generator)[:sample_count]

        cumulative_lengths = torch.cumsum(lengths, dim=0)
        self.station_ids = torch.searchsorted(
            cumulative_lengths, sampled_flat_indices, right=True
        )
        all_starts = torch.cat(valid_starts)
        self.starts = all_starts[sampled_flat_indices]
        self.total_candidate_samples = total_samples
        self.sample_count = sample_count

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        station_id = self.station_ids[index]
        start = self.starts[index]
        device = self.x_data.device
        station_id_dev = station_id.to(device)
        start_dev = start.to(device)

        short_idx = start_dev + torch.arange(self.T_short, device=device)
        target_idx = start_dev + self.T_short + torch.arange(self.pred_len, device=device)
        short_seg = self.x_data[station_id_dev, short_idx, :]
        short_mask = self.mask_data[station_id_dev, short_idx]
        target = self.x_data[station_id_dev, target_idx, :]
        future_met = target[:, :-1]
        return short_seg, short_mask, future_met, target, station_id, start


def pm25_loss(
        preds,
        targets,
        pol_mean,
        pol_std,
        gamma=0.25,
        k_ratio=0.1,
        include_trend=True,
        apply_scale=True,
        limit=2.0,
):
    """Compute the filtered mixed PM2.5 loss with optional trend and scaling."""
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = pm25_targets[valid_mask]

    if preds_filtered.shape[0] == 0:
        zero_val = preds.sum() * 0.0
        zero_detached = zero_val.detach()
        return zero_val, zero_detached, zero_detached, zero_detached, zero_detached

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
                     (gamma_w * loss_divergence_seq) + \
                     (delta_w * loss_topk_seq)
    if include_trend:
        total_loss_seq = total_loss_seq + beta * loss_trend_seq

    if apply_scale:
        scale_factors = torch.where(
            total_loss_seq > limit,
            limit / total_loss_seq.detach(),
            torch.ones_like(total_loss_seq)
        )
    else:
        scale_factors = torch.ones_like(total_loss_seq)

    final_loss = (total_loss_seq * scale_factors).mean()
    base_component = (loss_base_seq * scale_factors).mean().detach()
    if include_trend:
        trend_component = (loss_trend_seq * scale_factors).mean().detach()
    else:
        trend_component = torch.zeros_like(base_component)
    dtw_component = (loss_divergence_seq * scale_factors).mean().detach()
    topk_component = (loss_topk_seq * scale_factors).mean().detach()

    return final_loss, base_component, trend_component, dtw_component, topk_component


def test_loss(preds, targets, pol_mean, pol_std, gamma=0.25, k_ratio=0.1):
    """No scale and no trend loss!"""
    pm25_targets = targets[:, :, -1]
    pm25_targets_raw = pm25_targets * pol_std + pol_mean
    valid_mask = (pm25_targets_raw <= 1000).all(dim=1)

    preds_filtered = preds[valid_mask]
    targets_filtered = pm25_targets[valid_mask]
    if preds_filtered.shape[0] == 0:
        empty = preds.new_empty((0,))
        return valid_mask, empty, empty, empty, empty

    base_seq = torch.abs(preds_filtered - targets_filtered).mean(dim=1)

    sdtw_criterion = SoftDTWLossPyTorch(gamma=gamma)
    preds_3d = preds_filtered.unsqueeze(2)
    targets_3d = targets_filtered.unsqueeze(2)
    dtw_xy = sdtw_criterion(preds_3d, targets_3d)
    dtw_xx = sdtw_criterion(preds_3d, preds_3d)
    dtw_yy = sdtw_criterion(targets_3d, targets_3d)
    seq_len = targets_filtered.shape[1]
    dtw_seq = (dtw_xy - 0.5 * (dtw_xx + dtw_yy)).clamp(min=0.0) / seq_len

    k_num = max(1, int(seq_len * k_ratio))
    topk_target_vals, topk_indices = torch.topk(targets_filtered, k=k_num, dim=1)
    topk_pred_vals = torch.gather(preds_filtered, 1, topk_indices)
    topk_seq = torch.abs(topk_pred_vals - topk_target_vals).mean(dim=1)

    weighted_seq = base_seq + 2.4 * dtw_seq + 0.3 * topk_seq

    return (
        valid_mask,
        base_seq.detach(),
        dtw_seq.detach(),
        topk_seq.detach(),
        weighted_seq.detach(),
    )


def add_continuous_noise_mask(aux_seqs):
    """Add continuous PM2.5 noise spans to randomly selected auxiliary sequences."""
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




def prepare_global_index_bank(
        cache_paths,
        data_set,
):
    """Build sorted four-year valid-start indices for every station."""
    train_end = data_set.train_end
    val_end = data_set.val_end
    split_specs = (
        ("train", 0),
        ("val", train_end),
        ("test", val_end),
    )
    split_starts = []
    for split_name, offset in split_specs:
        cache_data = torch.load(cache_paths[split_name], map_location="cpu")
        starts_for_split = [
            starts.to(dtype=torch.long) + offset
            for starts in cache_data["valid_starts"]
        ]
        split_starts.append(starts_for_split)

    station_count = len(split_starts[0])
    station_starts = []
    total_records = 0
    for station_id in range(station_count):
        combined = torch.cat(
            [starts[station_id] for starts in split_starts],
            dim=0,
        )
        combined = torch.sort(combined).values
        station_starts.append(combined)
        total_records += combined.numel()

    print(
        "Prepared same-station random index bank: "
        f"stations={station_count}, records={total_records}."
    )
    return tuple(station_starts)




    
def train_main_model(
        run_id,
        index_bank,
        output_dir,
        cache_paths,
        data_set,
        device,
        seed=None,
):
    """Train the main model and save the EMA-selected validation checkpoint."""
    station_starts_global = index_bank

    T_short = 144
    pred_len = 48
    R_stations = 16
    d_model = 256
    in_channels = 10
    epochs = 100
    max_lr = 2e-4
    T_0 = 100
    d_profile = 256

    exp_dir = output_dir
    os.makedirs(exp_dir, exist_ok=True)
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    logger = Logger(os.path.join(exp_dir, f"main_training{seed_suffix}.log"))
    sys.stdout = logger
    print(f"Starting experiment (seed={seed})")

    print(f"Device: {device}")

    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std

    train_end = data_set.train_end
    val_end = data_set.val_end

    train_met_cpu = met_data[:, :train_end, :]
    train_pol_cpu = pol_data[:, :train_end]
    train_mask_cpu = mask_data[:, :train_end]

    # The static station profile is fixed from year 2, the final training year.
    profile_start = train_end - HOURS_PER_YEAR

    static_features_global = build_static_features(
        train_met_cpu[:, profile_start:train_end, :],
        train_pol_cpu[:, profile_start:train_end],
        train_mask_cpu[:, profile_start:train_end],
        pol_mean=pol_mean,
        pol_std=pol_std,
        geo_csv_path="station_features.csv"
    ).to(device)

    train_met = train_met_cpu.to(device)
    train_pol = train_pol_cpu.to(device)
    train_mask = train_mask_cpu.to(device)

    val_met = met_data[:, train_end:val_end, :].to(device)
    val_pol = pol_data[:, train_end:val_end].to(device)
    val_mask = mask_data[:, train_end:val_end].to(device)

    train_pred_ds = PredictionDatasetWithFuture(
        train_met, train_pol, train_mask,
        cache_file=cache_paths["train"],
        T_short=T_short, pred_len=pred_len,
        R_stations=R_stations, num_iterations=1500
    )

    val_pred_ds = PredictionDatasetWithFuture(
        val_met, val_pol, val_mask,
        cache_file=cache_paths["val"],
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
        e_layers=4,
        dropout=0.1,
    ).to(device)
    all_params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.AdamW(all_params, lr=max_lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    ema_alpha = 0.25
    ema_val_loss_no_trend = None
    best_ema_val_loss_no_trend = float('inf')
    best_model_path = os.path.join(exp_dir, f"best_model{seed_suffix}.pth")

    for epoch in range(epochs):
        encoder.train()
        predictor.train()

        total_loss_sum = 0.0
        prediction_loss_total = 0.0
        aux_loss_total_sum = 0.0
        train_base_total = 0.0
        train_trend_total = 0.0
        train_dtw_total = 0.0
        train_topk_total = 0.0
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
                random_matched_seqs, auxiliary_station_ids = (
                    retrieve_random_same_station_sequences(
                        station_ids,
                        starts_batch,
                        station_starts_global,
                        global_x_data,
                        T_short,
                        pred_len,
                    )
                )

            masked_matched_seqs, mask_info = add_continuous_noise_mask(
                random_matched_seqs
            )

            batch_static_feats = static_features_global[station_ids]
            aux_static_feats = static_features_global[auxiliary_station_ids]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                profiles = encoder(batch_static_feats)
                aux_profiles = encoder(aux_static_feats)
                preds, aux_reconstruct = predictor(
                    short_segs, profiles, future_met=future_mets, mask=short_masks,
                    matched_hist=masked_matched_seqs, aux_profiles=aux_profiles
                )
                loss, l_base, l_trend, l_dtw, l_topk = pm25_loss(
                    preds,
                    targets,
                    pol_mean=pol_mean,
                    pol_std=pol_std,
                    include_trend=True,
                    apply_scale=True,
                )

                aux_loss_val = torch.tensor(0.0, device=device)
                if mask_info:
                    target_list = []
                    pred_list = []
                    length_list = []

                    for b, k, start, end, actual_length in mask_info:
                        target_list.append(random_matched_seqs[b, k, start:end, :])
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

                        a_loss, _, _, _, _ = pm25_loss(
                            pred_batch_masked,
                            target_batch,
                            pol_mean=pol_mean,
                            pol_std=pol_std,
                            include_trend=True,
                            apply_scale=False,
                        )
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
            train_trend_total += l_trend.item()
            train_dtw_total += l_dtw.item()
            train_topk_total += l_topk.item()

        scheduler.step()

        num_train_batches = max(len(train_pred_loader), 1)
        avg_total_loss = total_loss_sum / num_train_batches
        avg_aux_loss = aux_loss_total_sum / num_train_batches
        avg_prediction_loss = prediction_loss_total / num_train_batches
        avg_t_base = train_base_total / num_train_batches
        avg_t_trend = train_trend_total / num_train_batches
        avg_t_dtw = train_dtw_total / num_train_batches
        avg_t_topk = train_topk_total / num_train_batches

        # Validation
        encoder.eval()
        predictor.eval()
        val_loss_total = 0.0
        val_base_total, val_dtw_total, val_topk_total = 0.0, 0.0, 0.0
        with torch.inference_mode():
            for short_segs, short_masks, future_mets, targets, station_ids, starts_batch in val_pred_loader:
                short_segs = short_segs.squeeze(0).to(device)
                short_masks = short_masks.squeeze(0).to(device)
                future_mets = future_mets.squeeze(0).to(device)
                targets = targets.squeeze(0).to(device)
                station_ids = station_ids.squeeze(0).to(device)
                starts_batch = starts_batch.squeeze(0).to(device)

                # Year 3 uses global indices and may retrieve any earlier sequence.
                starts_batch_global = starts_batch + train_end
                random_matched_seqs, auxiliary_station_ids = (
                    retrieve_random_same_station_sequences(
                        station_ids,
                        starts_batch_global,
                        station_starts_global,
                        global_x_data,
                        T_short,
                        pred_len,
                    )
                )

                batch_static_feats = static_features_global[station_ids]
                aux_static_feats = static_features_global[auxiliary_station_ids]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    profiles = encoder(batch_static_feats)
                    aux_profiles = encoder(aux_static_feats)
                    preds, _ = predictor(
                        short_segs, profiles, future_met=future_mets, mask=short_masks,
                        matched_hist=random_matched_seqs, aux_profiles=aux_profiles
                    )
                loss_v, lv_base, _, lv_dtw, lv_topk = pm25_loss(
                    preds,
                    targets,
                    pol_mean=pol_mean,
                    pol_std=pol_std,
                    include_trend=False,
                    apply_scale=True,
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
        if ema_val_loss_no_trend is None:
            ema_val_loss_no_trend = avg_val_loss
        else:
            ema_val_loss_no_trend = (
                ema_alpha * avg_val_loss
                + (1.0 - ema_alpha) * ema_val_loss_no_trend
            )

        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Total Loss: {avg_total_loss:.4f} | Aux Loss: {avg_aux_loss:.4f} | "
              f"Train Mixed Loss: {avg_prediction_loss:.4f} "
              f"(Base: {avg_t_base:.4f}, Trend: {avg_t_trend:.4f}, "
              f"DTW: {avg_t_dtw:.4f}, TopK: {avg_t_topk:.4f}) | "
              f"Val Mixed Loss (No Trend): {avg_val_loss:.4f} "
              f"(Base: {avg_v_base:.4f}, DTW: {avg_v_dtw:.4f}, TopK: {avg_v_topk:.4f}) | "
              f"EMA Val Loss (No Trend): {ema_val_loss_no_trend:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save only the weights selected by the smoothed no-trend validation loss.
        if ema_val_loss_no_trend < best_ema_val_loss_no_trend:
            torch.save({
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'epoch': epoch + 1,
                'seed': seed,
                'val_loss_no_trend': avg_val_loss,
                'ema_val_loss_no_trend': ema_val_loss_no_trend,
            }, best_model_path)
            best_ema_val_loss_no_trend = ema_val_loss_no_trend
            print(
                f"Saved new best weights to {os.path.basename(best_model_path)} "
                f"(EMA validation loss without trend: {ema_val_loss_no_trend:.6f})."
            )

    logger.close()
    return best_model_path



def evaluate_random_test_subset(
        run_id,
        output_dir,
        model_path,
        index_bank,
        cache_paths,
        data_set,
        device,
        test_fraction=0.1,
        batch_size=16,
        T_short=144,
        pred_len=48,
        seed=None,
):
    """Evaluate an exact random fraction of cached test samples and write one CSV report."""
    if seed is None:
        seed = 100_000 + run_id
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix
    pol_mean = data_set.pol_mean
    pol_std = data_set.pol_std
    train_end = data_set.train_end
    val_end = data_set.val_end

    profile_start = train_end - HOURS_PER_YEAR
    static_features_global = build_static_features(
        met_data[:, profile_start:train_end, :],
        pol_data[:, profile_start:train_end],
        mask_data[:, profile_start:train_end],
        pol_mean=pol_mean,
        pol_std=pol_std,
        geo_csv_path="station_features.csv"
    ).to(device)

    test_met = met_data[:, val_end:, :].to(device)
    test_pol = pol_data[:, val_end:].to(device)
    test_mask = mask_data[:, val_end:].to(device)
    test_ds = RandomPredictionSubset(
        test_met, test_pol, test_mask,
        cache_file=cache_paths["test"],
        fraction=test_fraction,
        seed=seed,
        T_short=T_short,
        pred_len=pred_len,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    encoder = StaticProfileEncoder(
        in_features=static_features_global.shape[1], d_profile=256, dropout=0.2
    ).to(device)
    predictor = ShortTermPredictorWithFuture(
        seq_len_short=T_short,
        pred_len=pred_len,
        in_channels=10,
        met_channels=9,
        d_profile=256,
        d_model=256,
        n_heads=8,
        e_layers=4,
        dropout=0.1,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    encoder.eval()
    predictor.eval()

    station_starts_global = index_bank
    global_x_data = torch.cat(
        [met_data.to(device), pol_data.to(device).unsqueeze(-1)], dim=-1
    )

    base_sum = 0.0
    dtw_sum = 0.0
    topk_sum = 0.0
    total_loss_sum = 0.0
    valid_sample_count = 0
    with torch.inference_mode():
        for short_segs, short_masks, future_mets, targets, station_ids, starts_batch in test_loader:
            short_segs = short_segs.to(device)
            short_masks = short_masks.to(device)
            future_mets = future_mets.to(device)
            targets = targets.to(device)
            station_ids = station_ids.to(device)
            starts_batch = starts_batch.to(device)
            starts_batch_global = starts_batch + val_end

            matched_seqs, matched_station_ids = (
                retrieve_random_same_station_sequences(
                    station_ids,
                    starts_batch_global,
                    station_starts_global,
                    global_x_data,
                    T_short,
                    pred_len,
                )
            )
            profiles = encoder(static_features_global[station_ids])
            aux_profiles = encoder(static_features_global[matched_station_ids])
            preds, _ = predictor(
                short_segs,
                profiles,
                future_met=future_mets,
                mask=short_masks,
                matched_hist=matched_seqs,
                aux_profiles=aux_profiles,
            )

            valid_mask, base_seq, dtw_seq, topk_seq, total_seq = test_loss(
                preds, targets, pol_mean=pol_mean, pol_std=pol_std
            )
            valid_sample_count += int(valid_mask.sum().item())
            base_sum += base_seq.sum().item()
            dtw_sum += dtw_seq.sum().item()
            topk_sum += topk_seq.sum().item()
            total_loss_sum += total_seq.sum().item()

    if valid_sample_count == 0:
        raise RuntimeError("No valid test samples were produced for the loss report.")

    mean_base = base_sum / valid_sample_count
    mean_dtw = dtw_sum / valid_sample_count
    mean_topk = topk_sum / valid_sample_count
    mean_total_loss = total_loss_sum / valid_sample_count
    common_report_values = {
        "valid_sample_count": valid_sample_count,
        "sampled_sample_count": test_ds.sample_count,
        "total_candidate_samples": test_ds.total_candidate_samples,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    rows = [
        {
            "loss_item": "base",
            "nominal_weight": 1.0,
            "mean_loss": mean_base,
            "sum_loss": base_sum,
            **common_report_values,
        },
        {
            "loss_item": "dtw",
            "nominal_weight": 2.4,
            "mean_loss": mean_dtw,
            "sum_loss": dtw_sum,
            **common_report_values,
        },
        {
            "loss_item": "topk",
            "nominal_weight": 0.3,
            "mean_loss": mean_topk,
            "sum_loss": topk_sum,
            **common_report_values,
        },
        {
            "loss_item": "total_without_trend",
            "nominal_weight": "combined",
            "mean_loss": mean_total_loss,
            "sum_loss": total_loss_sum,
            **common_report_values,
        },
    ]

    report_path = os.path.join(output_dir, f"test_loss_report_seed{seed}.csv")
    fieldnames = list(rows[0].keys())
    with open(report_path, "w", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Run {run_id} test report: valid={valid_sample_count}, "
        f"sampled={test_ds.sample_count}, candidates={test_ds.total_candidate_samples}, "
        f"Base={mean_base:.6f}, DTW={mean_dtw:.6f}, TopK={mean_topk:.6f}, "
        f"No-trend total={mean_total_loss:.6f}, path={report_path}"
    )
    return report_path


def main():
    raise SystemExit(
        "Use `python run_parallel_pipeline_same_station_random.py "
        "--seeds SEED [SEED ...]`."
    )


if __name__ == '__main__':
    main()
