import os
import torch
import numpy as np
from tqdm import tqdm
from mult_model import DataSet


def generate_and_save_cache(met_data, pol_data, mask_data, T_short, pred_len, cache_file):
    N_stations = met_data.shape[0]
    T_total = met_data.shape[1]
    total_window = T_short + pred_len

    valid_starts_per_station = []
    eps = 1e-4

    for n in tqdm(range(N_stations), desc=f"扫描站点 (总时间步={T_total})"):
        mask_n = mask_data[n]
        pol_n = pol_data[n]

        max_start = T_total - total_window
        if max_start <= 0:
            valid_starts_per_station.append(torch.tensor([], dtype=torch.long))
            continue

        cumsum = torch.cumsum(mask_n, dim=0)
        start_indices = torch.arange(0, max_start + 1)

        sp_start = start_indices
        sp_end = sp_start + total_window - 1

        window_sums = cumsum[sp_end].clone()
        need_sub = (sp_start > 0)
        window_sums[need_sub] -= cumsum[sp_start[need_sub] - 1]

        valid_mask = (window_sums >= total_window - 0.5)
        valid_starts = start_indices[valid_mask]

        # Fraud filter
        if len(valid_starts) > 0:
            short_starts = valid_starts
            max_short_repeats = T_short // 6
            max_target_repeats = pred_len // 6
            chunk_size = 72
            fraud_mask_list = []

            for i in range(0, len(short_starts), chunk_size):
                chunk_starts = short_starts[i: i + chunk_size]
                short_idx = chunk_starts.unsqueeze(1) + torch.arange(T_short)
                chunk_short_windows = pol_n[short_idx]
                s_diff = torch.abs(chunk_short_windows.unsqueeze(2) - chunk_short_windows.unsqueeze(1))
                s_max_freq = (s_diff < eps).sum(dim=2).max(dim=1).values

                target_starts = chunk_starts + T_short
                target_idx = target_starts.unsqueeze(1) + torch.arange(pred_len)
                chunk_target_windows = pol_n[target_idx]
                t_diff = torch.abs(chunk_target_windows.unsqueeze(2) - chunk_target_windows.unsqueeze(1))
                t_max_freq = (t_diff < eps).sum(dim=2).max(dim=1).values

                chunk_mask = (s_max_freq <= max_short_repeats) & (t_max_freq <= max_target_repeats)
                fraud_mask_list.append(chunk_mask)

            fraud_mask = torch.cat(fraud_mask_list, dim=0)
            valid_starts = valid_starts[fraud_mask]

        valid_starts_per_station.append(valid_starts)

    eligible_stations = []
    for n in range(N_stations):
        if len(valid_starts_per_station[n]) >= 1:
            eligible_stations.append(n)
    eligible_stations = torch.tensor(eligible_stations, dtype=torch.long)

    torch.save({
        'valid_starts': valid_starts_per_station,
        'eligible_stations': eligible_stations
    }, cache_file)


if __name__ == "__main__":
    T_short = 144
    pred_len = 48

    print("\n[1/3] 正在载入原始数据...")
    data_set = DataSet('data_matrix.npy')
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    total_time_steps = met_data.shape[1]
    split_idx = int(total_time_steps * (3 / 4))

    train_met = met_data[:, :split_idx, :]
    train_pol = pol_data[:, :split_idx]
    train_mask = mask_data[:, :split_idx]

    val_met = met_data[:, split_idx:, :]
    val_pol = pol_data[:, split_idx:]
    val_mask = mask_data[:, split_idx:]

    print("\n[2/3] 正在处理【训练集】...")
    train_cache_file = f"dataset_cache_train_T{T_short}_P{pred_len}.pt"
    generate_and_save_cache(train_met, train_pol, train_mask, T_short, pred_len, train_cache_file)

    print("\n[3/3] 正在处理【验证集】...")
    val_cache_file = f"dataset_cache_val_T{T_short}_P{pred_len}.pt"
    generate_and_save_cache(val_met, val_pol, val_mask, T_short, pred_len, val_cache_file)