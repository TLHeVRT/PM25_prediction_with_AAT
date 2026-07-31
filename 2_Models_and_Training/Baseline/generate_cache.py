import os
import torch
from tqdm import tqdm
from mult_model import DataSet


def max_consecutive_repeat_count(windows, eps):
    positions = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0)
    positions = positions.expand(windows.shape[0], -1)
    changes = torch.cat([
        torch.ones((windows.shape[0], 1), dtype=torch.bool, device=windows.device),
        torch.abs(windows[:, 1:] - windows[:, :-1]) >= eps
    ], dim=1)
    run_starts = torch.where(changes, positions, torch.full_like(positions, -1))
    latest_run_starts = torch.cummax(run_starts, dim=1).values
    return (positions - latest_run_starts + 1).max(dim=1).values


def generate_and_save_cache(met_data, pol_data, mask_data, pol_mean, pol_std,
                            T_short, pred_len, cache_file):
    print(f"Generating cache: {cache_file}")
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

        # Quality filter
        if len(valid_starts) > 0:
            short_starts = valid_starts
            max_short_repeats = T_short // 6
            max_target_repeats = pred_len // 6
            chunk_size = 72
            quality_mask_list = []

            for i in range(0, len(short_starts), chunk_size):
                chunk_starts = short_starts[i: i + chunk_size]
                short_idx = chunk_starts.unsqueeze(1) + torch.arange(T_short)
                chunk_short_windows = pol_n[short_idx]
                s_max_repeats = max_consecutive_repeat_count(chunk_short_windows, eps)

                target_starts = chunk_starts + T_short
                target_idx = target_starts.unsqueeze(1) + torch.arange(pred_len)
                chunk_target_windows = pol_n[target_idx]
                t_max_repeats = max_consecutive_repeat_count(chunk_target_windows, eps)
                chunk_target_raw = chunk_target_windows * pol_std + pol_mean

                chunk_mask = ((s_max_repeats <= max_short_repeats) &
                              (t_max_repeats <= max_target_repeats) &
                              (chunk_target_raw <= 1000).all(dim=1))
                quality_mask_list.append(chunk_mask)

            quality_mask = torch.cat(quality_mask_list, dim=0)
            valid_starts = valid_starts[quality_mask]

        valid_starts_per_station.append(valid_starts)

    eligible_stations = []
    for n in range(N_stations):
        if len(valid_starts_per_station[n]) >= 1:
            eligible_stations.append(n)
    eligible_stations = torch.tensor(eligible_stations, dtype=torch.long)

    print(f"Eligible stations: {len(eligible_stations)}/{N_stations}")

    torch.save({
        'valid_starts': valid_starts_per_station,
        'eligible_stations': eligible_stations
    }, cache_file)
    print(f"Saved cache to: {cache_file}")


def generate_all_caches(data_path, cache_dir, seed, T_short=144, pred_len=48):
    print("Starting preprocessing and cache generation...")

    print("Loading raw data...")
    data_set = DataSet(data_path)
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    one_year_steps = 365 * 24
    train_end = one_year_steps * 2
    val_end = one_year_steps * 3

    train_met = met_data[:, :train_end, :]
    train_pol = pol_data[:, :train_end]
    train_mask = mask_data[:, :train_end]

    val_met = met_data[:, train_end:val_end, :]
    val_pol = pol_data[:, train_end:val_end]
    val_mask = mask_data[:, train_end:val_end]

    test_met = met_data[:, val_end:, :]
    test_pol = pol_data[:, val_end:]
    test_mask = mask_data[:, val_end:]

    print("Processing training set...")
    train_cache_file = os.path.join(
        cache_dir,
        f"dataset_cache_train_2y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )
    generate_and_save_cache(
        train_met, train_pol, train_mask, data_set.pol_mean, data_set.pol_std,
        T_short, pred_len, train_cache_file
    )

    print("Processing validation set...")
    val_cache_file = os.path.join(
        cache_dir,
        f"dataset_cache_val_1y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )
    generate_and_save_cache(
        val_met, val_pol, val_mask, data_set.pol_mean, data_set.pol_std,
        T_short, pred_len, val_cache_file
    )

    print("Processing test set...")
    test_cache_file = os.path.join(
        cache_dir,
        f"dataset_cache_test_1y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )
    generate_and_save_cache(
        test_met, test_pol, test_mask, data_set.pol_mean, data_set.pol_std,
        T_short, pred_len, test_cache_file
    )


if __name__ == "__main__":
    generate_all_caches('data_matrix.npy', '.', seed=0)
