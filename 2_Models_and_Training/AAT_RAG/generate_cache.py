import os
import torch
import numpy as np
from tqdm import tqdm
from mult_model import DataSet


def max_consecutive_repeat_length(windows, eps=1e-4):
    """Return each row's longest run of adjacent approximately equal values."""
    if windows.shape[1] == 0:
        return torch.zeros(windows.shape[0], dtype=torch.long, device=windows.device)

    current_run = torch.ones(windows.shape[0], dtype=torch.long, device=windows.device)
    longest_run = current_run.clone()
    for index in range(1, windows.shape[1]):
        same_as_previous = torch.abs(windows[:, index] - windows[:, index - 1]) < eps
        current_run = torch.where(
            same_as_previous,
            current_run + 1,
            torch.ones_like(current_run),
        )
        longest_run = torch.maximum(longest_run, current_run)
    return longest_run


def generate_and_save_cache(
        met_data,
        pol_data,
        mask_data,
        T_short,
        pred_len,
        cache_file,
        pol_mean,
        pol_std,
        max_pm25=1000.0,
):
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
                s_max_run = max_consecutive_repeat_length(chunk_short_windows, eps=eps)

                target_starts = chunk_starts + T_short
                target_idx = target_starts.unsqueeze(1) + torch.arange(pred_len)
                chunk_target_windows = pol_n[target_idx]
                t_max_run = max_consecutive_repeat_length(chunk_target_windows, eps=eps)

                target_windows_raw = chunk_target_windows * pol_std + pol_mean
                target_quality_mask = (target_windows_raw <= max_pm25).all(dim=1)

                chunk_mask = (
                    (s_max_run <= max_short_repeats)
                    & (t_max_run <= max_target_repeats)
                    & target_quality_mask
                )
                fraud_mask_list.append(chunk_mask)

            fraud_mask = torch.cat(fraud_mask_list, dim=0)
            valid_starts = valid_starts[fraud_mask]

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


def get_split_cache_paths(output_dir, T_short=144, pred_len=48, seed=None):
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    cache_dir = os.path.join(output_dir, f"cache{seed_suffix}")
    return {
        "cache_dir": cache_dir,
        "train": os.path.join(
            cache_dir,
            f"dataset_cache_train_y1_y2_T{T_short}_P{pred_len}{seed_suffix}.pt",
        ),
        "val": os.path.join(
            cache_dir,
            f"dataset_cache_val_y3_T{T_short}_P{pred_len}{seed_suffix}.pt",
        ),
        "test": os.path.join(
            cache_dir,
            f"dataset_cache_test_y4_T{T_short}_P{pred_len}{seed_suffix}.pt",
        ),
    }


def generate_split_caches(
        output_dir, data_set=None, T_short=144, pred_len=48, seed=None
):
    """Generate one run's private train/validation/test window caches."""
    os.makedirs(output_dir, exist_ok=True)
    cache_paths = get_split_cache_paths(
        output_dir, T_short=T_short, pred_len=pred_len, seed=seed
    )
    os.makedirs(cache_paths["cache_dir"], exist_ok=True)

    print(f"Starting private cache generation in: {cache_paths['cache_dir']}")
    if data_set is None:
        print("Loading raw data...")
        data_set = DataSet('data_matrix.npy')

    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    train_end = data_set.train_end
    val_end = data_set.val_end

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
    generate_and_save_cache(
        train_met, train_pol, train_mask, T_short, pred_len, cache_paths["train"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    print("Processing validation set...")
    generate_and_save_cache(
        val_met, val_pol, val_mask, T_short, pred_len, cache_paths["val"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    print("Processing test set...")
    generate_and_save_cache(
        test_met, test_pol, test_mask, T_short, pred_len, cache_paths["test"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    return cache_paths


if __name__ == "__main__":
    raise SystemExit(
        "Run the complete seeded workflow with "
        "`python run_parallel_pipeline.py --seeds SEED [SEED ...]`."
    )
