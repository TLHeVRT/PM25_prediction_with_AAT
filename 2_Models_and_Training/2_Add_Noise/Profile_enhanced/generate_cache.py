import os
import torch
from tqdm import tqdm
from mult_model import get_base_skill_site_indices


def max_consecutive_repeat_count(windows, eps=1e-4):
    """Return the longest consecutive near-equal run in each row."""
    batch_size, window_len = windows.shape
    if window_len == 0:
        return torch.zeros(
            batch_size, dtype=torch.long, device=windows.device
        )

    current_runs = torch.ones(
        batch_size, dtype=torch.long, device=windows.device
    )
    max_runs = current_runs.clone()
    consecutive_equal = torch.abs(
        windows[:, 1:] - windows[:, :-1]
    ) < eps
    for step in range(window_len - 1):
        current_runs = torch.where(
            consecutive_equal[:, step],
            current_runs + 1,
            torch.ones_like(current_runs),
        )
        max_runs = torch.maximum(max_runs, current_runs)
    return max_runs


def generate_and_save_cache(
        met_data,
        pol_data,
        mask_data,
        T_short,
        pred_len,
        cache_file,
        pol_mean,
        pol_std,
        station_indices,
):
    N_stations = met_data.shape[0]
    T_total = met_data.shape[1]
    total_window = T_short + pred_len

    valid_starts_per_station = [
        torch.tensor([], dtype=torch.long) for _ in range(N_stations)
    ]
    eps = 1e-4

    for n in tqdm(station_indices, desc=f"扫描站点 (总时间步={T_total})"):
        mask_n = mask_data[n]
        pol_n = pol_data[n]

        max_start = T_total - total_window
        if max_start <= 0:
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
            max_short_run = T_short // 6
            max_target_run = pred_len // 6
            chunk_size = 72
            fraud_mask_list = []

            for i in range(0, len(short_starts), chunk_size):
                chunk_starts = short_starts[i: i + chunk_size]
                short_idx = chunk_starts.unsqueeze(1) + torch.arange(T_short)
                chunk_short_windows = pol_n[short_idx]
                s_max_run = max_consecutive_repeat_count(
                    chunk_short_windows, eps=eps
                )

                target_starts = chunk_starts + T_short
                target_idx = target_starts.unsqueeze(1) + torch.arange(pred_len)
                chunk_target_windows = pol_n[target_idx]
                t_max_run = max_consecutive_repeat_count(
                    chunk_target_windows, eps=eps
                )
                chunk_target_raw = (
                    chunk_target_windows * pol_std + pol_mean
                )

                chunk_mask = (
                    (s_max_run <= max_short_run)
                    & (t_max_run <= max_target_run)
                    & (chunk_target_raw <= 1000).all(dim=1)
                )
                fraud_mask_list.append(chunk_mask)

            fraud_mask = torch.cat(fraud_mask_list, dim=0)
            valid_starts = valid_starts[fraud_mask]

        valid_starts_per_station[n] = valid_starts

    eligible_stations = [
        n for n in station_indices if len(valid_starts_per_station[n]) >= 1
    ]
    eligible_stations = torch.tensor(eligible_stations, dtype=torch.long)

    torch.save({
        'valid_starts': valid_starts_per_station,
        'eligible_stations': eligible_stations
    }, cache_file)


def generate_split_caches(
        data_set, output_dir, seed, T_short=144, pred_len=48
):
    hours_per_year = 365 * 24

    os.makedirs(output_dir, exist_ok=True)
    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    total_time_steps = met_data.shape[1]
    expected_time_steps = 4 * hours_per_year
    if total_time_steps != expected_time_steps:
        raise ValueError(
            f"期望最近四年共 {expected_time_steps} 个小时时间步，实际得到 {total_time_steps}。"
        )

    train_end = 2 * hours_per_year
    val_end = 3 * hours_per_year
    site_groups = get_base_skill_site_indices()
    test_station_indices = {
        group: sorted(site_groups[group]) for group in ("low", "mid", "high")
    }
    test_station_set = set(
        site_groups["low"] + site_groups["mid"] + site_groups["high"]
    )
    train_station_indices = [
        n for n in range(met_data.shape[0]) if n not in test_station_set
    ]

    train_met = met_data[:, :train_end, :]
    train_pol = pol_data[:, :train_end]
    train_mask = mask_data[:, :train_end]

    val_met = met_data[:, train_end:val_end, :]
    val_pol = pol_data[:, train_end:val_end]
    val_mask = mask_data[:, train_end:val_end]

    test_met = met_data[:, val_end:, :]
    test_pol = pol_data[:, val_end:]
    test_mask = mask_data[:, val_end:]

    cache_files = {
        'train': os.path.join(
            output_dir,
            f"dataset_cache_train_2y_T{T_short}_P{pred_len}_seed{seed}.pt",
        ),
        'val': os.path.join(
            output_dir,
            f"dataset_cache_val_y3_T{T_short}_P{pred_len}_seed{seed}.pt",
        ),
        'test': {
            group: os.path.join(
                output_dir,
                f"dataset_cache_test_{group}_y4_"
                f"T{T_short}_P{pred_len}_seed{seed}.pt",
            )
            for group in ("low", "mid", "high")
        },
    }

    print("\n[1/3] 正在处理【训练集：第1-2年】...")
    train_cache_file = cache_files['train']
    generate_and_save_cache(
        train_met,
        train_pol,
        train_mask,
        T_short,
        pred_len,
        train_cache_file,
        pol_mean=data_set.pol_mean,
        pol_std=data_set.pol_std,
        station_indices=train_station_indices,
    )

    print("\n[2/3] 正在处理【验证集：第3年】...")
    val_cache_file = cache_files['val']
    generate_and_save_cache(
        val_met,
        val_pol,
        val_mask,
        T_short,
        pred_len,
        val_cache_file,
        pol_mean=data_set.pol_mean,
        pol_std=data_set.pol_std,
        station_indices=train_station_indices,
    )

    for skill_group in ("low", "mid", "high"):
        print(f"\n正在处理【{skill_group} 测试集：第4年】...")
        generate_and_save_cache(
            test_met,
            test_pol,
            test_mask,
            T_short,
            pred_len,
            cache_files['test'][skill_group],
            pol_mean=data_set.pol_mean,
            pol_std=data_set.pol_std,
            station_indices=test_station_indices[skill_group],
        )

    return cache_files
