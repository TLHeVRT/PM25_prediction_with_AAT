import os
import torch
from mult_model import DataSet


def rolling_sum(values, window_size):
    """Sum every fixed-width window along the time axis."""
    prefix = torch.cat([
        torch.zeros(
            values.shape[0], 1, dtype=values.dtype, device=values.device
        ),
        torch.cumsum(values, dim=1, dtype=values.dtype),
    ], dim=1)
    return prefix[:, window_size:] - prefix[:, :-window_size]


def rolling_count(values, window_size):
    """Count true values in every fixed-width window along the time axis."""
    return rolling_sum(values.to(torch.int32), window_size)


def windows_with_excessive_repeats(
        adjacent_repeats,
        window_size,
        repeat_limit,
):
    """Mark windows containing a value run longer than repeat_limit."""
    repeated_blocks = (
        rolling_count(adjacent_repeats, repeat_limit) == repeat_limit
    )
    return rolling_count(
        repeated_blocks, window_size - repeat_limit
    ) > 0


def generate_and_save_cache(
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
    N_stations = pol_data.shape[0]
    T_total = pol_data.shape[1]
    total_window = T_short + pred_len
    eps = 1e-4
    candidate_count = T_total - total_window + 1

    complete_windows = (
        rolling_sum(mask_data, total_window) >= total_window - 0.5
    )
    adjacent_repeats = (
        torch.abs(pol_data[:, 1:] - pol_data[:, :-1]) < eps
    )
    short_repeat_filter = windows_with_excessive_repeats(
        adjacent_repeats,
        T_short,
        T_short // 6,
    )[:, :candidate_count]
    target_repeat_filter = windows_with_excessive_repeats(
        adjacent_repeats,
        pred_len,
        pred_len // 6,
    )[:, T_short:T_short + candidate_count]
    target_quality = (
        pol_data * pol_std + pol_mean <= max_pm25
    )
    target_quality_filter = (
        rolling_count(target_quality, pred_len)
        [:, T_short:T_short + candidate_count]
        == pred_len
    )

    valid_windows = (
        complete_windows
        & ~short_repeat_filter
        & ~target_repeat_filter
        & target_quality_filter
    )
    valid_starts_per_station = [
        torch.nonzero(valid_windows[station_id], as_tuple=False)
        .squeeze(1)
        .cpu()
        for station_id in range(N_stations)
    ]
    eligible_stations = torch.nonzero(
        valid_windows.any(dim=1), as_tuple=False
    ).squeeze(1).cpu()

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
        "history": os.path.join(
            cache_dir,
            f"dataset_cache_history_y1_y2_T{T_short}_P{pred_len}{seed_suffix}.pt",
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
    """Generate private caches for historical, validation, and test years."""
    os.makedirs(output_dir, exist_ok=True)
    cache_paths = get_split_cache_paths(
        output_dir, T_short=T_short, pred_len=pred_len, seed=seed
    )
    os.makedirs(cache_paths["cache_dir"], exist_ok=True)

    print(f"Starting private cache generation in: {cache_paths['cache_dir']}")
    if data_set is None:
        print("Loading raw data...")
        data_set = DataSet('data_matrix.npy')

    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix

    history_end = data_set.history_end
    val_end = data_set.val_end

    history_pol = pol_data[:, :history_end]
    history_mask = mask_data[:, :history_end]

    val_pol = pol_data[:, history_end:val_end]
    val_mask = mask_data[:, history_end:val_end]

    test_pol = pol_data[:, val_end:]
    test_mask = mask_data[:, val_end:]

    print("Processing historical years 1-2...")
    generate_and_save_cache(
        history_pol, history_mask,
        T_short, pred_len, cache_paths["history"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    print("Processing validation set...")
    generate_and_save_cache(
        val_pol, val_mask, T_short, pred_len, cache_paths["val"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    print("Processing test set...")
    generate_and_save_cache(
        test_pol, test_mask, T_short, pred_len, cache_paths["test"],
        pol_mean=data_set.pol_mean, pol_std=data_set.pol_std
    )

    return cache_paths


if __name__ == "__main__":
    raise SystemExit(
        "Run full-test inference with "
        "`python run_inference.py --seeds SEED [SEED ...]`."
    )
