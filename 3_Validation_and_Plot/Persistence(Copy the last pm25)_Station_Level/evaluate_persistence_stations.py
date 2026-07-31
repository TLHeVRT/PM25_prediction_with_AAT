"""Evaluate a 48-hour persistence PM2.5 baseline on the filtered test set.

This file is intentionally self-contained: it reproduces the data split,
normalization, window filters, losses, and station CSV report without importing
any other project script.
"""

import csv
import os

DATA_PATH = "data_matrix.npy"
STATION_FEATURES_PATH = "station_features.csv"
OUTPUT_PATH = "Persistence_test_losses.csv"
DEVICE = "cuda"
BATCH_SIZE = 16384
CPU_THREADS = 8

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_NUM_THREADS"] = str(CPU_THREADS)

import numpy as np
import torch
from tqdm import tqdm
from tslearn.metrics import SoftDTWLossPyTorch


HOURS_PER_YEAR = 365 * 24
TRAIN_END = 2 * HOURS_PER_YEAR
VAL_END = 3 * HOURS_PER_YEAR
TOTAL_HOURS = 4 * HOURS_PER_YEAR
PM25_CHANNEL = 9
T_SHORT = 144
PRED_LEN = 48
TOTAL_WINDOW = T_SHORT + PRED_LEN
REPEAT_EPS = 1e-4
MAX_PM25 = 1000.0
SOFT_DTW_GAMMA = 0.25


def rolling_sum(values, width):
    """Return sums of every width-long interval along dimension 1."""
    cumulative = torch.cumsum(values, dim=1, dtype=torch.int32)
    zero = torch.zeros((values.shape[0], 1), dtype=torch.int32)
    cumulative = torch.cat((zero, cumulative), dim=1)
    return cumulative[:, width:] - cumulative[:, :-width]


def load_test_pm25(data_path):
    """Reproduce the existing global PM2.5 normalization exactly."""
    raw_matrix = np.load(data_path, mmap_mode="r")
    # Existing DataSet logic transposes first, keeps the final four 365-day
    # years, converts float16 to float32, then selects PM2.5 channel 9.
    pm25 = torch.from_numpy(
        np.array(
            raw_matrix[:, PM25_CHANNEL, -TOTAL_HOURS:],
            dtype=np.float32,
            copy=True,
        )
    )
    training_pm25 = pm25[:, :TRAIN_END]
    valid_for_statistics = (~torch.isnan(training_pm25)) & (
        training_pm25 <= MAX_PM25
    )
    valid_values = training_pm25[valid_for_statistics]

    # torch.std uses Bessel's correction by default, matching mult_model.py.
    pm25_mean = valid_values.mean()
    pm25_std = valid_values.std()

    test_raw = pm25[:, VAL_END:].clone()
    del pm25, training_pm25, valid_values
    test_mask = ~torch.isnan(test_raw)
    test_normalized = (
        torch.nan_to_num(test_raw, nan=0.0) - pm25_mean
    ) / (pm25_std + 1e-8)
    test_normalized *= test_mask
    return test_normalized, test_mask, pm25_mean.item(), pm25_std.item()


def valid_test_pairs(test_pm25, test_mask, pm25_mean, pm25_std):
    """Return every station/start pair passing the existing test filters."""
    time_count = test_pm25.shape[1]
    max_start = time_count - TOTAL_WINDOW
    sample_count = max_start + 1

    # Every PM2.5 value in the complete 144+48 window must be observed.
    missing_ok = rolling_sum(test_mask, TOTAL_WINDOW) == TOTAL_WINDOW

    # A run longer than L exists iff the window contains L adjacent "same"
    # edges. This is exactly equivalent to max_consecutive_repeat_length in
    # evaluate_stations.py, including its strict < 1e-4 comparison.
    same_as_previous = (
        torch.abs(test_pm25[:, 1:] - test_pm25[:, :-1]) < REPEAT_EPS
    )

    history_limit = T_SHORT // 6  # longest allowed run: 24 values
    history_bad_anchors = rolling_sum(
        same_as_previous, history_limit
    ) == history_limit
    history_anchor_span = T_SHORT - history_limit
    history_has_long_repeat = rolling_sum(
        history_bad_anchors, history_anchor_span
    ) > 0
    history_ok = ~history_has_long_repeat[:, :sample_count]

    target_limit = PRED_LEN // 6  # longest allowed run: 8 values
    target_bad_anchors = rolling_sum(
        same_as_previous, target_limit
    ) == target_limit
    target_anchor_span = PRED_LEN - target_limit
    target_has_long_repeat = rolling_sum(
        target_bad_anchors, target_anchor_span
    ) > 0
    target_ok = ~target_has_long_repeat[:, T_SHORT:T_SHORT + sample_count]

    # Match the old filter's normalized -> raw reconstruction before <= 1000.
    reconstructed_pm25 = test_pm25 * pm25_std + pm25_mean
    within_limit = reconstructed_pm25 <= MAX_PM25
    target_quality_ok = (
        rolling_sum(within_limit, PRED_LEN)[:, T_SHORT:T_SHORT + sample_count]
        == PRED_LEN
    )

    valid = missing_ok & history_ok & target_ok & target_quality_ok
    pairs = valid.nonzero(as_tuple=False)
    counts = valid.sum(dim=1, dtype=torch.long)
    return pairs, counts


def test_loss(predictions, targets, pm25_mean, pm25_std, criterion):
    """Return the exact per-sample no-trend test loss used by training code."""
    pm25_targets = targets[:, :, -1]
    raw_targets = pm25_targets * pm25_std + pm25_mean
    valid_mask = (raw_targets <= MAX_PM25).all(dim=1)
    predictions = predictions[valid_mask]
    pm25_targets = pm25_targets[valid_mask]
    if predictions.shape[0] == 0:
        empty = predictions.new_empty((0,))
        return valid_mask, empty, empty, empty, empty

    base_loss = torch.abs(predictions - pm25_targets).mean(dim=1)
    predictions_3d = predictions.unsqueeze(2)
    targets_3d = pm25_targets.unsqueeze(2)
    dtw_xy = criterion(predictions_3d, targets_3d)
    dtw_xx = criterion(predictions_3d, predictions_3d)
    dtw_yy = criterion(targets_3d, targets_3d)
    dtw_loss = (
        dtw_xy - 0.5 * (dtw_xx + dtw_yy)
    ).clamp(min=0.0) / pm25_targets.shape[1]

    topk_count = max(1, int(pm25_targets.shape[1] * 0.1))
    top_values, top_indices = torch.topk(
        pm25_targets, k=topk_count, dim=1
    )
    predicted_top_values = torch.gather(predictions, 1, top_indices)
    topk_loss = torch.abs(predicted_top_values - top_values).mean(dim=1)
    total_loss = base_loss + 2.4 * dtw_loss + 0.3 * topk_loss
    return valid_mask, base_loss, dtw_loss, topk_loss, total_loss


def calculate_losses(
    test_pm25,
    pairs,
    station_count,
    pm25_mean,
    pm25_std,
    device,
    batch_size,
):
    """Calculate the four existing loss columns for persistence predictions."""
    criterion = SoftDTWLossPyTorch(gamma=SOFT_DTW_GAMMA)
    test_pm25 = test_pm25.to(device)
    target_offsets = torch.arange(PRED_LEN, device=device)
    component_sums = {
        name: torch.zeros(station_count, dtype=torch.float64)
        for name in ("total", "base", "dtw", "topk")
    }

    with torch.inference_mode():
        for offset in tqdm(
            range(0, pairs.shape[0], batch_size),
            desc="Evaluating persistence baseline",
        ):
            batch_pairs = pairs[offset:offset + batch_size]
            station_ids_cpu = batch_pairs[:, 0]
            station_ids = station_ids_cpu.to(device)
            starts = batch_pairs[:, 1].to(device)

            target_indices = (
                starts.unsqueeze(1) + T_SHORT + target_offsets.unsqueeze(0)
            )
            targets = test_pm25[station_ids.unsqueeze(1), target_indices]
            last_pm25 = test_pm25[station_ids, starts + T_SHORT - 1]
            predictions = last_pm25.unsqueeze(1).expand(-1, PRED_LEN).contiguous()

            valid, base_loss, dtw_loss, topk_loss, total_loss = test_loss(
                predictions,
                targets.unsqueeze(2),
                pm25_mean,
                pm25_std,
                criterion,
            )
            station_ids_cpu = station_ids_cpu[valid.cpu()]

            for name, values in (
                ("total", total_loss),
                ("base", base_loss),
                ("dtw", dtw_loss),
                ("topk", topk_loss),
            ):
                component_sums[name].scatter_add_(
                    0, station_ids_cpu, values.double().cpu()
                )

    return component_sums


def load_station_labels(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        labels = [row[0] for row in reader if row]
    return labels


def write_station_report(output_path, station_labels, counts, component_sums):
    fieldnames = [
        "Station_ID",
        "Valid_Samples",
        "Total_Loss",
        "Base_Loss",
        "DTW_Loss",
        "TopK_Loss",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for station, station_label in enumerate(station_labels):
            count = int(counts[station])
            if count:
                values = {
                    name: float(component_sums[name][station] / count)
                    for name in component_sums
                }
            else:
                values = {name: "" for name in component_sums}
            writer.writerow(
                {
                    "Station_ID": station_label,
                    "Valid_Samples": count,
                    "Total_Loss": values["total"],
                    "Base_Loss": values["base"],
                    "DTW_Loss": values["dtw"],
                    "TopK_Loss": values["topk"],
                }
            )


def main():
    device = torch.device(DEVICE)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    print(f"Evaluation device: {device}")
    test_pm25, test_mask, pm25_mean, pm25_std = load_test_pm25(DATA_PATH)
    print(
        f"PM2.5 normalization: mean={pm25_mean:.9g}, std={pm25_std:.9g} "
        f"(first two years, non-NaN and <= {MAX_PM25:g})"
    )

    pairs, counts = valid_test_pairs(
        test_pm25, test_mask, pm25_mean, pm25_std
    )
    station_count = test_pm25.shape[0]
    print(
        f"Filtered test set: {pairs.shape[0]} samples across "
        f"{int((counts > 0).sum())}/{station_count} stations"
    )
    station_labels = load_station_labels(STATION_FEATURES_PATH)
    component_sums = calculate_losses(
        test_pm25,
        pairs,
        station_count,
        pm25_mean,
        pm25_std,
        device,
        BATCH_SIZE,
    )
    write_station_report(
        OUTPUT_PATH, station_labels, counts, component_sums
    )
    print(
        f"Saved {station_count} station rows and {pairs.shape[0]} valid samples "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
