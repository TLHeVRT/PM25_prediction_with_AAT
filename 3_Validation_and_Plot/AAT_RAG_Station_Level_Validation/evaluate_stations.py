"""Evaluate trained AAT-RAG weights on every valid test window, by station."""

import argparse
import csv
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("--cpu-threads", type=int, default=8)
parser.add_argument("--rag-model", default="best_encoder_seed1.pth")
parser.add_argument("--main-model", default="best_model_seed1.pth")
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_NUM_THREADS"] = str(args.cpu_threads)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tslearn.metrics import SoftDTWLossPyTorch

from mult_model import (
    CNN1DEncoder,
    DataSet,
    HOURS_PER_YEAR,
    ShortTermPredictorWithFuture,
    StaticProfileEncoder,
)

T_SHORT = 144
PRED_LEN = 48
IN_CHANNELS = 10
RAG_DIM = 128
PROFILE_DIM = 256
MODEL_DIM = 256
TOP_K = 10

DATA_PATH = "data_matrix.npy"
STATION_FEATURES_PATH = "station_features.csv"
RAG_MODEL_PATH = args.rag_model
MAIN_MODEL_PATH = args.main_model
CACHE_DIR = "evaluation_cache"
OUTPUT_PATH = "station_test_losses.csv"
BATCH_SIZE = 16
BANK_BATCH_SIZE = 8192
RETRIEVAL_CANDIDATE_LIMIT = 1000000
NUM_WORKERS = 0
SEED = 1
DEVICE = "cuda"

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(args.cpu_threads)
torch.set_num_interop_threads(args.cpu_threads)


def _torch_load(path, map_location):
    """Load tensor-only project files on both old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def generate_window_cache(
    met_data,
    pol_data,
    mask_data,
    cache_file,
    pol_mean,
    pol_std,
    t_short=T_SHORT,
    pred_len=PRED_LEN,
    max_pm25=1000.0,
):
    """Apply the training pipeline's missing-value, repeat, and PM2.5 filters."""
    print(f"Generating window cache: {cache_file}")
    station_count = met_data.shape[0]
    total_time = met_data.shape[1]
    total_window = t_short + pred_len
    valid_starts_per_station = []

    for station in tqdm(range(station_count), desc=f"Scanning stations (T={total_time})"):
        station_mask = mask_data[station]
        station_pm25 = pol_data[station]
        max_start = total_time - total_window
        if max_start <= 0:
            valid_starts_per_station.append(torch.empty(0, dtype=torch.long))
            continue

        start_indices = torch.arange(max_start + 1)
        cumulative_mask = torch.cumsum(station_mask, dim=0)
        end_indices = start_indices + total_window - 1
        window_sums = cumulative_mask[end_indices].clone()
        nonzero_start = start_indices > 0
        window_sums[nonzero_start] -= cumulative_mask[start_indices[nonzero_start] - 1]
        valid_starts = start_indices[window_sums >= total_window - 0.5]

        if len(valid_starts) > 0:
            keep_chunks = []
            for chunk_start in range(0, len(valid_starts), 72):
                starts = valid_starts[chunk_start:chunk_start + 72]
                short_idx = starts.unsqueeze(1) + torch.arange(t_short)
                target_idx = (starts + t_short).unsqueeze(1) + torch.arange(pred_len)
                short_windows = station_pm25[short_idx]
                target_windows = station_pm25[target_idx]

                short_ok = (
                    max_consecutive_repeat_length(short_windows) <= t_short // 6
                )
                target_ok = (
                    max_consecutive_repeat_length(target_windows) <= pred_len // 6
                )
                target_raw = target_windows * pol_std + pol_mean
                quality_ok = (target_raw <= max_pm25).all(dim=1)
                keep_chunks.append(short_ok & target_ok & quality_ok)

            valid_starts = valid_starts[torch.cat(keep_chunks)]

        valid_starts_per_station.append(valid_starts.cpu())

    eligible_stations = torch.tensor(
        [
            station
            for station, starts in enumerate(valid_starts_per_station)
            if len(starts) > 0
        ],
        dtype=torch.long,
    )
    torch.save(
        {
            "valid_starts": valid_starts_per_station,
            "eligible_stations": eligible_stations,
        },
        cache_file,
    )
    print(f"Cached {len(eligible_stations)}/{station_count} eligible stations.")


def prepare_window_caches(data_set, cache_dir):
    """Create the year-1/2, year-3, and year-4 window caches."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_paths = {
        "train": os.path.join(
            cache_dir, f"dataset_cache_train_y1_y2_T{T_SHORT}_P{PRED_LEN}.pt"
        ),
        "val": os.path.join(
            cache_dir, f"dataset_cache_val_y3_T{T_SHORT}_P{PRED_LEN}.pt"
        ),
        "test": os.path.join(
            cache_dir, f"dataset_cache_test_y4_T{T_SHORT}_P{PRED_LEN}.pt"
        ),
    }
    splits = {
        "train": slice(0, data_set.train_end),
        "val": slice(data_set.train_end, data_set.val_end),
        "test": slice(data_set.val_end, None),
    }

    for split_name, time_slice in splits.items():
        cache_path = cache_paths[split_name]
        generate_window_cache(
            data_set.met_data_normalized[:, time_slice, :],
            data_set.pol_data_normalized[:, time_slice],
            data_set.pol_mask_matrix[:, time_slice],
            cache_path,
            pol_mean=data_set.pol_mean,
            pol_std=data_set.pol_std,
        )
    return cache_paths


def cached_pairs(cache_file):
    cache = _torch_load(cache_file, map_location="cpu")
    valid_starts = cache["valid_starts"]
    lengths = torch.tensor([len(starts) for starts in valid_starts], dtype=torch.long)
    if int(lengths.sum()) == 0:
        raise RuntimeError(f"No valid samples in cache: {cache_file}")
    station_ids = torch.repeat_interleave(
        torch.arange(len(valid_starts), dtype=torch.long), lengths
    )
    starts = torch.cat(valid_starts).long()
    return station_ids, starts


class FullTestDataset(Dataset):
    """Every cached station/window pair in the test year, exactly once."""

    def __init__(self, met_data, pol_data, mask_data, cache_file):
        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data
        self.station_ids, self.starts = cached_pairs(cache_file)
        self.station_ids = self.station_ids.to(self.x_data.device)
        self.starts = self.starts.to(self.x_data.device)
        self.short_offsets = torch.arange(T_SHORT, device=self.x_data.device)
        self.target_offsets = torch.arange(PRED_LEN, device=self.x_data.device)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        station_id = self.station_ids[index]
        start = self.starts[index]
        short_idx = start + self.short_offsets
        target_idx = start + T_SHORT + self.target_offsets
        short_seg = self.x_data[station_id, short_idx, :]
        short_mask = self.mask_data[station_id, short_idx]
        target = self.x_data[station_id, target_idx, :]
        return short_seg, short_mask, target[:, :-1], target, station_id, start


def build_static_features(
    train_met,
    train_pol,
    train_mask,
    pol_mean,
    pol_std,
    station_features_path,
):
    """Reproduce the static station profiles built from training year 2."""
    station_count, time_count, met_channels = train_met.shape
    if time_count >= HOURS_PER_YEAR:
        met_year = train_met[:, -HOURS_PER_YEAR:, :]
        pol_year = train_pol[:, -HOURS_PER_YEAR:]
        mask_year = train_mask[:, -HOURS_PER_YEAR:]
    else:
        met_year, pol_year, mask_year = train_met, train_pol, train_mask

    pol_raw = pol_year * pol_std + pol_mean
    valid = (mask_year > 0) & (pol_raw <= 1000)
    valid_float = valid.float()
    chunk_size = met_year.shape[1] // 12
    pm25_means, pm25_stds, met_means = [], [], []

    for month in range(12):
        start = month * chunk_size
        end = (month + 1) * chunk_size if month < 11 else met_year.shape[1]
        pol_chunk = pol_year[:, start:end]
        met_chunk = met_year[:, start:end, :]
        chunk_mask = valid_float[:, start:end]

        count = chunk_mask.sum(dim=1).clamp(min=1)
        mean = (pol_chunk * chunk_mask).sum(dim=1) / count
        variance = (
            ((pol_chunk - mean.unsqueeze(1)) * chunk_mask).square().sum(dim=1)
            / count
        )
        pm25_means.append(mean)
        pm25_stds.append(torch.sqrt(variance.clamp(min=1e-8)))

        met_mask = chunk_mask.unsqueeze(-1)
        met_means.append(
            (met_chunk * met_mask).sum(dim=1) / met_mask.sum(dim=1).clamp(min=1)
        )

    correlation_columns = []
    for channel in range(met_channels):
        met_channel = met_year[:, :, channel]
        correlations = torch.zeros(station_count)
        for station in range(station_count):
            station_valid = valid[station]
            if station_valid.sum() > 2:
                x = met_channel[station, station_valid]
                y = pol_year[station, station_valid]
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                denominator = torch.sqrt(
                    (x_centered.square().sum() * y_centered.square().sum()).clamp(
                        min=1e-12
                    )
                )
                if denominator > 1e-8:
                    correlations[station] = (
                        x_centered * y_centered
                    ).sum() / denominator
        correlation_columns.append(correlations)

    historical_features = torch.cat(
        [
            torch.stack(pm25_means, dim=1),
            torch.stack(pm25_stds, dim=1),
            torch.cat(met_means, dim=1),
            torch.stack(correlation_columns, dim=1),
        ],
        dim=1,
    )
    historical_features = (
        historical_features - historical_features.mean(dim=0, keepdim=True)
    ) / historical_features.std(dim=0, keepdim=True).clamp(min=1e-8)

    if not os.path.exists(station_features_path):
        raise FileNotFoundError(
            "The trained main model requires the same station feature CSV used during "
            f"training, but it was not found: {station_features_path}"
        )
    station_frame = pd.read_csv(station_features_path)
    if len(station_frame) != station_count:
        raise ValueError(
            f"Station feature rows ({len(station_frame)}) do not match data stations "
            f"({station_count})."
        )
    geographic_values = station_frame.iloc[:, 1:].astype(float).to_numpy()
    if not np.isfinite(geographic_values).all():
        raise ValueError("Station feature columns contain NaN or infinite values.")
    geographic_features = torch.tensor(geographic_values, dtype=torch.float32)
    station_labels = station_frame.iloc[:, 0].tolist()
    return torch.cat([geographic_features, historical_features], dim=1), station_labels


@torch.inference_mode()
def build_memory_bank(
    split_x_data,
    cache_file,
    output_file,
    rag_encoder,
    device,
    batch_size,
):
    """Encode all valid cached histories for one temporal split."""
    station_ids, starts = cached_pairs(cache_file)
    vectors = []
    split_x_data = split_x_data.to(device)
    rag_encoder.eval()
    for offset in tqdm(
        range(0, len(starts), batch_size), desc=f"Building {os.path.basename(output_file)}"
    ):
        batch_stations = station_ids[offset:offset + batch_size].to(device)
        batch_starts = starts[offset:offset + batch_size].to(device)
        short_indices = batch_starts.unsqueeze(1) + torch.arange(T_SHORT, device=device)
        inputs = split_x_data[batch_stations.unsqueeze(1), short_indices, :]
        vectors.append(rag_encoder(inputs).to(torch.float16))

    payload = {
        "vectors": torch.cat(vectors),
        "stids": station_ids.to(device),
        "starts": starts.to(device),
    }
    torch.save(payload, output_file)
    del split_x_data
    return payload


def prepare_global_memory_bank(
    data_set,
    cache_paths,
    cache_dir,
    rag_encoder,
    device,
    batch_size,
):
    """Create the training pipeline's four-year RAG vector bank."""
    split_specs = {
        "train": (slice(0, data_set.train_end), 0, "train_y1_y2"),
        "val": (
            slice(data_set.train_end, data_set.val_end),
            data_set.train_end,
            "val_y3",
        ),
        "test": (slice(data_set.val_end, None), data_set.val_end, "test_y4"),
    }
    vector_parts, station_parts, start_parts = [], [], []

    for split_name, (time_slice, global_offset, file_label) in split_specs.items():
        bank_file = os.path.join(
            cache_dir, f"memory_bank_{file_label}_T{T_SHORT}.pt"
        )
        split_x_data = torch.cat(
            [
                data_set.met_data_normalized[:, time_slice, :],
                data_set.pol_data_normalized[:, time_slice].unsqueeze(-1),
            ],
            dim=-1,
        )
        payload = build_memory_bank(
            split_x_data,
            cache_paths[split_name],
            bank_file,
            rag_encoder,
            device,
            batch_size,
        )
        vector_parts.append(payload["vectors"])
        station_parts.append(payload["stids"].long())
        start_parts.append(payload["starts"].long() + global_offset)

    return (
        torch.cat(vector_parts).to(device),
        torch.cat(station_parts).to(device),
        torch.cat(start_parts).to(device),
    )


@torch.inference_mode()
def compute_bank_norms(bank_vectors, chunk_size=1_000_000):
    norm_chunks = []
    for start in range(0, bank_vectors.shape[0], chunk_size):
        vectors = bank_vectors[start:start + chunk_size].float()
        norm_chunks.append(vectors.square().sum(dim=1))
    return torch.cat(norm_chunks)


@torch.inference_mode()
def retrieve_topk_sequences(
    query_vectors,
    query_starts,
    bank_vectors,
    bank_norms,
    bank_station_ids,
    bank_starts,
    global_x_data,
    candidate_limit,
):
    """Use the trained pipeline's sampled L2 retrieval and temporal masks."""
    batch_count = query_vectors.shape[0]
    total_window = T_SHORT + PRED_LEN
    device = query_vectors.device
    sample_size = min(candidate_limit, bank_vectors.shape[0])
    sampled_indices = torch.randint(
        0, bank_vectors.shape[0], (sample_size,), device=device
    )
    sampled_vectors = bank_vectors[sampled_indices].float()
    sampled_norms = bank_norms[sampled_indices]
    sampled_station_ids = bank_station_ids[sampled_indices]
    sampled_starts = bank_starts[sampled_indices]
    result_indices = []
    k_count = min(TOP_K, sample_size)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for chunk_start in range(0, batch_count, 32):
            query_chunk = query_vectors[chunk_start:chunk_start + 32]
            start_chunk = query_starts[chunk_start:chunk_start + 32]
            distances = sampled_norms.unsqueeze(0) - 2 * torch.matmul(
                query_chunk, sampled_vectors.T
            )
            standard_valid = sampled_starts.unsqueeze(0) <= (
                start_chunk.unsqueeze(1) - total_window
            )
            is_year_one = start_chunk + total_window <= HOURS_PER_YEAR
            year_two_valid = (
                (sampled_starts.unsqueeze(0) >= HOURS_PER_YEAR)
                & (sampled_starts.unsqueeze(0) < 2 * HOURS_PER_YEAR)
                & (
                    sampled_starts.unsqueeze(0)
                    >= start_chunk.unsqueeze(1) + total_window
                )
            )
            valid = torch.where(is_year_one.unsqueeze(1), year_two_valid, standard_valid)
            distances.masked_fill_(~valid, float("inf"))
            result_indices.append(
                torch.topk(distances, k=k_count, dim=1, largest=False).indices
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    selected = torch.cat(result_indices)
    matched_station_ids = sampled_station_ids[selected]
    matched_starts = sampled_starts[selected]
    time_offsets = torch.arange(total_window, device=device).view(1, 1, -1)
    gather_starts = matched_starts.unsqueeze(2) + time_offsets
    gather_stations = matched_station_ids.unsqueeze(2).expand_as(gather_starts)
    return global_x_data[gather_stations, gather_starts, :], matched_station_ids


def test_loss(predictions, targets, pol_mean, pol_std, criterion):
    """Return the exact per-sample no-trend test loss used by training code."""
    pm25_targets = targets[:, :, -1]
    raw_targets = pm25_targets * pol_std + pol_mean
    valid_mask = (raw_targets <= 1000).all(dim=1)
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

    k_count = max(1, int(pm25_targets.shape[1] * 0.1))
    top_values, top_indices = torch.topk(pm25_targets, k=k_count, dim=1)
    predicted_top_values = torch.gather(predictions, 1, top_indices)
    topk_loss = torch.abs(predicted_top_values - top_values).mean(dim=1)
    total_loss = base_loss + 2.4 * dtw_loss + 0.3 * topk_loss
    return valid_mask, base_loss, dtw_loss, topk_loss, total_loss


def load_models(rag_path, model_path, static_feature_count, device):
    rag_encoder = CNN1DEncoder(in_channels=IN_CHANNELS, d_model=RAG_DIM).to(device)
    rag_encoder.load_state_dict(_torch_load(rag_path, map_location=device))
    rag_encoder.eval()

    checkpoint = _torch_load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or not {"encoder", "predictor"} <= checkpoint.keys():
        raise ValueError(
            f"Main checkpoint must contain 'encoder' and 'predictor' states: {model_path}"
        )
    expected_features = checkpoint["encoder"]["net.0.weight"].shape[1]
    if static_feature_count != expected_features:
        raise ValueError(
            f"Static feature width is {static_feature_count}, but the checkpoint expects "
            f"{expected_features}. Use the exact station_features.csv from training."
        )

    profile_encoder = StaticProfileEncoder(
        in_features=static_feature_count, d_profile=PROFILE_DIM, dropout=0.2
    ).to(device)
    predictor = ShortTermPredictorWithFuture(
        seq_len_short=T_SHORT,
        pred_len=PRED_LEN,
        in_channels=IN_CHANNELS,
        met_channels=IN_CHANNELS - 1,
        d_profile=PROFILE_DIM,
        d_model=MODEL_DIM,
        n_heads=8,
        e_layers=4,
        dropout=0.1,
    ).to(device)
    profile_encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    profile_encoder.eval()
    predictor.eval()
    return rag_encoder, profile_encoder, predictor


def write_station_report(output_path, station_labels, counts, component_sums):
    """Write per-station sample counts and sample-weighted mean losses."""
    fieldnames = [
        "Station_ID",
        "Valid_Samples",
        "Total_Loss",
        "Base_Loss",
        "DTW_Loss",
        "TopK_Loss",
    ]
    output_parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_parent, exist_ok=True)
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


def evaluate():
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    required_files = {
        "data matrix": DATA_PATH,
        "station features": STATION_FEATURES_PATH,
        "RAG checkpoint": RAG_MODEL_PATH,
        "main-model checkpoint": MAIN_MODEL_PATH,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required evaluation file(s):\n  " + "\n  ".join(missing))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Evaluation device: {device}")

    data_set = DataSet(DATA_PATH)
    profile_start = data_set.train_end - HOURS_PER_YEAR
    static_features, station_labels = build_static_features(
        data_set.met_data_normalized[:, profile_start:data_set.train_end, :],
        data_set.pol_data_normalized[:, profile_start:data_set.train_end],
        data_set.pol_mask_matrix[:, profile_start:data_set.train_end],
        pol_mean=data_set.pol_mean,
        pol_std=data_set.pol_std,
        station_features_path=STATION_FEATURES_PATH,
    )
    rag_encoder, profile_encoder, predictor = load_models(
        RAG_MODEL_PATH, MAIN_MODEL_PATH, static_features.shape[1], device
    )
    window_caches = prepare_window_caches(data_set, CACHE_DIR)
    bank_vectors, bank_station_ids, bank_starts = prepare_global_memory_bank(
        data_set,
        window_caches,
        CACHE_DIR,
        rag_encoder,
        device,
        batch_size=BANK_BATCH_SIZE,
    )
    bank_norms = compute_bank_norms(bank_vectors)

    test_dataset = FullTestDataset(
        data_set.met_data_normalized[:, data_set.val_end:, :].to(device),
        data_set.pol_data_normalized[:, data_set.val_end:].to(device),
        data_set.pol_mask_matrix[:, data_set.val_end:].to(device),
        window_caches["test"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    static_features = static_features.to(device)
    global_x_data = torch.cat(
        [
            data_set.met_data_normalized.to(device),
            data_set.pol_data_normalized.to(device).unsqueeze(-1),
        ],
        dim=-1,
    )
    station_count = data_set.met_data_normalized.shape[0]
    counts = torch.zeros(station_count, dtype=torch.long)
    component_sums = {
        name: torch.zeros(station_count, dtype=torch.float64)
        for name in ("total", "base", "dtw", "topk")
    }
    criterion = SoftDTWLossPyTorch(gamma=0.25)

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating full test set"):
            short_segs, short_masks, future_mets, targets, station_ids, starts = batch
            short_segs = short_segs.to(device, non_blocking=True)
            short_masks = short_masks.to(device, non_blocking=True)
            future_mets = future_mets.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            global_starts = starts + data_set.val_end

            query_vectors = rag_encoder(short_segs)
            matched_histories, matched_station_ids = retrieve_topk_sequences(
                query_vectors,
                global_starts,
                bank_vectors,
                bank_norms,
                bank_station_ids,
                bank_starts,
                global_x_data,
                candidate_limit=RETRIEVAL_CANDIDATE_LIMIT,
            )
            profiles = profile_encoder(static_features[station_ids])
            auxiliary_profiles = profile_encoder(
                static_features[matched_station_ids]
            )
            predictions, _ = predictor(
                short_segs,
                profiles,
                future_met=future_mets,
                mask=short_masks,
                matched_hist=matched_histories,
                aux_profiles=auxiliary_profiles,
            )
            valid, base, dtw, topk, total = test_loss(
                predictions,
                targets,
                pol_mean=data_set.pol_mean,
                pol_std=data_set.pol_std,
                criterion=criterion,
            )
            valid_station_ids = station_ids[valid].long().cpu()
            counts.scatter_add_(
                0, valid_station_ids, torch.ones_like(valid_station_ids)
            )
            for name, values in (
                ("total", total),
                ("base", base),
                ("dtw", dtw),
                ("topk", topk),
            ):
                component_sums[name].scatter_add_(
                    0, valid_station_ids, values.double().cpu()
                )

    if int(counts.sum()) == 0:
        raise RuntimeError("No valid test samples were evaluated.")
    write_station_report(OUTPUT_PATH, station_labels, counts, component_sums)
    print(
        f"Saved {station_count} station rows and {int(counts.sum())} valid samples "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate()
