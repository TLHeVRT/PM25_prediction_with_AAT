import os
import csv
import pandas as pd
import torch

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
LIGHT_ENCODER_DIM = 128
PROFILE_DIM = 256
MODEL_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
PROFILE_DROPOUT = 0.2
PREDICTOR_DROPOUT = 0.1
LOSS_GAMMA = 0.25
LOSS_TOPK_RATIO = 0.1
LOSS_DTW_WEIGHT = 2.4
LOSS_TOPK_WEIGHT = 0.3
SKILL_GROUP_NAMES = ("low", "mid", "high")


def build_static_features(history_met, history_pol, history_mask, pol_mean, pol_std,
                          geo_csv_path="station_features.csv"):
    """Build station profiles from the last historical year and geographic data."""
    print("Building static profile features from historical data...")
    N, T, C_met = history_met.shape
    hours_per_year = 365 * 24

    if T >= hours_per_year:
        met_y = history_met[:, -hours_per_year:, :]
        pol_y = history_pol[:, -hours_per_year:]
        mask_y = history_mask[:, -hours_per_year:]
    else:
        met_y = history_met
        pol_y = history_pol
        mask_y = history_mask

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


@torch.no_grad()
def build_memory_bank(
        databank,
        encoder,
        device,
        cache_path,
        batch_size=1024,
        storage_dtype=torch.float16,
):
    """Encode every valid cached window and save its vector, station, and start."""
    print(f"Building memory bank (Stations: {len(databank.eligible_stations)})...")
    encoder.eval()
    encoder.to(device)

    all_vecs, all_stids, all_starts = [], [], []
    samples = []

    for sid in databank.eligible_stations.tolist():
        starts = databank.valid_starts_per_station[sid]
        for st in starts:
            samples.append((sid, st.item()))

    x_data = databank.x_data.to(device)
    for i in tqdm(range(0, len(samples), batch_size), desc="Building Bank"):
        batch = samples[i:i + batch_size]
        sids_t = torch.tensor([s[0] for s in batch], device=device)
        sts_t = torch.tensor([s[1] for s in batch], device=device)

        short_idx = sts_t.unsqueeze(1) + torch.arange(databank.T_short, device=device)
        inputs = x_data[sids_t.unsqueeze(1), short_idx, :]

        vecs = encoder(inputs).to(dtype=storage_dtype)

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


@torch.no_grad()
def compute_bank_norms(bank_vectors, chunk_size=1_000_000):
    """Compute FP32 L2 norms without materializing a full FP32 bank copy."""
    norm_chunks = []
    for start in range(0, bank_vectors.shape[0], chunk_size):
        vectors_fp32 = bank_vectors[start:start + chunk_size].float()
        norm_chunks.append(vectors_fp32.square().sum(dim=1))
        del vectors_fp32
    return torch.cat(norm_chunks, dim=0)


def retrieve_top10_sequences(curr_vecs, starts_batch, bank_vectors, bank_norms, bank_stids, bank_starts, x_data, T_long, T_pred):
    """Return the nearest auxiliaries from a random subset of one group bank."""
    B = curr_vecs.shape[0]
    T_total = T_long + T_pred
    device = curr_vecs.device

    sample_size = min(1_000_000, bank_vectors.shape[0])
    random_indices = torch.randint(
        0, bank_vectors.shape[0], (sample_size,), device=device
    )
    sampled_vectors = bank_vectors[random_indices].float()
    sampled_norms = bank_norms[random_indices]
    sampled_station_ids = bank_stids[random_indices]
    sampled_starts = bank_starts[random_indices]

    query_chunk_size = 32
    all_topk_idx = []
    k_num = min(10, sample_size)

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    for query_start in range(0, B, query_chunk_size):
        query_end = min(query_start + query_chunk_size, B)
        query_vectors = curr_vecs[query_start:query_end]
        query_starts = starts_batch[query_start:query_end]

        distances = (
            sampled_norms.unsqueeze(0)
            - 2 * torch.matmul(query_vectors, sampled_vectors.T)
        )
        valid_mask = sampled_starts.unsqueeze(0) <= (
            query_starts.unsqueeze(1) - T_total
        )
        distances.masked_fill_(~valid_mask, float("inf"))

        _, topk_indices = torch.topk(
            distances, k=k_num, dim=1, largest=False
        )
        all_topk_idx.append(topk_indices)
    torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    topk_idx = torch.cat(all_topk_idx, dim=0)  # shape: [B, 10]

    t10_stids = sampled_station_ids[topk_idx]
    t10_starts = sampled_starts[topk_idx]

    batch_idx = torch.arange(T_total, device=device).view(1, 1, -1)

    target_starts = t10_starts.unsqueeze(2) + batch_idx
    target_sids = t10_stids.unsqueeze(2).expand(-1, -1, T_total)

    matched_seqs_batch = x_data[target_sids, target_starts, :]

    return matched_seqs_batch, t10_stids


class CachedPredictionWindows:
    """Expose filtered windows from one cache for retrieval-bank encoding."""

    def __init__(
            self,
            met_data,
            pol_data,
            cache_file,
            station_ids,
            T_short=T_SHORT,
    ):
        self.T_short = T_short
        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.N_stations = met_data.shape[0]

        cache_data = torch.load(cache_file, map_location="cpu")
        self.valid_starts_per_station = cache_data["valid_starts"]
        self.eligible_stations = torch.tensor([
            station_id
            for station_id in station_ids
            if len(self.valid_starts_per_station[station_id]) > 0
        ], dtype=torch.long)


class StationSubsetTestDataset:
    """Enumerate filtered final-year windows for one station group."""

    def __init__(
            self,
            met_data,
            pol_data,
            mask_data,
            cache_file,
            station_ids,
            T_short=T_SHORT,
            pred_len=PRED_LEN,
    ):
        self.T_short = T_short
        self.pred_len = pred_len
        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data

        valid_starts = torch.load(
            cache_file, map_location="cpu"
        )["valid_starts"]
        station_ids = torch.as_tensor(station_ids, dtype=torch.long)
        selected_starts = [valid_starts[station_id] for station_id in station_ids]
        lengths = torch.tensor(
            [len(starts) for starts in selected_starts], dtype=torch.long
        )
        self.station_ids = torch.repeat_interleave(
            station_ids, lengths
        )
        self.starts = torch.cat(selected_starts)

    def __len__(self):
        return len(self.starts)

    def iter_batches(self, batch_size):
        device = self.x_data.device
        short_offsets = torch.arange(self.T_short, device=device)
        target_offsets = self.T_short + torch.arange(
            self.pred_len, device=device
        )

        for offset in range(0, len(self), batch_size):
            station_ids = self.station_ids[offset:offset + batch_size].to(device)
            starts = self.starts[offset:offset + batch_size].to(device)
            station_index = station_ids.unsqueeze(1)
            short_index = starts.unsqueeze(1) + short_offsets
            target_index = starts.unsqueeze(1) + target_offsets

            short_segments = self.x_data[station_index, short_index, :]
            short_masks = self.mask_data[station_index, short_index]
            targets = self.x_data[station_index, target_index, :]
            future_mets = targets[:, :, :-1]
            yield (
                short_segments,
                short_masks,
                future_mets,
                targets,
                station_ids,
                starts,
            )


def inference_loss(
        preds,
        targets,
        pol_mean,
        pol_std,
        gamma=LOSS_GAMMA,
        k_ratio=LOSS_TOPK_RATIO,
):
    """Return per-sample components of the established test loss."""
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

    weighted_seq = (
        base_seq
        + LOSS_DTW_WEIGHT * dtw_seq
        + LOSS_TOPK_WEIGHT * topk_seq
    )

    return (
        valid_mask,
        base_seq.detach(),
        dtw_seq.detach(),
        topk_seq.detach(),
        weighted_seq.detach(),
    )
def prepare_grouped_memory_banks(
        output_dir,
        encoder_path,
        cache_paths,
        station_groups,
        data_set=None,
        T_short=T_SHORT,
        pred_len=PRED_LEN,
        seed=None,
):
    """Build one four-year retrieval bank for each station-skill group."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_set is None:
        data_set = DataSet("data_matrix.npy")

    met_data = data_set.met_data_normalized.to(device)
    pol_data = data_set.pol_data_normalized.to(device)
    history_end = data_set.history_end
    val_end = data_set.val_end
    split_data = {
        "history": (
            met_data[:, :history_end, :],
            pol_data[:, :history_end],
            0,
        ),
        "val": (
            met_data[:, history_end:val_end, :],
            pol_data[:, history_end:val_end],
            history_end,
        ),
        "test": (
            met_data[:, val_end:, :],
            pol_data[:, val_end:],
            val_end,
        ),
    }

    light_encoder = CNN1DEncoder(
        in_channels=IN_CHANNELS, d_model=LIGHT_ENCODER_DIM
    ).to(device)
    light_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    light_encoder.eval()

    cache_dir = cache_paths["cache_dir"]
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    grouped_banks = {}
    for group in SKILL_GROUP_NAMES:
        vector_chunks = []
        station_chunks = []
        start_chunks = []
        print(
            f"Preparing {group} retrieval bank from "
            f"{len(station_groups[group])} stations..."
        )
        for split_name, (split_met, split_pol, global_offset) in split_data.items():
            split_dataset = CachedPredictionWindows(
                split_met,
                split_pol,
                cache_file=cache_paths[split_name],
                station_ids=station_groups[group],
                T_short=T_short,
            )
            bank_cache_path = os.path.join(
                cache_dir,
                (
                    f"memory_bank_{group}_{split_name}_"
                    f"T{T_short}_P{pred_len}{seed_suffix}.pt"
                ),
            )
            vectors, station_ids, starts = build_memory_bank(
                split_dataset,
                light_encoder,
                device,
                bank_cache_path,
                batch_size=8192,
            )
            vector_chunks.append(vectors)
            station_chunks.append(station_ids)
            start_chunks.append(starts + global_offset)
            del split_dataset

        grouped_banks[group] = (
            torch.cat(vector_chunks, dim=0),
            torch.cat(station_chunks, dim=0),
            torch.cat(start_chunks, dim=0),
        )
        print(
            f"Prepared {group} retrieval bank: "
            f"records={grouped_banks[group][0].shape[0]}."
        )

    light_encoder.cpu()
    torch.cuda.empty_cache()
    return grouped_banks
def evaluate_cross_combination(
        test_dataset,
        bank,
        bank_norms,
        light_encoder,
        predictor,
        encoded_static_profiles,
        global_x_data,
        test_start,
        pol_mean,
        pol_std,
        batch_size,
        description,
):
    """Evaluate one auxiliary-group and test-group pairing."""
    bank_vectors, bank_stids, bank_starts = bank
    loss_sums = {
        "total_loss": 0.0,
        "base_loss": 0.0,
        "dtw_loss": 0.0,
        "topk_loss": 0.0,
    }
    valid_sample_count = 0

    with torch.inference_mode():
        for (
            short_segs,
            short_masks,
            future_mets,
            targets,
            station_ids,
            starts_batch,
        ) in tqdm(
            test_dataset.iter_batches(batch_size),
            total=(len(test_dataset) + batch_size - 1) // batch_size,
            desc=description,
        ):
            curr_vecs = light_encoder(short_segs)
            matched_seqs, matched_station_ids = retrieve_top10_sequences(
                curr_vecs,
                starts_batch + test_start,
                bank_vectors,
                bank_norms,
                bank_stids,
                bank_starts,
                global_x_data,
                T_SHORT,
                PRED_LEN,
            )
            preds, _ = predictor(
                short_segs,
                encoded_static_profiles[station_ids],
                future_met=future_mets,
                mask=short_masks,
                matched_hist=matched_seqs,
                aux_profiles=encoded_static_profiles[matched_station_ids],
            )
            valid_mask, base, dtw, topk, total = inference_loss(
                preds,
                targets,
                pol_mean=pol_mean,
                pol_std=pol_std,
            )
            valid_sample_count += int(valid_mask.sum().item())
            loss_sums["total_loss"] += total.sum().item()
            loss_sums["base_loss"] += base.sum().item()
            loss_sums["dtw_loss"] += dtw.sum().item()
            loss_sums["topk_loss"] += topk.sum().item()

    return {
        loss_name: loss_sum / valid_sample_count
        for loss_name, loss_sum in loss_sums.items()
    }


def evaluate_skill_cross_experiment(
        run_id,
        output_dir,
        encoder_path,
        model_path,
        grouped_banks,
        cache_paths,
        station_groups,
        data_set=None,
        batch_size=16,
        seed=0,
        geo_csv_path="station_features.csv",
):
    """Evaluate all nine auxiliary-group and test-group combinations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_set is None:
        data_set = DataSet("data_matrix.npy")

    met_data = data_set.met_data_normalized
    pol_data = data_set.pol_data_normalized
    mask_data = data_set.pol_mask_matrix
    history_end = data_set.history_end
    test_start = data_set.val_end

    profile_start = history_end - HOURS_PER_YEAR
    static_features = build_static_features(
        met_data[:, profile_start:history_end, :],
        pol_data[:, profile_start:history_end],
        mask_data[:, profile_start:history_end],
        pol_mean=data_set.pol_mean,
        pol_std=data_set.pol_std,
        geo_csv_path=geo_csv_path,
    ).to(device)

    light_encoder = CNN1DEncoder(
        in_channels=IN_CHANNELS, d_model=LIGHT_ENCODER_DIM
    ).to(device)
    light_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    light_encoder.eval()

    profile_encoder = StaticProfileEncoder(
        in_features=static_features.shape[1],
        d_profile=PROFILE_DIM,
        dropout=PROFILE_DROPOUT,
    ).to(device)
    predictor = ShortTermPredictorWithFuture(
        seq_len_short=T_SHORT,
        pred_len=PRED_LEN,
        in_channels=IN_CHANNELS,
        met_channels=IN_CHANNELS - 1,
        d_profile=PROFILE_DIM,
        d_model=MODEL_DIM,
        n_heads=NUM_HEADS,
        e_layers=NUM_LAYERS,
        dropout=PREDICTOR_DROPOUT,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    profile_encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    profile_encoder.eval()
    predictor.eval()

    with torch.inference_mode():
        encoded_static_profiles = profile_encoder(static_features)
    profile_encoder.cpu()
    del static_features
    torch.cuda.empty_cache()

    global_x_data = torch.cat(
        [met_data.to(device), pol_data.to(device).unsqueeze(-1)], dim=-1
    )
    test_met_data = met_data[:, test_start:, :].to(device)
    test_pol_data = pol_data[:, test_start:].to(device)
    test_mask_data = mask_data[:, test_start:].to(device)
    test_datasets = {
        group: StationSubsetTestDataset(
            test_met_data,
            test_pol_data,
            test_mask_data,
            cache_file=cache_paths["test"],
            station_ids=station_groups[group],
            T_short=T_SHORT,
            pred_len=PRED_LEN,
        )
        for group in SKILL_GROUP_NAMES
    }
    for group, test_dataset in test_datasets.items():
        print(
            f"Prepared {group} test subset: "
            f"stations={len(station_groups[group])}, "
            f"samples={len(test_dataset)}."
        )

    grouped_bank_norms = {
        group: compute_bank_norms(grouped_banks[group][0])
        for group in SKILL_GROUP_NAMES
    }

    rows = []
    for auxiliary_group in SKILL_GROUP_NAMES:
        for test_group in SKILL_GROUP_NAMES:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            experiment = f"{auxiliary_group}+{test_group}"
            means = evaluate_cross_combination(
                test_datasets[test_group],
                grouped_banks[auxiliary_group],
                grouped_bank_norms[auxiliary_group],
                light_encoder,
                predictor,
                encoded_static_profiles,
                global_x_data,
                test_start,
                data_set.pol_mean,
                data_set.pol_std,
                batch_size,
                description=f"Seed {seed} {experiment}",
            )
            rows.append({
                "experiment": experiment,
                "auxiliary_group": auxiliary_group,
                "test_group": test_group,
                **means,
            })
            print(
                f"Run {run_id} {experiment}: "
                f"Base={means['base_loss']:.6f}, "
                f"DTW={means['dtw_loss']:.6f}, "
                f"TopK={means['topk_loss']:.6f}, "
                f"Total={means['total_loss']:.6f}."
            )

    report_path = os.path.join(
        output_dir, f"skill_cross_3x3_loss_means_seed{seed}.csv"
    )
    fieldnames = [
        "experiment",
        "auxiliary_group",
        "test_group",
        "total_loss",
        "base_loss",
        "dtw_loss",
        "topk_loss",
    ]
    with open(report_path, "w", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Run {run_id} completed all nine skill combinations: "
        f"path={report_path}."
    )
    return report_path
def main():
    raise SystemExit(
        "Use `python run_inference.py --seeds SEED [SEED ...]`."
    )


if __name__ == '__main__':
    main()
