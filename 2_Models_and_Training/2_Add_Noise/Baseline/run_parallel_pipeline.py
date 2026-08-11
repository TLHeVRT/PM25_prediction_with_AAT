import argparse
import csv
import multiprocessing as mp
import os
import random
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


def sample_cached_windows(cache_file, sample_fraction, random_seed):
    import numpy as np
    import torch

    cache_data = torch.load(cache_file, map_location='cpu')
    valid_starts = cache_data['valid_starts']
    station_counts = np.asarray(
        [len(starts) for starts in valid_starts], dtype=np.int64
    )
    cumulative_counts = np.cumsum(station_counts)
    total_windows = int(cumulative_counts[-1]) if len(cumulative_counts) else 0

    sampled_count = max(1, int(total_windows * sample_fraction))
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


class RandomTestSubset:
    def __init__(self, met_data, pol_data, mask_data, cache_file,
                 seed, T_short=144, pred_len=48):
        import torch

        self.T_short = T_short
        self.pred_len = pred_len
        self.x_data = torch.cat([met_data, pol_data.unsqueeze(-1)], dim=-1)
        self.mask_data = mask_data

        station_ids, starts, total_windows = sample_cached_windows(
            cache_file=cache_file,
            sample_fraction=0.1,
            random_seed=seed,
        )
        self.station_ids = torch.from_numpy(station_ids)
        self.starts = torch.from_numpy(starts)
        self.total_windows = total_windows

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        import torch

        station_id = self.station_ids[index]
        start = self.starts[index]
        short_idx = start + torch.arange(self.T_short)
        target_idx = start + self.T_short + torch.arange(self.pred_len)

        short_seg = self.x_data[station_id, short_idx, :]
        short_mask = self.mask_data[station_id, short_idx]
        targets = self.x_data[station_id, target_idx, :]
        future_mets = targets[:, :-1]
        return short_seg, short_mask, future_mets, targets, station_id


def evaluate_test_subset(worker_dir, data_path, best_model_path, seed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from mult_model import DataSet
    from run_models import (
        add_noise_to_future_weather,
        build_model,
        pm25_no_trend_loss_components,
    )

    T_short = 144
    pred_len = 48
    one_year_steps = 365 * 24
    test_start = one_year_steps * 3

    data_set = DataSet(data_path)
    test_met = data_set.met_data_normalized[:, test_start:, :]
    test_pol = data_set.pol_data_normalized[:, test_start:]
    test_mask = data_set.pol_mask_matrix[:, test_start:]
    cache_file = os.path.join(
        worker_dir,
        f"dataset_cache_test_1y_T{T_short}_P{pred_len}_seed_{seed}.pt"
    )

    test_dataset = RandomTestSubset(
        test_met, test_pol, test_mask, cache_file,
        seed=seed, T_short=T_short, pred_len=pred_len
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = build_model(device, T_short=T_short, pred_len=pred_len)
    predictor.load_state_dict(torch.load(best_model_path, map_location=device))
    predictor.eval()
    noise_rng = np.random.default_rng(seed)

    base_sum = 0.0
    dtw_sum = 0.0
    topk_sum = 0.0
    total_sum = 0.0
    evaluated_count = 0

    with torch.no_grad():
        for (
            short_segs,
            short_masks,
            future_mets,
            targets,
            station_ids,
        ) in test_loader:
            short_segs = short_segs.to(device)
            short_masks = short_masks.to(device)
            future_mets = future_mets.to(device)
            targets = targets.to(device)
            future_mets = add_noise_to_future_weather(
                future_mets,
                station_ids,
                data_set.met_mean,
                data_set.met_std,
                noise_rng,
            )

            preds = predictor(short_segs, future_met=future_mets, mask=short_masks)
            base, dtw, topk, total = pm25_no_trend_loss_components(
                preds,
                targets,
                pol_mean=data_set.pol_mean,
                pol_std=data_set.pol_std
            )
            base_sum += base.sum().item()
            dtw_sum += dtw.sum().item()
            topk_sum += topk.sum().item()
            total_sum += total.sum().item()
            evaluated_count += len(total)

    mean_base_loss = base_sum / evaluated_count
    mean_dtw_loss = dtw_sum / evaluated_count
    mean_topk_loss = topk_sum / evaluated_count
    mean_weighted_total_loss = total_sum / evaluated_count
    filtered_count = len(test_dataset) - evaluated_count

    report_path = os.path.join(
        worker_dir, f"test_no_trend_loss_report_seed{seed}.csv"
    )
    fieldnames = [
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
    with open(report_path, 'w', newline='', encoding='utf-8-sig') as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'mean_base_loss': f"{mean_base_loss:.4f}",
            'mean_dtw_loss': f"{mean_dtw_loss:.4f}",
            'mean_topk_loss': f"{mean_topk_loss:.4f}",
            'mean_weighted_total_loss': f"{mean_weighted_total_loss:.4f}",
            'sampled_count': len(test_dataset),
            'evaluated_count': evaluated_count,
            'filtered_count': filtered_count,
            'available_test_windows': test_dataset.total_windows,
            'sample_fraction': 0.1,
            'random_seed': seed,
        })

    legacy_report_path = os.path.join(
        worker_dir, f"test_loss_report_seed_{seed}.txt"
    )
    if os.path.exists(legacy_report_path):
        os.remove(legacy_report_path)

    return {
        'sampled_count': len(test_dataset),
        'evaluated_count': evaluated_count,
        'filtered_count': filtered_count,
        'mean_base_loss': mean_base_loss,
        'mean_dtw_loss': mean_dtw_loss,
        'mean_topk_loss': mean_topk_loss,
        'mean_weighted_total_loss': mean_weighted_total_loss,
        'report_file': report_path,
    }


def run_worker(worker_id, seed, cpu_threads):
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    os.environ["NUMBA_NUM_THREADS"] = str(cpu_threads)

    import numpy as np
    import torch

    from generate_cache import generate_all_caches
    from run_models import run_experiment

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data_matrix.npy")
    worker_dir = os.path.join(
        base_dir, "parallel_runs", f"worker_{worker_id}_seed_{seed}"
    )
    os.makedirs(worker_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generate_all_caches(data_path, worker_dir, seed)
    best_model_path = run_experiment(
        exp_dir=worker_dir,
        cache_dir=worker_dir,
        data_path=data_path,
        seed=seed
    )
    evaluate_test_subset(worker_dir, data_path, best_model_path, seed)
    print(f"Worker {worker_id} completed: {worker_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True
    )
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    indexed_seeds = list(enumerate(args.seeds, start=1))
    for round_start in range(0, len(indexed_seeds), args.max_parallel):
        round_seeds = indexed_seeds[
            round_start:round_start + args.max_parallel
        ]
        processes = [
            mp.Process(
                target=run_worker,
                args=(worker_id, seed, args.cpu_threads)
            )
            for worker_id, seed in round_seeds
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    print("All workers completed.")


if __name__ == '__main__':
    main()
