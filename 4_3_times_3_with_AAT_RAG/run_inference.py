import argparse
import csv
import multiprocessing as mp
import os
import random
import shutil
import sys


sys.dont_write_bytecode = True


OUTPUT_ROOT = "inference_runs_skill_cross_3x3"
DATA_PATH = "data_matrix.npy"
STATION_FEATURES_PATH = "station_features.csv"
DEFAULT_SKILL_PATH = "base_skill.csv"
SKILL_GROUP_RANGES = {
    "low": (0.1, 0.3),
    "mid": (0.4, 0.6),
    "high": (0.7, 0.9),
}


def load_skill_station_groups(skill_csv_path):
    """Sort rows by base_skill and select the three requested rank ranges."""
    with open(skill_csv_path, newline="", encoding="utf-8-sig") as skill_file:
        rows = list(csv.DictReader(skill_file))

    skills = [float(row["base_skill"]) for row in rows]
    ranked_station_ids = sorted(range(len(skills)), key=skills.__getitem__)
    station_groups = {}
    for group, (lower, upper) in SKILL_GROUP_RANGES.items():
        rank_start = int(len(ranked_station_ids) * lower)
        rank_end = int(len(ranked_station_ids) * upper)
        selected_ids = tuple(ranked_station_ids[rank_start:rank_end])
        station_groups[group] = selected_ids
        selected_skills = [skills[station_id] for station_id in selected_ids]
        print(
            f"{group}: ranks=[{rank_start}, {rank_end}), "
            f"stations={len(selected_ids)}, "
            f"skill=[{min(selected_skills):.6f}, {max(selected_skills):.6f}]."
        )
    return station_groups


def _run_directory(output_root, run_id, seed):
    return os.path.abspath(
        os.path.join(output_root, f"run_{run_id}_seed{seed}")
    )


def _cleanup_cache_and_bank(output_root, run_id, seed):
    """Delete one completed process's private dataset caches and memory banks."""
    run_dir = _run_directory(output_root, run_id, seed)
    cache_dir = os.path.abspath(os.path.join(run_dir, f"cache_seed{seed}"))
    if os.path.commonpath([run_dir, cache_dir]) != run_dir:
        raise RuntimeError(f"Refusing to clean path outside run directory: {cache_dir}")

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[Run {run_id}, seed {seed}] Cleaned cache and memory-bank files.")
    else:
        print(f"[Run {run_id}, seed {seed}] No cache directory to clean.")


def _configure_reproducibility(seed, run_dir, cpu_threads):
    """Configure one process before importing project modules."""
    runtime_cache_dir = os.path.join(run_dir, f"runtime_cache_seed{seed}")
    temp_dir = os.path.join(run_dir, f"temp_seed{seed}")
    os.makedirs(runtime_cache_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Keep process-created runtime/cache files private as well as model artifacts.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TORCH_HOME"] = runtime_cache_dir
    os.environ["XDG_CACHE_HOME"] = runtime_cache_dir
    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    # SoftDTWLossPyTorch transfers its dynamic program to Numba on CPU.
    # Fixing these counts keeps parallel execution reproducible for a fixed command.
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    os.environ["NUMBA_NUM_THREADS"] = str(cpu_threads)

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)
    return torch


def run_inference(
        run_id,
        seed,
        output_root,
        checkpoint_root,
        station_groups,
        cpu_threads,
        batch_size,
):
    """Run one model's 3x3 station-skill cross experiment."""
    run_dir = _run_directory(output_root, run_id, seed)
    os.makedirs(run_dir, exist_ok=True)
    torch = _configure_reproducibility(seed, run_dir, cpu_threads)

    # Import only after the process-specific runtime directories are configured.
    from generate_cache import generate_split_caches
    from mult_model import DataSet
    from run_models import (
        evaluate_skill_cross_experiment,
        prepare_grouped_memory_banks,
    )

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(
        f"[Run {run_id}, seed {seed}] Device: {device_name}; "
        f"CPU threads: {cpu_threads}."
    )
    print(f"[Run {run_id}, seed {seed}] Loading and normalizing four-year data...")
    data_set = DataSet(DATA_PATH)

    print(f"[Run {run_id}, seed {seed}] Generating private split caches...")
    cache_paths = generate_split_caches(
        run_dir, data_set=data_set, seed=seed
    )

    encoder_path = os.path.join(
        checkpoint_root, f"best_encoder_seed{seed}.pth"
    )
    model_path = os.path.join(checkpoint_root, f"best_model_seed{seed}.pth")

    print(
        f"[Run {run_id}, seed {seed}] "
        "Building low/mid/high retrieval banks..."
    )
    grouped_banks = prepare_grouped_memory_banks(
        run_dir,
        encoder_path=encoder_path,
        cache_paths=cache_paths,
        data_set=data_set,
        station_groups=station_groups,
        seed=seed,
    )

    print(f"[Run {run_id}, seed {seed}] Evaluating all nine skill combinations...")
    report_path = evaluate_skill_cross_experiment(
        run_id,
        output_dir=run_dir,
        encoder_path=encoder_path,
        model_path=model_path,
        grouped_banks=grouped_banks,
        cache_paths=cache_paths,
        data_set=data_set,
        station_groups=station_groups,
        batch_size=batch_size,
        seed=seed,
        geo_csv_path=STATION_FEATURES_PATH,
    )

    print(
        f"[Run {run_id}, seed {seed}] Complete. encoder={encoder_path}, "
        f"model={model_path}, report={report_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 3x3 station-skill cross experiment for AAT-RAG checkpoints."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        required=True,
        help="checkpoint suffixes and random-subbank sampling seeds",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=".",
        help="directory containing best_encoder_seedN.pth and best_model_seedN.pth",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--skill-csv",
        default=DEFAULT_SKILL_PATH,
        help="CSV whose row order matches the data_matrix.npy station axis",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    output_root = os.path.abspath(OUTPUT_ROOT)
    checkpoint_root = os.path.abspath(args.checkpoint_root)
    station_groups = load_skill_station_groups(os.path.abspath(args.skill_csv))
    os.makedirs(output_root, exist_ok=True)
    runs = list(enumerate(args.seeds, start=1))
    print(
        f"Scheduling {len(runs)} run(s): max_parallel={args.max_parallel}, "
        f"cpu_threads_per_process={args.cpu_threads}."
    )
    failed = []
    for wave_start in range(0, len(runs), args.max_parallel):
        wave = runs[wave_start:wave_start + args.max_parallel]
        processes = []
        for run_id, seed in wave:
            os.environ["PYTHONHASHSEED"] = str(seed)
            process = mp.Process(
                target=run_inference,
                args=(
                    run_id,
                    seed,
                    output_root,
                    checkpoint_root,
                    station_groups,
                    args.cpu_threads,
                    args.batch_size,
                ),
                name=f"aat_inference_{run_id}_seed{seed}",
            )
            process.start()
            processes.append((run_id, seed, process))
            print(f"Started run {run_id} with seed {seed}: pid={process.pid}")

        for run_id, seed, process in processes:
            process.join()
            if process.exitcode != 0:
                failed.append((run_id, seed, process.exitcode))

        # Every process in this wave has finished using its private caches.
        # Remove both window-cache and memory-bank files before the next wave.
        for run_id, seed, _ in processes:
            _cleanup_cache_and_bank(output_root, run_id, seed)

    if failed:
        raise SystemExit(f"Parallel inference runs failed: {failed}")
    print(f"All {len(runs)} isolated inference runs completed successfully.")


if __name__ == "__main__":
    main()
