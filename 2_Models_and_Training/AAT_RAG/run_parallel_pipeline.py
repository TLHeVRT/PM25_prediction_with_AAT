import argparse
import multiprocessing as mp
import os
import random
import shutil
import sys


sys.dont_write_bytecode = True


OUTPUT_ROOT = "parallel_runs"
TEST_FRACTION = 0.1


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
    """Configure one process before any project module starts doing work."""
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


def run_complete_pipeline(run_id, seed, output_root, cpu_threads):
    """Run one seed-isolated cache -> encoder -> main model -> test pipeline."""
    run_dir = _run_directory(output_root, run_id, seed)
    os.makedirs(run_dir, exist_ok=True)
    torch = _configure_reproducibility(seed, run_dir, cpu_threads)

    # Import only after the process-specific runtime directories are configured.
    from generate_cache import generate_split_caches
    from mult_model import DataSet
    from run_models import (
        evaluate_random_test_subset,
        prepare_global_memory_bank,
        train_main_model,
    )
    from train_rag_encoder import train_rag_encoder

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(
        f"[Run {run_id}, seed {seed}] Device: {device_name}; "
        f"CPU threads: {cpu_threads}."
    )
    print(f"[Run {run_id}, seed {seed}] Loading and normalizing four-year data...")
    data_set = DataSet("data_matrix.npy")

    print(f"[Run {run_id}, seed {seed}] Generating private split caches...")
    cache_paths = generate_split_caches(
        run_dir, data_set=data_set, seed=seed
    )

    print(f"[Run {run_id}, seed {seed}] Training private RAG encoder...")
    encoder_path = train_rag_encoder(
        run_dir,
        data_set=data_set,
        cache_paths=cache_paths,
        seed=seed,
    )

    print(
        f"[Run {run_id}, seed {seed}] "
        "Building encoder-specific private retrieval caches..."
    )
    bank_global = prepare_global_memory_bank(
        run_dir,
        encoder_path=encoder_path,
        cache_paths=cache_paths,
        data_set=data_set,
        seed=seed,
    )

    print(f"[Run {run_id}, seed {seed}] Training main model...")
    model_path = train_main_model(
        run_id,
        bank_global=bank_global,
        output_dir=run_dir,
        encoder_path=encoder_path,
        cache_paths=cache_paths,
        data_set=data_set,
        seed=seed,
    )

    print(f"[Run {run_id}, seed {seed}] Evaluating random 10% test sample...")
    report_path = evaluate_random_test_subset(
        run_id,
        output_dir=run_dir,
        encoder_path=encoder_path,
        model_path=model_path,
        bank_global=bank_global,
        cache_paths=cache_paths,
        data_set=data_set,
        test_fraction=TEST_FRACTION,
        seed=seed,
    )

    print(
        f"[Run {run_id}, seed {seed}] Complete. encoder={encoder_path}, "
        f"model={model_path}, report={report_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run seeded AAT-RAG experiments."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help="parent directory for the isolated run directories",
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
    return parser.parse_args()


def main():
    args = parse_args()
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    output_root = os.path.abspath(args.output_root)
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
            # The spawned interpreter reads this before importing Python modules.
            os.environ["PYTHONHASHSEED"] = str(seed)
            process = mp.Process(
                target=run_complete_pipeline,
                args=(run_id, seed, output_root, args.cpu_threads),
                name=f"aat_pipeline_{run_id}_seed{seed}",
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
        raise SystemExit(f"Parallel pipelines failed: {failed}")
    print(f"All {len(runs)} isolated pipelines completed successfully.")


if __name__ == "__main__":
    main()
