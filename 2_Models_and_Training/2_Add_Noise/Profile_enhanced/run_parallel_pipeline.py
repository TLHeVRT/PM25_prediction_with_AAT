import argparse
import multiprocessing as mp
import os
import random
import sys


sys.dont_write_bytecode = True

OUTPUT_ROOT = "parallel_runs"
TEST_SAMPLE_FRACTION = 1.0


def validate_seed(value):
    seed = int(value)
    if not 0 <= seed <= 2 ** 32 - 1:
        raise argparse.ArgumentTypeError(
            "seed 必须位于 0 到 4294967295。"
        )
    return seed


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("该参数必须是大于等于 1 的整数。")
    return parsed


def configure_cpu_threads(cpu_threads):
    thread_count = str(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count


def seed_everything(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_worker(worker_id, seed, output_root, cpu_threads):
    configure_cpu_threads(cpu_threads)

    # Import numerical libraries only after applying this worker's thread limit.
    import numpy as np
    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    from generate_cache import generate_split_caches
    from mult_model import DataSet
    from run_models import train_main_model, evaluate_random_test_subset

    seed_everything(seed, np, torch)

    worker_dir = os.path.abspath(
        os.path.join(output_root, f"worker_{worker_id}_seed{seed}")
    )
    os.makedirs(worker_dir, exist_ok=True)

    print(
        f"[Worker {worker_id} | seed={seed} | CPU threads={cpu_threads}] "
        "载入并归一化四年数据..."
    )
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_set = DataSet(os.path.join(project_dir, "data_matrix.npy"))

    print(f"[Worker {worker_id} | seed={seed}] 生成独享缓存到 {worker_dir}")
    cache_files = generate_split_caches(
        data_set,
        output_dir=worker_dir,
        seed=seed,
        T_short=144,
        pred_len=48,
    )

    print(f"[Worker {worker_id} | seed={seed}] 训练主模型...")
    training_result = train_main_model(
        data_set=data_set,
        output_dir=worker_dir,
        cache_files=cache_files,
        seed=seed,
    )

    report_file = os.path.join(
        worker_dir, f"test_no_trend_loss_report_seed{seed}.csv"
    )
    summaries = evaluate_random_test_subset(
        data_set=data_set,
        test_cache_files=cache_files["test"],
        training_result=training_result,
        report_file=report_file,
        sample_fraction=TEST_SAMPLE_FRACTION,
        random_seed=seed,
        batch_size=32,
    )

    for skill_group in ("low", "mid", "high"):
        summary = summaries[skill_group]
        print(
            f"[Worker {worker_id} | seed={seed} | {skill_group}] "
            f"完成：{TEST_SAMPLE_FRACTION:.0%} 测试抽样 "
            f"{summary['sampled_count']}，按统一规则过滤 "
            f"{summary['filtered_count']}，Base "
            f"{summary['mean_base_loss']:.4f}，DTW "
            f"{summary['mean_dtw_loss']:.4f}，TopK "
            f"{summary['mean_topk_loss']:.4f}，加权总损失 "
            f"{summary['mean_weighted_total_loss']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="多种子独立缓存、训练与测试流水线。"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=validate_seed,
        required=True,
        metavar="SEED",
        help="一个或多个独立进程的随机种子。",
    )
    parser.add_argument(
        "--max-parallel",
        type=positive_int,
        required=True,
        help="每轮最多同时运行的种子进程数。",
    )
    parser.add_argument(
        "--cpu-threads",
        type=positive_int,
        required=True,
        help="每个种子进程可使用的 CPU 线程数。",
    )
    return parser.parse_args()


def main(seeds, max_parallel, cpu_threads, output_root=OUTPUT_ROOT):
    if not seeds:
        raise ValueError("至少需要提供一个 seed。")

    mp.set_start_method("spawn", force=True)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    worker_specs = list(enumerate(seeds, start=1))
    failed_workers = []
    total_rounds = (len(worker_specs) + max_parallel - 1) // max_parallel

    for round_index, round_start in enumerate(
            range(0, len(worker_specs), max_parallel), start=1
    ):
        round_specs = worker_specs[round_start:round_start + max_parallel]
        print(
            f"启动第 {round_index}/{total_rounds} 轮："
            f"{len(round_specs)} 个种子并行，每个进程使用 "
            f"{cpu_threads} 个 CPU 线程。"
        )

        processes = []
        for worker_id, seed in round_specs:
            process = mp.Process(
                target=run_worker,
                args=(worker_id, seed, output_root, cpu_threads),
                name=f"model-worker-{worker_id}-seed{seed}",
            )
            process.start()
            processes.append((worker_id, seed, process))

        for worker_id, seed, process in processes:
            process.join()
            if process.exitcode != 0:
                failed_workers.append((worker_id, seed, process.exitcode))

    if failed_workers:
        failures = ", ".join(
            f"worker_{worker_id}_seed{seed}: exitcode={exitcode}"
            for worker_id, seed, exitcode in failed_workers
        )
        raise RuntimeError(f"并行任务失败：{failures}")

    print(f"全部 {len(seeds)} 个种子进程均已完成。")


if __name__ == "__main__":
    args = parse_args()
    main(
        seeds=args.seeds,
        max_parallel=args.max_parallel,
        cpu_threads=args.cpu_threads,
    )
