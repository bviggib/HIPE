import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from fire import Fire

ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = ROOT / "scripts" / "fig2_experiments.py"


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _to_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _to_cli_bool(value: bool) -> str:
    return "true" if value else "false"


def _default_max_workers() -> int:
    # Leave one core free by default for OS and interactive work.
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _default_test_experiment_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"test_{stamp}"


def _build_seed_command(
    *,
    python_exec: str,
    experiment_name: str,
    seed: int,
    q: int,
    num_batches: int,
    results_root: str,
    include_qpstd: bool,
    skip_completed: bool,
    resume_batches: bool,
    run: bool,
    methods,
    objectives,
):
    cmd = [
        python_exec,
        str(TARGET_SCRIPT),
        f"--experiment_name={experiment_name}",
        f"--seed_start={seed}",
        "--num_seeds=1",
        f"--q={q}",
        f"--num_batches={num_batches}",
        f"--results_root={results_root}",
        f"--include_qpstd={_to_cli_bool(include_qpstd)}",
        f"--skip_completed={_to_cli_bool(skip_completed)}",
        f"--resume_batches={_to_cli_bool(resume_batches)}",
        f"--run={_to_cli_bool(run)}",
    ]

    method_list = _to_list(methods)
    objective_list = _to_list(objectives)

    if method_list:
        cmd.append(f"--methods={','.join(method_list)}")
    if objective_list:
        cmd.append(f"--objectives={','.join(objective_list)}")

    return cmd


def _seed_worker(
    *,
    python_exec: str,
    experiment_name: str,
    seed: int,
    q: int,
    num_batches: int,
    results_root: str,
    include_qpstd: bool,
    skip_completed: bool,
    resume_batches: bool,
    run: bool,
    methods,
    objectives,
    limit_blas_threads: bool,
    dry_run: bool,
):
    cmd = _build_seed_command(
        python_exec=python_exec,
        experiment_name=experiment_name,
        seed=seed,
        q=q,
        num_batches=num_batches,
        results_root=results_root,
        include_qpstd=include_qpstd,
        skip_completed=skip_completed,
        resume_batches=resume_batches,
        run=run,
        methods=methods,
        objectives=objectives,
    )

    print(f"[launch] seed={seed} cmd={shlex.join(cmd)}")
    if dry_run:
        return {"seed": seed, "ok": True, "returncode": 0}

    env = os.environ.copy()
    if limit_blas_threads:
        # Prevent oversubscription when many seed subprocesses run concurrently.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    ok = completed.returncode == 0
    status = "done" if ok else "failed"
    print(f"[{status}] seed={seed} returncode={completed.returncode}")
    return {"seed": seed, "ok": ok, "returncode": completed.returncode}


def main(
    experiment_name: str | None = None,
    methods: str | list[str] | None = None,
    objectives: str | list[str] | None = None,
    num_seeds: int = 100,
    seed_start: int = 0,
    q: int = 16,
    num_batches: int = 4,
    results_root: str = "results",
    include_qpstd: bool | str = True,
    skip_completed: bool | str = True,
    resume_batches: bool | str = True,
    run: bool | str = True,
    parallel: bool | str = True,
    max_workers: int | None = None,
    python_exec: str | None = None,
    limit_blas_threads: bool | str = True,
    dry_run: bool | str = False,
):
    include_qpstd = _to_bool(include_qpstd)
    skip_completed = _to_bool(skip_completed)
    resume_batches = _to_bool(resume_batches)
    run = _to_bool(run)
    parallel = _to_bool(parallel)
    limit_blas_threads = _to_bool(limit_blas_threads)
    dry_run = _to_bool(dry_run)

    is_testing_mode = dry_run or (not run)

    if experiment_name is None:
        experiment_name = _default_test_experiment_name() if is_testing_mode else "fig2_seqpstdbald"
    elif is_testing_mode and not experiment_name.startswith("test_"):
        original_name = experiment_name
        experiment_name = f"test_{original_name}"
        print(
            "[testing mode] "
            f"remapped experiment_name from {original_name} to {experiment_name}"
        )

    if python_exec is None:
        python_exec = sys.executable

    if not TARGET_SCRIPT.exists():
        raise FileNotFoundError(f"Target script not found: {TARGET_SCRIPT}")

    seed_values = list(range(seed_start, seed_start + num_seeds))
    if max_workers is None:
        max_workers = _default_max_workers()
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    worker_count = min(max_workers, len(seed_values)) if seed_values else 1

    print("Parallel fig2 runner")
    print(f"Target script: {TARGET_SCRIPT}")
    print(f"Python executable: {python_exec}")
    print(f"Experiment name: {experiment_name}")
    print(f"Seeds: {seed_values[0]}..{seed_values[-1]} (count={len(seed_values)})")
    print(f"Parallel: {parallel}")
    print(f"Workers: {worker_count}")
    print(f"Limit BLAS threads: {limit_blas_threads}")
    print(f"Dry run: {dry_run}")

    results = []
    if parallel and len(seed_values) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _seed_worker,
                    python_exec=python_exec,
                    experiment_name=experiment_name,
                    seed=seed,
                    q=q,
                    num_batches=num_batches,
                    results_root=results_root,
                    include_qpstd=include_qpstd,
                    skip_completed=skip_completed,
                    resume_batches=resume_batches,
                    run=run,
                    methods=methods,
                    objectives=objectives,
                    limit_blas_threads=limit_blas_threads,
                    dry_run=dry_run,
                )
                for seed in seed_values
            ]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for seed in seed_values:
            results.append(
                _seed_worker(
                    python_exec=python_exec,
                    experiment_name=experiment_name,
                    seed=seed,
                    q=q,
                    num_batches=num_batches,
                    results_root=results_root,
                    include_qpstd=include_qpstd,
                    skip_completed=skip_completed,
                    resume_batches=resume_batches,
                    run=run,
                    methods=methods,
                    objectives=objectives,
                    limit_blas_threads=limit_blas_threads,
                    dry_run=dry_run,
                )
            )

    failed = sorted(item["seed"] for item in results if not item["ok"])
    succeeded = sorted(item["seed"] for item in results if item["ok"])

    print(
        f"Summary: succeeded={len(succeeded)}, failed={len(failed)}, total={len(results)}"
    )
    if failed:
        raise RuntimeError(f"Seed subprocesses failed for seeds: {failed}")


if __name__ == "__main__":
    Fire(main)
