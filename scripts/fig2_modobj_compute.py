import json
import os
import sys
import hashlib
from copy import deepcopy
from pathlib import Path

import torch
from fire import Fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.objective import get_objective_function
from active_init.registry.model import get_model


FIG2_OBJECTIVES = (
    "lcbench_australian_al",
    "lcbench_car_al",
    "hartmann6",
    "hartmann6_12",
    "ishigami",
)

FIG2_METHODS = (
    "hipe",
    "bald",
    "nipv",
    "sobol",
    "random",
)

FIG2_QPSTD_METHODS = (
    "hipe",
    "bald",
    "nipv",
    "sobol",
    "random",
    "qPSTD",
    "seq_PSTD_BALD",
    "seq_pstdhipe11",
    "seq_pstdhipe31",
)


def _parse_name_from_objective_config(objective: str) -> str:
    config_path = Path("configs") / "objective" / f"{objective}.yaml"
    if not config_path.exists():
        return objective

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("name:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            return value or objective
    return objective


def _to_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**31 - 1)


def _lhs_samples(num_samples: int, dim: int, seed: int) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    cut = torch.linspace(0.0, 1.0, num_samples + 1, dtype=torch.float64)
    u = torch.rand((num_samples, dim), generator=generator, dtype=torch.float64)
    a = cut[:-1].unsqueeze(1)
    b = cut[1:].unsqueeze(1)
    points = a + (b - a) * u
    for dim_idx in range(dim):
        order = torch.randperm(num_samples, generator=generator)
        points[:, dim_idx] = points[order, dim_idx]
    return points


def _ensure_test_set(
    *,
    results_root: Path,
    objective_key: str,
    objective_save_name: str,
    objective,
    test_size: int,
    test_seed: int,
    overwrite: bool,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    test_set_path = results_root / objective_save_name / "test_set.json"
    if test_set_path.exists() and not overwrite:
        payload = _load_json(test_set_path)
        test_X = torch.tensor(payload["test_X"], dtype=torch.float64)
        test_f = torch.tensor(payload["test_f"], dtype=torch.float64)
        return test_X, test_f, test_set_path

    lhs_seed = _stable_seed(objective_save_name, test_seed)
    test_X = _lhs_samples(test_size, objective.dim, lhs_seed)

    bounds = getattr(objective, "bounds", None)
    if bounds is not None and hasattr(bounds, "shape") and bounds.shape[0] == 2:
        bounds = bounds.to(dtype=torch.float64)
        test_X = bounds[0] + (bounds[1] - bounds[0]) * test_X

    noiseless_objective = deepcopy(objective)
    if hasattr(noiseless_objective, "objective"):
        noiseless_objective.objective.noise_std = 0
    if hasattr(noiseless_objective, "noise_std"):
        noiseless_objective.noise_std = 0
    test_f = noiseless_objective(test_X).unsqueeze(-1).to(torch.float64)

    payload = {
        "objective": objective_key,
        "objective_name": objective_save_name,
        "test_size": test_size,
        "dim": int(objective.dim),
        "lhs_seed": lhs_seed,
        "test_X": test_X.tolist(),
        "test_f": test_f.tolist(),
    }
    _save_json(test_set_path, payload)
    return test_X, test_f, test_set_path


def _load_training_data(al_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        data = _load_json(al_path)
    except (OSError, json.JSONDecodeError):
        return None

    train_data = data.get("TrainingData", {})
    if "train_X" not in train_data or "train_Y" not in train_data:
        return None

    try:
        train_X = torch.tensor(train_data["train_X"], dtype=torch.float64)
        train_Y = torch.tensor(train_data["train_Y"], dtype=torch.float64)
    except (TypeError, ValueError):
        return None

    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    if train_X.shape[0] != train_Y.shape[0]:
        return None
    return train_X, train_Y


def _load_existing_smape(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return [], []

    smape = payload.get("sMAPE", {})
    means = list(smape.get("Mean") or [])
    maxes = list(smape.get("Max") or [])
    if len(means) != len(maxes):
        return [], []
    return means, maxes


def _compute_smape(
    *,
    objective,
    model_kwargs,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    test_X: torch.Tensor,
    test_f: torch.Tensor,
    q: int,
    num_batches: int,
    smape_eps: float,
    seed: int,
    resume_from: int = 0,
    existing_means: list[float] | None = None,
    existing_maxes: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    means = list(existing_means or [])
    maxes = list(existing_maxes or [])

    torch.manual_seed(seed)

    total_points = train_X.shape[0]
    max_batches = min(num_batches, total_points // q)

    for batch_idx in range(resume_from, max_batches):
        end_idx = (batch_idx + 1) * q
        batch_X = train_X[:end_idx]
        batch_Y = train_Y[:end_idx]

        model, _ = get_model(
            objective=objective,
            train_X=batch_X,
            train_Y=batch_Y,
            model_kwargs=model_kwargs,
            skip_kwargs=True,
        )
        model.eval()
        with torch.no_grad():
            pred = model.posterior(test_X).mean.squeeze(-1)

        actual = test_f.squeeze(-1)
        denom = pred.abs() + actual.abs() + smape_eps
        smape = (pred - actual).abs() / denom * 200.0

        means.append(float(smape.mean().item()))
        maxes.append(float(smape.max().item()))

    return means, maxes


def main(
    experiment_name: str = "fig2_seqq8b8",
    methods: str | list[str] | None = None,
    objectives: str | list[str] | None = None,
    num_seeds: int = 50,
    seed_start: int = 0,
    q: int = 8,
    num_batches: int = 8,
    results_root: str = "results",
    include_qpstd: bool | str = False,
    skip_completed: bool | str = True,
    resume_batches: bool | str = True,
    test_size: int = 10000,
    test_seed: int = 0,
    smape_eps: float = 1e-8,
    model_config: str = "fb",
    overwrite_test_set: bool | str = False,
    run: bool | str = True,
):
    include_qpstd = _to_bool(include_qpstd)
    skip_completed = _to_bool(skip_completed)
    resume_batches = _to_bool(resume_batches)
    overwrite_test_set = _to_bool(overwrite_test_set)
    run = _to_bool(run)

    default_methods = FIG2_QPSTD_METHODS if include_qpstd else FIG2_METHODS
    methods = _to_list(methods) or list(default_methods)
    objectives = _to_list(objectives) or list(FIG2_OBJECTIVES)

    results_root_path = Path(results_root) / experiment_name
    seed_values = list(range(seed_start, seed_start + num_seeds))

    print(
        "Computing sMAPE with q={}, num_batches={}, test_size={}, num_seeds={}".format(
            q, num_batches, test_size, num_seeds
        )
    )
    print(f"Experiment: {results_root_path}")
    print(f"Methods: {methods}")
    print(f"Objectives: {objectives}")

    processed = 0
    skipped = 0
    missing = 0

    for objective_key in objectives:
        objective_save_name = _parse_name_from_objective_config(objective_key)
        objective_cfg_path = Path("configs") / "objective" / f"{objective_key}.yaml"
        if not objective_cfg_path.exists():
            print(f"[skip objective] missing config: {objective_cfg_path}")
            continue

        objective_cfg = OmegaConf.load(objective_cfg_path)
        objective = get_objective_function(objective_cfg, seed=test_seed)

        test_X, test_f, test_set_path = _ensure_test_set(
            results_root=results_root_path,
            objective_key=objective_key,
            objective_save_name=objective_save_name,
            objective=objective,
            test_size=test_size,
            test_seed=test_seed,
            overwrite=overwrite_test_set,
        )

        model_cfg_path = Path("configs") / "model" / f"{model_config}.yaml"
        if not model_cfg_path.exists():
            print(f"[skip objective] missing model config: {model_cfg_path}")
            continue
        model_kwargs = OmegaConf.load(model_cfg_path)

        for method in methods:
            for seed in seed_values:
                seed_dir = (
                    results_root_path
                    / objective_save_name
                    / method
                    / f"seed{seed}"
                )
                al_path = seed_dir / "al.json"
                mod_path = seed_dir / "al_modobj.json"
                if not al_path.exists():
                    missing += 1
                    print(
                        "[skip seed] missing al.json: "
                        f"objective={objective_key} (saved_as={objective_save_name}), "
                        f"method={method}, seed={seed}"
                    )
                    continue

                existing_means, existing_maxes = _load_existing_smape(mod_path)
                if skip_completed and len(existing_means) >= num_batches:
                    skipped += 1
                    continue

                if not resume_batches:
                    existing_means, existing_maxes = [], []

                train_data = _load_training_data(al_path)
                if train_data is None:
                    missing += 1
                    print(
                        "[skip seed] invalid TrainingData: "
                        f"objective={objective_key} (saved_as={objective_save_name}), "
                        f"method={method}, seed={seed}"
                    )
                    continue

                train_X, train_Y = train_data
                if train_X.shape[0] < q:
                    missing += 1
                    print(
                        "[skip seed] insufficient train_X: "
                        f"objective={objective_key} (saved_as={objective_save_name}), "
                        f"method={method}, seed={seed}, points={train_X.shape[0]}"
                    )
                    continue

                available_batches = min(num_batches, train_X.shape[0] // q)
                if available_batches < num_batches:
                    print(
                        "[warning] fewer batches than requested: "
                        f"objective={objective_key} (saved_as={objective_save_name}), "
                        f"method={method}, seed={seed}, "
                        f"available={available_batches}, requested={num_batches}"
                    )

                start_batch = len(existing_means)
                if skip_completed and start_batch >= available_batches:
                    skipped += 1
                    continue

                if not run:
                    print(
                        "[dry-run] "
                        f"objective={objective_key} (saved_as={objective_save_name}), "
                        f"method={method}, seed={seed}, batches={available_batches}"
                    )
                    continue

                means, maxes = _compute_smape(
                    objective=objective,
                    model_kwargs=model_kwargs,
                    train_X=train_X,
                    train_Y=train_Y,
                    test_X=test_X,
                    test_f=test_f,
                    q=q,
                    num_batches=available_batches,
                    smape_eps=smape_eps,
                    seed=seed,
                    resume_from=start_batch,
                    existing_means=existing_means,
                    existing_maxes=existing_maxes,
                )

                payload = {
                    "Objective": objective_key,
                    "ObjectiveName": objective_save_name,
                    "Method": method,
                    "Seed": seed,
                    "BatchSize": q,
                    "NumBatches": available_batches,
                    "sMAPE": {
                        "Mean": means,
                        "Max": maxes,
                    },
                    "TestSet": {
                        "path": os.path.relpath(test_set_path, seed_dir),
                        "size": test_size,
                        "seed": test_seed,
                        "eps": smape_eps,
                    },
                }
                _save_json(mod_path, payload)
                processed += 1

    print(
        "Summary: processed={}, skipped={}, missing={}"
        .format(processed, skipped, missing)
    )


if __name__ == "__main__":
    Fire(main)
