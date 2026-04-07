import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from fire import Fire
from omegaconf import OmegaConf

from active_init.al import get_rmse_and_mll
from active_init.registry.acquisition import get_acquisition_function
from active_init.registry.model import get_model
from experiments.evaluation import get_model_hyperparameters
from experiments.objective import get_objective_function


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

# fig2_seq_qpstd change start
FIG2_QPSTD_METHODS = (
    "hipe",
    "bald",
    "nipv",
    "sobol",
    "random",
    "qPSTD",
    "seq_PSTD_BALD",
)

QPSTD_METHOD = "qPSTD"
BALD_METHOD = "bald"
SEQ_PSTD_BALD_METHOD = "seq_PSTD_BALD"
# fig2_seq_qpstd change end

# fig2_resume_batches change start
DEFAULT_MODEL_CONFIG = "fb"
DEFAULT_ACQ_OPT_CONFIG = "default"
# fig2_resume_batches change end


def _parse_name_from_objective_config(objective: str) -> str:
    """Resolve results folder name (objective.name in Hydra config) from objective key."""
    config_path = Path("configs") / "objective" / f"{objective}.yaml"
    if not config_path.exists():
        return objective

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("name:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            return value or objective
    return objective


def _is_seed_complete(seed_dir: Path, expected_steps: int | None = None) -> bool:
    """A seed is considered complete when both init and al metrics are present and readable."""
    init_file = seed_dir / "init.json"
    al_file = seed_dir / "al.json"

    if not init_file.exists() or not al_file.exists():
        return False

    try:
        with al_file.open("r", encoding="utf-8") as f:
            al_metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(al_metrics, dict):
        return False

    if "RMSE" not in al_metrics or "MLL" not in al_metrics:
        return False

    rmse = al_metrics["RMSE"]
    mll = al_metrics["MLL"]
    if not isinstance(rmse, list) or not isinstance(mll, list):
        return False

    if expected_steps is not None and (
        len(rmse) != expected_steps or len(mll) != expected_steps
    ):
        return False

    return True


def _to_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


# fig2_resume_batches change start
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
# fig2_resume_batches change end


# fig2_seq_qpstd change start
def _run_method_seed(
    *,
    experiment_name: str,
    objective: str,
    method: str,
    q: int,
    budget: int,
    seed: int,
    run: bool,
):
    cmd = [
        sys.executable,
        "main.py",
        "task=al",
        f"experiment.name={experiment_name}",
        f"objective={objective}",
        f"init={method}",
        f"acq_opt.q={q}",
        f"experiment.budget={budget}",
        f"seed={seed}",
    ]
    print(" ".join(cmd))
    if run:
        subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# fig2_resume_batches change start
def _continue_method_seed(
    *,
    results_root: str,
    experiment_name: str,
    objective_key: str,
    objective_save_name: str,
    method: str,
    seed: int,
    q: int,
    num_batches: int,
    run: bool,
) -> bool:
    if not run:
        return False

    seed_dir = (
        Path(results_root)
        / experiment_name
        / objective_save_name
        / method
        / f"seed{seed}"
    )
    init_file = seed_dir / "init.json"
    al_file = seed_dir / "al.json"
    if not init_file.exists() or not al_file.exists():
        return False

    try:
        al_metrics = _load_json(al_file)
    except (OSError, json.JSONDecodeError):
        return False

    rmse = al_metrics.get("RMSE")
    mll = al_metrics.get("MLL")
    train_data = al_metrics.get("TrainingData", {})
    if not isinstance(rmse, list) or not isinstance(mll, list):
        return False
    if len(rmse) == 0 or len(rmse) != len(mll) or len(rmse) >= num_batches:
        return False

    try:
        train_X = torch.tensor(train_data["train_X"], dtype=torch.float64)
        train_Y = torch.tensor(train_data["train_Y"], dtype=torch.float64)
    except (KeyError, TypeError, ValueError):
        return False

    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    completed_steps = len(rmse)
    if completed_steps == 0 or train_X.shape[0] % completed_steps != 0:
        return False

    previous_q = train_X.shape[0] // completed_steps
    if previous_q != q:
        print(
            "[resume disabled] "
            f"objective={objective_key}, method={method}, seed={seed}, "
            f"existing_q={previous_q}, requested_q={q}"
        )
        return False

    objective_cfg_path = Path("configs") / "objective" / f"{objective_key}.yaml"
    init_cfg_path = Path("configs") / "init" / f"{method}.yaml"
    model_cfg_path = Path("configs") / "model" / f"{DEFAULT_MODEL_CONFIG}.yaml"
    acq_opt_cfg_path = Path("configs") / "acq_opt" / f"{DEFAULT_ACQ_OPT_CONFIG}.yaml"
    config_cfg_path = Path("configs") / "config.yaml"
    if not (
        objective_cfg_path.exists()
        and init_cfg_path.exists()
        and model_cfg_path.exists()
        and acq_opt_cfg_path.exists()
        and config_cfg_path.exists()
    ):
        return False

    objective_cfg = OmegaConf.load(objective_cfg_path)
    init_cfg = OmegaConf.load(init_cfg_path)
    model_kwargs = OmegaConf.load(model_cfg_path)
    acq_opt_kwargs = OmegaConf.to_container(OmegaConf.load(acq_opt_cfg_path), resolve=True)
    config_cfg = OmegaConf.load(config_cfg_path)

    objective = get_objective_function(objective_cfg, seed=seed)
    acq_opt_kwargs["q"] = q
    num_test_points = int(config_cfg.evaluation.test_set_size)
    budget = q * num_batches

    rmses = list(rmse)
    mlls = list(mll)
    model, _ = get_model(
        objective=objective,
        train_X=train_X,
        train_Y=train_Y,
        model_kwargs=model_kwargs,
        skip_kwargs=True,
    )

    print(
        "[resume seed] "
        f"objective={objective_key} (saved_as={objective_save_name}), method={method}, "
        f"seed={seed}, steps={completed_steps}->{num_batches}"
    )

    while len(train_X) < budget:
        init_kwargs_dict = OmegaConf.to_container(init_cfg, resolve=False)
        if init_kwargs_dict["name"] not in ["sobol", "random"]:
            acq_func = get_acquisition_function(
                objective=objective,
                acq_name=init_kwargs_dict["name"],
                model=model,
                acq_kwargs=dict(init_kwargs_dict.get("acq_kwargs") or {}),
                bounds=objective.bounds,
                mc_strategy=init_kwargs_dict.get("dist", None),
            )
            from gpytorch import settings

            with settings.detach_test_caches(False):
                candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=objective.bounds,
                    **acq_opt_kwargs,
                )
        elif init_kwargs_dict["name"] == "sobol":
            candidates = draw_sobol_samples(
                bounds=objective.bounds,
                q=acq_opt_kwargs["q"],
                n=1,
            ).squeeze(0)
        else:
            candidates = torch.rand(
                (acq_opt_kwargs["q"], objective.bounds.shape[-1])
            ).to(objective.bounds.device)

        new_X = candidates.detach().to(train_X)
        new_Y = objective(new_X).unsqueeze(-1).to(train_Y)
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        model, _ = get_model(
            objective=objective,
            train_X=train_X,
            train_Y=train_Y,
            model_kwargs=model_kwargs,
            skip_kwargs=True,
        )
        rmse_val, mll_val = get_rmse_and_mll(
            num_test_points=num_test_points,
            model=model,
            objective=objective,
        )
        rmses.append(rmse_val)
        mlls.append(mll_val)

    noiseless_objective = deepcopy(objective)
    noiseless_objective.objective.noise_std = 0
    noiseless_objective.noise_std = 0
    train_f = noiseless_objective(train_X).unsqueeze(-1)

    al_metrics["RMSE"] = rmses
    al_metrics["MLL"] = mlls
    al_metrics["Hyperparameters"] = get_model_hyperparameters(model)
    al_metrics["TrainingData"] = {
        "train_X": train_X.tolist(),
        "train_Y": train_Y.tolist(),
        "train_f": train_f.tolist(),
    }
    _save_json(al_file, al_metrics)
    return True


def _run_or_resume_method_seed(
    *,
    results_root: str,
    experiment_name: str,
    objective: str,
    objective_save_name: str,
    method: str,
    q: int,
    budget: int,
    num_batches: int,
    seed: int,
    run: bool,
    resume_batches: bool,
) -> bool:
    if resume_batches and _continue_method_seed(
        results_root=results_root,
        experiment_name=experiment_name,
        objective_key=objective,
        objective_save_name=objective_save_name,
        method=method,
        seed=seed,
        q=q,
        num_batches=num_batches,
        run=run,
    ):
        return True

    _run_method_seed(
        experiment_name=experiment_name,
        objective=objective,
        method=method,
        q=q,
        budget=budget,
        seed=seed,
        run=run,
    )
    return False
# fig2_resume_batches change end


def _compose_seq_pstd_bald_seed(
    *,
    results_root: str,
    experiment_name: str,
    objective_save_name: str,
    seed: int,
    num_batches: int,
) -> bool:
    qpstd_seed_dir = (
        Path(results_root)
        / experiment_name
        / objective_save_name
        / QPSTD_METHOD
        / f"seed{seed}"
    )
    bald_seed_dir = (
        Path(results_root)
        / experiment_name
        / objective_save_name
        / BALD_METHOD
        / f"seed{seed}"
    )
    seq_seed_dir = (
        Path(results_root)
        / experiment_name
        / objective_save_name
        / SEQ_PSTD_BALD_METHOD
        / f"seed{seed}"
    )

    if not _is_seed_complete(qpstd_seed_dir, expected_steps=num_batches):
        return False
    if not _is_seed_complete(bald_seed_dir, expected_steps=num_batches):
        return False

    qpstd_init = _load_json(qpstd_seed_dir / "init.json")
    qpstd_al = _load_json(qpstd_seed_dir / "al.json")
    bald_al = _load_json(bald_seed_dir / "al.json")

    split_step = math.ceil(num_batches / 2)
    qpstd_rmse = list(qpstd_al["RMSE"])
    qpstd_mll = list(qpstd_al["MLL"])
    bald_rmse = list(bald_al["RMSE"])
    bald_mll = list(bald_al["MLL"])

    seq_rmse = qpstd_rmse[:split_step] + bald_rmse[split_step:num_batches]
    seq_mll = qpstd_mll[:split_step] + bald_mll[split_step:num_batches]

    seq_al = dict(qpstd_al)
    seq_al["RMSE"] = seq_rmse
    seq_al["MLL"] = seq_mll
    seq_al["SeqMethod"] = {
        "name": SEQ_PSTD_BALD_METHOD,
        "phase1_method": QPSTD_METHOD,
        "phase2_method": BALD_METHOD,
        "split_step": split_step,
        "num_steps": num_batches,
    }

    _save_json(seq_seed_dir / "init.json", qpstd_init)
    _save_json(seq_seed_dir / "al.json", seq_al)
    print(
        "[compose seq] "
        f"objective={objective_save_name}, seed={seed}, "
        f"split_step={split_step}, saved_method={SEQ_PSTD_BALD_METHOD}"
    )
    return True
# fig2_seq_qpstd change end


def main(
    experiment_name: str = "fig2_al",
    methods: str | list[str] | None = None,
    objectives: str | list[str] | None = None,
    num_seeds: int = 4,
    seed_start: int = 0,
    q: int = 16,
    num_batches: int = 4,
    results_root: str = "results",
    include_qpstd: bool | str = False,
    skip_completed: bool | str = True,
    resume_batches: bool | str = True,
    run: bool | str = True,
):
    # fig2_resume_batches change start
    include_qpstd = _to_bool(include_qpstd)
    skip_completed = _to_bool(skip_completed)
    resume_batches = _to_bool(resume_batches)
    run = _to_bool(run)
    # fig2_resume_batches change end

    # fig2_seq_qpstd change start
    default_methods = FIG2_QPSTD_METHODS if include_qpstd else FIG2_METHODS
    methods = _to_list(methods) or list(default_methods)
    # fig2_seq_qpstd change end
    objectives = _to_list(objectives) or list(FIG2_OBJECTIVES)
    budget = q * num_batches
    seed_values = list(range(seed_start, seed_start + num_seeds))

    print(f"Running task=al with q={q}, num_batches={num_batches}, budget={budget}, num_seeds={num_seeds}")
    print(f"Experiment name: {experiment_name}")
    print(f"Methods: {methods}")
    print(f"Objectives: {objectives}")
    print(f"Include qPSTD methods: {include_qpstd}")
    print(f"Skip completed: {skip_completed}")
    print(f"Resume batches: {resume_batches}")

    launched = 0
    resumed = 0
    skipped = 0

    for objective in objectives:
        objective_save_name = _parse_name_from_objective_config(objective)
        objective_complete = True

        for method in methods:
            method_complete = True

            for seed in seed_values:
                seed_dir = Path(results_root) / experiment_name / objective_save_name / method / f"seed{seed}"
                if skip_completed and _is_seed_complete(seed_dir, expected_steps=num_batches):
                    skipped += 1
                    print(
                        "[skip seed] "
                        f"objective={objective} (saved_as={objective_save_name}), method={method}, seed={seed}"
                    )
                    continue

                method_complete = False
                objective_complete = False

                # fig2_seq_qpstd change start
                if method != SEQ_PSTD_BALD_METHOD:
                    used_resume = _run_or_resume_method_seed(
                        results_root=results_root,
                        experiment_name=experiment_name,
                        objective=objective,
                        objective_save_name=objective_save_name,
                        method=method,
                        q=q,
                        budget=budget,
                        num_batches=num_batches,
                        seed=seed,
                        run=run,
                        resume_batches=resume_batches,
                    )
                    if used_resume:
                        resumed += 1
                    else:
                        launched += 1
                    continue

                qpstd_seed_dir = (
                    Path(results_root)
                    / experiment_name
                    / objective_save_name
                    / QPSTD_METHOD
                    / f"seed{seed}"
                )
                bald_seed_dir = (
                    Path(results_root)
                    / experiment_name
                    / objective_save_name
                    / BALD_METHOD
                    / f"seed{seed}"
                )

                if not (skip_completed and _is_seed_complete(qpstd_seed_dir, expected_steps=num_batches)):
                    used_resume = _run_or_resume_method_seed(
                        results_root=results_root,
                        experiment_name=experiment_name,
                        objective=objective,
                        objective_save_name=objective_save_name,
                        method=QPSTD_METHOD,
                        q=q,
                        budget=budget,
                        num_batches=num_batches,
                        seed=seed,
                        run=run,
                        resume_batches=resume_batches,
                    )
                    if used_resume:
                        resumed += 1
                    else:
                        launched += 1
                else:
                    print(
                        "[skip phase] "
                        f"objective={objective} (saved_as={objective_save_name}), method={QPSTD_METHOD}, seed={seed}"
                    )

                if not (skip_completed and _is_seed_complete(bald_seed_dir, expected_steps=num_batches)):
                    used_resume = _run_or_resume_method_seed(
                        results_root=results_root,
                        experiment_name=experiment_name,
                        objective=objective,
                        objective_save_name=objective_save_name,
                        method=BALD_METHOD,
                        q=q,
                        budget=budget,
                        num_batches=num_batches,
                        seed=seed,
                        run=run,
                        resume_batches=resume_batches,
                    )
                    if used_resume:
                        resumed += 1
                    else:
                        launched += 1
                else:
                    print(
                        "[skip phase] "
                        f"objective={objective} (saved_as={objective_save_name}), method={BALD_METHOD}, seed={seed}"
                    )

                composed = _compose_seq_pstd_bald_seed(
                    results_root=results_root,
                    experiment_name=experiment_name,
                    objective_save_name=objective_save_name,
                    seed=seed,
                    num_batches=num_batches,
                )
                if not composed:
                    raise RuntimeError(
                        "Unable to compose seq_PSTD_BALD seed after phase runs. "
                        f"objective={objective}, seed={seed}"
                    )
                # fig2_seq_qpstd change end

            if skip_completed and method_complete:
                print(
                    "[skip method] "
                    f"objective={objective} (saved_as={objective_save_name}), method={method}, "
                    "all requested seeds are complete"
                )

        if skip_completed and objective_complete:
            print(
                f"[skip objective] objective={objective} (saved_as={objective_save_name}), "
                "all requested methods/seeds are complete"
            )

    print(
        f"Summary: launched={launched}, resumed={resumed}, "
        f"skipped={skipped}, total={launched + resumed + skipped}"
    )


if __name__ == "__main__":
    Fire(main)
