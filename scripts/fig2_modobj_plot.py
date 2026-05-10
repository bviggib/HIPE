import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.constants import (
    AL_METHOD_ORDER,
    BENCHMARKS,
    COLORS,
    METRIC_NAMES,
    NAMES,
)
from experiments.performance import plot_relative_ranking


FIG2_FUNCTIONS = (
    "Australian",
    "car",
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
    "random",
    "seq_PSTD_BALD",
    "seq_pstdhipe11",
    "seq_pstdhipe31",
)


METRIC_SPECS = (
    {
        "name": "MLL",
        "label": METRIC_NAMES.get("MLL", "MLL"),
        "file": "al.json",
        "path": ("MLL",),
        "reduce": "logmeanexp",
    },
    {
        "name": "RMSE",
        "label": METRIC_NAMES.get("RMSE", "RMSE"),
        "file": "al.json",
        "path": ("RMSE",),
        "reduce": "mean",
    },
    {
        "name": "NRMSE",
        "label": "NRMSE",
        "file": "al_modobj.json",
        "path": ("NRMSE", "Mean"),
        "reduce": "identity",
    },
    {
        "name": "MaxNormErr",
        "label": "Max Norm Err",
        "file": "al_modobj.json",
        "path": ("NRMSE", "Max"),
        "reduce": "identity",
    },
)


def _extract_metric(data: dict, path: tuple[str, ...]):
    value = data
    for key in path:
        if key not in value:
            return None
        value = value[key]
    return value


def _reduce_metric(metric_data: np.ndarray, reduce_kind: str) -> np.ndarray:
    if reduce_kind == "logmeanexp" and metric_data.ndim >= 2:
        return -np.log(np.mean(np.exp(metric_data), axis=-1))
    if reduce_kind == "mean" and metric_data.ndim >= 2:
        return np.mean(metric_data, axis=-1)
    return metric_data


def plot_metrics_over_time(
    *,
    base_path: str,
    metric_specs: tuple[dict, ...],
    functions: list[str] | None,
    methods: list[str] | None,
    output_file: str | None,
    end_at: int | None,
    batch_size: int,
    relative_ranking: bool,
    show: bool,
):
    if base_path.endswith("/"):
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(base_path))
    elif isinstance(functions, str):
        functions = [functions]

    functions = list(filter(lambda x: "." not in x, functions))
    num_cols = len(functions)
    num_rows = len(metric_specs)

    fig, axes_all = plt.subplots(
        num_rows,
        num_cols + int(relative_ranking),
        figsize=(16, 4.75 * num_rows),
    )
    if not isinstance(axes_all, np.ndarray):
        axes_all = np.array([axes_all])
    axes_all = axes_all.reshape(num_rows, -1)

    for metric_idx, spec in enumerate(metric_specs):
        axes = axes_all[metric_idx]
        all_regrets = {}
        for ax_idx, (ax, func) in enumerate(zip(axes, functions)):
            all_regrets[func] = {}
            if not methods:
                methods = sorted(os.listdir(f"{base_path}/{func}/"))
            elif isinstance(methods, str):
                methods = [methods]
            methods = [m for m in AL_METHOD_ORDER if m in methods]

            x = None

            method_seeds = [
                set(
                    s.split("seed")[-1].split("/")[0]
                    for s in glob(
                        os.path.join(
                            base_path, func, method, f"seed*/{spec['file']}"
                        )
                    )
                )
                for method in methods
            ]
            if not method_seeds:
                continue

            common_seeds = set.intersection(*method_seeds)
            if not common_seeds:
                print(f"Warning: No common seeds for function '{func}'. Skipping.")
                continue

            seeds = sorted(list(common_seeds))
            for method in methods:
                metric_values = []
                for seed in seeds:
                    file_path = os.path.join(
                        base_path, func, method, f"seed{seed}", spec["file"]
                    )
                    if not os.path.exists(file_path):
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    metric_payload = _extract_metric(data, spec["path"])
                    if metric_payload is None:
                        continue

                    metric_data = np.array(metric_payload)
                    if metric_data.ndim == 0:
                        metric_data = np.array([metric_data.item()])
                    metric_data = _reduce_metric(metric_data, spec["reduce"])
                    metric_values.append(metric_data)

                if not metric_values:
                    continue

                metric_values = np.array(metric_values)
                if end_at:
                    metric_values = metric_values[..., :end_at]

                all_regrets[func][method] = metric_values

                avg_metric = metric_values.mean(axis=0)
                std_error = metric_values.std(axis=0) / np.sqrt(len(metric_values))

                x = np.arange(len(avg_metric)) + 1
                label = NAMES[method] if (ax_idx == 0 and metric_idx == 0) else "__nolabel__"
                ax.errorbar(
                    x,
                    avg_metric,
                    yerr=std_error,
                    label=label,
                    linestyle=":",
                    linewidth=1.5,
                    color=COLORS[method],
                    fmt="o",
                    elinewidth=2,
                    markersize=5,
                    capsize=4,
                )

            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis="both", which="major", labelsize=15)
            if x is not None:
                ax.set_xticks(x)
            if metric_idx == 0:
                ax.set_title(BENCHMARKS[func], fontsize=20)
            if metric_idx == num_rows - 1:
                ax.set_xlabel("Batch", fontsize=18)

            axes[0].set_ylabel(spec["label"], fontsize=18)

        if relative_ranking:
            plot_relative_ranking(
                all_regrets,
                maximize=False,
                axes=axes[-1],
                colors=COLORS,
                order=AL_METHOD_ORDER,
                position="only",
                include_first=1,
            )

    handles, labels = axes_all[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="lower center",
        ncol=len(methods),
        fontsize=12,
        bbox_to_anchor=(0.5, -0.012),
    )
    fig.tight_layout(rect=[-0.005, 0.05, 1.005, 1.01])
    fig.subplots_adjust(wspace=0.33)
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()


def main(
    path: str = "results/fig2_seqq8b8",
    output_dir: str = "figures",
    include_qpstd: bool = False,
    q: int = 8,
    num_batches: int = 8,
    show: bool = False,
):
    COLORS.setdefault("sobol", COLORS.get("fb_sobol", "dodgerblue"))
    COLORS.setdefault("random", COLORS.get("fb_random", "purple"))

    os.makedirs(output_dir, exist_ok=True)
    methods = FIG2_QPSTD_METHODS if include_qpstd else FIG2_METHODS
    file_name = "fig2_modobj_qpstd.pdf" if include_qpstd else "fig2_modobj.pdf"
    output_file = os.path.join(output_dir, file_name)

    plot_metrics_over_time(
        base_path=path,
        metric_specs=METRIC_SPECS,
        functions=list(FIG2_FUNCTIONS),
        methods=list(methods),
        output_file=output_file,
        end_at=num_batches,
        batch_size=q,
        relative_ranking=True,
        show=show,
    )
    print(f"Saved {output_file}")


if __name__ == "__main__":
    Fire(main)
