import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire import Fire
from experiments.constants import COLORS
from experiments.performance import plot_rmse_nmll_over_time


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
    # fig2_seq_qpstd change start
    "hipe",
    "bald",
    "nipv",
    "sobol",
    "random",
    "qPSTD",
    "seq_PSTD_BALD",
    # fig2_seq_qpstd change end
)


def main(
    path: str = "results/fig2_al",
    output_dir: str = "figures",
    include_qpstd: bool = False,
    q: int = 16,
    num_batches: int = 4,
    show: bool = False,
):
    # Reuse existing fb_* colors for AL method keys without changing constants.py.
    COLORS.setdefault("sobol", COLORS.get("fb_sobol", "dodgerblue"))
    COLORS.setdefault("random", COLORS.get("fb_random", "purple"))

    os.makedirs(output_dir, exist_ok=True)
    methods = FIG2_QPSTD_METHODS if include_qpstd else FIG2_METHODS
    file_name = "fig2_qpstd.pdf" if include_qpstd else "fig2.pdf"
    output_file = os.path.join(output_dir, file_name)

    plot_rmse_nmll_over_time(
        base_path=path,
        metrics=["MLL", "RMSE"],
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
