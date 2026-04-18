import os
import json
import math
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from json import JSONDecodeError
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.axes import Axes
# fig2_qpstd change start
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
# fig2_qpstd change end
from experiments.constants import (
    COLORS, 
    NAMES, 
    BENCHMARKS, 
    METRIC_NAMES,
    BO_METHOD_ORDER,
    AL_METHOD_ORDER,
)
from scipy.stats import rankdata
import pandas as pd
mpl.rcParams["font.family"] = "serif"


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_relative_ranking(all_regrets: dict, axes: Axes, maximize: bool = True, colors: dict = None, order: tuple = BO_METHOD_ORDER, position: str = "only", include_first: bool = False):
    """
    Plot relative average ranking of methods across multiple functions and seeds.

    Args:
        all_regrets (dict): Nested dictionary of form {function: {method: np.array of regrets per seed}}.
        maximize (bool): If True, higher regret is better (maximize objective); else lower is better.
        colors (dict): Optional color mapping for methods.
    """
    #include_first = False
    method_ranks = defaultdict(list)
    rankings = []
    #fix issue
    err_rankings = []
    for func, method_dict in all_regrets.items():
        methods = list(method_dict.keys())

        # filter and sort methods based on the order
        methods = [method for method in order if method in methods]

        # Skip functions with fewer than 2 methods
        if len(method_dict) < 2:
            continue
        # For each seed, compute ranks
        method_arrays = [np.array(method_dict[m]) for m in methods]
        num_seeds = [len(v) for v in method_arrays]
        
        if any(seed != num_seeds[0] for seed in num_seeds):
            print(f"Warning: Different number of seeds for function {func}. Skipping.")
            continue
        # TODO: make it work even if #seeds are different
        all_method_arr = np.concatenate([item[np.newaxis, ...] for item in method_arrays], axis=0) 
        all_method_arr = all_method_arr.round(decimals=5)
        if maximize:
            all_method_arr = -all_method_arr
        ranks = rankdata(all_method_arr, axis=0, method="average")

        dict_  = {method: rank for method, rank in zip(methods, ranks)}
        #print(methods)
        #np.save(f"ranks/lcbench_{func}_ranks.npz", ranks)
        avg_ranks = np.mean(ranks, axis=1)
        #fix issue
        err_ranks = np.std(ranks, axis=1) / np.sqrt(ranks.shape[1])
        ranks = ranks.reshape(-1, ranks.shape[-1])
        #avg_ranks[..., 0] = len(methods) / 2 + 0.5
        # NOTE there seems to be a rounding error here, not sure why. Hardcoding rank at 0
        rankings.append(avg_ranks)
        #fix issue
        err_rankings.append(err_ranks)
    #fix issue
    if not rankings:
        #fix issue
        print("Warning: No valid rankings available to plot.")
        #fix issue
        return
    
    # Compute average ranks across all functions
    rankings = np.array(rankings)
    agg_ranks = np.mean(rankings, axis=0).T
    #fix issue
    err_ranks = np.mean(np.array(err_rankings), axis=0).T
    if not include_first:
        agg_ranks = agg_ranks[1:] 
        err_ranks = err_ranks[1:]
    for i, method in enumerate(methods):
        color = colors.get(method, "black")
        x_vals = 1 + np.arange(len(agg_ranks))
        y_vals = agg_ranks[..., i]
        y_err = err_ranks[..., i]
        axes.errorbar(
            x_vals, 
            y_vals,
            yerr=y_err, 
            linestyle=":", 
            linewidth=1.5, 
            color=color,
            fmt='o',  # Plot markers only at data points
            elinewidth=2, 
            markersize=5, # Thicker error bar lines
            capsize=4,       # Add caps to the error bars
                
        )
    if position != "bottom":
        axes.set_title("Relative Rank", fontsize=20, weight="bold")
    if position != "top":
        axes.set_xlabel("Batch", fontsize=18)

    axes.invert_yaxis()
    axes.grid(True, linestyle="--", alpha=0.6)
    axes.set_xticks(x_vals)
    
    axes.tick_params(axis='both', which='major', labelsize=15)

    
# Custom y-axis tick formatter
def make_regret_formatter(reference):
    def formatter(y, pos):
        if y == 0:
            return f"{reference:.1f}"  # Show reference value with 2 decimal places
        else:
            return f"+{y:.1f}"  # Show y value with 2 decimal places
    return ticker.FuncFormatter(formatter)

class SmartRegretFormatter(ticker.Formatter):
    def __init__(self, axis, reference):
        self.axis = axis  # Reference to the axis (used to check limits dynamically)
        self.reference = reference

    def __call__(self, y, pos=None):
        # Dynamically determine axis upper limit
        max_val = self.axis.get_view_interval()[1]
        if y == 0:
            return f"{self.reference:.1f}"  # Reference with one decimal
            
        if max_val >= 10:
            return f"+{int(round(y))}"  # Other values as integers
        else:
            return f"+{y:.1f}"  # Other values with one decimal

def plot_average_simple_regret(
    base_path: str,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: int | list[int] | None = None,
    output_file: str = None,
    start_at: int = 0,
    end_at: int | None = None,
    batch_size: int = 1,  # New argument for batch size
    split: bool = False,
    infer: bool = True,
    subtract_baseline: bool = False,
    in_sample: bool = False,
    relative_ranking: bool = False,
    show: bool = False,
    smaller: bool = False
):
    """
    Plot the average simple regret across seeds for multiple methods and functions.

    Args:
        base_path (str): Base directory containing the results.
        functions (List[str]): List of objective functions (e.g., ["hartmann6"]).
        methods (List[str]): List of methods to compare (e.g., ["fb_tsig", "random"]).
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Optional path to save the resulting plot.
        batch_size (int): Size of each batch (number of iterations per batch).
    """
    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))

    else:
        functions = functions.split(",")
    
    functions = list(filter(lambda x: "." not in x, functions))
    num_rows = 1 #max(len(functions) // 3, 1)
    
    num_cols = math.ceil(len(functions) / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols + relative_ranking, figsize=(16.5 - 9.33 * smaller, 5 * num_rows))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)
    all_regrets = {}
    for ax_idx, (ax, func) in enumerate(zip(axes, functions)):
        ylim_min = np.inf
        all_regrets[func] = {}
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))
        
        elif isinstance(methods, str):
            methods = [methods]    
        methods = [m for m in BO_METHOD_ORDER if m in methods]
        method_seed_sets = []
        for method in methods:
            method_seed_paths = glob(os.path.join(base_path, func, method, f"seed*"))
            method_seeds = set(
                s.split("seed")[-1].split("/")[0] for s in method_seed_paths
            )
            method_seed_sets.append(method_seeds)

        # Compute intersection of all seed sets (common seeds)
        common_seeds = sorted(
            list(set.intersection(*method_seed_sets)), 
            key=lambda x: int(x)
        )
        print(len(common_seeds))
        if not common_seeds:
            print(f"Warning: No common seeds found for function '{func}'. Skipping...")

            continue  # Skip this function if no common seeds are found
        for m_idx, method in enumerate(methods):
            if ax_idx == 0:
                label = NAMES.get(method, method)
            else:
                label = "__nolabel__"
            regrets = []
            #if not seeds:
            seeds = common_seeds
                #seeds = seeds[0:50]
            for seed in seeds:
                file_path = os.path.join(
                    base_path, func, method, f"seed{seed}", "bo.json"
                )
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}")
                    continue
                with open(file_path, "r") as f:
                    data = json.load(f)
                train_f = np.array(data["TrainingData"]["train_f"]).squeeze()[start_at:end_at]
                # Aggregate regrets per batch directly from train_f
                num_batches = len(train_f) // batch_size
                if infer:
                    if in_sample:
                        train_f = np.array(data["InSampleBestValues"]).squeeze()
                        regrets.append(train_f)
                    else:
                        init_f = np.array(data["InSampleBestValues"][0]).squeeze()
    
                        train_f = np.array(data["OutOfSampleBestValues"]).squeeze()
                        train_f =  np.insert(train_f, 0, init_f)
                        regrets.append(train_f)
                else:
                    batched_regret = [
                        train_f[i * batch_size:(i + 1) * batch_size].max()
                        for i in range(num_batches)
                    ]
                    batched_regret = [train_f[0]] + batched_regret
                    regrets.append(np.maximum.accumulate(batched_regret))  # Ensure cumulative max
            
            regrets = np.array(regrets)
            
            all_regrets[func][method] = regrets
            if split:
                x = np.arange(len(regrets.T)) + 1
                ax.plot(x, regrets.T, alpha=1, linewidth=0.5, color=COLORS[method])
            
            else:
                if end_at:
                    regrets = regrets[..., :end_at]

                if subtract_baseline:
                    ref_regret = regrets[..., 0:1].mean()
                    regrets = regrets - regrets[..., 0:1]
                avg_regret = regrets.mean(axis=0)
                
                std_error = regrets.std(axis=0) / np.sqrt(len(seeds))    
                ylim_min = min(ylim_min, avg_regret.min())
                
                # Plot the results with error bars
                
                x = np.arange(len(avg_regret)) + 1
                if batch_size > 1 or infer:
                    #ax.plot((x - 1)[1:], avg_regret[1:], linestyle="--", linewidth=1.0, color=COLORS[method])
                    OFFSET = 0.06
                    # Plot the thicker error bars on top
                    ax.errorbar(
                        -((0.5 -len(methods)) * OFFSET / 2 + m_idx * OFFSET) + (x - 1)[1:], avg_regret[1:], yerr=std_error[1:], 
                        fmt='o',  # Plot markers only at data points
                        elinewidth=1,  # Thicker error bar lines
                        capsize=3,       # Add caps to the error bars
                        color=COLORS[method], 
                        label=label
                    )
                    ax.set_xticks([1, 2])
                    #ax.axhline(avg_regret[0].mean(), color="k", linestyle="--", linewidth=0.75)
                else:
                    ax.plot(x, avg_regret, alpha=1, linewidth=1.6, color=COLORS[method], label=label)
                    ax.fill_between(
                        x,
                        avg_regret - std_error,
                        avg_regret + std_error,
                        alpha=0.2,
                        color=COLORS[method],
                    )
                    ax.plot(x, avg_regret - std_error, alpha=0.3, linewidth=0.5, color=COLORS[method])
                    ax.plot(x, avg_regret + std_error, alpha=0.3, linewidth=0.5, color=COLORS[method])
                  
        ax.set_xlabel("Batch", fontsize=18)  # Updated label
        ax.set_title(BENCHMARKS[func], fontsize=20)
        
        ax.grid(True, linestyle="--", alpha=0.6)
        if subtract_baseline:
            reference_val = ref_regret
            formatter = SmartRegretFormatter(ax.yaxis, reference=reference_val)
            ax.yaxis.set_major_formatter(formatter)
        
        ax.tick_params(axis='both', which='major', labelsize=15)
            
    if subtract_baseline:
        axes[0].set_ylabel("Inferred improvement over baseline", fontsize=18)
    else:
        axes[0].set_ylabel("Value of infered maximum", fontsize=18)
    if relative_ranking:
        plot_relative_ranking(all_regrets, maximize=True, axes=axes[-1], colors=COLORS)
    handles, labels = axes[0].get_legend_handles_labels()
    
    if smaller:
        fig.legend(
            handles[::-1],
            labels[::-1],
            loc='lower center',  # Equivalent to loc=8
            ncol=((len(methods) + 1) // 2),
            fontsize=18,
            bbox_to_anchor=(0.5, -0.027) 
        )
        fig.tight_layout(rect=[-0.005, 0.16, 1.005, 1.03])
    else:
        fig.legend(
            handles[::-1],
            labels[::-1],
            loc='lower center',  # Equivalent to loc=8
            ncol=len(methods),
            fontsize=19,
            bbox_to_anchor=(0.5, -0.027) 
        )
        fig.tight_layout(rect=[-0.005, 0.08, 0.99, 1.03])

    fig.subplots_adjust(wspace=0.33)
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()


def plot_rmse_and_mll(
    base_path: str,
    plot_after_init: bool = True,
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: int | list[int] | None = None,
    output_file: str = None,
):
    """
    Plot the average RMSEs and MLLs per method and benchmark.

    Args:
        base_path (str): Base directory containing the results.
        functions (List[str]): List of objective functions (e.g., ["hartmann6"]).
        methods (List[str]): List of methods to compare (e.g., ["fb_tsig", "random"]).
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Optional path to save the resulting plot.
    """

    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))

    elif isinstance(functions, str):
        functions = [functions]
    rmse_results = {func: {} for func in functions}
    mll_results = {func: {} for func in functions}
    fig, axes = plt.subplots(2, len(functions), figsize=(len(functions) * 2.5, 6.0))
    axes = axes.reshape(2, -1)
    for func in functions:
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))

        elif isinstance(methods, str):
            methods = [methods]

        rmse_results[func] = {method: [] for method in methods}
        mll_results[func] = {method: [] for method in methods}
        for c, method in zip(COLORS, methods):
            if not seeds:
                seeds = [
                    s.split("seed")[-1].split("/")[0]
                    for s in glob(os.path.join(base_path, func, method, f"seed*"))
                ]
            for seed in seeds:
                if plot_after_init:
                    prefix = "init.json"
                else:
                    prefix = "bo.json"
                file_path = os.path.join(base_path, func, method, f"seed{seed}", prefix)
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}")
                    continue
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    rmse_results[func][method].append(data["RMSE"])
                    mll_results[func][method].append(data["MLL"])

                except JSONDecodeError:
                    continue

    all_rmse = {
        func: {method: np.array(rmse_results[func][method]) for method in methods}
        for func in functions
    }
    all_mll = {
        func: {method: np.array(mll_results[func][method]) for method in methods}
        for func in functions
    }

    # Plot RMSE and MLL
    axes[0, 0].set_ylabel("RMSE", fontsize=14)
    axes[1, 0].set_ylabel("Negative MLL", fontsize=14)

    for idx, func in enumerate(functions):
        # Plot RMSE
        mll_res = []
        rmse_res = []
        for method in methods:
            if len(all_rmse[func][method]) > 0:
                rmse_res.append(np.mean(all_rmse[func][method], axis=0))
                mll_res.append(-np.log(np.mean(np.exp(all_mll[func][method]), axis=0)))

            else:
                rmse_res.append([])
                mll_res.append([])
        # Plot RMSE
        bplot_rmse = axes[0, idx].boxplot(rmse_res, patch_artist=True, labels=methods, showfliers=False)
        for patch, method in zip(bplot_rmse["boxes"], methods):
            patch.set_facecolor(COLORS[method])
        axes[0, idx].grid(True, linestyle="--", alpha=0.6)

        # Plot MLL
        bplot_mll = axes[1, idx].boxplot(mll_res, patch_artist=True, labels=methods)
        for patch, method in zip(bplot_mll["boxes"], methods):
            patch.set_facecolor(COLORS[method])

        axes[0, idx].set_title(f"{BENCHMARKS[func]}", fontsize=16)
        axes[1, idx].tick_params(axis="x", rotation=0)
        axes[1, idx].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")

    plt.show()


def plot_lengthscale_distributions(
    base_path: str,
    plot_after_init: bool = True,
    functions: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    output_file: str | None = None,
    stacked: bool = False,
    metric: str = "median",
):
    """
    Plot the distribution of lengthscales per method and benchmark.

    Args:
        base_path (str): Base directory containing the results.
        plot_after_init: Whether to plot after init or after BO.
        functions (List[str]): List of objective functions (e.g., ["hartmann6"]).
        methods (List[str]): List of methods to compare (e.g., ["fb_tsig", "random"]).
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Optional path to save the resulting plot.
    """

    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))
    elif isinstance(functions, str):
        functions = [functions]
    num_methods = len(functions)
    lengthscale_results = {func: {} for func in functions}

    if stacked:
        num_methods = 1
    fig, axes = plt.subplots(

        len(functions), num_methods, figsize=(7.5 * num_methods, len(functions) * 5), sharey=True
    )
    all_data = {}
    if len(functions) == 1:
        axes = np.array([axes])  # Ensure axes is always iterable

    axes = axes.reshape(len(functions), -1)
    
    for func_idx, func in enumerate(functions):
        all_data[func] = {} 
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))
        elif isinstance(methods, str):
            methods = [methods]

        lengthscale_results[func] = {method: [] for method in methods}
        for method_idx, method in enumerate(methods):
            if not seeds:
                seeds = [
                    int(s.split("seed")[-1].split("/")[0])
                    for s in glob(os.path.join(base_path, func, method, f"seed*"))
                ]

            for seed in seeds:
                if plot_after_init:
                    prefix = "init.json"
                else:
                    prefix = "bo.json"
                file_path = os.path.join(base_path, func, method, f"seed{seed}", prefix)
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}")
                    continue
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Extract lengthscale(s) and add to results
                    lengthscales = np.array(data["Hyperparameters"]["lengthscale"])
                    lengthscale_results[func][method].append(lengthscales)
                except JSONDecodeError:
                    continue
                # Plot the distributions

            lengthscale_data = np.array(lengthscale_results[func][method])
            if len(lengthscale_data) == 0:
                continue
            lengthscale_data = np.squeeze(lengthscale_data, axis=-2)
            if metric == "mean":
                lengthscale_data = np.exp(np.log(lengthscale_data).mean(axis=-2))
            elif metric == "median":
                lengthscale_data = np.median(lengthscale_data, axis=-2)
            else:
                raise ValueError("Invalid metric. Choose 'mean' or 'median'.")
            all_data[func][method] = lengthscale_data
            dims = np.array(lengthscale_data).shape[-1]
            if not stacked:
                bplot = axes[func_idx, method_idx].boxplot(
                    lengthscale_data,
                    patch_artist=True,
                    labels=[f"{i+1}" for i in range(dims)],
                    showfliers=False,
    
                )

                for patch in bplot["boxes"]:
                    patch.set_facecolor(COLORS[method])

                axes[func_idx, method_idx].set_title(
                    f"{BENCHMARKS[func]} - Lengthscale Distribution", fontsize=19
                )
                axes[func_idx, method_idx].tick_params(axis="x", rotation=45)
                
                axes[func_idx, method_idx].grid(True, linestyle="--", alpha=0.6)
                axes[func_idx, method_idx].set_yscale("log")
        
        if stacked:
            axes = axes.flatten()
            for i, (method, lengthscale_data) in enumerate(all_data[func].items()):
                if i == 0:
                    labels = [f"{j+1}" if (j+1) % 5 == 0 else "" for j in range(dims)]
                else:
                    labels = [""] * dims
                bplot = axes[func_idx].boxplot(
                    lengthscale_data,
                    patch_artist=True,
                    labels=labels,
                    showfliers=False,
                    whis=True,
                    positions=(0.3 * i) + np.arange(len(lengthscale_data.T)),
                    widths=0.3,
                )

                for patch in bplot["boxes"]:
                    patch.set_facecolor(COLORS[method])

                axes[func_idx].set_title(
                    f"{BENCHMARKS[func]} - Lengthscale Distribution", fontsize=22
                )
                axes[func_idx].tick_params(axis="both", labelsize=18)
                axes[func_idx].grid(True, linestyle="--", alpha=0.6)
                axes[func_idx].set_yscale("log")
            axes[func_idx].set_ylabel("Lengthscale", fontsize=21)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")

    plt.show()


def plot_lengthscale_distributions_swarm(
    base_path: str,
    plot_after_init: bool = True,
    functions: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    output_file: str | None = None,
):
    """
    Plot the distribution of lengthscales per method and benchmark using swarmplots.

    Args:
        base_path (str): Base directory containing the results.
        plot_after_init: Whether to plot after init or after BO.
        functions (List[str]): List of objective functions (e.g., ["hartmann6"]).
        methods (List[str]): List of methods to compare (e.g., ["fb_tsig", "random"]).
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Optional path to save the resulting plot.
    """
    # fig2_qpstd change start
    if sns is None:
        raise ModuleNotFoundError(
            "seaborn is required for plot_lengthscale_distributions_swarm. "
            "Install it with `pip install seaborn`."
        )
    # fig2_qpstd change end

    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))
    elif isinstance(functions, str):
        functions = [functions]
    functions = ["hartmann6_12", "hartmann6_25", "hartmann4_8"]

    lengthscale_results = {func: {} for func in functions}

    num_methods = len(sorted(os.listdir(f"{base_path}/{functions[0]}/")))
    fig, axes = plt.subplots(
        len(functions), num_methods, figsize=(9 * num_methods, len(functions) * 6)
    )
    axes = axes.reshape(len(functions), -1)
    if len(functions) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for func_idx, func in enumerate(functions):
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))
        elif isinstance(methods, str):
            methods = [methods]

        lengthscale_results[func] = {method: [] for method in methods}

        for method_idx, method in enumerate(methods):
            print(func, method)
            if not seeds:
                seeds = [
                    int(s.split("seed")[-1].split("/")[0])
                    for s in glob(os.path.join(base_path, func, method, f"seed*"))
                ]

            for seed in seeds:
                if plot_after_init:
                    prefix = "init.json"
                else:
                    prefix = "bo.json"
                file_path = os.path.join(base_path, func, method, f"seed{seed}", prefix)
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}")
                    continue
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Extract lengthscale(s) and add to results
                    lengthscales = np.array(data["Hyperparameters"]["lengthscale"])
                    lengthscale_results[func][method].extend(lengthscales)
                except JSONDecodeError:
                    continue

            lengthscale_data = np.array(lengthscale_results[func][method])
            if len(lengthscale_data) == 0:
                continue
            lengthscale_data = np.squeeze(lengthscale_data, axis=-2)
            dims = np.array(lengthscale_data).shape[-1]

            # Prepare data for swarmplot
            plot_data = []
            dim_labels = []
            for dim_idx in range(dims):
                plot_data.extend(lengthscale_data[:, dim_idx])
                dim_labels.extend([f"dim_{dim_idx+1}"] * len(lengthscale_data[:, dim_idx]))

            # Create swarmplot
            print("Plotting")
            sns.swarmplot(
                x=dim_labels,
                y=plot_data,
                ax=axes[func_idx, method_idx],
                palette="Set2",
                dodge=True
            )
            print("Plotted")
            axes[func_idx, method_idx].set_title(
                f"{BENCHMARKS[func]} - Lengthscale Distribution", fontsize=16
            )
            axes[func_idx, method_idx].tick_params(axis="x", rotation=45)
            axes[func_idx, method_idx].grid(True, linestyle="--", alpha=0.6)
            axes[func_idx, method_idx].set_yscale("log")

        axes[func_idx, 0].set_ylabel("Lengthscale", fontsize=14)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()


def plot_sobol_indices(
    base_path: str,
    plot_after_init: bool = True,
    functions: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    output_file: str | None = None,
):
    """
    Plot the distribution of lengthscales per method and benchmark.

    Args:
        base_path (str): Base directory containing the results.
        plot_after_init: Whether to plot after init or after BO.
        functions (List[str]): List of objective functions (e.g., ["hartmann6"]).
        methods (List[str]): List of methods to compare (e.g., ["fb_tsig", "random"]).
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Optional path to save the resulting plot.
    """

    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))
    elif isinstance(functions, str):
        functions = [functions]

    index_results = {func: {} for func in functions}
    real_index_results = {func: {} for func in functions}
    if methods is None:
        num_methods = len(sorted(os.listdir(f"{base_path}/{functions[0]}/")))
    else:
        num_methods = len(methods)
    fig, axes = plt.subplots(
        len(functions), num_methods, figsize=(5 * num_methods, len(functions) * 4), sharey=True
    )
    axes = axes.reshape(len(functions), -1)
    for func_idx, func in enumerate(functions):
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))
        elif isinstance(methods, str):
            methods = [methods]

        index_results[func] = {method: [] for method in methods}
        real_index_results[func] = {method: [] for method in methods}

        for method_idx, method in enumerate(methods):
            if not seeds:
                seeds = [
                    int(s.split("seed")[-1].split("/")[0])
                    for s in glob(os.path.join(base_path, func, method, f"seed*"))
                ]

            for seed in seeds:
                if plot_after_init:
                    prefix = "init.json"
                else:
                    prefix = "bo.json"
                file_path = os.path.join(base_path, func, method, f"seed{seed}", prefix)
                if not os.path.exists(file_path):
                    print(f"Warning: File not found {file_path}")
                    continue
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Extract indices and add to results
                    indices = np.array(data["SobolIndices"])
                    real_indices = np.array(data["RealIndices"])
                    index_results[func][method].append(indices)
                    real_index_results[func][method].append(real_indices)
                except JSONDecodeError:
                    continue
                
                # Plot the distributions

            index_results[func][method] = np.array(index_results[func][method])
            real_index_results[func][method] = np.array(real_index_results[func][method])
            if len(index_results[func][method]) == 0:
                continue

            dims = np.array(index_results[func][method]).shape[-1]
            real_sobol = real_index_results[func][method].mean(axis=0)[np.newaxis, :]
            
            bplot = axes[func_idx, method_idx].boxplot(
                index_results[func][method],
                patch_artist=True,
                labels=[f"{i+1}" for i in range(dims)],
                showfliers=False,
                whis=(0, 100),
                widths=0.75,
                medianprops={"linewidth": 0, "color": "darkgrey"}
            )
            trueplot = axes[func_idx, method_idx].boxplot(
                real_sobol,
                patch_artist=True,
                widths=0.95,
            )

            for patch in bplot["boxes"]:
                patch.set_facecolor(COLORS[method])
                patch.set_edgecolor(COLORS[method])
            for patch in bplot["whiskers"]:
                patch.set_color(COLORS[method])
                patch.set_linewidth(1.4)
            for patch in bplot["caps"]:
                patch.set_color(COLORS[method])
                patch.set_linewidth(1.4)

            for (patch, sobol_idx) in zip(trueplot["medians"], real_sobol.flatten()):
                if sobol_idx < 0.05:
                    patch.set_linewidth(False)
                else:
                    patch.set_linewidth(2.6)
                    patch.set_color("black")

            axes[func_idx, method_idx].set_title(
                f"{BENCHMARKS[func]} - {NAMES[method]}", fontsize=16
            )
            axes[func_idx, method_idx].tick_params(axis="x", rotation=0)
            axes[func_idx, method_idx].grid(True, linestyle="--", alpha=0.6)

        axes[func_idx, 0].set_ylabel("Sobol Index", fontsize=14)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")

    plt.show()


def plot_rmse_nmll_over_time(
    base_path: str,
    metrics: str | list[str] = "MLL",  # Metrics to plot
    functions: str | list[str] | None = None,
    methods: str | list[str] | None = None,
    seeds: int | list[int] | None = None,
    output_file: str = None,
    start_at: int = 0,
    end_at: int | None = None,
    batch_size: int = 1,
    position: str = "only",
    relative_ranking: bool = False,
    show: bool = False,
):
    """
    Plot RMSE and NMLL over time across seeds for multiple methods and functions.

    Args:
        base_path (str): Base directory containing the results.
        metrics (List[str]): Metrics to plot. Supported: "rmse", "nmll".
        functions (List[str]): List of objective functions.
        methods (List[str]): List of methods to compare.
        seeds (List[int]): List of seeds to aggregate over.
        output_file (str): Path to save the plot.
        batch_size (int): Number of iterations per batch.
    """
    if base_path[-1] == "/":
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))
    elif isinstance(functions, str):
        functions = [functions]

    functions = list(filter(lambda x: "." not in x, functions))
    num_cols = len(functions)
    if isinstance(metrics, str):
        metrics = [metrics]
    num_rows = len(metrics)

    fig, axes_all = plt.subplots(num_rows, num_cols + relative_ranking, figsize=(16, 4.75 * num_rows))
    if not isinstance(axes_all, np.ndarray):
        axes_all = np.array([axes_all])
    axes_all = axes_all.reshape(num_rows, -1)

    all_regrets = {}
    pos = "only" if len(metrics) == 1 else ["top", "bottom"]
    for metric_idx, (metric, position) in enumerate(zip(metrics, pos)):
        axes = axes_all[metric_idx]
        for ax_idx, (ax, func) in enumerate(zip(axes, functions)):
            all_regrets[func] = {}
            if not methods:
                methods = sorted(os.listdir(f"{base_path}/{func}/"))
            elif isinstance(methods, str):
                methods = [methods]
            methods = [m for m in AL_METHOD_ORDER if m in methods]
            # Find common seeds across all methods
            method_seeds = [
                set(
                    s.split("seed")[-1].split("/")[0]
                    for s in glob(os.path.join(base_path, func, method, "seed*/al.json"))
                ) for method in methods
            ]

            common_seeds = set.intersection(*method_seeds)
            if not common_seeds:
                print(f"Warning: No common seeds for function '{func}'. Skipping.")
                continue  # Skip this function if no common seeds found

            seeds = sorted(list(common_seeds))  # Use only common seeds from now on
            print(len(common_seeds))
            for method in methods:
                metric_values = []
                if not seeds:
                    seeds = [
                        s.split("seed")[-1].split("/")[0]
                        for s in glob(os.path.join(base_path, func, method, f"seed*"))
                    ]
                for seed in seeds:
                    file_path = os.path.join(
                        base_path, func, method, f"seed{seed}", "al.json"
                    )
                    if not os.path.exists(file_path):
                        print(f"Warning: File not found {file_path}")
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if metric not in data:
                        print(f"Warning: Metric '{metric}' not found in {file_path}")
                        continue

                    metric_data = np.array(data[metric]).squeeze()
                    metric_values.append(metric_data)

                metric_values = np.array(metric_values)
                if end_at:
                    metric_values = metric_values[..., :end_at]
                
                if metric == "MLL":
                    metric_values = -np.log(np.mean(np.exp(metric_values), axis=-1))
                else:
                    metric_values = np.mean(metric_values, axis=-1)
                all_regrets[func][method] = metric_values
                
                avg_metric = metric_values.mean(axis=0) 
                std_error = metric_values.std(axis=0) / np.sqrt(len(seeds))

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
                    fmt='o',  # Plot markers only at data points
                    elinewidth=2, 
                    markersize=5, # Thicker error bar lines
                    capsize=4,       # Add caps to the error bars
                        
                )
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=15)  
            ax.set_xticks(x)
            if position != "bottom":
                ax.set_title(BENCHMARKS[func], fontsize=20)
            if position != "top":
                ax.set_xlabel("Batch", fontsize=18)    
        
            axes[0].set_ylabel(METRIC_NAMES[metric], fontsize=18)

        if relative_ranking:
            plot_relative_ranking(
                all_regrets, 
                maximize=False, 
                axes=axes[-1], 
                colors=COLORS, 
                order=AL_METHOD_ORDER, 
                position=position,
                include_first=1,
            )
    handels, labels = axes_all[0, 0].get_legend_handles_labels()
    if len(metrics) == 1:
        fig.legend(
            handels[::-1],
            labels[::-1],
            loc='lower center',  # Equivalent to loc=8
            ncol=len(methods),
            fontsize=19,
            bbox_to_anchor=(0.5, -0.027) 
        )
        fig.tight_layout(rect=[-0.005, 0.08, 1.005, 1.03])
    else:
        fig.legend(
            handels[::-1],
            labels[::-1],
            loc='lower center',  # Equivalent to loc=8
            ncol=len(methods),
            fontsize=12,
            bbox_to_anchor=(0.5, -0.012) 
        )
        fig.tight_layout(rect=[-0.005, 0.05, 1.005, 1.01])
    fig.subplots_adjust(wspace=0.33)
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()


def plot_accuracy_regret_correlation(
    base_path: str,
    accuracy_metric: str = "sobol",  # Options: "sobol", "rmse", "mll"
    functions: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    output_file: str | None = None,
    correlation_metric: str = "pearson",  # Options: "pearson", "spearman"
    after_bo: bool = True,
):
    """
    Plot the correlation between accuracy (Sobol error, RMSE, or MLL) and regret performance.

    Args:
        base_path (str): Base directory for results.
        accuracy_metric (str): "sobol", "rmse", or "mll".
        functions (List[str]): List of benchmarks.
        methods (List[str]): List of methods.
        seeds (List[int]): List of seeds.
        output_file (str): Path to save the plot.
        correlation_metric (str): 'pearson' or 'spearman'.
        after_bo (bool): Whether to use results after BO or after init.
    """
    from scipy.stats import pearsonr, spearmanr

    if base_path.endswith("/"):
        base_path = base_path[:-1]

    if not functions:
        functions = sorted(os.listdir(f"{base_path}/"))
    if isinstance(functions, str):
        functions = [functions]

    accuracy_vals = []
    final_regrets = []
    method_labels = []
    function_labels = []

    for func in functions:
        if not methods:
            methods = sorted(os.listdir(f"{base_path}/{func}/"))
        if isinstance(methods, str):
            methods = [methods]

        for method in methods:
            if not seeds:
                seeds = [
                    s.split("seed")[-1].split("/")[0]
                    for s in glob(os.path.join(base_path, func, method, "seed*"))
                ]
            for seed in seeds:
                file_name = "bo.json" if after_bo else "init.json"
                file_path = os.path.join(base_path, func, method, f"seed{seed}", file_name)
                if not os.path.exists(file_path):
                    continue
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if accuracy_metric == "sobol":
                        real_indices = np.array(data.get("RealIndices"))
                        estimated_indices = np.array(data.get("SobolIndices"))
                        if real_indices is None or estimated_indices is None:
                            continue
                        accuracy_value = np.mean(np.abs(real_indices - estimated_indices))
                    elif accuracy_metric == "rmse":
                        accuracy_value = np.mean(data.get("RMSE", np.nan))
                    elif accuracy_metric == "mll":
                        mll_vals = np.array(data.get("MLL", np.nan))
                        accuracy_value = -np.log(np.mean(np.exp(mll_vals)))  # Convert back to log scale
                    else:
                        raise ValueError(f"Invalid accuracy_metric '{accuracy_metric}'")

                    if np.isnan(accuracy_value):
                        continue

                    train_f = np.array(data["TrainingData"]["train_f"]).squeeze()
                    final_regret = np.max(train_f)

                    accuracy_vals.append(accuracy_value)
                    final_regrets.append(final_regret)
                    method_labels.append(method)
                    function_labels.append(func)

                except (JSONDecodeError, KeyError):
                    continue

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5))
    accuracy_vals = np.array(accuracy_vals)
    final_regrets = np.array(final_regrets)

    scatter = ax.scatter(
        accuracy_vals,
        final_regrets,
        c=[COLORS.get(m, 'gray') for m in method_labels],
        alpha=0.7,
        edgecolors='k',
        linewidths=0.5
    )

    # Axis Labels
    accuracy_label_map = {
        "sobol": "Sobol Index Estimation Error (MAE)",
        "rmse": "RMSE",
        "mll": "Negative MLL"
    }
    ax.set_xlabel(accuracy_label_map.get(accuracy_metric, "Accuracy"), fontsize=14)
    ax.set_ylabel("Final Performance", fontsize=14)
    ax.set_title(f"Correlation between {accuracy_metric.upper()} Accuracy and Regret", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Correlation Calculation
    if correlation_metric == "pearson":
        corr, _ = pearsonr(accuracy_vals, final_regrets)
        metric_name = "Pearson"
    else:
        corr, _ = spearmanr(accuracy_vals, final_regrets)
        metric_name = "Spearman"

    ax.text(
        0.52, 0.97,
        f"$\\rho = {-corr:.2f}$",
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top'
    )

    # Custom Legend
    handles = []
    seen_methods = set()
    for method in method_labels:
        if method not in seen_methods:
            handles.append(plt.Line2D([], [], marker='o', linestyle='', color=COLORS.get(method, 'gray'), label=NAMES.get(method, method)))
            seen_methods.add(method)
    ax.legend(handles=handles, title="Methods", fontsize=12)

    fig.tight_layout()
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()