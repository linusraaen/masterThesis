import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ALGO_COLORS = {"RWM": "#3a7eca", "HMC": "#f58c2a", "NUTS": "#3daa62"}
ALGO_ORDER  = ["RWM", "HMC", "NUTS"]
PRIOR_COLORS = {0.1: "#e41a1c", 1.0: "#377eb8", 10.0: "#4daf4a"}


def _agg(df, metric, group_cols=None):
    if group_cols is None:
        group_cols = []
    return (
        df.groupby(["algorithm", "dimension"] + group_cols)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


def _agg_by(df, metric, group_cols):
    """Aggregate metric by arbitrary group columns (no dimension)."""
    return (
        df.groupby(group_cols)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


# ── Existing plots ────────────────────────────────────────────────────────────

def plot_metric_with_errors(
    df, metric, ylabel, title,
    save_path=None, log_x=True, log_y=True, group_cols=None
):
    plt.figure(figsize=(8, 5))
    agg = _agg(df, metric, group_cols)

    for algorithm in ALGO_ORDER:
        subset = agg[agg["algorithm"] == algorithm].sort_values("dimension")
        if subset.empty:
            continue
        plt.errorbar(
            subset["dimension"], subset["mean"], yerr=subset["std"],
            marker="o", linewidth=2, label=algorithm,
            color=ALGO_COLORS[algorithm], capsize=4, capthick=1.5,
        )

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ess(df, title="ESS vs Dimension",
             save_path="results/figures/ess_vs_dimension.png"):
    plot_metric_with_errors(df, "ess", "Effective Sample Size (ESS)",
                            title, save_path)


def plot_runtime(df, title="Runtime vs Dimension",
                 save_path="results/figures/runtime_vs_dimension.png"):
    plot_metric_with_errors(df, "runtime", "Runtime (seconds)",
                            title, save_path)


def plot_ess_per_sec(df, title="ESS per Second vs Dimension",
                     save_path="results/figures/ess_per_second_vs_dimension.png"):
    plot_metric_with_errors(df, "ess_per_sec", "ESS per Second",
                            title, save_path)


def plot_rhat(df, title="R-hat vs Dimension",
              save_path="results/figures/rhat_vs_dimension.png"):
    plt.figure(figsize=(8, 5))
    agg = _agg(df, "rhat")

    for algorithm in ALGO_ORDER:
        subset = agg[agg["algorithm"] == algorithm].sort_values("dimension")
        if subset.empty:
            continue
        plt.errorbar(
            subset["dimension"], subset["mean"], yerr=subset["std"],
            marker="o", linewidth=2, label=algorithm,
            color=ALGO_COLORS[algorithm], capsize=4, capthick=1.5,
        )

    plt.axhline(y=1.1, color="red", linestyle="--", linewidth=1,
                label="R-hat = 1.1 (warning threshold)")
    plt.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Dimension")
    plt.ylabel("R-hat")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_divergences(df, title="Divergences vs Dimension",
                     save_path="results/figures/divergences_vs_dimension.png"):
    df_div = df[df["divergences"].notna()]
    if df_div.empty:
        print("No divergence data — skipping plot")
        return

    agg = (
        df_div.groupby(["algorithm", "dimension"])["divergences"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    has_data = False
    for algorithm in ["HMC", "NUTS"]:
        subset = agg[agg["algorithm"] == algorithm].sort_values("dimension")
        if subset.empty:
            continue
        plt.errorbar(
            subset["dimension"], subset["mean"], yerr=subset["std"],
            marker="o", linewidth=2, label=algorithm,
            color=ALGO_COLORS[algorithm], capsize=4, capthick=1.5,
        )
        has_data = True

    if not has_data:
        plt.close()
        return

    plt.xscale("log")
    plt.xlabel("Dimension")
    plt.ylabel("Number of Divergences")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_logistic_regression_bar(
        df_lr,
        title="ESS/s — Bayesian Logistic Regression",
        save_path="results/figures/ess_per_sec_logistic.png"):
    """
    ESS/s bar chart. Pass pre-filtered dataframe — no internal filtering.
    Works for logistic regression and 8-schools.
    """
    agg = df_lr.groupby("algorithm")["ess_per_sec"].agg(
        ["mean", "std"]).reset_index()
    agg = agg[agg["algorithm"].isin(ALGO_ORDER)]

    plt.figure(figsize=(6, 5))
    colors = [ALGO_COLORS[a] for a in agg["algorithm"]]
    plt.bar(agg["algorithm"], agg["mean"], yerr=agg["std"],
            color=colors, capsize=6, edgecolor="white", linewidth=0.5)
    plt.ylabel("ESS per Second")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rhat_bar(
        df_lr,
        title="R-hat — Bayesian Logistic Regression",
        save_path="results/figures/rhat_logistic.png"):
    """
    R-hat bar chart. Pass pre-filtered dataframe — no internal filtering.
    Works for logistic regression and 8-schools.
    """
    agg = df_lr.groupby("algorithm")["rhat"].agg(["mean", "std"]).reset_index()
    agg = agg[agg["algorithm"].isin(ALGO_ORDER)]

    plt.figure(figsize=(6, 5))
    colors = [ALGO_COLORS[a] for a in agg["algorithm"]]
    plt.bar(agg["algorithm"], agg["mean"], yerr=agg["std"],
            color=colors, capsize=6, edgecolor="white", linewidth=0.5)
    plt.axhline(y=1.1, color="red", linestyle="--", linewidth=1,
                label="R-hat = 1.1 (warning threshold)")
    plt.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.ylabel("R-hat")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ── Prior scale plots ─────────────────────────────────────────────────────────

def plot_prior_scale_comparison(
        df_lr, metric="ess_per_sec",
        ylabel="ESS per Second",
        title="Effect of Prior Scale on ESS/s",
        save_path="results/figures/prior_scale_ess_per_sec.png"):
    """Grouped bar chart: x-axis = algorithm, groups = prior_std."""
    _grouped_bar(
        df_lr, metric, "prior_std", PRIOR_COLORS,
        ylabel=ylabel, title=title, save_path=save_path,
        group_label_fmt="σ={}",
    )


def plot_prior_scale_rhat(
        df_lr,
        save_path="results/figures/prior_scale_rhat.png"):
    """Grouped bar chart of R-hat across prior scales."""
    _grouped_bar(
        df_lr, "rhat", "prior_std", PRIOR_COLORS,
        ylabel="R-hat",
        title="Effect of Prior Scale on R-hat",
        save_path=save_path,
        threshold_line=1.1,
        group_label_fmt="σ={}",
    )




# ── Tau scale plots (8-schools) ───────────────────────────────────────────────

TAU_COLORS = {1.0: "#e41a1c", 10.0: "#377eb8", 100.0: "#4daf4a"}


def _grouped_bar(df, metric, group_col, group_colors, ylabel, title,
                 save_path, threshold_line=None, group_label_fmt="{}"):
    """
    Generic grouped bar chart helper.
    x-axis = algorithm, groups = unique values of group_col.
    """
    groups = sorted(df[group_col].unique())
    agg = df.groupby(["algorithm", group_col])[metric].agg(
        ["mean", "std"]).reset_index()

    algos = [a for a in ALGO_ORDER if a in agg["algorithm"].values]
    n_algos  = len(algos)
    n_groups = len(groups)
    width = 0.25
    x = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(groups):
        sub = agg[agg[group_col] == g]
        means = [sub[sub["algorithm"] == a]["mean"].values[0]
                 if len(sub[sub["algorithm"] == a]) > 0 else 0
                 for a in algos]
        stds  = [sub[sub["algorithm"] == a]["std"].values[0]
                 if len(sub[sub["algorithm"] == a]) > 0 else 0
                 for a in algos]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=group_label_fmt.format(g),
               color=group_colors.get(g, f"C{i}"),
               capsize=4, alpha=0.85)

    if threshold_line is not None:
        ax.axhline(y=threshold_line, color="red", linestyle="--",
                   linewidth=1, label=f"threshold = {threshold_line}")

    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def plot_tau_scale_comparison(
        df_8s,
        parameterisation="centred",
        save_path="results/figures/tau_scale_ess_per_sec_centred.png"):
    """
    Grouped bar chart showing ESS/s across tau_scale values for one
    parameterisation of the 8-schools model.
    """
    data = df_8s[df_8s["parameterisation"] == parameterisation]
    param_label = "Centred" if parameterisation == "centred" else "Non-Centred"
    _grouped_bar(
        data, "ess_per_sec", "tau_scale", TAU_COLORS,
        ylabel="ESS per Second",
        title=f"Effect of Prior Scale on ESS/s — 8-Schools ({param_label})",
        save_path=save_path,
        group_label_fmt="τ scale={}",
    )


def plot_tau_scale_rhat(
        df_8s,
        parameterisation="centred",
        save_path="results/figures/tau_scale_rhat_centred.png"):
    """
    Grouped bar chart showing R-hat across tau_scale values for one
    parameterisation of the 8-schools model.
    """
    data = df_8s[df_8s["parameterisation"] == parameterisation]
    param_label = "Centred" if parameterisation == "centred" else "Non-Centred"
    _grouped_bar(
        data, "rhat", "tau_scale", TAU_COLORS,
        ylabel="R-hat",
        title=f"Effect of Prior Scale on R-hat — 8-Schools ({param_label})",
        save_path=save_path,
        threshold_line=1.1,
        group_label_fmt="τ scale={}",
    )


# ── Panel plots ───────────────────────────────────────────────────────────────

def plot_experiment_panel(df_fun):
    """2-panel summary: funnel centred and funnel non-centred."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Neal's Funnel: Effect of Parameterisation", fontsize=14, fontweight="bold")

    datasets = [
        (axes[0], df_fun[df_fun["parameterisation"] == "centred"],
         "Neal's Funnel (centred)"),
        (axes[1], df_fun[df_fun["parameterisation"] == "noncentred"],
         "Neal's Funnel (non-centred)"),
    ]

    for ax, data, title in datasets:
        agg = _agg(data, "ess_per_sec")
        for algo in ALGO_ORDER:
            sub = agg[agg["algorithm"] == algo].sort_values("dimension")
            if sub.empty:
                continue
            ax.errorbar(
                sub["dimension"], sub["mean"], yerr=sub["std"],
                marker="o", label=algo, color=ALGO_COLORS[algo],
                linewidth=2, capsize=3, capthick=1.2,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dimension", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("ESS per Second", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/experiment_panel.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("Saved results/figures/experiment_panel.png")


def plot_funnel_comparison(df_fun):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Neal's Funnel: Centred vs Non-Centred",
                 fontsize=13, fontweight="bold")

    for ax, param in zip(axes, ["centred", "noncentred"]):
        sub = df_fun[df_fun["parameterisation"] == param]
        agg = _agg(sub, "ess_per_sec")
        for algo in ALGO_ORDER:
            s = agg[agg["algorithm"] == algo].sort_values("dimension")
            if s.empty:
                continue
            ax.errorbar(
                s["dimension"], s["mean"], yerr=s["std"],
                marker="o", label=algo, color=ALGO_COLORS[algo],
                linewidth=2, capsize=3, capthick=1.2,
            )
        ax.set_title(param.capitalize(), fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dimension", fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("ESS per Second", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/funnel_comparison.png", dpi=300)
    plt.close()
    print("Saved results/figures/funnel_comparison.png")