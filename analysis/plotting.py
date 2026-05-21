import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ALGO_COLORS = {"RWM": "#3a7eca", "HMC": "#f58c2a", "NUTS": "#3daa62"}
ALGO_ORDER  = ["RWM", "HMC", "NUTS"]


def _agg(df, metric, group_cols=None):
    """Aggregate metric by algorithm + dimension (+ any extra group cols)."""
    if group_cols is None:
        group_cols = []
    return (
        df.groupby(["algorithm", "dimension"] + group_cols)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


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


def plot_logistic_regression_bar(df_lr, save_path="results/figures/ess_per_sec_logistic.png"):
    """
    Bar chart for logistic regression — single dimension (d=30)
    so a line plot doesn't make sense. Shows mean ± std across seeds.
    """
    agg = df_lr.groupby("algorithm")["ess_per_sec"].agg(["mean", "std"]).reset_index()
    agg = agg[agg["algorithm"].isin(ALGO_ORDER)]

    plt.figure(figsize=(6, 5))
    colors = [ALGO_COLORS[a] for a in agg["algorithm"]]
    plt.bar(agg["algorithm"], agg["mean"], yerr=agg["std"],
            color=colors, capsize=6, edgecolor="white", linewidth=0.5)
    plt.ylabel("ESS per Second")
    plt.title("ESS/s — Bayesian Logistic Regression (d=30)")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rhat_bar(df_lr, save_path="results/figures/rhat_logistic.png"):
    """R-hat bar chart for logistic regression."""
    agg = df_lr.groupby("algorithm")["rhat"].agg(["mean", "std"]).reset_index()
    agg = agg[agg["algorithm"].isin(ALGO_ORDER)]

    plt.figure(figsize=(6, 5))
    colors = [ALGO_COLORS[a] for a in agg["algorithm"]]
    bars = plt.bar(agg["algorithm"], agg["mean"], yerr=agg["std"],
                   color=colors, capsize=6, edgecolor="white", linewidth=0.5)
    plt.axhline(y=1.1, color="red", linestyle="--", linewidth=1,
                label="R-hat = 1.1 (warning threshold)")
    plt.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.ylabel("R-hat")
    plt.title("R-hat — Bayesian Logistic Regression (d=30)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_experiment_panel(df_iso, df_fun):
    """3-panel summary: isotropic, funnel centred, funnel non-centred."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Scalability of MCMC Methods", fontsize=14, fontweight="bold")

    datasets = [
        (axes[0], df_iso,
         "Isotropic Gaussian"),
        (axes[1], df_fun[df_fun["parameterisation"] == "centred"],
         "Neal's Funnel (centred)"),
        (axes[2], df_fun[df_fun["parameterisation"] == "noncentred"],
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
    fig.savefig("results/figures/experiment_panel.png", dpi=300, bbox_inches="tight")
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