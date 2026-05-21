import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


ALGO_COLORS = {"RWM": "#3a7eca", "HMC": "#f58c2a", "NUTS": "#3daa62"}
ALGO_ORDER  = ["RWM", "HMC", "NUTS"]


def plot_metric(
    df,
    metric,
    ylabel,
    title,
    save_path=None,
    log_x=True,
    log_y=True
):
    plt.figure(figsize=(8, 5))

    for algorithm in ALGO_ORDER:
        subset = df[df["algorithm"] == algorithm]
        if subset.empty:
            continue
        plt.plot(
            subset["dimension"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=algorithm,
            color=ALGO_COLORS[algorithm]
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_ess(df):
    plot_metric(
        df=df, metric="ess",
        ylabel="Effective Sample Size (ESS)",
        title="ESS vs Dimension",
        save_path="results/figures/ess_vs_dimension.png"
    )


def plot_runtime(df):
    plot_metric(
        df=df, metric="runtime",
        ylabel="Runtime (seconds)",
        title="Runtime vs Dimension",
        save_path="results/figures/runtime_vs_dimension.png"
    )


def plot_ess_per_sec(df, title="ESS per Second vs Dimension", 
                     save_path="results/figures/ess_per_second_vs_dimension.png"):
    plot_metric(
        df=df, metric="ess_per_sec",
        ylabel="ESS per Second",
        title=title,
        save_path=save_path
    )


def plot_experiment_panel(df_iso, df_ill, df_fun, condition_number=100):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Scalability of MCMC Methods", fontsize=14, fontweight="bold")

    datasets = [
    (axes[0], df_iso, "Isotropic Gaussian"),
    (axes[1], df_ill[df_ill["condition_number"] == 100], "Ill-Conditioned (κ=100)"),
    (axes[2], df_fun[df_fun["parameterisation"] == "centred"], "Neal's Funnel (centred)"),
    (axes[3], df_fun[df_fun["parameterisation"] == "noncentred"], "Neal's Funnel (non-centred)"),
]

    for ax, data, title in datasets:
        for algo in ALGO_ORDER:
            sub = data[data["algorithm"] == algo].sort_values("dimension")
            if sub.empty:
                continue
            ax.plot(
                sub["dimension"], sub["ess_per_sec"],
                marker="o", label=algo,
                color=ALGO_COLORS[algo], linewidth=2
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
    plt.show()
    print("Saved results/figures/experiment_panel.png")


def plot_condition_number_sweep(df_ill, dimension=100):
    data = df_ill[df_ill["dimension"] == dimension]

    plt.figure(figsize=(8, 5))
    for algo in ALGO_ORDER:
        sub = data[data["algorithm"] == algo].sort_values("condition_number")
        if sub.empty:
            continue
        plt.plot(
            sub["condition_number"], sub["ess_per_sec"],
            marker="o", label=algo,
            color=ALGO_COLORS[algo], linewidth=2
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Condition Number κ")
    plt.ylabel("ESS per Second")
    plt.title(f"ESS/sec vs Condition Number (d={dimension})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/condition_sweep_d{dimension}.png", dpi=300)
    plt.show()


def plot_funnel_comparison(df_fun):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Neal's Funnel: Centred vs Non-Centred", fontsize=13, fontweight="bold")

    for ax, param in zip(axes, ["centred", "noncentred"]):
        sub = df_fun[df_fun["parameterisation"] == param]
        for algo in ALGO_ORDER:
            s = sub[sub["algorithm"] == algo].sort_values("dimension")
            if s.empty:
                continue
            ax.plot(
                s["dimension"], s["ess_per_sec"],
                marker="o", label=algo,
                color=ALGO_COLORS[algo], linewidth=2
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
    plt.show()
    print("Saved results/figures/funnel_comparison.png")