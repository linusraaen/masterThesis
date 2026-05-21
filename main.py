import numpyro
numpyro.set_host_device_count(4)

import jax
print(f"Devices available: {jax.local_device_count()}")

from experiment import (
    run_isotropic_experiment,
    run_funnel_experiment,
    run_logistic_regression_experiment,
)

from analysis.plotting import (
    plot_ess,
    plot_runtime,
    plot_ess_per_sec,
    plot_rhat,
    plot_divergences,
    plot_experiment_panel,
    plot_funnel_comparison,
    plot_logistic_regression_bar,
    plot_rhat_bar,
)

import pandas as pd
import os


def ensure_directories():
    os.makedirs("results/csv",     exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)


def main():
    ensure_directories()

    # ── Experiment 1: Isotropic Gaussian ──────────────────────────────────────
    print("\n=== Experiment 1: Isotropic Gaussian ===")
    df_iso = run_isotropic_experiment()
    df_iso.to_csv("results/csv/isotropic_results.csv", index=False)

    plot_ess(df_iso)
    plot_runtime(df_iso)
    plot_ess_per_sec(
        df_iso,
        title="ESS/s vs Dimension — Isotropic Gaussian",
        save_path="results/figures/ess_per_sec_isotropic.png"
    )
    plot_rhat(
        df_iso,
        title="R-hat vs Dimension — Isotropic Gaussian",
        save_path="results/figures/rhat_isotropic.png"
    )
    plot_divergences(
        df_iso,
        title="Divergences vs Dimension — Isotropic Gaussian",
        save_path="results/figures/divergences_isotropic.png"
    )

    # ── Experiment 2: Neal's Funnel ────────────────────────────────────────────
    print("\n=== Experiment 2: Neal's Funnel ===")
    df_fun = run_funnel_experiment()
    df_fun.to_csv("results/csv/funnel_results.csv", index=False)

    plot_funnel_comparison(df_fun)
    plot_ess_per_sec(
        df_fun[df_fun["parameterisation"] == "centred"],
        title="ESS/s vs Dimension — Neal's Funnel (Centred)",
        save_path="results/figures/ess_per_sec_funnel_centred.png"
    )
    plot_ess_per_sec(
        df_fun[df_fun["parameterisation"] == "noncentred"],
        title="ESS/s vs Dimension — Neal's Funnel (Non-Centred)",
        save_path="results/figures/ess_per_sec_funnel_noncentred.png"
    )
    plot_rhat(
        df_fun,
        title="R-hat vs Dimension — Neal's Funnel",
        save_path="results/figures/rhat_funnel.png"
    )
    plot_divergences(
        df_fun,
        title="Divergences vs Dimension — Neal's Funnel",
        save_path="results/figures/divergences_funnel.png"
    )

    # ── Experiment 3: Bayesian Logistic Regression ─────────────────────────────
    print("\n=== Experiment 3: Bayesian Logistic Regression ===")
    df_lr = run_logistic_regression_experiment()
    df_lr.to_csv("results/csv/logistic_regression_results.csv", index=False)

    plot_logistic_regression_bar(df_lr)
    plot_rhat_bar(df_lr)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Generating summary panel ===")
    plot_experiment_panel(df_iso, df_fun)

    df_all = pd.concat([df_iso, df_fun, df_lr], ignore_index=True)
    df_all.to_csv("results/csv/all_results.csv", index=False)
    print("\nDone. All results saved.")


if __name__ == "__main__":
    main()