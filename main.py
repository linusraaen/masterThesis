import numpyro
numpyro.set_host_device_count(4)

import jax
print(f"Devices available: {jax.local_device_count()}")

import os
import pandas as pd

from experiment import (
    run_isotropic_experiment,
    run_funnel_experiment,
    run_logistic_regression_experiment,
    run_eight_schools_experiment,
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
    plot_prior_scale_comparison,
    plot_prior_scale_rhat,
    plot_tau_scale_comparison,
    plot_tau_scale_rhat,
)


def ensure_directories():
    os.makedirs("results/csv",     exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)


def main():
    ensure_directories()

    # ── Experiment 1: Isotropic Gaussian ──────────────────────────────────────
    print("\n=== Experiment 1: Isotropic Gaussian ===")
    df_iso = run_isotropic_experiment()
    plot_ess(df_iso)
    plot_runtime(df_iso)
    plot_ess_per_sec(df_iso,
        title="ESS/s vs Dimension — Isotropic Gaussian",
        save_path="results/figures/ess_per_sec_isotropic.png")
    plot_rhat(df_iso,
        title="R-hat vs Dimension — Isotropic Gaussian",
        save_path="results/figures/rhat_isotropic.png")
    plot_divergences(df_iso,
        title="Divergences vs Dimension — Isotropic Gaussian",
        save_path="results/figures/divergences_isotropic.png")

    # ── Experiment 2: Neal's Funnel ────────────────────────────────────────────
    print("\n=== Experiment 2: Neal's Funnel ===")
    df_fun = run_funnel_experiment()
    plot_funnel_comparison(df_fun)
    plot_ess_per_sec(
        df_fun[df_fun["parameterisation"] == "centred"],
        title="ESS/s vs Dimension — Neal's Funnel (Centred)",
        save_path="results/figures/ess_per_sec_funnel_centred.png")
    plot_ess_per_sec(
        df_fun[df_fun["parameterisation"] == "noncentred"],
        title="ESS/s vs Dimension — Neal's Funnel (Non-Centred)",
        save_path="results/figures/ess_per_sec_funnel_noncentred.png")
    plot_rhat(df_fun,
        title="R-hat vs Dimension — Neal's Funnel",
        save_path="results/figures/rhat_funnel.png")
    plot_divergences(df_fun,
        title="Divergences vs Dimension — Neal's Funnel",
        save_path="results/figures/divergences_funnel.png")

    # ── Experiment 3: Bayesian Logistic Regression ─────────────────────────────
    print("\n=== Experiment 3: Bayesian Logistic Regression ===")
    df_lr = run_logistic_regression_experiment()

    # Baseline (sigma=1.0)
    df_lr_base = df_lr[df_lr["prior_std"] == 1.0]
    plot_logistic_regression_bar(df_lr_base,
        title="ESS/s — Bayesian Logistic Regression (σ=1.0)",
        save_path="results/figures/ess_per_sec_logistic.png")
    plot_rhat_bar(df_lr_base,
        title="R-hat — Bayesian Logistic Regression (σ=1.0)",
        save_path="results/figures/rhat_logistic.png")

    # Prior scale comparison
    plot_prior_scale_comparison(df_lr)
    plot_prior_scale_rhat(df_lr)

    # ── Experiment 4: 8-Schools ────────────────────────────────────────────────
    print("\n=== Experiment 4: 8-Schools Hierarchical Model ===")
    df_8s = run_eight_schools_experiment()

    # Centred vs non-centred baseline (tau_scale=10.0)
    for param in ["centred", "noncentred"]:
        label = "Centred" if param == "centred" else "Non-Centred"
        df_8s_base = df_8s[
            (df_8s["parameterisation"] == param) &
            (df_8s["tau_scale"] == 10.0)
        ]
        plot_logistic_regression_bar(
            df_8s_base,
            title=f"ESS/s — 8-Schools ({label}, τ scale=10)",
            save_path=f"results/figures/ess_per_sec_8schools_{param}.png")
        plot_rhat_bar(
            df_8s_base,
            title=f"R-hat — 8-Schools ({label}, τ scale=10)",
            save_path=f"results/figures/rhat_8schools_{param}.png")

    # Tau scale comparison
    plot_tau_scale_comparison(df_8s, parameterisation="centred",
        save_path="results/figures/tau_scale_ess_centred.png")
    plot_tau_scale_rhat(df_8s, parameterisation="centred",
        save_path="results/figures/tau_scale_rhat_centred.png")
    plot_tau_scale_comparison(df_8s, parameterisation="noncentred",
        save_path="results/figures/tau_scale_ess_noncentred.png")
    plot_tau_scale_rhat(df_8s, parameterisation="noncentred",
        save_path="results/figures/tau_scale_rhat_noncentred.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Summary panel ===")
    plot_experiment_panel(df_iso, df_fun)

    pd.concat([df_iso, df_fun, df_lr, df_8s], ignore_index=True).to_csv(
        "results/csv/all_results.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()