from experiment import (
    run_isotropic_experiment,
    run_ill_conditioned_experiment,
    run_funnel_experiment,
)

from analysis.plotting import (
    plot_ess,
    plot_runtime,
    plot_ess_per_sec,
    plot_experiment_panel,
    plot_condition_number_sweep,
    plot_funnel_comparison,
)

import pandas as pd
import os


def ensure_directories():
    os.makedirs("results/csv",     exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)


def main():
    ensure_directories()
    print("\n=== Experiment 1: Isotropic Gaussian ===")
    df_iso = run_isotropic_experiment()
    df_iso.to_csv("results/csv/isotropic_results.csv", index=False)

    plot_ess(df_iso)
    plot_runtime(df_iso)
    plot_ess_per_sec(df_iso)

    print("\n=== Experiment 2: Ill-Conditioned Gaussian ===")
    df_ill = run_ill_conditioned_experiment()
    df_ill.to_csv("results/csv/ill_conditioned_results.csv", index=False)

    plot_condition_number_sweep(df_ill, dimension=100)

    print("\n=== Experiment 3: Neal's Funnel ===")
    df_fun = run_funnel_experiment()
    df_fun.to_csv("results/csv/funnel_results.csv", index=False)

    plot_funnel_comparison(df_fun)

    print("\n=== Generating summary panel ===")
    plot_experiment_panel(df_iso, df_ill, df_fun)

    df_all = pd.concat([df_iso, df_ill, df_fun], ignore_index=True)
    df_all.to_csv("results/csv/all_results.csv", index=False)
    print("\nDone. All results saved to results/csv/ and figures to results/figures/")


if __name__ == "__main__":
    main()