import numpyro
numpyro.set_host_device_count(4)

import jax
import pandas as pd
import jax.numpy as jnp
from jax import random
from functools import partial

from algorithms.rwm import RandomWalkMetropolis, LOG_DENSITY_MAP
from algorithms.hmc import run_hmc
from algorithms.nuts import run_nuts

from datasets import (
    gaussian_model,
    funnel_model,
    funnel_model_noncentred,
    prepare_breast_cancer_data,
    bayesian_logistic_regression,
)


from analysis.metrics import (
    compute_ess,
    compute_rhat,
    compute_ess_per_sec,
    measure_runtime
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

CONFIG = {
    "dimensions":        [2, 10, 50, 100, 200, 500],
    "funnel_dimensions": [2, 5, 10, 30, 100, 200, 500],
    "num_samples":       2000,
    "warmup_steps":      1000,
    "num_chains":        4,
    "seeds":             [18, 42, 123],
    "proposal_std":      2.38,
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run_dimension(d, rng_key, hmc_model, nuts_model, rwm_log_density,
                   model_kwargs, rwm_kwargs, num_samples, warmup_steps,
                   num_chains, proposal_std):
    rows = []

    # RWM
    rwm = RandomWalkMetropolis(proposal_std=proposal_std)
    (rwm_samples, acceptance_rate), rwm_runtime = measure_runtime(
        rwm.sample,
        rng_key, d, num_samples, warmup_steps,
        num_chains=num_chains,
        log_density_fn=rwm_log_density,
        model_kwargs=rwm_kwargs
    )
    rwm_ess  = compute_ess(rwm_samples)
    rwm_rhat = compute_rhat(rwm_samples, num_chains=num_chains)
    rows.append({
        "algorithm":       "RWM",
        "dimension":       d,
        "runtime":         rwm_runtime,
        "ess":             rwm_ess,
        "ess_per_sec":     compute_ess_per_sec(rwm_ess, rwm_runtime),
        "rhat":            rwm_rhat,
        "divergences":     None,
        "acceptance_rate": acceptance_rate,
    })

    # HMC
    hmc_samples, hmc_runtime, hmc_divergences, _ = run_hmc(
        rng_key, d, num_samples, warmup_steps,
        num_chains=num_chains,
        model=hmc_model, model_kwargs=model_kwargs
    )
    hmc_ess  = compute_ess(hmc_samples)
    hmc_rhat = compute_rhat(hmc_samples, num_chains=num_chains)
    rows.append({
        "algorithm":       "HMC",
        "dimension":       d,
        "runtime":         hmc_runtime,
        "ess":             hmc_ess,
        "ess_per_sec":     compute_ess_per_sec(hmc_ess, hmc_runtime),
        "rhat":            hmc_rhat,
        "divergences":     hmc_divergences,
        "acceptance_rate": None,
    })

    # NUTS
    nuts_samples, nuts_runtime, nuts_divergences, _ = run_nuts(
        rng_key, d, num_samples, warmup_steps,
        num_chains=num_chains,
        model=nuts_model, model_kwargs=model_kwargs
    )
    nuts_ess  = compute_ess(nuts_samples)
    nuts_rhat = compute_rhat(nuts_samples, num_chains=num_chains)
    rows.append({
        "algorithm":       "NUTS",
        "dimension":       d,
        "runtime":         nuts_runtime,
        "ess":             nuts_ess,
        "ess_per_sec":     compute_ess_per_sec(nuts_ess, nuts_runtime),
        "rhat":            nuts_rhat,
        "divergences":     nuts_divergences,
        "acceptance_rate": None,
    })

    return rows


def _run_experiment_single_seed(seed, dimensions, hmc_model, nuts_model,
                                 rwm_log_density, model_kwargs, rwm_kwargs,
                                 extra_fields, cfg):
    results = []
    rng_key = random.PRNGKey(seed)

    for d in dimensions:
        rng_key, key = random.split(rng_key)
        rows = _run_dimension(
            d, key,
            hmc_model=hmc_model,
            nuts_model=nuts_model,
            rwm_log_density=rwm_log_density,
            model_kwargs=model_kwargs,
            rwm_kwargs=rwm_kwargs,
            num_samples=cfg["num_samples"],
            warmup_steps=cfg["warmup_steps"],
            num_chains=cfg["num_chains"],
            proposal_std=cfg["proposal_std"],
        )
        for r in rows:
            r["seed"] = seed
            for k, v in extra_fields.items():
                r[k] = v
        results.extend(rows)

    return results


# ── Experiment 1: Isotropic Gaussian ─────────────────────────────────────────

def run_isotropic_experiment():
    cfg = CONFIG
    results = []
    for seed in cfg["seeds"]:
        print(f"[isotropic] seed={seed}")
        rows = _run_experiment_single_seed(
            seed=seed,
            dimensions=cfg["dimensions"],
            hmc_model=gaussian_model,
            nuts_model=gaussian_model,
            rwm_log_density=LOG_DENSITY_MAP["isotropic"],
            model_kwargs={},
            rwm_kwargs={},
            extra_fields={"experiment": "isotropic"},
            cfg=cfg,
        )
        results.extend(rows)
    return pd.DataFrame(results)


# ── Experiment 2: Neal's Funnel ───────────────────────────────────────────────

def run_funnel_experiment():
    cfg = CONFIG
    results = []
    for param, hmc_model, nuts_model, rwm_key in [
        ("centred",    funnel_model,           funnel_model,           "funnel"),
        ("noncentred", funnel_model_noncentred, funnel_model_noncentred, "funnel_noncentred"),
    ]:
        for seed in cfg["seeds"]:
            print(f"[funnel {param}] seed={seed}")
            rows = _run_experiment_single_seed(
                seed=seed,
                dimensions=cfg["funnel_dimensions"],
                hmc_model=hmc_model,
                nuts_model=nuts_model,
                rwm_log_density=LOG_DENSITY_MAP[rwm_key],
                model_kwargs={},
                rwm_kwargs={},
                extra_fields={"experiment": "funnel",
                              "parameterisation": param},
                cfg=cfg,
            )
            results.extend(rows)
    return pd.DataFrame(results)


# ── Experiment 3: Bayesian Logistic Regression ───────────────────────────────

def run_logistic_regression_experiment():
    cfg = CONFIG
    results = []

    X, y = prepare_breast_cancer_data()
    dimension = int(X.shape[1])  # 30
    model = partial(bayesian_logistic_regression, X=X, y=y)

    # Log density for RWM
    def logistic_log_density(w, **kwargs):
        log_prior = -0.5 * jnp.sum(w ** 2)
        logits = X @ w
        log_lik = jnp.sum(
            y * jax.nn.log_sigmoid(logits) +
            (1 - y) * jax.nn.log_sigmoid(-logits)
        )
        return log_prior + log_lik

    print(f"\n[logistic_regression] d={dimension}, n={X.shape[0]}")

    for seed in cfg["seeds"]:
        print(f"[logistic_regression] seed={seed}")
        rng_key = random.PRNGKey(seed)

        # HMC
        rng_key, key = random.split(rng_key)
        hmc_samples, hmc_runtime, hmc_divergences, _ = run_hmc(
            key, dimension,
            num_samples=cfg["num_samples"],
            warmup_steps=cfg["warmup_steps"],
            num_chains=cfg["num_chains"],
            model=model,
            model_kwargs={}
        )
        hmc_ess  = compute_ess(hmc_samples)
        hmc_rhat = compute_rhat(hmc_samples, num_chains=cfg["num_chains"])
        results.append({
            "algorithm":       "HMC",
            "dimension":       dimension,
            "runtime":         hmc_runtime,
            "ess":             hmc_ess,
            "ess_per_sec":     compute_ess_per_sec(hmc_ess, hmc_runtime),
            "rhat":            hmc_rhat,
            "divergences":     hmc_divergences,
            "acceptance_rate": None,
            "experiment":      "logistic_regression",
            "seed":            seed,
        })

        # NUTS
        rng_key, key = random.split(rng_key)
        nuts_samples, nuts_runtime, nuts_divergences, _ = run_nuts(
            key, dimension,
            num_samples=cfg["num_samples"],
            warmup_steps=cfg["warmup_steps"],
            num_chains=cfg["num_chains"],
            model=model,
            model_kwargs={}
        )
        nuts_ess  = compute_ess(nuts_samples)
        nuts_rhat = compute_rhat(nuts_samples, num_chains=cfg["num_chains"])
        results.append({
            "algorithm":       "NUTS",
            "dimension":       dimension,
            "runtime":         nuts_runtime,
            "ess":             nuts_ess,
            "ess_per_sec":     compute_ess_per_sec(nuts_ess, nuts_runtime),
            "rhat":            nuts_rhat,
            "divergences":     nuts_divergences,
            "acceptance_rate": None,
            "experiment":      "logistic_regression",
            "seed":            seed,
        })

        # RWM
        rwm = RandomWalkMetropolis(proposal_std=cfg["proposal_std"])
        rng_key, key = random.split(rng_key)
        (rwm_samples, acceptance_rate), rwm_runtime = measure_runtime(
            rwm.sample,
            key, dimension,
            cfg["num_samples"],
            cfg["warmup_steps"],
            num_chains=cfg["num_chains"],
            log_density_fn=logistic_log_density,
            model_kwargs={}
        )
        rwm_ess  = compute_ess(rwm_samples)
        rwm_rhat = compute_rhat(rwm_samples, num_chains=cfg["num_chains"])
        results.append({
            "algorithm":       "RWM",
            "dimension":       dimension,
            "runtime":         rwm_runtime,
            "ess":             rwm_ess,
            "ess_per_sec":     compute_ess_per_sec(rwm_ess, rwm_runtime),
            "rhat":            rwm_rhat,
            "divergences":     None,
            "acceptance_rate": acceptance_rate,
            "experiment":      "logistic_regression",
            "seed":            seed,
        })

    df = pd.DataFrame(results)
    df.to_csv("results/csv/logistic_regression_results.csv", index=False)
    print("Saved results/csv/logistic_regression_results.csv")
    return df


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all_experiments():
    df_iso = run_isotropic_experiment()
    df_fun = run_funnel_experiment()
    df_lr  = run_logistic_regression_experiment()

    df_all = pd.concat([df_iso, df_fun, df_lr], ignore_index=True)
    df_all.to_csv("results/csv/all_results.csv", index=False)

    df_iso.to_csv("results/csv/isotropic_results.csv", index=False)
    df_fun.to_csv("results/csv/funnel_results.csv", index=False)

    print("Saved all CSVs")
    return df_iso, df_fun, df_lr


if __name__ == "__main__":
    df_iso, df_fun, df_lr = run_all_experiments()

    # Isotropic plots
    plot_ess(df_iso)
    plot_runtime(df_iso)
    plot_ess_per_sec(
        df_iso,
        title="ESS/s vs Dimension — Isotropic Gaussian",
        save_path="results/figures/ess_per_sec_isotropic.png"
    )
    plot_rhat(df_iso,
              title="R-hat vs Dimension — Isotropic Gaussian",
              save_path="results/figures/rhat_isotropic.png")
    plot_divergences(df_iso,
                     title="Divergences vs Dimension — Isotropic Gaussian",
                     save_path="results/figures/divergences_isotropic.png")

    # Funnel plots
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
    plot_rhat(df_fun,
              title="R-hat vs Dimension — Neal's Funnel",
              save_path="results/figures/rhat_funnel.png")
    plot_divergences(df_fun,
                     title="Divergences vs Dimension — Neal's Funnel",
                     save_path="results/figures/divergences_funnel.png")

    # Logistic regression plots
    plot_logistic_regression_bar(df_lr)
    plot_rhat_bar(df_lr)

    # Summary panel
    plot_experiment_panel(df_iso, df_fun)