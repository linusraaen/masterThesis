import numpyro
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import pandas as pd
from jax import random
from functools import partial

from algorithms.rwm import RandomWalkMetropolis, LOG_DENSITY_MAP
from algorithms.hmc import run_hmc
from algorithms.nuts import run_nuts

from datasets import (
    funnel_model,
    funnel_model_noncentred,
    prepare_breast_cancer_data,
    bayesian_logistic_regression,
    eight_schools_centred,
    eight_schools_noncentred,
)

from analysis.metrics import (
    compute_ess,
    compute_rhat,
    compute_ess_per_sec,
    measure_runtime,
)

CONFIG = {
    "funnel_dimensions": [2, 5, 10, 30, 100, 200, 500],
    "num_samples":       2000,
    "warmup_steps":      1000,
    "num_chains":        4,
    "seeds":             [18, 42, 123],
    "proposal_std":      2.38,
    "prior_stds":        [0.1, 1.0, 10.0],
    "tau_scales":        [1.0, 10.0, 100.0],
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def _make_row(algorithm, dimension, runtime, ess, rhat, divergences,
              acceptance_rate, seed, extra_fields):
    return {
        "algorithm":       algorithm,
        "dimension":       dimension,
        "runtime":         runtime,
        "ess":             ess,
        "ess_per_sec":     compute_ess_per_sec(ess, runtime),
        "rhat":            rhat,
        "divergences":     divergences,
        "acceptance_rate": acceptance_rate,
        "seed":            seed,
        **extra_fields,
    }


def _run_all_algorithms(rng_key, dimension, seed, extra_fields, cfg,
                        hmc_model, nuts_model,
                        rwm_log_density, rwm_model_kwargs=None,
                        hmc_model_kwargs=None,
                        include_rwm=True):
    """Runs RWM, HMC, and NUTS for a single configuration."""
    if rwm_model_kwargs is None:
        rwm_model_kwargs = {}
    if hmc_model_kwargs is None:
        hmc_model_kwargs = {}

    rows = []
    num_chains  = cfg["num_chains"]
    num_samples = cfg["num_samples"]
    warmup      = cfg["warmup_steps"]

    # RWM
    if include_rwm:
        rwm = RandomWalkMetropolis(proposal_std=cfg["proposal_std"])
        (rwm_samples, acc), rwm_runtime = measure_runtime(
            rwm.sample, rng_key, dimension, num_samples, warmup,
            num_chains=num_chains,
            log_density_fn=rwm_log_density,
            model_kwargs=rwm_model_kwargs,
        )
        rows.append(_make_row(
            "RWM", dimension, rwm_runtime,
            compute_ess(rwm_samples),
            compute_rhat(rwm_samples, num_chains=num_chains),
            None, acc, seed, extra_fields
        ))

    # HMC
    rng_key, key = random.split(rng_key)
    hmc_samples, hmc_runtime, hmc_div, _ = run_hmc(
        key, dimension, num_samples, warmup,
        num_chains=num_chains,
        model=hmc_model, model_kwargs=hmc_model_kwargs,
    )
    rows.append(_make_row(
        "HMC", dimension, hmc_runtime,
        compute_ess(hmc_samples),
        compute_rhat(hmc_samples, num_chains=num_chains),
        hmc_div, None, seed, extra_fields
    ))

    # NUTS
    rng_key, key = random.split(rng_key)
    nuts_samples, nuts_runtime, nuts_div, _ = run_nuts(
        key, dimension, num_samples, warmup,
        num_chains=num_chains,
        model=nuts_model, model_kwargs=hmc_model_kwargs,
    )
    rows.append(_make_row(
        "NUTS", dimension, nuts_runtime,
        compute_ess(nuts_samples),
        compute_rhat(nuts_samples, num_chains=num_chains),
        nuts_div, None, seed, extra_fields
    ))

    return rows


# ── Experiment 1: Neal's Funnel ───────────────────────────────────────────────

def run_funnel_experiment():
    cfg = CONFIG
    results = []
    for param, hmc_model, nuts_model, rwm_key in [
        ("centred",    funnel_model,           funnel_model,           "funnel"),
        ("noncentred", funnel_model_noncentred, funnel_model_noncentred, "funnel_noncentred"),
    ]:
        for seed in cfg["seeds"]:
            print(f"[funnel {param}] seed={seed}")
            rng_key = random.PRNGKey(seed)
            for d in cfg["funnel_dimensions"]:
                rng_key, key = random.split(rng_key)
                rows = _run_all_algorithms(
                    key, d, seed,
                    extra_fields={"experiment": "funnel",
                                  "parameterisation": param},
                    cfg=cfg,
                    hmc_model=hmc_model,
                    nuts_model=nuts_model,
                    rwm_log_density=LOG_DENSITY_MAP[rwm_key],
                )
                results.extend(rows)
    return pd.DataFrame(results)


# ── Experiment 3: Bayesian Logistic Regression — prior scale sweep ───────────

def run_logistic_regression_experiment():
    """
    Tests how prior scale affects sampler performance on a real posterior.
    Uses the standardised Breast Cancer Wisconsin dataset (d=30).
    prior_std in {0.1, 1.0, 10.0} — tight, standard, diffuse.
    """
    cfg = CONFIG
    results = []
    X, y = prepare_breast_cancer_data(standardise=True)
    dimension = int(X.shape[1])

    for prior_std in cfg["prior_stds"]:
        model = partial(bayesian_logistic_regression, X=X, y=y,
                        prior_std=prior_std)

        def make_rwm_log_density(ps):
            def logistic_log_density(w, **kwargs):
                log_prior = -0.5 * jnp.sum((w / ps) ** 2) - \
                            dimension * jnp.log(ps)
                logits = jnp.clip(X @ w, -30, 30)
                log_lik = jnp.sum(
                    y * jax.nn.log_sigmoid(logits) +
                    (1 - y) * jax.nn.log_sigmoid(-logits)
                )
                return log_prior + log_lik
            return logistic_log_density

        rwm_log_density = make_rwm_log_density(prior_std)

        for seed in cfg["seeds"]:
            print(f"[logistic] prior_std={prior_std} seed={seed}")
            rng_key = random.PRNGKey(seed)
            rows = _run_all_algorithms(
                rng_key, dimension, seed,
                extra_fields={"experiment": "logistic_regression",
                              "prior_std": prior_std},
                cfg=cfg,
                hmc_model=model,
                nuts_model=model,
                rwm_log_density=rwm_log_density,
            )
            results.extend(rows)

    df = pd.DataFrame(results)
    df.to_csv("results/csv/logistic_regression_results.csv", index=False)
    print("Saved results/csv/logistic_regression_results.csv")
    return df


# ── Experiment 4: 8-Schools — centred vs non-centred, prior scale sweep ──────

def run_eight_schools_experiment():
    """
    Tests two parameterisation axes on the real 8-schools hierarchical model:
    1. Centred vs non-centred — shows funnel geometry in a real model
    2. Prior scale on tau (tau_scale in {1.0, 10.0, 100.0}) — shows how
       the prior on the group-level variance affects sampler performance
    RWM excluded — posterior geometry too complex for isotropic proposals.
    """
    cfg = CONFIG
    results = []
    dimension = 10  # mu, tau, theta_1..theta_8

    for param, model_fn in [
        ("centred",    eight_schools_centred),
        ("noncentred", eight_schools_noncentred),
    ]:
        for tau_scale in cfg["tau_scales"]:
            model = partial(model_fn, tau_scale=tau_scale)
            for seed in cfg["seeds"]:
                print(f"[8-schools {param}] tau_scale={tau_scale} seed={seed}")
                rng_key = random.PRNGKey(seed)
                rows = _run_all_algorithms(
                    rng_key, dimension, seed,
                    extra_fields={"experiment":       "eight_schools",
                                  "parameterisation": param,
                                  "tau_scale":        tau_scale},
                    cfg=cfg,
                    hmc_model=model,
                    nuts_model=model,
                    rwm_log_density=None,
                    include_rwm=False,
                )
                results.extend(rows)

    df = pd.DataFrame(results)
    df.to_csv("results/csv/eight_schools_results.csv", index=False)
    print("Saved results/csv/eight_schools_results.csv")
    return df


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all_experiments():
    df_fun = run_funnel_experiment()
    df_lr  = run_logistic_regression_experiment()
    df_8s  = run_eight_schools_experiment()

    for df, path in [
        (df_fun, "results/csv/funnel_results.csv"),
        (df_lr,  "results/csv/logistic_regression_results.csv"),
        (df_8s,  "results/csv/eight_schools_results.csv"),
    ]:
        df.to_csv(path, index=False)

    pd.concat([df_fun, df_lr, df_8s], ignore_index=True).to_csv(
        "results/csv/all_results.csv", index=False
    )
    print("Saved all CSVs")
    return df_fun, df_lr, df_8s