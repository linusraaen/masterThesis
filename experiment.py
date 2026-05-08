import pandas as pd
from jax import random

from algorithms.rwm import RandomWalkMetropolis, LOG_DENSITY_MAP
from algorithms.hmc import run_hmc
from algorithms.nuts import run_nuts

from datasets import (
    gaussian_model,
    ill_conditioned_gaussian_model,
    funnel_model,
    funnel_model_noncentred,
)

from analysis.metrics import (
    compute_ess,
    compute_ess_per_sec,
    measure_runtime
)

from analysis.plotting import (
    plot_ess,
    plot_runtime,
    plot_ess_per_sec,
    plot_experiment_panel
)

CONFIG = {
    "dimensions":        [2, 10, 50, 100, 200, 500], 
    "funnel_dimensions": [2, 5, 10, 30, 100, 200, 500],  
    "condition_numbers": [10, 100, 1000], 
    "num_samples":       2000,
    "warmup_steps":      1000,
    "seed":              18,
    "proposal_std":      2.38 #Roberts et al. (1997)
}



def _run_dimension(d, rng_key, hmc_model, nuts_model, rwm_log_density,
                   model_kwargs, rwm_kwargs, num_samples, warmup_steps, proposal_std):
    rows = []

    rwm = RandomWalkMetropolis(proposal_std=proposal_std)
    (rwm_samples, acceptance_rate), rwm_runtime = measure_runtime(
        rwm.sample,
        rng_key, d, num_samples, warmup_steps,
        log_density_fn=rwm_log_density,
        model_kwargs=rwm_kwargs
    )
    rwm_ess = compute_ess(rwm_samples)
    rows.append({
        "algorithm":       "RWM",
        "dimension":       d,
        "runtime":         rwm_runtime,
        "ess":             rwm_ess,
        "ess_per_sec":     compute_ess_per_sec(rwm_ess, rwm_runtime),
        "acceptance_rate": acceptance_rate,
    })

    hmc_samples, hmc_runtime = run_hmc(
        rng_key, d, num_samples, warmup_steps,
        model=hmc_model, model_kwargs=model_kwargs
    )
    hmc_ess = compute_ess(hmc_samples)
    rows.append({
        "algorithm":       "HMC",
        "dimension":       d,
        "runtime":         hmc_runtime,
        "ess":             hmc_ess,
        "ess_per_sec":     compute_ess_per_sec(hmc_ess, hmc_runtime),
        "acceptance_rate": None,
    })

    nuts_samples, nuts_runtime = run_nuts(
        rng_key, d, num_samples, warmup_steps,
        model=nuts_model, model_kwargs=model_kwargs
    )
    nuts_ess = compute_ess(nuts_samples)
    rows.append({
        "algorithm":       "NUTS",
        "dimension":       d,
        "runtime":         nuts_runtime,
        "ess":             nuts_ess,
        "ess_per_sec":     compute_ess_per_sec(nuts_ess, nuts_runtime),
        "acceptance_rate": None,
    })

    return rows


def run_isotropic_experiment():
    results = []
    cfg = CONFIG
    rng_key = random.PRNGKey(cfg["seed"])

    for d in cfg["dimensions"]:
        print(f"[isotropic] dimension={d}")
        rng_key, key = random.split(rng_key)
        rows = _run_dimension(
            d, key,
            hmc_model=gaussian_model,
            nuts_model=gaussian_model,
            rwm_log_density=LOG_DENSITY_MAP["isotropic"],
            model_kwargs={},
            rwm_kwargs={},
            num_samples=cfg["num_samples"],
            warmup_steps=cfg["warmup_steps"],
            proposal_std=cfg["proposal_std"]
        )
        for r in rows:
            r["experiment"] = "isotropic"
        results.extend(rows)

    return pd.DataFrame(results)

def run_ill_conditioned_experiment():
    results = []
    cfg = CONFIG
    rng_key = random.PRNGKey(cfg["seed"])

    for cond in cfg["condition_numbers"]:
        for d in cfg["dimensions"]:
            print(f"[ill_conditioned] kappa={cond}  dimension={d}")
            rng_key, key = random.split(rng_key)
            rows = _run_dimension(
                d, key,
                hmc_model=ill_conditioned_gaussian_model,
                nuts_model=ill_conditioned_gaussian_model,
                rwm_log_density=LOG_DENSITY_MAP["ill_conditioned"],
                model_kwargs={"condition_number": cond},
                rwm_kwargs={"condition_number": cond},
                num_samples=cfg["num_samples"],
                warmup_steps=cfg["warmup_steps"],
                proposal_std=cfg["proposal_std"]
            )
            for r in rows:
                r["experiment"]      = "ill_conditioned"
                r["condition_number"] = cond
            results.extend(rows)

    return pd.DataFrame(results)

def run_funnel_experiment():
    results = []
    cfg = CONFIG
    rng_key = random.PRNGKey(cfg["seed"])

    for param, hmc_model, nuts_model, rwm_key in [
        ("centred",    funnel_model,           funnel_model,           "funnel"),
        ("noncentred", funnel_model_noncentred, funnel_model_noncentred, "funnel_noncentred"),
    ]:
        for d in cfg["funnel_dimensions"]:
            print(f"[funnel {param}] dimension={d}")
            rng_key, key = random.split(rng_key)
            rows = _run_dimension(
                d, key,
                hmc_model=hmc_model,
                nuts_model=nuts_model,
                rwm_log_density=LOG_DENSITY_MAP[rwm_key],
                model_kwargs={},
                rwm_kwargs={},
                num_samples=cfg["num_samples"],
                warmup_steps=cfg["warmup_steps"],
                proposal_std=cfg["proposal_std"]
            )
            for r in rows:
                r["experiment"]       = "funnel"
                r["parameterisation"] = param
            results.extend(rows)

    return pd.DataFrame(results)

def run_all_experiments():
    df_iso  = run_isotropic_experiment()
    df_ill  = run_ill_conditioned_experiment()
    df_fun  = run_funnel_experiment()

    df_all = pd.concat([df_iso, df_ill, df_fun], ignore_index=True)
    df_all.to_csv("results/csv/all_results.csv", index=False)
    print("Saved results/csv/all_results.csv")

    df_iso.to_csv("results/csv/isotropic_results.csv", index=False)
    df_ill.to_csv("results/csv/ill_conditioned_results.csv", index=False)
    df_fun.to_csv("results/csv/funnel_results.csv", index=False)

    return df_iso, df_ill, df_fun


if __name__ == "__main__":
    df_iso, df_ill, df_fun = run_all_experiments()
    plot_ess(df_iso)
    plot_runtime(df_iso)
    plot_ess_per_sec(df_iso)
    plot_experiment_panel(df_iso, df_ill, df_fun)