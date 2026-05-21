import time
import numpy as np
import pandas as pd
import arviz as az


def compute_ess(samples):
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples[None, :, :]
    ess = az.ess(samples, method="bulk")
    if hasattr(ess, 'values'):
        ess_values = ess.values
    else:
        ess_values = np.asarray(ess)
    return float(np.mean(ess_values))


def compute_rhat(samples, num_chains=4):
    samples = np.asarray(samples)
    if samples.ndim == 2:
        n = samples.shape[0] // num_chains
        samples = samples.reshape(num_chains, n, -1)
    rhat = az.rhat(samples)
    if hasattr(rhat, 'values'):
        return float(np.mean(rhat.values))
    return float(np.mean(np.asarray(rhat)))


def compute_divergences(mcmc):
    try:
        diverging = mcmc.get_extra_fields()["diverging"]
        return int(diverging.sum())
    except (KeyError, AttributeError):
        return None


def measure_runtime(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    runtime = time.time() - start
    return result, runtime


def compute_ess_per_sec(ess, runtime):
    return ess / runtime


def summarize_results(df):
    agg_dict = {
        "ess":         ["mean", "std"],
        "runtime":     ["mean", "std"],
        "ess_per_sec": ["mean", "std"],
        "rhat":        ["mean", "std"],
    }
    if "divergences" in df.columns:
        agg_dict["divergences"] = ["mean", "std"]
    return (
        df.groupby(["algorithm", "dimension"])
        .agg(agg_dict)
    )