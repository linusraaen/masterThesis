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

def measure_runtime(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    runtime = time.time() - start
    return result, runtime

def compute_ess_per_sec(ess, runtime):

    return ess / runtime

def summarize_results(df):

    return (
        df.groupby(["algorithm", "dimension"])
        .agg({
            "ess": ["mean", "std"],
            "runtime": ["mean", "std"],
            "ess_per_sec": ["mean", "std"]
        })
    )