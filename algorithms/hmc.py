import numpyro
from numpyro.infer import MCMC, HMC
import time


def run_hmc(
    rng_key,
    dimension,
    num_samples=2000,
    warmup_steps=1000,
    num_chains=4,
    model=None,
    model_kwargs=None
):
    if model is None:
        raise ValueError("model must be provided")

    if model_kwargs is None:
        model_kwargs = {}

    kernel = HMC(model)

    mcmc = MCMC(
        kernel,
        num_warmup=warmup_steps,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False
    )

    start = time.time()
    mcmc.run(rng_key, dimension=dimension, **model_kwargs)
    runtime = time.time() - start

    # samples shape: (num_chains * num_samples, dimension)
    samples = mcmc.get_samples()["x"]

    # divergences
    try:
        divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    except (KeyError, AttributeError):
        divergences = None

    return samples, runtime, divergences, mcmc