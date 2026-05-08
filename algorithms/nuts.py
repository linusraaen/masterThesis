import numpyro
from numpyro.infer import MCMC, NUTS
from datasets import gaussian_model
import time


def run_nuts(
    rng_key,
    dimension,
    num_samples=2000,
    warmup_steps=1000,
    model=None,
    model_kwargs=None
):

    if model is None:
        model = gaussian_model

    if model_kwargs is None:
        model_kwargs = {}

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        num_warmup=warmup_steps,
        progress_bar=False
    )

    start = time.time()
    mcmc.run(rng_key, dimension=dimension, **model_kwargs)
    runtime = time.time() - start

    samples = mcmc.get_samples()["x"]

    return samples, runtime