import numpy as np
import jax.numpy as jnp
from jax import random


def isotropic_log_density(x, **kwargs):
    return -0.5 * jnp.sum(x ** 2)


def ill_conditioned_log_density(x, condition_number=100, **kwargs):
    dimension = x.shape[0]
    eigenvalues = jnp.linspace(1.0, float(condition_number), dimension)
    return -0.5 * jnp.sum(x ** 2 / eigenvalues)


def funnel_log_density(x, **kwargs):
    v = x[0]
    z = x[1:]
    log_p_v = -0.5 * (v / 3.0) ** 2 - jnp.log(3.0)
    log_p_z = -0.5 * jnp.sum(z ** 2 * jnp.exp(-v)) - 0.5 * (x.shape[0] - 1) * v
    return log_p_v + log_p_z



LOG_DENSITY_MAP = {
    "isotropic":       isotropic_log_density,
    "ill_conditioned": ill_conditioned_log_density,
    "funnel":          funnel_log_density,
    "funnel_noncentred": isotropic_log_density,  
}


class RandomWalkMetropolis:

    def __init__(self, proposal_std=0.5): 
        self.proposal_std = proposal_std

    def sample(
        self,
        rng_key,
        dimension,
        num_samples,
        warmup_steps=1000,
        log_density_fn=None,
        model_kwargs=None
    ):

        if log_density_fn is None:
            log_density_fn = isotropic_log_density

        if model_kwargs is None:
            model_kwargs = {}

        current = jnp.zeros(dimension)
        current_log_prob = log_density_fn(current, **model_kwargs)

        samples = []
        accepted = 0
        total_steps = warmup_steps + num_samples

        for i in range(total_steps):

            rng_key, subkey = random.split(rng_key)
            proposal = current + (self.proposal_std / jnp.sqrt(dimension)) * random.normal(
                subkey, shape=(dimension,)
            )

            proposal_log_prob = log_density_fn(proposal, **model_kwargs)
            log_alpha = proposal_log_prob - current_log_prob

            rng_key, subkey = random.split(rng_key)
            u = jnp.log(random.uniform(subkey))

            if u < log_alpha:
                current = proposal
                current_log_prob = proposal_log_prob
                accepted += 1

            if i >= warmup_steps:
                samples.append(np.array(current))

        acceptance_rate = accepted / total_steps
        return np.array(samples), acceptance_rate