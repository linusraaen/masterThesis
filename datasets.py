import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def gaussian_model(dimension):
    numpyro.sample(
        "x",
        dist.MultivariateNormal(
            loc=jnp.zeros(dimension),
            covariance_matrix=jnp.eye(dimension)
        )
    )


def ill_conditioned_gaussian_model(dimension, condition_number=100):
    eigenvalues = jnp.linspace(1.0, float(condition_number), dimension)
    covariance = jnp.diag(eigenvalues)
    numpyro.sample(
        "x",
        dist.MultivariateNormal(
            loc=jnp.zeros(dimension),
            covariance_matrix=covariance
        )
    )


def funnel_model(dimension):
    v = numpyro.sample("v", dist.Normal(0.0, 3.0))
    numpyro.sample(
        "x",
        dist.Normal(
            jnp.zeros(dimension - 1),
            jnp.exp(v / 2.0) * jnp.ones(dimension - 1)
        )
    )


def funnel_model_noncentred(dimension):
    v = numpyro.sample("v", dist.Normal(0.0, 3.0))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(dimension - 1), 1.0))
    numpyro.deterministic("x", jnp.exp(v / 2.0) * z)