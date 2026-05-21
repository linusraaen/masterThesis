import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np


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

def prepare_breast_cancer_data():
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target.astype(np.float32)
    return jnp.array(X), jnp.array(y)

def bayesian_logistic_regression(X, y, dimension=None):
    """
    Bayesian logistic regression.
    dimension argument kept for API compatibility but ignored —
    dimension is determined by X.shape[1].
    """
    d = X.shape[1]
    # Prior over weights
    weights = numpyro.sample(
        "x",  # keep "x" as sample name for compatibility with existing code
        dist.Normal(jnp.zeros(d), jnp.ones(d))
    )
    # Likelihood
    logits = X @ weights
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
