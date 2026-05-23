import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np


# ── Synthetic distributions ───────────────────────────────────────────────────


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


# ── Bayesian logistic regression ──────────────────────────────────────────────

def prepare_breast_cancer_data(standardise=True):
    """
    Loads the Breast Cancer Wisconsin dataset.
    569 samples, 30 features, binary outcome (0=malignant, 1=benign).
    Standardisation puts all features on the same scale, which is important
    for MCMC — without it the posterior is extremely ill-conditioned.
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    if standardise:
        X = StandardScaler().fit_transform(X)
    y = data.target.astype(np.float32)
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)


def bayesian_logistic_regression(X, y, prior_std=1.0, dimension=None):
    """
    Bayesian logistic regression model.
    prior_std controls the width of the weight prior — smaller values
    constrain weights more tightly, larger values allow more diffuse posteriors.
    Logits are clipped to prevent numerical overflow.
    """
    d = X.shape[1]
    weights = numpyro.sample(
        "x",
        dist.Normal(jnp.zeros(d), prior_std * jnp.ones(d))
    )
    logits = jnp.clip(X @ weights, -30, 30)
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


# ── 8-schools hierarchical model ──────────────────────────────────────────────

# Estimated treatment effects and standard errors from a study of SAT
# coaching programs across 8 schools (Rubin, 1981).
EIGHT_SCHOOLS_Y = jnp.array(
    [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
    dtype=jnp.float32
)
EIGHT_SCHOOLS_SIGMA = jnp.array(
    [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    dtype=jnp.float32
)


def eight_schools_centred(dimension=None, tau_scale=10.0):
    """
    Centred parameterisation of the 8-schools hierarchical model.

    Each school effect theta_j is sampled directly from N(mu, tau).
    This creates strong posterior dependence between tau and theta,
    producing funnel geometry that is hard for MCMC to explore.

    tau_scale controls the prior on the group-level variance tau.
    Smaller values constrain the variance more tightly.
    dimension is accepted for API compatibility but ignored.
    """
    mu  = numpyro.sample("mu",  dist.Normal(0.0, 10.0))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))

    with numpyro.plate("schools", 8):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))

    numpyro.sample("obs", dist.Normal(theta, EIGHT_SCHOOLS_SIGMA),
                   obs=EIGHT_SCHOOLS_Y)
    numpyro.deterministic(
        "x", jnp.concatenate([mu[None], tau[None], theta])
    )


def eight_schools_noncentred(dimension=None, tau_scale=10.0):
    """
    Non-centred parameterisation of the 8-schools hierarchical model.

    Instead of sampling theta directly from N(mu, tau), we sample
    a standardised offset theta_tilde ~ N(0, 1) and recover
    theta = mu + tau * theta_tilde deterministically.
    This decouples mu/tau from theta, removing the funnel geometry.

    tau_scale controls the prior on the group-level variance tau.
    dimension is accepted for API compatibility but ignored.
    """
    mu  = numpyro.sample("mu",  dist.Normal(0.0, 10.0))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))

    with numpyro.plate("schools", 8):
        theta_tilde = numpyro.sample("theta_tilde", dist.Normal(0.0, 1.0))

    theta = numpyro.deterministic("theta", mu + tau * theta_tilde)
    numpyro.sample("obs", dist.Normal(theta, EIGHT_SCHOOLS_SIGMA),
                   obs=EIGHT_SCHOOLS_Y)
    numpyro.deterministic(
        "x", jnp.concatenate([mu[None], tau[None], theta])
    )