import numpy as np


def sample_ref_exp(N, rate=8.0, xmax=1.0, rng=None):
    """Sample from an exponential truncated at xmax."""
    rng = np.random.default_rng() if rng is None else rng
    Z = 1.0 - np.exp(-rate * xmax)
    u = rng.random(N)
    return -(1.0 / rate) * np.log(1.0 - u * Z)


def sample_signal_gauss(N, mu=0.8, sigma=0.02, rng=None):
    """Sample signal from a Gaussian."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mu, scale=sigma, size=N)


def make_data_sample_poisson(NR=2000, NS=10, rng=None):
    """
    Generate a pseudo-dataset with Poisson-fluctuated background and signal counts.

    Returns
    -------
    x : ndarray
        Shuffled pooled data sample (background + signal).
    xb : ndarray
        Background-only component.
    xs : ndarray
        Signal-only component.
    N_B : int
        Realized background count.
    N_S : int
        Realized signal count.
    """
    rng = np.random.default_rng() if rng is None else rng
    N_B = int(rng.poisson(NR))
    N_S = int(rng.poisson(NS))

    xb = sample_ref_exp(N_B, rng=rng)
    xs = sample_signal_gauss(N_S, rng=rng)

    x = np.concatenate([xb, xs])
    rng.shuffle(x)
    return x, xb, xs, N_B, N_S