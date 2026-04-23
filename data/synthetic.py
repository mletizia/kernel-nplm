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



def build_pooled_sample(x_ref, x_dat, dtype=np.float64):
    """
    Build a pooled binary classification dataset from two samples.

    Parameters
    ----------
    x_ref : array-like
        Reference sample.
    x_dat : array-like
        Data sample.
    dtype : data-type, optional
        Output dtype for X and y. Default is np.float64.

    Returns
    -------
    X : ndarray of shape (n_ref + n_dat, 1)
        Pooled feature matrix.
    y : ndarray of shape (n_ref + n_dat,)
        Binary labels:
        0 for reference, 1 for data.
    """
    x_ref = np.asarray(x_ref)
    x_dat = np.asarray(x_dat)

    n_ref = x_ref.shape[0]
    n_dat = x_dat.shape[0]

    X = np.empty((n_ref + n_dat, 1), dtype=dtype)
    y = np.empty(n_ref + n_dat, dtype=dtype)

    X[:n_ref, 0] = x_ref
    X[n_ref:, 0] = x_dat

    y[:n_ref] = 0.0
    y[n_ref:] = 1.0

    return X, y