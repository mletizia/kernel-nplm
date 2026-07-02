# Kernel NPLM

Kernel NPLM is a compact Python implementation of the New Physics Learning Machine
(NPLM) statistic using kernel logistic classification through Falkon.

The repository now separates the core model from statistical-test wrappers:

- `nplm/` contains the core LogisticFalkon NPLM implementation and plotting helpers.
- `stat_tests/` contains NPLM-based permutation and reference-resampling tests.
- `data/` contains small data-generation, preprocessing, and pooling utilities.
- `examples/` contains runnable examples.

## Repository Layout

```text
kernel-nplm/
├── data/
│   ├── datasets.py                 # Build pooled reference/data samples and labels
│   ├── preprocessing.py            # Standardization helpers
│   └── synthetic.py                # 1D toy generators
├── examples/
│   ├── gaussian_1d_generator.py    # Generator-based 1D null/alternative toys
│   ├── 1DGaussian.py               # Legacy 1D Gaussian example
│   └── event_weighted_loss.py      # Event-weighted loss checks
├── nplm/
│   ├── logfalkon_nplm.py           # Core LogFalkonNPLM class
│   ├── event_weighted_cross_entropy.py
│   └── plotting.py
├── stat_tests/
│   ├── permutation.py              # nplm_permutation_test
│   └── resampling.py               # nplm_resampling_test
├── papers/
└── readme.md
```

## Requirements

The code expects a Python environment with:

- `numpy`
- `scipy`
- `torch`
- `falkon`
- `matplotlib`, for plotting utilities and examples

The code is written with Python 3.9-compatible syntax. The intended runtime is
Python 3.11 or newer.

## Core Statistic

Use `LogFalkonNPLM` when you already have a reference sample, a data sample, and
want the raw NPLM statistic for one split.

```python
import numpy as np

from data import build_pooled_sample
from nplm import LogFalkonNPLM

x_ref = np.random.normal(0.0, 1.0, size=(1000, 1))
x_data = np.random.normal(0.2, 1.0, size=(200, 1))
x, y = build_pooled_sample(x_ref, x_data)

sigma = LogFalkonNPLM.estimate_sigma_median(x, seed=123)
config = {
    "sigma": sigma,
    "NR": len(x_data),
    "M": "sqrt",
    "lambda": [1e-6],
    "iter": [1000],
    "cpu": True,
    "keops": "no",
    "verbose": 0,
    "seed": 123,
}

nplm = LogFalkonNPLM(config)
t_obs = nplm.compute_statistic(x, y)
```

`NR` is required. It represents the expected data count under the null
hypothesis and is used to set the reference-event weight.

## Permutation Test

Use `nplm_permutation_test` when the two samples are exchangeable under the
null. The test takes two samples and returns a finite-sample corrected p-value,
a Gaussian Z-score, and the observed statistic.

```python
from stat_tests import nplm_permutation_test

result = nplm_permutation_test(
    x_ref,
    x_data,
    config,
    n_permutations=100,
    seed=123,
    return_null=True,
)

print(result.p_value, result.z_score, result.t_obs)
```

Notes:

- `model_config` must specify `NR` and `sigma`.
- The empirical null excludes the initial observed split.
- By default, Nystrom centers are resampled at every fit through fresh model
  seeds. Set `resample_nystrom=False` to freeze the model seed across fits.

## Reference-Resampling Test

Use `nplm_resampling_test` when you have a large reference pool and want to
calibrate the NPLM statistic using pseudo-experiments drawn from that reference.

```python
from stat_tests import nplm_resampling_test

result = nplm_resampling_test(
    x_ref,
    x_data,
    config,
    n_ref=1000,
    n_data=200,
    n_null=100,
    null_sampling="disjoint",
    seed=123,
    return_null=True,
)
```

Null sampling modes:

- `disjoint` draws `n_ref + n_data` events from `x_ref` without replacement,
  then splits them into pseudo-reference and pseudo-data samples. This is the
  default.
- `independent` draws pseudo-reference and pseudo-data independently with
  replacement from `x_ref`.

If `n_data == len(x_data)`, the function tests the full data sample once and
returns the corresponding p-value and Z-score. If `n_data < len(x_data)`, it
estimates an alternative ensemble by repeatedly sampling from `x_ref` and
`x_data`, then reports the 0.16, 0.50, and 0.84 quantiles of the alternative
test statistic and their empirical p-values/Z-scores against the null.

The resampling test warns when the available sample pool is less than a factor
of 10 larger than the requested sampled size.

## Hyperparameter Tuning

Use `nplm_resampling_hyperparameter_scan` to inspect how the null test statistic
changes across a grid of Falkon penalties and Nystrom-center counts. This
workflow is built on the same reference-resampling method as
`nplm_resampling_test`: for each `(lambda, M)` point it draws null toys from the
reference pool and computes `n_trials` NPLM statistics. It does not select a
working point automatically.

```python
from tuning import nplm_resampling_hyperparameter_scan, save_scan_plots

config = {
    "sigma": sigma,          # fixed before the scan
    "NR": n_data,
    "iter": [100_000],
    "cg_tol": 3.16e-4,
    "cpu": False,
    "keops": "yes",
    "verbose": 0,
}

scan = nplm_resampling_hyperparameter_scan(
    x_ref,
    config,
    lambda_values=[1e-10, 1e-9, 1e-8],
    m_values=[500, 1000, 2000],
    n_ref=20_000,
    n_data=5_000,
    n_trials=10,
    poisson_fluctuate_n_data=True,
    seed=123,
    progress=True,
)

paths = save_scan_plots(scan, "results_tuning")
print(paths)
```

The saved plots show the average reference-null test statistic versus `M` for
each `lambda`, a grid heatmap, and the average training time per toy. Inspect
these plots to choose the working point yourself. The underlying paper fixes
`sigma` from reference-distance scales, scans `M` for stability versus cost, and
takes `lambda` as small as possible while keeping training numerically stable.
When `poisson_fluctuate_n_data=True`, each resampled null toy draws its realized
pseudo-data count from `Poisson(n_data)` while `NR` stays fixed as the expected
count used in the NPLM reference-event weight.

## Examples

Run the generator-based one-dimensional example:

```bash
python3 examples/gaussian_1d_generator.py --cpu --n-null 10 --n-alt 10
```

This script samples both null and alternative pseudo-experiments directly from
known data-generating distributions, computes NPLM statistics, saves arrays, and
optionally plots the resulting distributions.

Run the legacy one-dimensional example:

```bash
python3 examples/1DGaussian.py
```

Run the event-weighted loss checks:

```bash
python3 examples/event_weighted_loss.py
```

## Reproducibility

The wrappers in `stat_tests/` use local NumPy generators for sampling and store
the model seeds used for each fit in the returned result objects. They also
restore NumPy and Torch global RNG state after running, so callers do not
inherit hidden RNG side effects from a test run.

For deterministic runs, pass an explicit `seed` and keep the same model
configuration. For more robust calibration, leave `resample_nystrom=True`, which
is the default.

## References

Main reference:

- [Learning New Physics Efficiently with Nonparametric Methods](https://arxiv.org/abs/2204.02317)

Falkon:

- https://github.com/FalkonML/falkon

## Author

Marco Letizia
