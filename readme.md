# Kernel NPLM

Kernel NPLM is a compact Python implementation of the New Physics Learning Machine
(NPLM) statistic using kernel logistic classification through Falkon.

The repository now separates the core model from statistical-test wrappers:

- `nplm/` contains the core LogisticFalkon NPLM implementation and plotting helpers.
- `stat_tests/` contains NPLM-based permutation and reference-resampling tests.
- `data/` contains small data-generation, preprocessing, and pooling utilities.
- `local-data/` is a git-ignored place for local datasets and large inputs.
- `examples/` contains runnable examples.

## Repository Layout

```text
kernel-nplm/
├── data/
│   ├── datasets.py                 # Build pooled reference/data samples and labels
│   ├── preprocessing.py            # Standardization helpers
│   ├── sampling.py                 # Null, mixture, and full-alternative pool samplers
│   └── synthetic.py                # 1D toy generators
├── local-data/                     # Local datasets; contents are ignored by Git
├── examples/
│   ├── gaussian_1d_generator.py    # Generator-based 1D null/alternative toys
│   └── event_weighted_loss.py      # Event-weighted loss checks
├── nplm/
│   ├── logfalkon_nplm.py           # Core LogFalkonNPLM class
│   ├── event_weighted_cross_entropy.py
│   └── plotting.py
├── stat_tests/
│   ├── permutation.py              # nplm_permutation_test
│   ├── resampling.py               # Separate ensembles and combined convenience test
│   └── comparison.py               # Empirical and checked chi-square calibration
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
- An exception or nonfinite statistic stops the test and identifies the observed
  fit or permutation number and model seed. No p-value is returned for an
  incomplete run. Finite negative statistics remain valid for empirical calibration.

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

### Separate reconstruction and comparison

For mixtures or several alternatives sharing one null, reconstruct ensembles
independently and compare afterwards. These are parts of the same reference
resampling method. The combined `nplm_resampling_test` remains available with its
existing signature, sampling modes, result fields, flags and warnings. The
permutation and hyperparameter-scan public interfaces are also retained.

```python
from data import make_null_sampler, make_mixture_sampler, make_alternative_sampler
from stat_tests import (
    nplm_resampling_null, nplm_resampling_alternative, compare_nplm_results,
)

# Pools are already in the same feature coordinates / preprocessing convention.
# component_pool contains only the added component.
B, S, n_ref = 2000, 10, 20000
config = dict(config, NR=B)
null_sampler = make_null_sampler(reference_pool, n_ref=n_ref, expected_data=B)
mixture_sampler = make_mixture_sampler(
    reference_pool, component_pool, n_ref=n_ref,
    expected_background=B, expected_component=S,
)
null = nplm_resampling_null(null_sampler, config, n_null=300, seed=123)
alternative = nplm_resampling_alternative(
    mixture_sampler, config, n_alternative=100, seed=456,
)
comparison = compare_nplm_results(null, {"mixture": alternative}, fit_chi2=True, seed=0)
print(comparison.alternatives["mixture"].z_scores)  # at t quantiles 0.16, 0.50, 0.84
print(comparison.chi2_fit)

# A full-alternative pool already includes its background contribution.
full_sampler = make_alternative_sampler(
    reference_pool, full_alternative_pool, n_ref=n_ref, expected_data=expected_total,
)
full = nplm_resampling_alternative(full_sampler, config, seed=789)
comparison = compare_nplm_results(null, {"mixture": alternative, "full": full})
```

The two reconstruction functions and the combined wrapper share one toy loop:
sample, prepare pooled inputs, set realized sizes, fit NPLM, validate, and collect.
The combined wrapper supplies its existing RNG and model seeds to preserve its
seeded sampling order.

Both reconstruction functions default to 100 toys, `seed=0`,
`resample_nystrom=True`, and `dtype=np.float64`. Each calls its sampler once per
toy, then fits `LogFalkonNPLM` and collects an `NPLMResamplingEnsemble` with:

- `statistics`, `reference_counts`, and `data_counts` arrays;
- `metadata`: one copied dictionary per toy;
- `seed` (the ensemble master seed), `model_seeds`, `model_config`,
  `resample_nystrom`, and `dtype` for provenance.

`NR` stays fixed at the expected null yield, including for alternatives with a
different total yield. Each fit sets `N_R` and `N_D` from the realized arrays,
overriding those two entries in the supplied configuration. Pools and the
caller's configuration are not modified.

The sampler factories return callbacks with the contract
`x_ref, x_data, metadata = sampler(rng)`:

| Factory | Counts and event selection |
| --- | --- |
| `make_null_sampler(reference_pool, n_ref=..., expected_data=...)` | Poisson data count; one draw of reference + data rows without replacement, then split. |
| `make_mixture_sampler(reference_pool, component_pool, n_ref=..., expected_background=..., expected_component=...)` | Independent Poisson background and component counts; jointly draw reference/background and split, then append component rows sampled without replacement. |
| `make_alternative_sampler(reference_pool, alternative_pool, n_ref=..., expected_data=...)` | Reference from its pool; all data from a separate full-alternative pool at the expected total count. |

Reference size is fixed. Indices never repeat within a pool draw; rows can be
reused across toys. Separate pools must represent separate event collections;
the code does not infer event identity from matching feature values. Factories
check feature-column compatibility and capacity for each realized draw.
An oversized draw raises an error instead of truncating or redrawing it.
Full-alternative metadata records the total without assigning background or
component counts. Mixture metadata includes both counts and their expectations.

Poisson counts are nonnegative, including zero. A zero injected component is
valid. An entirely empty pseudo-dataset cannot be fitted and raises an error
identifying the toy (numbered from one) and model seed. This also corrects the combined
wrapper's former clamp of zero Poisson counts to one. Failed and nonfinite fits
abort the ensemble; they are never silently dropped or retried.

For fixed counts, synthetic generators, or file-backed sampling, supply a custom
callback with the same contract. Preprocess the pools first or apply a previously
chosen transformation inside the callback. The fitting loop neither chooses nor
fits preprocessing. Dataset-specific loading and normalization belong in the
calling experiment; see the DIMUON pilot example below. EFT rate normalization
and benchmark tuning remain separate physics inputs.

### Calibration and reuse

`compare_nplm_results` accepts a null ensemble and a mapping of names to
alternative ensembles. It performs no event sampling or NPLM fitting. It always
returns empirical right-tail p-values `(1 + n_extreme) / (n_null + 1)` and
Gaussian significances `norm.isf(p)` at alternative statistic quantiles
0.16, 0.50, and 0.84. This interval is the **spread across toys**, not uncertainty
on the median. `p_value_resolution` is `1 / (n_null + 1)` and
`z_score_resolution` is the corresponding largest empirical Z. A toy sample of
100 or 300 null statistics cannot resolve 3 sigma empirically. No clipping is
applied: p=1 gives Z=-infinity.

Comparison requires only NumPy and SciPy: importing `compare_nplm_results` and
the result containers does not load Torch or Falkon. You can fit on a remote
machine and compare saved statistics locally. Empirical p-values use the same
numerical helper in permutation, resampling, comparison, and plotting.

With `fit_chi2=True` (default false), the null mean estimates the chi-square
degrees of freedom. A parametric-bootstrap KS test uses 999 replicates and
re-estimates the degrees of freedom in every replicate. Its corrected p-value
is compared with 0.05. One fit/check is reused for all alternatives in that
comparison. `chi2_fit` contains the degrees of freedom, KS statistic, bootstrap
p-value, acceptance flag, reason, replicate count and threshold. Invalid or
rejected fits leave `chi2_p_values` and `chi2_z_scores` as `None`, while empirical
results remain available. Acceptance is a compatibility check, not a guarantee
of accurate far-tail extrapolation. `seed=0` controls bootstrap randomness.

Reuse a null only with compatible model settings, preprocessing, expected null
yield, reference sampling, and randomness policy. The caller checks compatibility;
the comparison cannot verify the physical meaning of custom samplers. With
`resample_nystrom=False`, one model seed is repeated within an ensemble. Different
master seeds generally give different fixed model seeds, so separately generated
ensembles are not automatically calibrated for the same fixed-seed experiment.
The same master seed aligns fixed model seeds but also couples the sampling
streams. Prefer fresh model seeds per toy for independent ensembles. A fixed
seed does not freeze center coordinates when rows change.
Use distinct master seeds for independent ensembles under the default policy.

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

The 1D example uses the separate reconstruction/comparison API with generator
callbacks. Its CLI and existing saved `.npz` fields are retained, with realized
reference/data counts and `provenance_json` added. The latter records each
ensemble's master seed, model configuration, dtype, randomness policy and toy
metadata. Per-model seeds retain their existing saved fields. Saving is handled
by the example, not the statistical functions. The optional plot shows a
chi-square curve only when the compatibility check passes.

Run the fast sampling/calibration/regression tests (NumPy, SciPy and Torch;
model fits are mocked and Falkon is optional):

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

On a Falkon-capable machine, run the real-model check of one null reused for
zero and nonzero signal, including saved provenance:

```bash
python -m unittest discover -s tests -p 'integration_*.py' -v
python examples/gaussian_1d_generator.py --cpu --n-reference 400 \
    --expected-background 80 --expected-signal 10 --n-null 4 --n-alt 4 \
    --nystrom-centers 20 --penalty 1e-5 --iterations 1000 --no-plot
```

Run the event-weighted loss checks:

```bash
python3 examples/event_weighted_loss.py
```

### DIMUON pilot experiments

[`examples/dimuon_resampling.py`](examples/dimuon_resampling.py) keeps DIMUON
loading, preprocessing and plotting in one file and uses the existing resampling
API. It reads the combined `DiLepton_*.h5` files in `local-data/DIMUON` (override
with `--data-dir`). HDF5 loading requires `h5py`; fitting requires Falkon and Torch,
and plotting requires Matplotlib. First check the actual inputs without fitting:

```bash
python examples/dimuon_resampling.py --validate-only
```

Run a small GPU pilot with the SM null, Z' 300 GeV with expected signal yield 40,
and Z' 600 GeV with expected signal yield 15:

```bash
python examples/dimuon_resampling.py --gpu --n-null 5 --n-alt 5 \
    --output-dir results_dimuon_pilot
```

The reference contains 100,000 events and the expected SM yield is 20,000 by
default. Reference/background events are jointly sampled without replacement
and then split; background and injected signal counts fluctuate independently
with Poisson distributions. `NR` stays at the expected SM yield. The default
1,000 Nyström centers are a reduced pilot setting: use `--nystrom-centers` to
change it. Such small toy counts are execution checks, not a significance study
or a validated reproduction of the paper's working point.

The solver defaults are `iter=[1_000_000]` and `cg_tol=np.sqrt(1e-7)`;
override them with `--iterations` and `--cg-tol` if needed.

Inputs are ordered as `pt1, pt2, eta1, eta2, delta_phi`; `mll` is excluded. Each
toy pools its reference and pseudo-data before applying the
[legacy normalization](https://github.com/FalkonHEP/falkonhep/blob/main/falkonhep/utils/data_utils.py):
columns containing negatives are standardized, other columns exceeding one are
divided by their mean, and columns already in [0, 1] are unchanged. This rule is
recomputed per toy, including injected events. The generic preprocessing helper
is unchanged. **No additional mass cut is applied by default**; select one with
`--mll-min` if required. All expected yields must correspond to the chosen
selection. The combined SM file contains events below 100 GeV, so matching the
paper's event selection and rates still needs an explicit physics choice.

Generate the null separately with `--cases null`. Reuse a saved null while
changing signal yields or running additional alternatives:

```bash
python examples/dimuon_resampling.py --gpu --cases zprime600 \
    --zprime600-yield 15 --n-alt 100 \
    --null-results results_dimuon_pilot/null.npz \
    --output-dir results_dimuon_600
```

Reuse requires the same model settings, reference size, selection, preprocessing
and SM file identity (absolute path, size and modification time). `--n-null` is
ignored when loading a null. Each case has a stable, separate seed derived from
`--seed`, so changing the case list does not change its draws. Repeating the
same case/seed/toy count reproduces a run; use a new seed for additional runs.

EFT files are sampled as full alternatives, not as added signal pools. Select
`eft06`, `eft06_2` or `eft06_5` with `--cases` and supply a positive numerical
`--eft-yield CASE=TOTAL` for each. `TOTAL` is the expected full pseudo-data yield
after selection. The script does not infer a Wilson coefficient, EFT rate or
signal yield from a filename; those inputs must be established separately.

Each completed ensemble is saved immediately as `null.npz` or `CASE.npz`, with
statistics, realized counts, model seeds and JSON provenance (configuration,
pool information, sampler counts and per-toy normalization). A reused null is
also copied into the output directory. `comparison.npz` stores quantile
calibration and chi-square diagnostics; `dimuon_distributions.png` and `.pdf`
overlay the distributions. Use separate output directories to retain runs.
The comparison fits the null once and only reports chi-square significances
when its bootstrap compatibility check passes. Empirical results are always
saved; the 16–84% interval describes the spread across toys. `--no-chi2` and
`--no-plot` disable the optional fit/check and plot respectively.

### Direct DIMUON baseline

[`examples/dimuon_simple.py`](examples/dimuon_simple.py) is a standalone version
of the supplied baseline: one loop samples events, normalizes them, constructs
`LogisticFalkon`, calls `fit`/`predict`, and computes the full statistic with
NumPy. It imports no repository model, sampling or statistical-test modules.
The defaults are M=20,000, sigma=3, penalty=1e-6, iter=[1,000,000],
cg_tol=sqrt(1e-7), 100,000 reference events and expected background 20,000.
It uses the legacy `higgs` normalization and no additional cuts. Counts are
Poisson; reference and background are sampled jointly and split without overlap.

```bash
python examples/dimuon_simple.py --gpu --case null --n-toys 50 \
    --output-dir results_dimuon_simple_null
```

For the discrepancy investigation, replay only the first three of the saved
null toys, with their saved sampling settings, model configuration and seeds:

```bash
python examples/dimuon_simple.py --replay-null results_dimuon_null/null.npz \
    --n-toys 3 --output-dir results_dimuon_simple_replay
```

Replay consumes the original complete model-seed sequence before drawing events,
so requesting fewer toys still reproduces the original inputs. It checks their
counts and normalization parameters. Explicit model/device options can override
saved settings for controlled comparisons; sampling sizes and seed remain fixed.
Use the same SM file, with its original row order. Add `--validate-only` to check
sampling/preprocessing without Torch or Falkon and without writing outputs.

Use `--case zprime300` or `--case zprime600` for independently fluctuating SM
and signal mixtures, with expected signal yields 40 and 15 respectively;
`--signal-yield` overrides these. `--normalization scaler` or `none` permits
explicit preprocessing comparisons. This script does not yet cover EFT.
Fresh runs default to `--seed 0`; use distinct seeds for independent ensembles.

`toys.csv` records each completed toy immediately, including both contributions
to t, training time, and saved t/difference during replay. `config.json` records
the run settings; `results.npz` contains statistics, counts, model seeds and
normalization parameters. `distribution.pdf` shows the raw histogram (disable
with `--no-plot`). Calibration is intentionally outside this diagnostic script.
Use a separate output directory per run; a new run replaces that directory's
result files. Failed fits abort and preserve the CSV rows already written.

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
