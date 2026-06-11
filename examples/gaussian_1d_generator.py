"""Run a generator-based one-dimensional Gaussian NPLM example."""

import argparse
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

#########################################################################################################
# Import path setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_pooled_sample, make_data_sample_poisson, sample_ref_exp
from nplm import LogFalkonNPLM
from nplm.plotting import emp_pvalues_mc, plot_nplm_distributions, z_from_p


#########################################################################################################
# Constants and result containers

MAX_MODEL_SEED = 2**32 - 1


ToyEnsemble = namedtuple(
    "ToyEnsemble",
    ["statistics", "background_counts", "signal_counts", "model_seeds"],
)
ToyEnsemble.__doc__ = "Collection of NPLM toy statistics and metadata."


RunSummary = namedtuple(
    "RunSummary",
    [
        "seed",
        "n_reference",
        "expected_background",
        "expected_signal",
        "n_null",
        "n_alt",
        "sigma",
        "nystrom_centers",
        "penalty",
        "iterations",
        "cg_tol",
        "cpu",
        "keops",
    ],
)
RunSummary.__doc__ = "Serializable summary of a Gaussian generator run."


#########################################################################################################
# Model and toy helpers

def make_model_config(args):
    """Build a LogFalkonNPLM configuration from parsed CLI arguments.

    :param args: Parsed command-line arguments.
    :returns: NPLM model configuration dictionary.
    """
    return {
        "sigma": float(args.sigma),
        "NR": float(args.expected_background),
        "N_R": int(args.n_reference),
        "M": int(args.nystrom_centers),
        "lambda": [float(args.penalty)],
        "iter": [int(args.iterations)],
        "cg_tol": float(args.cg_tol),
        "cpu": bool(args.cpu),
        "keops": args.keops,
        "verbose": int(args.model_verbose),
    }


def compute_statistic(*, x_reference, x_data, base_model_config, model_seed):
    """Compute one NPLM statistic for a generated reference/data pair.

    :param x_reference: Reference sample with shape ``(n_reference, 1)``.
    :param x_data: Data sample with shape ``(n_data, 1)``.
    :param base_model_config: Model configuration without the per-toy seed.
    :param model_seed: Per-toy model seed.
    :returns: Scalar NPLM statistic.
    """
    x_pooled, y = build_pooled_sample(x_reference, x_data)
    model_config = dict(base_model_config)
    model_config["seed"] = int(model_seed)

    nplm = LogFalkonNPLM(model_config)
    return float(nplm.compute_statistic(x_pooled, y))


def run_toy_ensemble(
    *,
    label,
    n_toys,
    n_reference,
    data_sampler,
    base_model_config,
    rng,
    progress_every,
):
    """Run a set of generator toys and collect their statistics.

    :param label: Progress label printed for this ensemble.
    :param n_toys: Number of toys to generate.
    :param n_reference: Reference sample size per toy.
    :param data_sampler: Callable returning ``(x_data, n_background, n_signal)``.
    :param base_model_config: Model configuration without the per-toy seed.
    :param rng: NumPy random generator.
    :param progress_every: Print progress every this many toys; ``0`` disables progress.
    :returns: ``ToyEnsemble`` with statistic arrays and toy metadata.
    """
    statistics = np.empty(n_toys, dtype=np.float64)
    background_counts = np.empty(n_toys, dtype=np.int64)
    signal_counts = np.empty(n_toys, dtype=np.int64)
    model_seeds = np.empty(n_toys, dtype=np.int64)

    for toy_idx in range(n_toys):
        if should_print_progress(toy_idx, n_toys, progress_every):
            print(f"[{label}] toy {toy_idx + 1}/{n_toys}")

        x_reference = sample_ref_exp(n_reference, rng=rng)
        x_data, n_background, n_signal = data_sampler(rng)
        model_seed = int(rng.integers(0, MAX_MODEL_SEED, dtype=np.uint32))

        statistics[toy_idx] = compute_statistic(
            x_reference=x_reference,
            x_data=x_data,
            base_model_config=base_model_config,
            model_seed=model_seed,
        )
        background_counts[toy_idx] = n_background
        signal_counts[toy_idx] = n_signal
        model_seeds[toy_idx] = model_seed

    return ToyEnsemble(
        statistics=statistics,
        background_counts=background_counts,
        signal_counts=signal_counts,
        model_seeds=model_seeds,
    )


def make_data_sampler(*, expected_background, expected_signal):
    """Create a Poisson data sampler for the configured signal strength.

    :param expected_background: Expected background count.
    :param expected_signal: Expected signal count.
    :returns: Callable accepting an RNG and returning sampled data metadata.
    """
    def sample_data(rng):
        """Sample one data toy from the configured generator.

        :param rng: NumPy random generator.
        :returns: Tuple ``(x_data, n_background, n_signal)``.
        """
        x_data, _, _, n_background, n_signal = make_data_sample_poisson(
            NR=expected_background,
            NS=expected_signal,
            rng=rng,
        )
        return x_data, n_background, n_signal

    return sample_data


def should_print_progress(toy_idx, n_toys, progress_every):
    """Return whether progress should be printed for the current toy.

    :param toy_idx: Zero-based toy index.
    :param n_toys: Total number of toys.
    :param progress_every: Print frequency; ``0`` disables progress.
    :returns: Boolean progress-print decision.
    """
    if progress_every <= 0:
        return False
    return toy_idx == 0 or toy_idx + 1 == n_toys or (toy_idx + 1) % progress_every == 0


#########################################################################################################
# Output helpers

def save_results(*, output_dir, summary, null_result, alt_result):
    """Save toy outputs and run summary to a compressed NumPy file.

    :param output_dir: Directory where results are written.
    :param summary: ``RunSummary`` object.
    :param null_result: Null ``ToyEnsemble``.
    :param alt_result: Alternative ``ToyEnsemble``.
    :returns: Path to the saved ``.npz`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gaussian_1d_generator_results.npz"

    np.savez(
        output_path,
        t_null=null_result.statistics,
        t_alt=alt_result.statistics,
        null_background_counts=null_result.background_counts,
        null_signal_counts=null_result.signal_counts,
        alt_background_counts=alt_result.background_counts,
        alt_signal_counts=alt_result.signal_counts,
        null_model_seeds=null_result.model_seeds,
        alt_model_seeds=alt_result.model_seeds,
        summary_json=json.dumps(summary._asdict(), sort_keys=True),
    )
    return output_path


def alternative_quantile_z_scores(null_statistics, alt_statistics):
    """Compute empirical Z-scores at alternative statistic quantiles.

    :param null_statistics: Null statistics with shape ``(n_null,)``.
    :param alt_statistics: Alternative statistics with shape ``(n_alt,)``.
    :returns: Pair ``(quantile_levels, z_scores)`` with shape ``(3,)``.
    """
    quantile_levels = np.array([0.16, 0.50, 0.84], dtype=np.float64)
    alt_t_quantiles = np.quantile(alt_statistics, quantile_levels)
    p_values = emp_pvalues_mc(null_statistics, alt_t_quantiles)
    return quantile_levels, z_from_p(p_values)


def print_summary(null_result, alt_result):
    """Print a compact summary of null and alternative toy statistics.

    :param null_result: Null ``ToyEnsemble``.
    :param alt_result: Alternative ``ToyEnsemble``.
    :returns: ``None``.
    """
    print("\nGenerator-toy NPLM results")
    print("--------------------------")
    print(f"null toys: {len(null_result.statistics)}")
    print(f"alt toys:  {len(alt_result.statistics)}")

    _, z_scores = alternative_quantile_z_scores(
        null_result.statistics,
        alt_result.statistics,
    )
    z16, z50, z84 = z_scores
    print(
        "alt empirical Z at t quantiles [16%, 50%, 84%]: "
        f"[{z16:.3f}, {z50:.3f}, {z84:.3f}]"
    )
    print(f"median empirical Z: {z50:.3f} (-{z50 - z16:.3f}/+{z84 - z50:.3f})")


def maybe_plot(*, output_dir, null_result, alt_result, make_plot):
    """Optionally plot the generated null and alternative distributions.

    :param output_dir: Directory where the plot is written.
    :param null_result: Null ``ToyEnsemble``.
    :param alt_result: Alternative ``ToyEnsemble``.
    :param make_plot: If true, save and display the plot.
    :returns: ``None``.
    """
    if not make_plot:
        return

    plot_nplm_distributions(
        t_null=null_result.statistics,
        t_alt=alt_result.statistics,
        save_path=output_dir / "gaussian_1d_generator_distributions.png",
    )


#########################################################################################################
# Command-line parsing

def parse_args():
    """Parse command-line arguments for the Gaussian generator example.

    :returns: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generator-based 1D NPLM example. Use this when H0 and H1 "
            "pseudo-experiments can be sampled directly from known distributions."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results_gaussian_1d"))

    parser.add_argument("--n-reference", type=positive_int, default=200_000)
    parser.add_argument("--expected-background", type=positive_float, default=2_000)
    parser.add_argument("--expected-signal", type=nonnegative_float, default=10)
    parser.add_argument("--n-null", type=positive_int, default=10)
    parser.add_argument("--n-alt", type=positive_int, default=10)

    parser.add_argument("--sigma", type=positive_float, default=0.3)
    parser.add_argument("--nystrom-centers", type=positive_int, default=3_000)
    parser.add_argument("--penalty", type=positive_float, default=1e-10)
    parser.add_argument("--iterations", type=positive_int, default=1_000_000)
    parser.add_argument("--cg-tol", type=positive_float, default=np.sqrt(1e-7))
    parser.add_argument("--keops", choices=("yes", "no"), default="no")
    parser.add_argument("--model-verbose", type=int, default=0)

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", dest="cpu", action="store_true")
    device_group.add_argument("--gpu", dest="cpu", action="store_false")
    parser.set_defaults(cpu=True)

    parser.add_argument("--progress-every", type=nonnegative_int, default=1)
    parser.add_argument("--no-plot", action="store_true")

    return parser.parse_args()


def positive_int(value):
    """Parse a positive integer argument.

    :param value: Raw argument value.
    :returns: Positive integer.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value):
    """Parse a non-negative integer argument.

    :param value: Raw argument value.
    :returns: Non-negative integer.
    """
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value):
    """Parse a positive finite float argument.

    :param value: Raw argument value.
    :returns: Positive finite float.
    """
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite float")
    return parsed


def nonnegative_float(value):
    """Parse a non-negative finite float argument.

    :param value: Raw argument value.
    :returns: Non-negative finite float.
    """
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite float")
    return parsed


def build_summary(args):
    """Build a serializable run summary from parsed CLI arguments.

    :param args: Parsed command-line arguments.
    :returns: ``RunSummary`` object.
    """
    return RunSummary(
        seed=int(args.seed),
        n_reference=int(args.n_reference),
        expected_background=float(args.expected_background),
        expected_signal=float(args.expected_signal),
        n_null=int(args.n_null),
        n_alt=int(args.n_alt),
        sigma=float(args.sigma),
        nystrom_centers=int(args.nystrom_centers),
        penalty=float(args.penalty),
        iterations=int(args.iterations),
        cg_tol=float(args.cg_tol),
        cpu=bool(args.cpu),
        keops=args.keops,
    )


#########################################################################################################
# Entrypoint

def main():
    """Run the generator example from command-line arguments.

    :returns: ``None``.
    """
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base_model_config = make_model_config(args)

    null_result = run_toy_ensemble(
        label="null",
        n_toys=args.n_null,
        n_reference=args.n_reference,
        data_sampler=make_data_sampler(
            expected_background=args.expected_background,
            expected_signal=0.0,
        ),
        base_model_config=base_model_config,
        rng=rng,
        progress_every=args.progress_every,
    )

    alt_result = run_toy_ensemble(
        label="alt",
        n_toys=args.n_alt,
        n_reference=args.n_reference,
        data_sampler=make_data_sampler(
            expected_background=args.expected_background,
            expected_signal=args.expected_signal,
        ),
        base_model_config=base_model_config,
        rng=rng,
        progress_every=args.progress_every,
    )

    summary = build_summary(args)
    output_path = save_results(
        output_dir=args.output_dir,
        summary=summary,
        null_result=null_result,
        alt_result=alt_result,
    )

    print_summary(null_result, alt_result)
    print(f"saved arrays: {output_path}")

    maybe_plot(
        output_dir=args.output_dir,
        null_result=null_result,
        alt_result=alt_result,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
