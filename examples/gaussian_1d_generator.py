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

from data import make_data_sample_poisson, sample_ref_exp
from stat_tests import (
    nplm_resampling_null, nplm_resampling_alternative, compare_nplm_results,
)


#########################################################################################################
# Constants and result containers

MAX_MODEL_SEED = 2**32 - 1


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


def make_sampler(*, n_reference, expected_background, expected_signal,
                 label="toy", n_toys=100, progress_every=0):
    """Return a generator callback with the same contract as the pool samplers.

    Reference events are generated independently. Metadata records the actual
    background and signal counts; no preprocessing is fitted inside NPLM.
    """
    toy_idx = 0

    def sample(rng):
        nonlocal toy_idx
        if should_print_progress(toy_idx, n_toys, progress_every):
            print(f"[{label}] toy {toy_idx + 1}/{n_toys}")
        toy_idx += 1
        x_reference = sample_ref_exp(n_reference, rng=rng)
        x_data, _, _, n_background, n_signal = make_data_sample_poisson(
            NR=expected_background, NS=expected_signal, rng=rng,
        )
        return x_reference, x_data, {
            "sampling": "generator", "n_background": n_background,
            "n_component": n_signal, "expected_background": expected_background,
            "expected_component": expected_signal,
        }

    return sample


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
    """Save toy outputs and run summary to a NumPy file.

    :param output_dir: Directory where results are written.
    :param summary: ``RunSummary`` object.
    :param null_result: Null ``NPLMResamplingEnsemble``.
    :param alt_result: Alternative ``NPLMResamplingEnsemble``.
    :returns: Path to the saved ``.npz`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gaussian_1d_generator_results.npz"

    np.savez(
        output_path,
        t_null=null_result.statistics,
        t_alt=alt_result.statistics,
        null_background_counts=np.array([m["n_background"] for m in null_result.metadata], dtype=np.int64),
        null_signal_counts=np.array([m["n_component"] for m in null_result.metadata], dtype=np.int64),
        alt_background_counts=np.array([m["n_background"] for m in alt_result.metadata], dtype=np.int64),
        alt_signal_counts=np.array([m["n_component"] for m in alt_result.metadata], dtype=np.int64),
        null_model_seeds=null_result.model_seeds,
        alt_model_seeds=alt_result.model_seeds,
        summary_json=json.dumps(summary._asdict(), sort_keys=True),
        null_reference_counts=null_result.reference_counts,
        null_data_counts=null_result.data_counts,
        alt_reference_counts=alt_result.reference_counts,
        alt_data_counts=alt_result.data_counts,
        provenance_json=json.dumps({
            label: {
                "seed": result.seed, "model_config": result.model_config,
                "resample_nystrom": result.resample_nystrom, "dtype": result.dtype,
                "metadata": result.metadata,
            }
            for label, result in (("null", null_result), ("alternative", alt_result))
        }, sort_keys=True),
    )
    return output_path


def print_summary(null_result, alt_result, comparison):
    """Print empirical quantiles and optional chi-square compatibility diagnostics."""
    print(f"\nnull toys: {len(null_result.statistics)}")
    print(f"alt toys:  {len(alt_result.statistics)}")
    result = comparison.alternatives["alternative"]
    print("alt empirical Z at t quantiles [16%, 50%, 84%]: "
          + np.array2string(result.z_scores, precision=3))
    print("The 16–84% interval is the spread across toys, not uncertainty on the median.")
    print(f"empirical p-value resolution: {comparison.p_value_resolution:.4g}")
    if comparison.chi2_fit is not None:
        print(f"chi-square check: {comparison.chi2_fit.reason}; "
              f"bootstrap p={comparison.chi2_fit.p_value}")
    if result.chi2_z_scores is not None:
        print("alt chi-square Z at the same t quantiles: "
              + np.array2string(result.chi2_z_scores, precision=3))


def maybe_plot(*, output_dir, null_result, alt_result, comparison, make_plot):
    """Plot toy distributions and a chi-square curve only if the check passed."""
    if not make_plot:
        return
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.histogram_bin_edges(
        np.concatenate((null_result.statistics, alt_result.statistics)), bins=30
    )
    ax.hist(null_result.statistics, bins=edges, density=True, alpha=0.5, label="Null")
    ax.hist(alt_result.statistics, bins=edges, density=True, alpha=0.5, label="Alternative")
    fit = comparison.chi2_fit
    if fit is not None and fit.accepted:
        grid = np.linspace(max(0.0, edges[0]), edges[-1], 1000)
        ax.plot(grid, chi2.pdf(grid, df=fit.dof), label=f"chi-square (df={fit.dof:.2f})")
    result = comparison.alternatives["alternative"]
    for level, value in zip(result.quantile_levels, result.t_quantiles):
        ax.axvline(value, linestyle="--", linewidth=1, label=f"Alternative {level:.0%}")
    ax.set(xlabel="NPLM statistic", ylabel="Density",
           title=f"Median empirical Z = {result.z_scores[1]:.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / "gaussian_1d_generator_distributions.png"
    fig.savefig(path, dpi=150)
    print(f"saved plot: {path}")
    plt.show()


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

    # Separate master seeds make each ensemble reproducible on its own.
    null_seed, alt_seed = (
        int(value) for value in rng.integers(0, MAX_MODEL_SEED, size=2, dtype=np.uint32)
    )
    null_result = nplm_resampling_null(
        make_sampler(
            n_reference=args.n_reference, expected_background=args.expected_background,
            expected_signal=0.0, label="null", n_toys=args.n_null,
            progress_every=args.progress_every,
        ),
        base_model_config, n_null=args.n_null, seed=null_seed,
    )
    alt_result = nplm_resampling_alternative(
        make_sampler(
            n_reference=args.n_reference, expected_background=args.expected_background,
            expected_signal=args.expected_signal, label="alt", n_toys=args.n_alt,
            progress_every=args.progress_every,
        ),
        base_model_config, n_alternative=args.n_alt, seed=alt_seed,
    )
    comparison = compare_nplm_results(
        null_result, {"alternative": alt_result}, fit_chi2=True, seed=args.seed,
    )

    summary = build_summary(args)
    output_path = save_results(
        output_dir=args.output_dir,
        summary=summary,
        null_result=null_result,
        alt_result=alt_result,
    )

    print_summary(null_result, alt_result, comparison)
    print(f"saved arrays: {output_path}")

    maybe_plot(
        output_dir=args.output_dir,
        null_result=null_result,
        alt_result=alt_result,
        comparison=comparison,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
