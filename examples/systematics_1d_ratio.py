"""Validate finite-difference Falkon morphing on a one-dimensional nuisance toy."""

import argparse
import sys
from pathlib import Path

import numpy as np


#########################################################################################################
# Import path setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nplm_systematics import FalkonLogRatioEstimator, FiniteDifferenceMorpher
from nplm_systematics.toys import (
    sample_exponential_nuisance,
    truncated_exponential_log_ratio_derivative,
)


#########################################################################################################
# CLI helpers

def parse_args():
    """Parse command-line arguments.

    :returns: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-central", type=int, default=4000)
    parser.add_argument("--n-varied", type=int, default=4000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--rate0", type=float, default=8.0)
    parser.add_argument("--rate-log-step", type=float, default=0.08)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--M", default="sqrt")
    parser.add_argument("--penalty", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cg-tol", type=float, default=np.sqrt(1e-7))
    parser.add_argument("--keops", default="no")
    parser.add_argument("--cpu", dest="cpu", action="store_true", default=True)
    parser.add_argument("--gpu", dest="cpu", action="store_false")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def parse_m(value):
    """Parse the Falkon Nyström-center argument.

    :param value: Input value, either ``"sqrt"`` or an integer-like string.
    :returns: ``"sqrt"`` or an integer.
    """
    if isinstance(value, str) and value == "sqrt":
        return value
    return int(value)


#########################################################################################################
# Main example

def main():
    """Run the one-dimensional nuisance-ratio validation.

    :returns: ``None``.
    """
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x0 = sample_exponential_nuisance(
        args.n_central,
        0.0,
        rate0=args.rate0,
        rate_log_step=args.rate_log_step,
        xmax=args.xmax,
        rng=rng,
    ).reshape(-1, 1)
    xp = sample_exponential_nuisance(
        args.n_varied,
        args.epsilon,
        rate0=args.rate0,
        rate_log_step=args.rate_log_step,
        xmax=args.xmax,
        rng=rng,
    ).reshape(-1, 1)
    xm = sample_exponential_nuisance(
        args.n_varied,
        -args.epsilon,
        rate0=args.rate0,
        rate_log_step=args.rate_log_step,
        xmax=args.xmax,
        rng=rng,
    ).reshape(-1, 1)

    sigma = args.sigma
    if sigma is None:
        sigma = FalkonLogRatioEstimator.estimate_sigma_median(
            np.vstack([x0, xp, xm]),
            seed=args.seed,
        )

    config = {
        "sigma": float(sigma),
        "M": parse_m(args.M),
        "lambda": [float(args.penalty)],
        "iter": [int(args.iterations)],
        "cg_tol": float(args.cg_tol),
        "keops": args.keops,
        "cpu": bool(args.cpu),
        "seed": int(args.seed),
        "verbose": int(args.verbose),
    }

    morpher = FiniteDifferenceMorpher(args.epsilon, config)
    morpher.fit(x0, xp, xm)

    grid = np.linspace(0.0, args.xmax, args.grid_size, dtype=np.float64).reshape(-1, 1)
    g_plus, g_minus = morpher.predict_components(grid)
    delta_falkon = morpher.predict_delta(grid)
    delta_true = truncated_exponential_log_ratio_derivative(
        grid.reshape(-1),
        rate0=args.rate0,
        rate_log_step=args.rate_log_step,
        xmax=args.xmax,
    )

    residual = delta_falkon - delta_true
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    corr = float(np.corrcoef(delta_falkon, delta_true)[0, 1])

    print("\n[systematics_1d_ratio]")
    print(f"sigma:             {sigma:.6g}")
    print(f"epsilon:           {args.epsilon:.6g}")
    print(f"RMSE(delta):       {rmse:.6g}")
    print(f"MAE(delta):        {mae:.6g}")
    print(f"corr(delta):       {corr:.6g}")
    if morpher.summary_ is not None:
        print(f"train_time_plus:   {morpher.summary_.train_time_plus:.3f} s")
        print(f"train_time_minus:  {morpher.summary_.train_time_minus:.3f} s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            grid=grid.reshape(-1),
            g_plus=g_plus,
            g_minus=g_minus,
            delta_falkon=delta_falkon,
            delta_true=delta_true,
            residual=residual,
            rmse=np.asarray(rmse),
            mae=np.asarray(mae),
            corr=np.asarray(corr),
            config=np.asarray(str(config)),
        )
        print(f"saved:             {args.output}")


if __name__ == "__main__":
    main()
