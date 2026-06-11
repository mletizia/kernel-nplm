"""Legacy one-dimensional Gaussian NPLM toy example."""

import os

import numpy as np

from data import build_pooled_sample, make_data_sample_poisson, sample_ref_exp
from nplm import LogFalkonNPLM
from nplm.plotting import plot_nplm_distributions


#########################################################################################################
# Toy runners

def run_null_toys(model_cfg, N_R=200_000, NR=2_000, NS=0, B=300, rng=None):
    """Run null pseudo-experiments from the configured generators.

    :param model_cfg: NPLM model configuration dictionary.
    :param N_R: Reference sample size per toy.
    :param NR: Expected background count per data toy.
    :param NS: Expected signal count per data toy.
    :param B: Number of null toys.
    :param rng: Optional NumPy random generator.
    :returns: Null statistics with shape ``(B,)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    t_null = np.empty(B)

    for b in range(B):
        print(f"--- Null toy {b + 1}/{B} ---")
        x_ref = sample_ref_exp(N_R, rng=rng)
        x_dat, _, _, _, _ = make_data_sample_poisson(NR=NR, NS=NS, rng=rng)

        x_pooled, y = build_pooled_sample(x_ref, x_dat)
        nplm = LogFalkonNPLM(model_cfg)
        t_null[b] = nplm.compute_statistic(x_pooled, y)

    return t_null


def run_alt_toys(model_cfg, N_R=200_000, NR=2_000, NS=10, B=50, rng=None):
    """Run alternative pseudo-experiments from the configured generators.

    :param model_cfg: NPLM model configuration dictionary.
    :param N_R: Reference sample size per toy.
    :param NR: Expected background count per data toy.
    :param NS: Expected signal count per data toy.
    :param B: Number of alternative toys.
    :param rng: Optional NumPy random generator.
    :returns: Alternative statistics with shape ``(B,)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    t_alt = np.empty(B)

    for b in range(B):
        print(f"--- Alt toy {b + 1}/{B} ---")
        x_ref = sample_ref_exp(N_R, rng=rng)
        x_dat, _, _, _, _ = make_data_sample_poisson(NR=NR, NS=NS, rng=rng)

        x_pooled, y = build_pooled_sample(x_ref, x_dat)
        nplm = LogFalkonNPLM(model_cfg)
        t_alt[b] = nplm.compute_statistic(x_pooled, y)

    return t_alt


#########################################################################################################
# Entrypoint

def main():
    """Run the legacy one-dimensional Gaussian toy example.

    :returns: ``None``.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    output_folder = "results_1D"
    os.makedirs(output_folder, exist_ok=False)

    n_reference = 200_000
    expected_background = 2_000
    expected_signal = 10
    sigma = 0.3

    n_null = 300
    n_alt = 100

    cfg = {
        "sigma": float(sigma),
        "NR": float(expected_background),
        "M": 3000,
        "lambda": [1e-10],
        "cpu": False,
        "verbose": 0,
        "keops": "yes",
    }

    t_null = run_null_toys(
        model_cfg=cfg,
        N_R=n_reference,
        NR=expected_background,
        NS=0,
        B=n_null,
        rng=rng,
    )

    np.save(output_folder + "/nplm_null_stats.npy", t_null)

    t_alt = run_alt_toys(
        model_cfg=cfg,
        N_R=n_reference,
        NR=expected_background,
        NS=expected_signal,
        B=n_alt,
        rng=rng,
    )

    np.save(output_folder + "/nplm_alt_stats.npy", t_alt)

    print("Saved null statistics to nplm_null_stats.npy")
    print("Saved alternative statistics to nplm_alt_stats.npy")
    print(f"Null mean: {np.mean(t_null):.6g}")
    print(f"Alt mean:  {np.mean(t_alt):.6g}")

    plot_nplm_distributions(
        t_null=t_null,
        t_alt=t_alt,
        save_path=output_folder + "/nplm_distributions.png",
    )


if __name__ == "__main__":
    main()
