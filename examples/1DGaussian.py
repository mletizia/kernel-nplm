import os
import numpy as np

import matplotlib.pyplot as plt

from nplm import LogFalkonNPLM
from nplm.plotting import plot_nplm_distributions, plot_classifier_score_distributions

from data import (
    sample_ref_exp,
    make_data_sample_poisson,
    build_pooled_sample,
)


def run_null_toys(model_cfg, N_R=200_000, NR=2_000, NS=0, B=300, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t_null = np.empty(B)

    for b in range(B):
        print(f"--- Null toy {b + 1}/{B} ---")
        x_ref_b = sample_ref_exp(N_R, rng=rng)
        x_dat_b, _, _, _, _ = make_data_sample_poisson(NR=NR, NS=NS, rng=rng)

        X_b, y_b = build_pooled_sample(x_ref_b, x_dat_b)
        nplm = LogFalkonNPLM(model_cfg)
        t_null[b] = nplm.compute_statistic(X_b, y_b)

    return t_null


def run_alt_toys(model_cfg, N_R=200_000, NR=2_000, NS=10, B=50, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t_alt = np.empty(B)

    for b in range(B):
        print(f"--- Alt toy {b + 1}/{B} ---")
        x_ref_b = sample_ref_exp(N_R, rng=rng)
        x_dat_b, _, _, _, _ = make_data_sample_poisson(NR=NR, NS=NS, rng=rng)

        X_b, y_b = build_pooled_sample(x_ref_b, x_dat_b)
        nplm = LogFalkonNPLM(model_cfg)
        t_alt[b] = nplm.compute_statistic(X_b, y_b)

    return t_alt


def main():
    seed = 0
    rng = np.random.default_rng(seed)

    output_folder = "results_1D"
    os.makedirs(output_folder, exist_ok=False)

    # -----------------------------
    # User parameters
    # -----------------------------
    N_R = 200_000
    NR = 2_000
    NS = 10
    sigma = 0.3

    B_null = 300
    B_alt = 100

    cfg = {
        "sigma": float(sigma),
        "NR": float(NR),
        "M": 3000,
        "lambda": [1e-10],
        "cpu": False,
        "verbose": 0,
        "keops": "yes",
    }

    # -----------------------------
    # Null toys
    # -----------------------------
    t_null = run_null_toys(
        model_cfg=cfg,
        N_R=N_R,
        NR=NR,
        NS=0,
        B=B_null,
        rng=rng,
    )

    np.save(output_folder + "/nplm_null_stats.npy", t_null)

    # -----------------------------
    # Alternative toys
    # -----------------------------
    t_alt = run_alt_toys(
        model_cfg=cfg,
        N_R=N_R,
        NR=NR,
        NS=NS,
        B=B_alt,
        rng=rng,
    )

    np.save(output_folder + "/nplm_alt_stats.npy", t_alt)

    print("Saved null statistics to nplm_null_stats.npy")
    print("Saved alternative statistics to nplm_alt_stats.npy")
    print(f"Null mean: {np.mean(t_null):.6g}")
    print(f"Alt mean:  {np.mean(t_alt):.6g}")

    # -----------------------------
    # Plot null and alternative distributions
    # -----------------------------
    plot_nplm_distributions(
        t_null=t_null,
        t_alt=t_alt,
        save_path=output_folder + "/nplm_distributions.png",
    )

    # -----------------------------
    # Observed score distribution diagnostics
    # -----------------------------
    x_ref_obs = sample_ref_exp(N_R, rng=rng)
    x_dat_obs, _, _, _, _ = make_data_sample_poisson(NR=NR, NS=NS, rng=rng)
    X_obs, y_obs = build_pooled_sample(x_ref_obs, x_dat_obs)

    nplm_obs = LogFalkonNPLM(cfg)
    t_obs, details = nplm_obs.compute_statistic(X_obs, y_obs, return_details=True)
    scores_obs = nplm_obs.compute_scores(X_obs)

    scores_ref = scores_obs[y_obs == 0]
    scores_data = scores_obs[y_obs == 1]

    np.save(output_folder + "/scores_ref.npy", scores_ref)
    np.save(output_folder + "/scores_data.npy", scores_data)

    print(f"Observed statistic t = {t_obs:.6g}")
    print(f"Saved reference and data scores to {output_folder}/scores_ref.npy and scores_data.npy")

    plot_classifier_score_distributions(
        scores_ref=scores_ref,
        scores_data=scores_data,
        label_data=fr"NPLM - $t={t_obs:.2f}$",
        xlabel="NPLM output on observed sample",
        save_path=output_folder + "/nplm_score_distributions.png",
    )


if __name__ == "__main__":
    main()
