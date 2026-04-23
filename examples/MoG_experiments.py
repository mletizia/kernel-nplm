import numpy as np
import matplotlib.pyplot as plt
import os

from nplm import LogFalkonNPLM

from plotting import plot_nplm_distributions

def build_dataset(X1, X2):
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))
    return X, y


def run_null_toys(model_cfg, ref_sample, N_R=100_000, NR=10_000, B=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t_null = np.empty(B)

    for b in range(B):
        print(f"--- Null toy {b + 1}/{B} ---")

        x = rng.choice(ref_sample, size=N_R + NR, replace=True)

        x_ref_b = x[:N_R, :]
        x_dat_b = x[N_R:, :]

        X_b , y_b = build_dataset(x_ref_b, x_dat_b)

        nplm = LogFalkonNPLM(model_cfg)
        t_null[b] = nplm.compute_statistic(X_b, y_b)

    return t_null


def run_alt_toys(model_cfg, ref_sample, data_sample, N_R=100_000, NR=10_000, B=40, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t_alt = np.empty(B)

    for b in range(B):
        print(f"--- Alt toy {b + 1}/{B} ---")

        x_ref_b = rng.choice(ref_sample, size=N_R, replace=True)
        x_dat_b = rng.choice(data_sample, size=NR, replace=True)

        X_b , y_b = build_dataset(x_ref_b, x_dat_b)

        nplm = LogFalkonNPLM(model_cfg)
        t_alt[b] = nplm.compute_statistic(X_b, y_b)

    return t_alt


def main():
    seed = 0
    rng = np.random.default_rng(seed)

    output_folder = "MoG_4D_results"
    #os.makedirs(output_folder, exist_ok=False)

    # -----------------------------
    # User parameters
    # -----------------------------
    N_R = 100_000
    NR = 10_000
    sigma = 4.96

    B_null = 200
    B_alt = 40

    cfg = {
        "sigma": float(sigma),
        "NR": float(NR),
        "M": 10000,
        "lambda": [1e-10],
        "cpu": False,
        "verbose": 0,
        "keops": "yes",
    }

    ref_sample = np.load("4d_MoG/true.npy")
    data_sample = np.load("4d_MoG/nf500k.npy")

    # # -----------------------------
    # # Null toys
    # # -----------------------------
    # t_null = run_null_toys(
    #     model_cfg=cfg,
    #     ref_sample=ref_sample,
    #     N_R=N_R,
    #     NR=NR,
    #     B=B_null,
    #     rng=rng,
    # )
    # np.save(output_folder+"/nplm_null_stats.npy", t_null)

    # # -----------------------------
    # # Alternative toys
    # # -----------------------------
    # t_alt = run_alt_toys(
    #     model_cfg=cfg,
    #     ref_sample=ref_sample,
    #     data_sample=data_sample,
    #     N_R=N_R,
    #     NR=NR,
    #     B=B_alt,
    #     rng=rng,
    # )
    # np.save(output_folder+"/nplm_alt_stats_100k.npy", t_alt)

    t_null = np.load("MoG_4D_results/nplm_null_stats.npy")
    t_alt = np.load("MoG_4D_results/nplm_alt_stats_500k.npy")

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
        save_path=output_folder+"/nplm_distributions_500k.png",
        bins=20
    )

if __name__ == "__main__":
    main()