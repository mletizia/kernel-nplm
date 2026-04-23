import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


# -----------------------------
# Empirical statistics
# -----------------------------
def emp_pvalue_mc(t0, t1):
    B = len(t0)
    return (1.0 + np.sum(t0 >= t1)) / (B + 1.0)


def emp_pvalues_mc(t0, t_values):
    t0 = np.asarray(t0)
    t_values = np.asarray(t_values)
    B = len(t0)

    return (1.0 + np.sum(t0[:, None] >= t_values[None, :], axis=0)) / (B + 1.0)


def z_from_p(p):
    return norm.isf(p)


# -----------------------------
# Chi2 statistics
# -----------------------------
def chi2_pvalues(t_values, dof):
    p = chi2.sf(t_values, df=dof)
    return np.maximum(p, np.finfo(float).tiny)


# -----------------------------
# Plot
# -----------------------------
def plot_nplm_distributions(
    t_null,
    t_alt,
    save_path="nplm_distributions.png",
    bins=30,
    figsize=(7, 4.5),
):
    t_null = np.asarray(t_null, dtype=float)
    t_alt = np.asarray(t_alt, dtype=float)

    if len(t_null) == 0 or len(t_alt) == 0:
        raise ValueError("t_null and t_alt must be non-empty")

    # -----------------------------
    # DOF estimate
    # -----------------------------
    dof_chi2 = float(np.mean(t_null))

    # -----------------------------
    # Percentiles in t_alt
    # -----------------------------
    q16, q50, q84 = np.percentile(t_alt, [16, 50, 84])

    # -----------------------------
    # Empirical Z (via percentiles of t)
    # -----------------------------
    p_emp = emp_pvalues_mc(t_null, [q16, q50, q84])
    Z_emp = z_from_p(p_emp)

    Z_emp_16, Z_emp_50, Z_emp_84 = Z_emp
    dZ_emp_minus = Z_emp_50 - Z_emp_16
    dZ_emp_plus = Z_emp_84 - Z_emp_50

    # -----------------------------
    # Chi2 Z (via percentiles of t)
    # -----------------------------
    p_chi2 = chi2_pvalues([q16, q50, q84], dof_chi2)
    Z_chi2 = z_from_p(p_chi2)

    Z_chi2_16, Z_chi2_50, Z_chi2_84 = Z_chi2
    dZ_chi2_minus = Z_chi2_50 - Z_chi2_16
    dZ_chi2_plus = Z_chi2_84 - Z_chi2_50

    # -----------------------------
    # Binning
    # -----------------------------
    xmin = min(np.min(t_null), np.min(t_alt))
    xmax = max(np.max(t_null), np.max(t_alt))

    # extend range
    xmin_ext = xmin - 20
    xmax_ext = xmax + 20

    bins_arr = np.linspace(xmin_ext, xmax_ext, bins)

    # -----------------------------
    # Chi2 curve
    # -----------------------------
    xgrid = np.linspace(xmin_ext, xmax_ext, 1000)
    chi2_pdf = chi2.pdf(xgrid, df=dof_chi2)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=figsize)

    plt.hist(
        t_null,
        bins=bins_arr,
        density=True,
        alpha=0.5,
        label="Null distribution"
    )

    plt.hist(
        t_alt,
        bins=bins_arr,
        density=True,
        alpha=0.5,
        label="Alternative distribution"
    )

    plt.plot(
        xgrid,
        chi2_pdf,
        linewidth=2,
        label=fr"$\chi^2(\nu)$, $\nu = {dof_chi2:.2f}$"
    )

    # Vertical lines for percentiles
    for x, lab in [(q16, "16%"), (q50, "50%"), (q84, "84%")]:
        plt.axvline(
            x,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label=f"$t_{{alt}}$ {lab}"
        )

    # -----------------------------
    # Text box
    # -----------------------------
    textstr = "\n".join([
        fr"$N_{{\rm toys}}^{{\rm null}}$: {len(t_null)}",
        fr"$N_{{\rm toys}}^{{\rm alt}}$: {len(t_alt)}",
        fr"$t_{{alt}}$ percentiles:",
        fr"[{q16:.3g}, {q50:.3g}, {q84:.3g}]",
        "",
        fr"Empirical Z:",
        fr"{Z_emp_50:.2f}"
        + fr"$^{{+{dZ_emp_plus:.2f}}}_{{-{dZ_emp_minus:.2f}}}$",
        fr"$\chi^2$-based Z:",
        fr"{Z_chi2_50:.2f}"
        + fr"$^{{+{dZ_chi2_plus:.2f}}}_{{-{dZ_chi2_minus:.2f}}}$",
    ])

    plt.text(
        0.97, 0.6, textstr,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )

    plt.xlabel("t")
    plt.ylabel("Density")
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()



def plot_reconstruction_1d(
    *,
    df,
    data_1d,
    weight_data,
    ref_1d,
    weight_ref,
    t_obs,
    ref_scores,
    bins=24,
    x_range=(0.0, 1.5),
    logy=True,
    ratio_ylim=(0.0, 10.0),
    eps=1e-10,
    var_name="x",
    save=False,
    save_path="",
    file_name="",
    figsize=(8, 8),
    show=True,
):
    data_1d = np.asarray(data_1d).reshape(-1)
    ref_1d = np.asarray(ref_1d).reshape(-1)

    if np.isscalar(weight_ref):
        w_ref = np.full(len(ref_1d), float(weight_ref))
    else:
        w_ref = np.asarray(weight_ref)

    if np.isscalar(weight_data):
        w_data = np.full(len(data_1d), float(weight_data))
    else:
        w_data = np.asarray(weight_data)

    ref_scores = np.asarray(ref_scores)
    if ref_scores.ndim == 2:
        ref_scores = ref_scores[:, 0]

    if np.isscalar(bins):
        edges = np.linspace(x_range[0], x_range[1], bins + 1)
    else:
        edges = np.asarray(bins)

    centers = 0.5 * (edges[1:] + edges[:-1])

    hD, _ = np.histogram(data_1d, bins=edges, weights=w_data)
    hR, _ = np.histogram(ref_1d, bins=edges, weights=w_ref)
    hN, _ = np.histogram(ref_1d, bins=edges, weights=np.exp(ref_scores) * w_ref)

    yerr = np.sqrt(np.maximum(hD, 0.0))

    ztxt = ""
    if df is not None:
        Z = norm.ppf(chi2.cdf(float(t_obs), int(df)))
        ztxt = f", Z-score={Z:.2f}"

    fig = plt.figure(figsize=figsize)
    ax_top = fig.add_axes([0.1, 0.43, 0.8, 0.5])
    ax_ratio = fig.add_axes([0.1, 0.1, 0.8, 0.3], sharex=ax_top)

    ax_top.hist(data_1d, bins=edges, weights=w_data, histtype="step", color="black", lw=1.5, label="DATA", zorder=2)
    ax_top.hist(ref_1d, bins=edges, weights=w_ref, color="#a6cee3", edgecolor="#1f78b4", lw=1, label="REFERENCE", zorder=1)
    ax_top.errorbar(centers, hD, yerr=yerr, color="black", ls="", marker="o", ms=5, zorder=3)
    ax_top.scatter(centers, hN, edgecolor="black", color="#b2df8a", lw=1, s=30, label="RECO", zorder=4)

    title = f"t={float(t_obs):.2f}{ztxt}"
    ax_top.legend(title=title, ncol=2)
    ax_top.set_ylabel("events")
    ax_top.set_xlim(edges[0], edges[-1])
    if logy:
        ax_top.set_yscale("log")
    ax_top.tick_params(axis="x", labelbottom=False)

    denom = hR + eps
    ax_ratio.errorbar(centers, hD / denom, yerr=yerr / denom, ls="", marker="o", color="black", label="DATA/REF")
    ax_ratio.plot(centers, hN / denom, color="#b2df8a", lw=3, label="RECO")

    ax_ratio.set_xlabel(var_name)
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.set_xlim(edges[0], edges[-1])
    ax_ratio.grid()
    ax_ratio.legend()

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, file_name), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_top, ax_ratio)


def plot_reconstruction_marginals(
    *,
    df,
    data,
    weight_data,
    ref,
    weight_ref,
    t_obs,
    ref_scores,
    feature_names=None,
    bins=24,
    x_ranges=None,
    save=False,
    save_path="",
    file_prefix="reco_dim",
    show=True,
    **kwargs,
):
    data = np.asarray(data)
    ref = np.asarray(ref)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    if data.shape[1] != ref.shape[1]:
        raise ValueError("data and ref must have the same dimensionality")

    d = data.shape[1]
    if feature_names is None:
        feature_names = [f"x[{j}]" for j in range(d)]
    if len(feature_names) != d:
        raise ValueError("feature_names must have length d")

    if x_ranges is None:
        x_ranges = [None] * d
    elif len(x_ranges) != d:
        raise ValueError("x_ranges must be None or have length d")

    figs = []
    for j in range(d):
        xr = x_ranges[j]
        if xr is None:
            lo = float(np.min([data[:, j].min(), ref[:, j].min()]))
            hi = float(np.max([data[:, j].max(), ref[:, j].max()]))
            pad = 1e-6 + 0.02 * (hi - lo if hi > lo else 1.0)
            xr = (lo - pad, hi + pad)

        fname = ""
        if save:
            fname = f"{file_prefix}_{j}.pdf"

        fig, _ = plot_reconstruction_1d(
            df=df,
            data_1d=data[:, j],
            weight_data=weight_data,
            ref_1d=ref[:, j],
            weight_ref=weight_ref,
            t_obs=t_obs,
            ref_scores=ref_scores,
            bins=bins,
            x_range=xr,
            var_name=feature_names[j],
            save=save,
            save_path=save_path,
            file_name=fname,
            show=show,
            **kwargs,
        )
        figs.append(fig)

    return figs