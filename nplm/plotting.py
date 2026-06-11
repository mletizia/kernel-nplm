"""Plot empirical NPLM distributions and one-dimensional reconstructions."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm


#########################################################################################################
# Empirical statistic helpers

def emp_pvalue_mc(t0, t1):
    """Compute a right-tail Monte Carlo p-value.

    :param t0: Null statistics with shape ``(n_toys,)``.
    :param t1: Observed or alternative statistic.
    :returns: Finite-sample corrected right-tail p-value.
    """
    null_statistics = np.asarray(t0, dtype=float)
    if null_statistics.size == 0:
        raise ValueError("t0 must be non-empty")

    test_statistic = float(t1)
    n_toys = len(null_statistics)
    return float((1.0 + np.sum(null_statistics >= test_statistic)) / (n_toys + 1.0))


def emp_pvalues_mc(t0, t_values):
    """Compute right-tail Monte Carlo p-values for multiple statistics.

    :param t0: Null statistics with shape ``(n_toys,)``.
    :param t_values: Test statistics with shape ``(n_values,)``.
    :returns: P-values with shape ``(n_values,)``.
    """
    null_statistics = np.asarray(t0, dtype=float)
    statistic_values = np.asarray(t_values, dtype=float)
    if null_statistics.size == 0:
        raise ValueError("t0 must be non-empty")

    n_toys = len(null_statistics)
    counts = np.sum(null_statistics[:, None] >= statistic_values[None, :], axis=0)
    return (1.0 + counts) / (n_toys + 1.0)


def z_from_p(p):
    """Convert right-tail p-values to Gaussian Z-scores.

    :param p: P-value or array of p-values.
    :returns: Corresponding Gaussian Z-score or scores.
    """
    return norm.isf(p)


#########################################################################################################
# Chi-square statistic helpers

def chi2_pvalues(t_values, dof):
    """Compute right-tail chi-square p-values.

    :param t_values: Test statistics with shape ``(n_values,)``.
    :param dof: Positive chi-square degrees of freedom.
    :returns: P-values clipped away from zero.
    """
    dof = float(dof)
    if not np.isfinite(dof) or dof <= 0:
        raise ValueError("dof must be positive finite")

    p_values = chi2.sf(t_values, df=dof)
    return np.maximum(p_values, np.finfo(float).tiny)


#########################################################################################################
# Distribution plots

def plot_nplm_distributions(
    t_null,
    t_alt,
    save_path="nplm_distributions.png",
    bins=30,
    figsize=(7, 4.5),
):
    """Plot null and alternative NPLM statistic distributions.

    :param t_null: Null statistics with shape ``(n_null,)``.
    :param t_alt: Alternative statistics with shape ``(n_alt,)``.
    :param save_path: Output path, or ``None`` to skip saving.
    :param bins: Number of histogram bins.
    :param figsize: Matplotlib figure size.
    :returns: ``None``.
    """
    t_null = np.asarray(t_null, dtype=float)
    t_alt = np.asarray(t_alt, dtype=float)

    if len(t_null) == 0 or len(t_alt) == 0:
        raise ValueError("t_null and t_alt must be non-empty")
    if int(bins) < 1:
        raise ValueError("bins must be >= 1")

    dof_chi2 = float(np.mean(t_null))
    q16, q50, q84 = np.percentile(t_alt, [16, 50, 84])

    p_emp = emp_pvalues_mc(t_null, [q16, q50, q84])
    z_emp = z_from_p(p_emp)

    z_emp_16, z_emp_50, z_emp_84 = z_emp
    dz_emp_minus = z_emp_50 - z_emp_16
    dz_emp_plus = z_emp_84 - z_emp_50

    p_chi2 = chi2_pvalues([q16, q50, q84], dof_chi2)
    z_chi2 = z_from_p(p_chi2)

    z_chi2_16, z_chi2_50, z_chi2_84 = z_chi2
    dz_chi2_minus = z_chi2_50 - z_chi2_16
    dz_chi2_plus = z_chi2_84 - z_chi2_50

    xmin = min(np.min(t_null), np.min(t_alt))
    xmax = max(np.max(t_null), np.max(t_alt))
    xmin_ext = xmin - 20
    xmax_ext = xmax + 20

    bins_arr = np.linspace(xmin_ext, xmax_ext, int(bins))
    xgrid = np.linspace(xmin_ext, xmax_ext, 1000)
    chi2_pdf = chi2.pdf(xgrid, df=dof_chi2)

    plt.figure(figsize=figsize)
    plt.hist(t_null, bins=bins_arr, density=True, alpha=0.5, label="Null distribution")
    plt.hist(t_alt, bins=bins_arr, density=True, alpha=0.5, label="Alternative distribution")
    plt.plot(xgrid, chi2_pdf, linewidth=2, label=fr"$\chi^2(\nu)$, $\nu = {dof_chi2:.2f}$")

    for value, label in [(q16, "16%"), (q50, "50%"), (q84, "84%")]:
        plt.axvline(value, linestyle="--", linewidth=1, alpha=0.8, label=f"$t_{{alt}}$ {label}")

    textstr = "\n".join(
        [
            fr"$N_{{\rm toys}}^{{\rm null}}$: {len(t_null)}",
            fr"$N_{{\rm toys}}^{{\rm alt}}$: {len(t_alt)}",
            fr"$t_{{alt}}$ percentiles:",
            fr"[{q16:.3g}, {q50:.3g}, {q84:.3g}]",
            "",
            fr"Empirical Z:",
            fr"{z_emp_50:.2f}"
            + fr"$^{{+{dz_emp_plus:.2f}}}_{{-{dz_emp_minus:.2f}}}$",
            fr"$\chi^2$-based Z:",
            fr"{z_chi2_50:.2f}"
            + fr"$^{{+{dz_chi2_plus:.2f}}}_{{-{dz_chi2_minus:.2f}}}$",
        ]
    )

    plt.text(
        0.97,
        0.6,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    plt.xlabel("t")
    plt.ylabel("Density")
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


#########################################################################################################
# Reconstruction plots

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
    """Plot data, reference, and reconstructed reference in one dimension.

    :param df: Chi-square degrees of freedom, or ``None``.
    :param data_1d: Data sample with shape ``(n_data,)``.
    :param weight_data: Scalar or data weights with shape ``(n_data,)``.
    :param ref_1d: Reference sample with shape ``(n_ref,)``.
    :param weight_ref: Scalar or reference weights with shape ``(n_ref,)``.
    :param t_obs: Observed NPLM statistic.
    :param ref_scores: Reference scores with shape ``(n_ref,)`` or ``(n_ref, 1)``.
    :returns: Pair ``(fig, (ax_top, ax_ratio))``.
    """
    data_1d = np.asarray(data_1d).reshape(-1)
    ref_1d = np.asarray(ref_1d).reshape(-1)
    if data_1d.size == 0 or ref_1d.size == 0:
        raise ValueError("data_1d and ref_1d must be non-empty")
    if eps <= 0:
        raise ValueError("eps must be positive")

    if np.isscalar(weight_ref):
        w_ref = np.full(len(ref_1d), float(weight_ref))
    else:
        w_ref = np.asarray(weight_ref).reshape(-1)
    if len(w_ref) != len(ref_1d):
        raise ValueError("weight_ref must be scalar or have the same length as ref_1d")

    if np.isscalar(weight_data):
        w_data = np.full(len(data_1d), float(weight_data))
    else:
        w_data = np.asarray(weight_data).reshape(-1)
    if len(w_data) != len(data_1d):
        raise ValueError("weight_data must be scalar or have the same length as data_1d")

    ref_scores = np.asarray(ref_scores)
    if ref_scores.ndim == 2:
        ref_scores = ref_scores[:, 0]
    ref_scores = ref_scores.reshape(-1)
    if len(ref_scores) != len(ref_1d):
        raise ValueError("ref_scores must have the same length as ref_1d")

    if np.isscalar(bins):
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError("bins must be >= 1")
        edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("bins must be a scalar or 1D edge array with at least two entries")

    centers = 0.5 * (edges[1:] + edges[:-1])

    hist_data, _ = np.histogram(data_1d, bins=edges, weights=w_data)
    hist_ref, _ = np.histogram(ref_1d, bins=edges, weights=w_ref)
    hist_reco, _ = np.histogram(ref_1d, bins=edges, weights=np.exp(ref_scores) * w_ref)

    yerr = np.sqrt(np.maximum(hist_data, 0.0))

    z_text = ""
    if df is not None:
        z_score = norm.ppf(chi2.cdf(float(t_obs), int(df)))
        z_text = f", Z-score={z_score:.2f}"

    fig = plt.figure(figsize=figsize)
    ax_top = fig.add_axes([0.1, 0.43, 0.8, 0.5])
    ax_ratio = fig.add_axes([0.1, 0.1, 0.8, 0.3], sharex=ax_top)

    ax_top.hist(
        data_1d,
        bins=edges,
        weights=w_data,
        histtype="step",
        color="black",
        lw=1.5,
        label="DATA",
        zorder=2,
    )
    ax_top.hist(
        ref_1d,
        bins=edges,
        weights=w_ref,
        color="#a6cee3",
        edgecolor="#1f78b4",
        lw=1,
        label="REFERENCE",
        zorder=1,
    )
    ax_top.errorbar(centers, hist_data, yerr=yerr, color="black", ls="", marker="o", ms=5, zorder=3)
    ax_top.scatter(
        centers,
        hist_reco,
        edgecolor="black",
        color="#b2df8a",
        lw=1,
        s=30,
        label="RECO",
        zorder=4,
    )

    title = f"t={float(t_obs):.2f}{z_text}"
    ax_top.legend(title=title, ncol=2)
    ax_top.set_ylabel("events")
    ax_top.set_xlim(edges[0], edges[-1])
    if logy:
        ax_top.set_yscale("log")
    ax_top.tick_params(axis="x", labelbottom=False)

    denom = hist_ref + eps
    ax_ratio.errorbar(
        centers,
        hist_data / denom,
        yerr=yerr / denom,
        ls="",
        marker="o",
        color="black",
        label="DATA/REF",
    )
    ax_ratio.plot(centers, hist_reco / denom, color="#b2df8a", lw=3, label="RECO")

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
    """Plot one-dimensional reconstructions for each feature.

    :param df: Chi-square degrees of freedom, or ``None``.
    :param data: Data sample with shape ``(n_data, n_features)``.
    :param weight_data: Scalar or data weights.
    :param ref: Reference sample with shape ``(n_ref, n_features)``.
    :param weight_ref: Scalar or reference weights.
    :param t_obs: Observed NPLM statistic.
    :param ref_scores: Reference scores with shape ``(n_ref,)`` or ``(n_ref, 1)``.
    :returns: List of Matplotlib figures.
    """
    data = np.asarray(data)
    ref = np.asarray(ref)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    if data.ndim != 2 or ref.ndim != 2:
        raise ValueError("data and ref must be 1D or 2D arrays")
    if data.shape[1] != ref.shape[1]:
        raise ValueError("data and ref must have the same dimensionality")

    n_features = data.shape[1]
    if feature_names is None:
        feature_names = [f"x[{j}]" for j in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names must have length n_features")

    if x_ranges is None:
        x_ranges = [None] * n_features
    elif len(x_ranges) != n_features:
        raise ValueError("x_ranges must be None or have length n_features")

    figs = []
    for feature_idx in range(n_features):
        feature_range = x_ranges[feature_idx]
        if feature_range is None:
            low = float(np.min([data[:, feature_idx].min(), ref[:, feature_idx].min()]))
            high = float(np.max([data[:, feature_idx].max(), ref[:, feature_idx].max()]))
            pad = 1e-6 + 0.02 * (high - low if high > low else 1.0)
            feature_range = (low - pad, high + pad)

        file_name = ""
        if save:
            file_name = f"{file_prefix}_{feature_idx}.pdf"

        fig, _ = plot_reconstruction_1d(
            df=df,
            data_1d=data[:, feature_idx],
            weight_data=weight_data,
            ref_1d=ref[:, feature_idx],
            weight_ref=weight_ref,
            t_obs=t_obs,
            ref_scores=ref_scores,
            bins=bins,
            x_range=feature_range,
            var_name=feature_names[feature_idx],
            save=save,
            save_path=save_path,
            file_name=file_name,
            show=show,
            **kwargs,
        )
        figs.append(fig)

    return figs
