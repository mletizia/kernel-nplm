"""DIMUON pilot experiments; all dataset-specific choices stay in this file.

Examples (from the repository root):
    python examples/dimuon_resampling.py --validate-only
    python examples/dimuon_resampling.py --gpu --n-null 5 --n-alt 5
    python examples/dimuon_resampling.py --gpu --cases zprime600 \
        --null-results results_dimuon/null.npz --output-dir results_dimuon_600

The five inputs exclude mll. No extra event selection is applied by default.
All expected yields refer to events AFTER any requested --mll-min selection.
EFT files are full alternatives and require an explicit expected TOTAL yield.

Normalization follows the legacy pooled, per-toy rule in:
https://github.com/FalkonHEP/falkonhep/blob/main/falkonhep/utils/data_utils.py
Signed columns are standardized, nonnegative columns exceeding one are divided
by their mean, and columns already in [0, 1] are left unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import make_null_sampler, make_mixture_sampler, make_alternative_sampler
from stat_tests import (
    NPLMResamplingEnsemble, nplm_resampling_null, nplm_resampling_alternative,
    compare_nplm_results,
)

FEATURES = ("pt1", "pt2", "eta1", "eta2", "delta_phi")
FILES = {
    "null": "DiLepton_SM.h5",
    "zprime300": "DiLepton_Zprime300.h5",
    "zprime600": "DiLepton_Zprime600.h5",
    "eft06": "DiLepton_EFT06.h5",
    "eft06_2": "DiLepton_EFT06_2.h5",
    "eft06_5": "DiLepton_EFT06_5.h5",
}
PREPROCESSING = "legacy_pooled_per_toy_v1"


def load_pool(path, mll_min=None):
    """Load columns by name, preserving alignment, and optionally select on mll."""
    import h5py

    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        required = FEATURES + (("mll",) if mll_min is not None else ())
        for name in required:
            if name not in handle or not isinstance(handle[name], h5py.Dataset):
                raise ValueError(f"{path}: missing dataset {name}")
            if handle[name].ndim != 1:
                raise ValueError(f"{path}: {name} must be a 1D column")
        n_rows = len(handle[FEATURES[0]])
        if any(len(handle[name]) != n_rows for name in required):
            raise ValueError(f"{path}: columns have different lengths")
        pool = np.empty((n_rows, len(FEATURES)), dtype=np.float64)
        for index, name in enumerate(FEATURES):
            pool[:, index] = handle[name][:]
        if not np.all(np.isfinite(pool)):
            raise ValueError(f"{path}: input features contain nonfinite values")
        if mll_min is not None:
            mass = handle["mll"][:]
            if not np.all(np.isfinite(mass)):
                raise ValueError(f"{path}: mll contains nonfinite values")
            pool = pool[mass >= mll_min]
    if len(pool) == 0:
        raise ValueError(f"{path}: no events remain after selection")
    info = {
        "path": str(path), "file_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "input_rows": n_rows, "selected_rows": len(pool),
    }
    print(f"Loaded {path.name}: {len(pool):,}/{n_rows:,} events", flush=True)
    return pool, info


def preprocess_pair(x_ref, x_data):
    """Normalize a complete toy jointly; return copies and the applied affine map."""
    if len(x_data) == 0:
        raise ValueError("empty pseudo-dataset (realized data count is zero)")
    pooled = np.concatenate((x_ref, x_data), axis=0)
    means, stds = pooled.mean(axis=0), pooled.std(axis=0)
    signed = pooled.min(axis=0) < 0
    positive = (~signed) & (pooled.max(axis=0) > 1)
    offset = np.where(signed, means, 0.0)
    scale = np.where(signed, stds, np.where(positive, means, 1.0))
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Legacy normalization requires positive finite feature scales")
    pooled = (pooled - offset) / scale
    return pooled[:len(x_ref)], pooled[len(x_ref):], {
        "preprocessing": PREPROCESSING,
        "offset": offset.tolist(), "scale": scale.tolist(),
    }


def prepare_sampler(raw_sampler, case, n_toys, progress_every):
    """Apply DIMUON preprocessing after drawing the reference/data pair."""
    toy = 0

    def sample(rng):
        nonlocal toy
        toy += 1
        if progress_every and (toy == 1 or toy == n_toys or toy % progress_every == 0):
            print(f"[{case}] toy {toy}/{n_toys}", flush=True)
        reference, data, metadata = raw_sampler(rng)
        reference, data, normalization = preprocess_pair(reference, data)
        return reference, data, dict(metadata, **normalization)

    return sample


def save_ensemble(path, result, experiment):
    """Save each completed ensemble immediately so later failures do not lose it."""
    provenance = {
        "seed": result.seed, "model_config": result.model_config,
        "resample_nystrom": result.resample_nystrom, "dtype": result.dtype,
        "metadata": result.metadata, "experiment": experiment,
    }
    np.savez_compressed(
        path, statistics=result.statistics, reference_counts=result.reference_counts,
        data_counts=result.data_counts, model_seeds=result.model_seeds,
        provenance_json=json.dumps(provenance, sort_keys=True, allow_nan=False),
    )
    print(f"Saved {path}", flush=True)


def load_null(path, config, experiment, n_reference):
    """Reuse a null only with matching model, input pool and preprocessing settings."""
    with np.load(path, allow_pickle=False) as saved:
        provenance = json.loads(str(saved["provenance_json"]))
        if provenance["model_config"] != config or provenance["experiment"] != experiment:
            raise ValueError("Saved null has different model, source pool or preprocessing settings")
        if not provenance["resample_nystrom"] or provenance["dtype"] != "float64":
            raise ValueError("Saved null must use float64 and fresh model seeds per toy")
        result = NPLMResamplingEnsemble(
            saved["statistics"].copy(), saved["reference_counts"].copy(),
            saved["data_counts"].copy(), provenance["metadata"], provenance["seed"],
            saved["model_seeds"].copy(), provenance["model_config"], True, "float64",
        )
    if (result.statistics.ndim != 1 or len(result.statistics) == 0
            or not np.all(np.isfinite(result.statistics))
            or any(values.shape != result.statistics.shape for values in (
                result.reference_counts, result.data_counts, result.model_seeds))
            or len(result.metadata) != len(result.statistics)
            or np.any(result.data_counts <= 0)
            or np.any(result.reference_counts != n_reference)):
        raise ValueError("Saved null has invalid statistics, counts or provenance lengths")
    print(f"Reusing {len(result.statistics)} null toys from {path}", flush=True)
    return result


def save_comparison(output_dir, null, alternatives, labels, seed, fit_chi2, make_plot, bins):
    comparison = compare_nplm_results(null, alternatives, fit_chi2=fit_chi2, seed=seed)
    fit = comparison.chi2_fit
    arrays = {
        "n_null": comparison.n_null,
        "p_value_resolution": comparison.p_value_resolution,
        "z_score_resolution": comparison.z_score_resolution,
        "comparison_seed": seed,
        "chi2_fit_json": json.dumps(None if fit is None else fit._asdict()),
    }
    print(f"Empirical p-value resolution: {comparison.p_value_resolution:.5g}")
    if fit is not None:
        print(f"Chi-square check: {fit.reason}; bootstrap p={fit.p_value}")
    for name, result in comparison.alternatives.items():
        for field in ("quantile_levels", "t_quantiles", "p_values", "z_scores",
                      "chi2_p_values", "chi2_z_scores"):
            value = getattr(result, field)
            if value is not None:
                arrays[f"{name}_{field}"] = value
        print(f"{labels[name]}: t[16,50,84%]={result.t_quantiles}; empirical Z={result.z_scores}")
        if result.chi2_z_scores is not None:
            print(f"  chi-square Z={result.chi2_z_scores}")
    print("The 16–84% interval is the spread across toys, not uncertainty on the median.")
    np.savez_compressed(output_dir / "comparison.npz", **arrays)
    if not make_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    edges = np.histogram_bin_edges(
        np.concatenate([null.statistics] + [r.statistics for r in alternatives.values()]), bins=bins
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null.statistics, bins=edges, density=True, color="0.5", alpha=0.35, label="SM null")
    for index, (name, result) in enumerate(alternatives.items()):
        color = f"C{index}"
        ax.hist(result.statistics, bins=edges, density=True, histtype="step",
                linewidth=1.8, color=color, label=labels[name])
        ax.axvline(np.median(result.statistics), color=color, linestyle="--", linewidth=1)
    if fit is not None and fit.accepted:
        grid = np.linspace(max(np.finfo(float).eps, edges[0]), edges[-1], 1000)
        ax.plot(grid, chi2.pdf(grid, df=fit.dof), color="black",
                label=f"Null chi-square fit (df={fit.dof:.2f})")
    ax.set(xlabel="NPLM statistic", ylabel="Density", title="DIMUON toy experiments")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"dimuon_distributions.{extension}", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "local-data" / "DIMUON")
    parser.add_argument("--output-dir", type=Path, default=Path("results_dimuon"))
    parser.add_argument("--cases", nargs="+", choices=FILES, default=["zprime300", "zprime600"],
                        help="Alternatives to run; the null is always included. Use 'null' for null only.")
    parser.add_argument("--n-null", type=int, default=10)
    parser.add_argument("--n-alt", type=int, default=10)
    parser.add_argument("--n-reference", type=int, default=100_000)
    parser.add_argument("--expected-background", type=float, default=20_000)
    parser.add_argument("--zprime300-yield", type=float, default=40)
    parser.add_argument("--zprime600-yield", type=float, default=15)
    parser.add_argument("--eft-yield", action="append", default=[], metavar="CASE=TOTAL",
                        help="Expected full EFT yield after selection, e.g. eft06=TOTAL; no coefficient mapping is assumed.")
    parser.add_argument("--mll-min", type=float, default=None, help="Optional mll >= threshold in GeV; default: no extra cut.")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--nystrom-centers", type=int, default=1000, help="Reduced pilot default; configurable independently of toy counts.")
    parser.add_argument("--penalty", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--cg-tol", type=float, default=np.sqrt(1e-7))
    parser.add_argument("--keops", choices=("yes", "no"), default="no")
    parser.add_argument("--model-verbose", type=int, default=0)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", dest="cpu", action="store_true")
    device.add_argument("--gpu", dest="cpu", action="store_false")
    parser.set_defaults(cpu=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-chi2", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Load inputs and check one sampled/preprocessed toy per case, without fitting.")
    parser.add_argument("--null-results", type=Path, help="Reuse null.npz from an earlier compatible run; --n-null is then ignored.")
    args = parser.parse_args(argv)
    for name in ("n_null", "n_alt", "n_reference", "nystrom_centers", "iterations", "bins",
                 "expected_background", "sigma", "penalty", "cg_tol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("zprime300_yield", "zprime600_yield", "mll_min", "seed", "progress_every", "model_verbose"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    if args.nystrom_centers > args.n_reference:
        parser.error("Use --nystrom-centers <= --n-reference so every Poisson toy has enough rows")
    if len(args.cases) != len(set(args.cases)):
        parser.error("--cases must not contain duplicates")
    eft_yields = {}
    for entry in args.eft_yield:
        try:
            name, total = entry.split("=")
            total = float(total)
            if name not in FILES or not name.startswith("eft") or name in eft_yields:
                raise ValueError()
            if not np.isfinite(total) or total <= 0:
                raise ValueError()
        except ValueError:
            parser.error("Each --eft-yield must be a unique EFT case and positive total: CASE=TOTAL")
        eft_yields[name] = total
    if any(case.startswith("eft") and case not in eft_yields for case in args.cases):
        parser.error("Every selected EFT case requires --eft-yield CASE=TOTAL")
    args.eft_yields = eft_yields
    return args


def main(argv=None):
    args = parse_args(argv)
    for case in dict.fromkeys(["null"] + args.cases):
        path = args.data_dir / FILES[case]
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {case}: {path}")
    config = dict(sigma=args.sigma, NR=args.expected_background, M=args.nystrom_centers,
                  **{"lambda": [args.penalty], "iter": [args.iterations]},
                  cg_tol=args.cg_tol, cpu=args.cpu, keops=args.keops, verbose=args.model_verbose)
    sm, sm_info = load_pool(args.data_dir / FILES["null"], args.mll_min)
    experiment = dict(features=list(FEATURES), mll_min=args.mll_min,
                      preprocessing=PREPROCESSING, reference_pool=sm_info, case="null")
    # Assign every case a stable seed, independent of the selected alternatives/order.
    seeds = dict(zip(FILES, map(int, np.random.default_rng(args.seed).integers(
        0, 2**32 - 1, size=len(FILES), dtype=np.uint32))))
    raw_null = make_null_sampler(sm, n_ref=args.n_reference, expected_data=args.expected_background)
    null = None
    if args.null_results is not None:
        null = load_null(args.null_results, config, experiment, args.n_reference)
    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if null is None:
            null = nplm_resampling_null(
                prepare_sampler(raw_null, "null", args.n_null, args.progress_every),
                config, n_null=args.n_null, seed=seeds["null"],
            )
        save_ensemble(args.output_dir / "null.npz", null, experiment)
    else:
        reference, data, _ = prepare_sampler(raw_null, "null", 1, 0)(np.random.default_rng(seeds["null"]))
        print(f"Validated null: reference={len(reference)}, data={len(data)}; NR={config['NR']}")

    alternatives, labels = {}, {}
    for case in args.cases:
        if case == "null":
            continue
        pool, info = load_pool(args.data_dir / FILES[case], args.mll_min)
        if case.startswith("zprime"):
            signal = getattr(args, f"{case}_yield")
            raw_sampler = make_mixture_sampler(sm, pool, n_ref=args.n_reference,
                expected_background=args.expected_background, expected_component=signal)
            labels[case] = f"Z' {case[6:]} GeV, S={signal:g}"
        else:
            raw_sampler = make_alternative_sampler(sm, pool, n_ref=args.n_reference,
                                                   expected_data=args.eft_yields[case])
            labels[case] = f"{case}, total={args.eft_yields[case]:g}"
        sampler = prepare_sampler(raw_sampler, case, 1 if args.validate_only else args.n_alt,
                                  args.progress_every)
        if args.validate_only:
            reference, data, metadata = sampler(np.random.default_rng(seeds[case]))
            print(f"Validated {case}: reference={len(reference)}, data={len(data)}; {metadata}")
            continue
        alternatives[case] = nplm_resampling_alternative(
            sampler, config, n_alternative=args.n_alt, seed=seeds[case],
        )
        save_ensemble(args.output_dir / f"{case}.npz", alternatives[case],
                      dict(experiment, case=case, alternative_pool=info, label=labels[case]))
    if not args.validate_only:
        save_comparison(args.output_dir, null, alternatives, labels, args.seed,
                        not args.no_chi2, not args.no_plot, args.bins)


if __name__ == "__main__":
    main()
