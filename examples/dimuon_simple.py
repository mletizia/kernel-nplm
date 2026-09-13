"""Direct DIMUON experiments adapted from the supplied LogisticFalkon baseline.

One loop: sample, normalize, fit LogisticFalkon, predict, compute_t, save.
No imports from this repository's model, sampler or statistical-test modules.
The default normalization is the legacy 'higgs' rule, with no additional cuts.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np


FEATURES = ("pt1", "pt2", "eta1", "eta2", "delta_phi")
FILES = {"null": "DiLepton_SM.h5", "zprime300": "DiLepton_Zprime300.h5",
         "zprime600": "DiLepton_Zprime600.h5"}


def read_data(path):
    with h5py.File(path, "r") as handle:
        columns = [handle[name][:] for name in FEATURES]
    if any(column.ndim != 1 for column in columns):
        raise ValueError(f"{path}: expected one-dimensional feature columns")
    data = np.column_stack(columns).astype(np.float64, copy=False)
    if len(data) == 0 or not np.isfinite(data).all():
        raise ValueError(f"{path}: empty or nonfinite input")
    print(f"Loaded {path}: {len(data):,} events", flush=True)
    return data


def standardize(X, method):
    """The baseline's column-wise normalization; also record the applied map."""
    Xnorm = X.copy()
    offset, scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
    for j in range(X.shape[1]):
        column = X[:, j]
        if method == "scaler" or (method == "higgs" and np.min(column) < 0):
            offset[j], scale[j] = np.mean(column), np.std(column)
        elif method == "higgs" and np.max(column) > 1:
            scale[j] = np.mean(column)
        if scale[j] <= 0 or not np.isfinite(scale[j]):
            raise ValueError(f"Cannot normalize constant/invalid feature {FEATURES[j]}")
        Xnorm[:, j] = (column - offset[j]) * 1.0 / scale[j]
    return Xnorm, offset, scale


def get_logflk_config(config, weight, seed):
    from falkon.kernels import GaussianKernel
    from falkon.options import FalkonOptions
    from falkon.gsc_losses import WeightedCrossEntropyLoss

    return dict(
        kernel=GaussianKernel(sigma=config["sigma"]), M=config["M"],
        penalty_list=config["lambda"], iter_list=config["iter"], seed=seed,
        options=FalkonOptions(cg_tolerance=config["cg_tol"], keops_active="no",
                              use_cpu=config["cpu"], debug=False),
        loss=WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=config["sigma"]),
                                      neg_weight=weight),
    )


def compute_t(predictions, labels, weight):
    """Full statistic from the baseline, retaining both terms for diagnosis."""
    scores, labels = predictions.reshape(-1), labels.reshape(-1)
    reference_term = 2 * weight * np.sum(1 - np.exp(scores[labels == 0]))
    data_term = 2 * np.sum(scores[labels == 1])
    return float(reference_term + data_term), float(reference_term), float(data_term)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "local-data/DIMUON")
    parser.add_argument("--output-dir", type=Path, default=Path("results_dimuon_simple"))
    parser.add_argument("--case", choices=FILES, default="null")
    parser.add_argument("--n-toys", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Default: 0; replay uses the saved seed.")
    parser.add_argument("--n-reference", type=int, default=None, help="Default: 100000.")
    parser.add_argument("--expected-background", type=float, default=None, help="Default: 20000.")
    parser.add_argument("--signal-yield", type=float, default=None, help="Defaults: Zprime300=40, Zprime600=15.")
    parser.add_argument("--normalization", choices=("higgs", "scaler", "none"), default="higgs")
    parser.add_argument("--nystrom-centers", type=int, default=None, help="Default: 20000.")
    parser.add_argument("--sigma", type=float, default=None, help="Default: 3.")
    parser.add_argument("--penalty", type=float, default=None, help="Default: 1e-6.")
    parser.add_argument("--iterations", type=int, default=None, help="Default: 1000000.")
    parser.add_argument("--cg-tol", type=float, default=None, help="Default: sqrt(1e-7).")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", dest="cpu", action="store_true")
    device.add_argument("--gpu", dest="cpu", action="store_false")
    parser.set_defaults(cpu=None)
    parser.add_argument("--replay-null", type=Path,
                        help="Replay the first n-toys from a resampling null.npz, retaining its sampling and seeds.")
    parser.add_argument("--validate-only", action="store_true", help="Check sampled inputs without Torch/Falkon or output files.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.n_toys < 1:
        parser.error("--n-toys must be positive")
    config = dict(M=20000, sigma=3.0, **{"lambda": [1e-6], "iter": [1_000_000]},
                  cg_tol=float(np.sqrt(1e-7)), NR=20000.0, cpu=False, keops="no")
    n_ref, seed = 100000, 0
    saved_t, saved_metadata = None, None
    if args.replay_null:
        if args.case != "null" or any(v is not None for v in
                (args.seed, args.n_reference, args.expected_background, args.signal_yield)):
            parser.error("Replay requires --case null and uses saved sampling sizes and seed")
        with np.load(args.replay_null, allow_pickle=False) as saved:
            provenance = json.loads(str(saved["provenance_json"]))
            experiment = provenance["experiment"]
            if (experiment["case"] != "null" or experiment["mll_min"] is not None
                    or experiment["features"] != list(FEATURES)
                    or experiment["preprocessing"] != "legacy_pooled_per_toy_v1"
                    or not provenance["resample_nystrom"] or provenance["dtype"] != "float64"):
                parser.error("Replay requires an uncut float64 DIMUON null with fresh model seeds")
            config.update(provenance["model_config"])
            n_ref, seed = int(saved["reference_counts"][0]), provenance["seed"]
            saved_t = saved["statistics"].copy()
            saved_metadata = provenance["metadata"]
            if args.n_toys > len(saved_t) or np.any(saved["reference_counts"] != n_ref):
                parser.error("Replay needs enough saved toys with a fixed reference size")
            rng = np.random.default_rng(seed)
            # Consume ALL original seeds, even when replaying only the first few toys.
            model_seeds = rng.integers(0, 2**32 - 1, size=len(saved_t), dtype=np.uint32)
            np.testing.assert_array_equal(model_seeds, saved["model_seeds"])
    else:
        n_ref = n_ref if args.n_reference is None else args.n_reference
        seed = seed if args.seed is None else args.seed
        config["NR"] = config["NR"] if args.expected_background is None else args.expected_background
        rng = np.random.default_rng(seed)
        model_seeds = rng.integers(0, 2**32 - 1, size=args.n_toys, dtype=np.uint32)
    # Explicit model overrides permit controlled tests on the same saved events.
    for name, key in (("nystrom_centers", "M"), ("sigma", "sigma"), ("cg_tol", "cg_tol"), ("cpu", "cpu")):
        if getattr(args, name) is not None:
            config[key] = getattr(args, name)
    for name, key in (("penalty", "lambda"), ("iterations", "iter")):
        if getattr(args, name) is not None:
            config[key] = [getattr(args, name)]
    if (n_ref < 1 or config["M"] > n_ref or any(not np.isfinite(v) or v <= 0 for v in
            (config["M"], config["sigma"], config["NR"], config["cg_tol"], *config["lambda"], *config["iter"]))):
        parser.error("Sizes/model settings must be positive and finite, with M <= reference size")
    if config["keops"] != "no":
        parser.error("This baseline uses keops='no'")
    signal_yield = args.signal_yield if args.signal_yield is not None else {"null": 0, "zprime300": 40, "zprime600": 15}[args.case]
    if not np.isfinite(signal_yield) or signal_yield < 0 or (args.case == "null" and signal_yield != 0):
        parser.error("Signal yield must be nonnegative and zero for the null")

    sm = read_data(args.data_dir / FILES["null"])
    signal = None if args.case == "null" else read_data(args.data_dir / FILES[args.case])
    if saved_metadata is not None and len(sm) != experiment["reference_pool"]["selected_rows"]:
        raise ValueError("Replay SM pool has a different row count")
    weight = config["NR"] / n_ref
    print("Model configuration:", json.dumps(config, sort_keys=True), flush=True)
    print(f"Normalization={args.normalization}; weight={weight}; reference={n_ref}", flush=True)
    if not args.validate_only:
        import torch
        from falkon import LogisticFalkon

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "config.json").open("w") as output:
            json.dump(dict(model_config=config, arguments={k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()}, seed=seed, n_reference=n_ref,
                      expected_signal=signal_yield, features=FEATURES), output, indent=2)
        # Write each completed toy immediately, as in the supplied baseline.
        with (args.output_dir / "toys.csv").open("w") as output:
            csv.writer(output).writerow(["toy", "seed", "n_background", "n_signal", "t",
                                        "reference_term", "data_term", "train_seconds", "saved_t", "delta_t"])
    rows, offsets, scales = [], [], []
    for i in range(args.n_toys):
        try:
            nb = int(rng.poisson(config["NR"]))
            ns = 0 if signal is None else int(rng.poisson(signal_yield))
            if nb + ns == 0:
                raise ValueError("Empty pseudo-dataset")
            joint = rng.choice(len(sm), size=n_ref + nb, replace=False)
            X = sm[joint]
            if signal is not None:
                X = np.vstack((X, signal[rng.choice(len(signal), size=ns, replace=False)]))
            Y = np.zeros((len(X), 1), dtype=np.float64)
            Y[n_ref:] = 1
            Xnorm, offset, scale = standardize(X, args.normalization)
            if saved_metadata is not None:
                if nb != saved_metadata[i]["n_data"]:
                    raise ValueError("Replay data count differs from saved toy")
                # Verify the source sample even when testing another normalization.
                _, old_offset, old_scale = standardize(X, "higgs")
                np.testing.assert_allclose(old_offset, saved_metadata[i]["offset"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(old_scale, saved_metadata[i]["scale"], rtol=1e-12, atol=1e-12)
            model_seed = int(model_seeds[i])
            print(f"Toy {i + 1}/{args.n_toys}: seed={model_seed}, background={nb}, signal={ns}", flush=True)
            if args.validate_only:
                continue
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
            Xtorch, Ytorch = torch.from_numpy(Xnorm), torch.from_numpy(Y)
            model = LogisticFalkon(**get_logflk_config(config, weight, model_seed))
            started = time.perf_counter()
            model.fit(Xtorch, Ytorch)
            elapsed = time.perf_counter() - started
            predictions = model.predict(Xtorch).detach().cpu().numpy()
            t, reference_term, data_term = compute_t(predictions, Y, weight)
            if not np.isfinite([t, reference_term, data_term]).all():
                raise ValueError("Nonfinite statistic")
            previous = None if saved_t is None else float(saved_t[i])
            delta = None if previous is None else t - previous
            row = [i + 1, model_seed, nb, ns, t, reference_term, data_term, elapsed, previous, delta]
            with (args.output_dir / "toys.csv").open("a") as output:
                csv.writer(output).writerow(row)
            rows.append(row); offsets.append(offset); scales.append(scale)
            print(f"  t={t:.10f} = {reference_term:.10f} + {data_term:.10f}; fit={elapsed:.2f}s; delta={delta}", flush=True)
            del model, Xtorch, Ytorch, predictions
        except Exception as exc:
            raise RuntimeError(f"Toy {i + 1} failed (model seed={int(model_seeds[i])}): {exc}") from exc
    if args.validate_only:
        print("Sampling and preprocessing validated; no model fitting performed.")
        return
    results = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(args.output_dir / "results.npz", statistics=results[:, 4],
                        reference_terms=results[:, 5], data_terms=results[:, 6],
                        reference_counts=np.full(args.n_toys, n_ref), data_counts=results[:, 2:4].sum(1).astype(int),
                        model_seeds=model_seeds[:args.n_toys], offset=np.asarray(offsets), scale=np.asarray(scales))
    print(f"Mean t={results[:, 4].mean():.6f}; median t={np.median(results[:, 4]):.6f}")
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(results[:, 4], bins=min(20, max(3, args.n_toys // 3)), density=True, color="0.6")
        ax.set(xlabel="NPLM statistic", ylabel="Density", title=f"DIMUON {args.case}: direct LogisticFalkon")
        fig.tight_layout()
        fig.savefig(args.output_dir / "distribution.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
