"""Fast checks with mocked model fits; Falkon is not needed for this module."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import chi2, kstest, norm
import torch

from data import make_null_sampler, make_mixture_sampler, make_alternative_sampler
from stat_tests import resampling as rs
from stat_tests import comparison as cp
from stat_tests import nplm_permutation_test
from stat_tests._utils import _empirical_pvalues
from tuning import nplm_resampling_hyperparameter_scan


CONFIG = {"sigma": 0.3, "NR": 20, "lambda": [1e-6], "M": 5}


def fake_statistic(*, x, y, base_config, seed):
    """Order-sensitive deterministic fit substitute for regression snapshots."""
    return float(np.dot(x[:, 0], np.arange(1, len(x) + 1)) + seed % 1000 / 1000)


def mock_model(effect=fake_statistic):
    """Substitute only model fitting, leaving the complete toy procedure active."""
    compute = Mock(side_effect=effect)

    class Model:
        def __init__(self, config):
            self.config = config

        def compute_statistic(self, x, y, return_details=False):
            return compute(x=x, y=y, base_config=self.config, seed=self.config["seed"])

    return patch.dict(sys.modules, {"nplm": SimpleNamespace(LogFalkonNPLM=Model)})


def ensemble(statistics):
    statistics = np.asarray(statistics, dtype=float)
    n = len(statistics)
    return rs.NPLMResamplingEnsemble(
        statistics, np.full(n, 100), np.full(n, 20), [{} for _ in range(n)],
        0, np.arange(n), CONFIG.copy(), True, "float64",
    )


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.pool = np.arange(1000).reshape(-1, 1)
        self.component = np.arange(2000, 3000).reshape(-1, 1)

    def test_null_and_mixture_are_disjoint_and_do_not_mutate_pools(self):
        for sampler in (
            make_null_sampler(self.pool, n_ref=50, expected_data=20),
            make_mixture_sampler(self.pool, self.component, n_ref=50,
                                 expected_background=20, expected_component=5),
            make_mixture_sampler(self.pool, np.empty((0, 1)), n_ref=50,
                                 expected_background=20, expected_component=0),
        ):
            ref, data, metadata = sampler(np.random.default_rng(12))
            rows = np.concatenate((ref, data))[:, 0]
            self.assertEqual(len(rows), len(np.unique(rows)))
            self.assertEqual(len(ref), 50)
            self.assertEqual(len(data), metadata["n_data"])
            ref[:] = -1
            data[:] = -2
        np.testing.assert_array_equal(self.pool[:, 0], np.arange(1000))
        np.testing.assert_array_equal(self.component[:, 0], np.arange(2000, 3000))

    def test_independent_poisson_components(self):
        sampler = make_mixture_sampler(self.pool, self.component, n_ref=10,
                                       expected_background=20, expected_component=5)
        rng = np.random.default_rng(9)
        counts = np.array([
            [m["n_background"], m["n_component"]]
            for _, _, m in (sampler(rng) for _ in range(6000))
        ])
        np.testing.assert_allclose(counts.mean(axis=0), [20, 5], atol=0.2)
        np.testing.assert_allclose(counts.var(axis=0), [20, 5], atol=0.7)
        self.assertLess(abs(np.corrcoef(counts.T)[0, 1]), 0.04)
        self.assertTrue(np.any(counts[:, 1] == 0))

    def test_full_alternative_and_zero(self):
        sampler = make_alternative_sampler(self.pool, self.component, n_ref=10, expected_data=30)
        ref, data, metadata = sampler(np.random.default_rng(11))
        self.assertTrue(np.all(ref < 1000))
        self.assertTrue(np.all(data >= 2000))
        self.assertEqual(len(np.unique(data)), len(data))
        self.assertNotIn("n_background", metadata)
        self.assertNotIn("n_component", metadata)
        self.assertEqual(metadata["n_data"], len(data))
        _, empty, _ = make_null_sampler(self.pool, n_ref=10, expected_data=0)(np.random.default_rng(0))
        self.assertEqual(empty.shape, (0, 1))

    def test_validation_and_capacity(self):
        with self.assertRaisesRegex(ValueError, "feature columns"):
            make_alternative_sampler(self.pool, np.ones((100, 2)), n_ref=10, expected_data=3)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            make_null_sampler(self.pool, n_ref=1001, expected_data=3)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            make_null_sampler(self.pool, n_ref=10, expected_data=-1)
        for sampler in (
            make_null_sampler(self.pool[:10], n_ref=10, expected_data=20),
            make_mixture_sampler(self.pool, self.component[:1], n_ref=10,
                                 expected_background=20, expected_component=20),
            make_alternative_sampler(self.pool, self.component[:1], n_ref=10, expected_data=20),
        ):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                sampler(np.random.default_rng(0))


class ReconstructionTests(unittest.TestCase):
    def test_realized_sizes_fixed_nr_metadata_and_config_copy(self):
        calls = []
        metadata = {"values": []}
        def sampler(rng):
            metadata["values"].append(len(calls))
            return np.ones((12, 2)), np.ones((3 + len(calls), 2)), metadata
        class Model:
            def __init__(self, config):
                calls.append(config)
            def compute_statistic(self, x, y, return_details=False):
                config = calls[-1]
                self_outer.assertEqual(config["NR"], 20)
                self_outer.assertEqual(config["N_R"], int((y == 0).sum()))
                self_outer.assertEqual(config["N_D"], int((y == 1).sum()))
                config["lambda"].append(999)
                return 1.0
        self_outer = self
        config = dict(deepcopy(CONFIG), N_R=999, N_D=999)
        with patch.dict(sys.modules, {"nplm": SimpleNamespace(LogFalkonNPLM=Model)}):
            result = rs.nplm_resampling_alternative(sampler, config, n_alternative=3)
        np.testing.assert_array_equal(result.reference_counts, [12, 12, 12])
        np.testing.assert_array_equal(result.data_counts, [3, 4, 5])
        self.assertEqual([m["values"] for m in result.metadata], [[0], [0, 1], [0, 1, 2]])
        self.assertEqual(config["lambda"], [1e-6])
        config["lambda"].append(5)
        self.assertEqual(result.model_config["lambda"], [1e-6])

    def test_reproducibility_and_nystrom_modes(self):
        sampler = make_null_sampler(np.arange(1000), n_ref=30, expected_data=20)
        with mock_model():
            for resample in (True, False):
                kwargs = dict(n_null=5, seed=45, resample_nystrom=resample)
                a = rs.nplm_resampling_null(sampler, CONFIG, **kwargs)
                b = rs.nplm_resampling_null(sampler, CONFIG, **kwargs)
                np.testing.assert_array_equal(a.statistics, b.statistics)
                np.testing.assert_array_equal(a.model_seeds, b.model_seeds)
                self.assertEqual(len(np.unique(a.model_seeds)), 5 if resample else 1)
                self.assertEqual(a.dtype, "float64")
                self.assertEqual(a.seed, 45)

    def test_rng_restored_on_success_and_every_failure(self):
        def disturb():
            np.random.seed(2)
            torch.manual_seed(2)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(2)
        def sampler(rng):
            disturb()
            return np.ones(10), np.ones(4), {}
        def bad_sampler(rng):
            disturb()
            raise ValueError("sampling error")
        def fit(**kwargs):
            disturb()
            return 2.0
        for callback, effect, failure in (
            (sampler, fit, None), (bad_sampler, fit, "sampling error"),
            (sampler, ValueError("fit error"), "fit error"),
            (sampler, lambda **kw: np.nan, "nonfinite"),
            (make_null_sampler(np.arange(100), n_ref=10, expected_data=0), fit, "empty pseudo-dataset"),
        ):
            np_before = np.random.get_state()
            torch_before = torch.random.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            with mock_model(effect):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, f"null toy 1 failed \(model seed=[0-9]+\):.*{failure}"):
                        rs.nplm_resampling_null(callback, CONFIG, n_null=2)
                else:
                    rs.nplm_resampling_null(callback, CONFIG, n_null=2)
            np_after = np.random.get_state()
            np.testing.assert_array_equal(np_before[1], np_after[1])
            self.assertEqual(np_before[2:], np_after[2:])
            self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
            if cuda_before is not None:
                for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
                    self.assertTrue(torch.equal(before, after))


class ComparisonTests(unittest.TestCase):
    def test_import_and_comparison_without_fitting_dependencies(self):
        code = '''
import sys
from types import SimpleNamespace

class BlockFittingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'falkon', 'nplm'}:
            raise AssertionError('Unexpected fitting dependency: ' + fullname)

sys.meta_path.insert(0, BlockFittingImports())
from stat_tests import compare_nplm_results, NPLMResamplingEnsemble
result = compare_nplm_results(
    SimpleNamespace(statistics=[1, 2, 3, 4]),
    {'alternative': SimpleNamespace(statistics=[2, 3, 4])},
    fit_chi2=True,
)
assert result.n_null == 4
assert result.chi2_fit.n_bootstrap == 999
assert 'torch' not in sys.modules and 'falkon' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_pvalues_scalar_array_and_invalid_values(self):
        self.assertEqual(_empirical_pvalues([-2, -1, 0, 1], 0), 3 / 5)
        np.testing.assert_array_equal(
            _empirical_pvalues([-2, -1, 0, 1], [-3, 0, 2]), [1, 3 / 5, 1 / 5]
        )
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                _empirical_pvalues([0, 1], invalid)
            with self.assertRaises(ValueError):
                _empirical_pvalues([0, invalid], 1)

    def test_empirical_quantiles_ties_and_resolution(self):
        null = ensemble([0, 1, 2, 3])
        alt = ensemble([1, 2, 3, 8])
        with mock_model(AssertionError("must not fit")):
            result = cp.compare_nplm_results(null, {"a": alt, "b": ensemble([100, 100])})
        a = result.alternatives["a"]
        q = np.quantile(alt.statistics, [0.16, 0.5, 0.84])
        p = [(1 + np.sum(null.statistics >= value)) / 5 for value in q]
        np.testing.assert_array_equal(a.t_quantiles, q)
        np.testing.assert_array_equal(a.p_values, p)
        np.testing.assert_array_equal(a.z_scores, norm.isf(p))
        np.testing.assert_array_equal(_empirical_pvalues(null.statistics, [2]), [3 / 5])
        np.testing.assert_array_equal(result.alternatives["b"].p_values, [0.2] * 3)
        self.assertEqual(result.z_score_resolution, norm.isf(0.2))
        self.assertIsNone(a.chi2_z_scores)

    def test_chi2_acceptance_rejection_and_shared_fit(self):
        values = chi2.ppf((np.arange(300) + 0.5) / 300, df=5)
        with patch.object(cp, "_fit_chi2", wraps=cp._fit_chi2) as fit:
            result = cp.compare_nplm_results(ensemble(values), {"a": ensemble([5, 8]), "b": ensemble([6, 9])}, fit_chi2=True)
            self.assertEqual(fit.call_count, 1)
        self.assertTrue(result.chi2_fit.accepted)
        self.assertEqual(result.chi2_fit.n_bootstrap, 999)
        self.assertEqual(result.chi2_fit.dof, values.mean())
        self.assertIsNotNone(result.alternatives["a"].chi2_z_scores)
        again = cp.compare_nplm_results(ensemble(values), {}, fit_chi2=True)
        self.assertEqual(result.chi2_fit, again.chi2_fit)
        for values in (np.full(100, 5), [-1, 2], [0, 0], [1]):
            rejected = cp.compare_nplm_results(ensemble(values), {"a": ensemble([3, 5])}, fit_chi2=True)
            self.assertFalse(rejected.chi2_fit.accepted)
            self.assertIsNone(rejected.alternatives["a"].chi2_p_values)
            self.assertIsNotNone(rejected.alternatives["a"].p_values)

    def test_bootstrap_refits_mean_each_time(self):
        values = np.random.default_rng(15).chisquare(4, size=40)
        fit = cp._fit_chi2(values, 42)
        rng = np.random.default_rng(42)
        observed = kstest(values, "chi2", args=(values.mean(),)).statistic
        count = 0
        for _ in range(999):
            toy = rng.chisquare(values.mean(), size=len(values))
            count += kstest(toy, "chi2", args=(toy.mean(),)).statistic >= observed
        self.assertAlmostEqual(fit.ks_statistic, observed)
        self.assertEqual(fit.p_value, (1 + count) / 1000)

    def test_invalid_statistics(self):
        for values in ([], [np.nan], [np.inf]):
            with self.assertRaisesRegex(ValueError, "finite 1D"):
                cp.compare_nplm_results(ensemble(values), {})


class LegacyTests(unittest.TestCase):
    def test_separate_null_preserves_combined_sampling_sequence(self):
        pool = np.arange(1000)
        for resample in (True, False):
            options = dict(seed=12, resample_nystrom=resample)
            with mock_model():
                separate = rs.nplm_resampling_null(
                    make_null_sampler(pool, n_ref=30, expected_data=20),
                    CONFIG, n_null=3, **options,
                )
                combined = rs.nplm_resampling_test(
                    pool, pool, CONFIG, n_ref=30, n_data=20, n_null=3,
                    n_alternative=0, poisson_fluctuate_n_data=True,
                    return_null=True, **options,
                )
            np.testing.assert_array_equal(separate.statistics, combined.null_statistics)
            np.testing.assert_array_equal(separate.data_counts, combined.null_data_counts)
            np.testing.assert_array_equal(separate.model_seeds, combined.null_seeds)

    def test_frozen_legacy_results(self):
        fixtures = json.loads(Path(__file__).with_name("resampling_legacy.json").read_text())
        with mock_model():
            for case in fixtures:
                with self.subTest(options=case["options"]):
                    result = rs.nplm_resampling_test(
                        np.arange(1000), np.arange(2000, 2000 + case["pool_size"]),
                        CONFIG, n_ref=30, n_data=20, n_null=3, seed=123,
                        return_null=True, return_alternative=True, **case["options"],
                    )
                    for key, value in case["expected"].items():
                        if isinstance(value, list):
                            np.testing.assert_array_equal(getattr(result, key), value)
                        else:
                            self.assertEqual(getattr(result, key), value)

    def test_flags_warnings_and_zero_count_correction(self):
        with mock_model():
            with self.assertWarnsRegex(UserWarning, "n_alternative is ignored"):
                result = rs.nplm_resampling_test(
                    np.arange(1000), np.arange(20), CONFIG, n_ref=30, n_data=20,
                    n_null=2, n_alternative=5, return_observed=False,
                )
            self.assertIsNone(result.t_obs)
            self.assertIsNone(result.null_statistics)
            self.assertIsNone(result.alternative_statistics)
            self.assertIsNotNone(result.p_value)
            self.assertEqual(result.observed_data_count, 20)
            with self.assertWarnsRegex(UserWarning, "Sample pool factor"):
                rs.nplm_resampling_test(np.arange(50), np.arange(20), CONFIG,
                                        n_ref=30, n_data=20, n_null=1)
            # Seed 0 gives a genuine zero after drawing one model seed.
            with self.assertRaisesRegex(RuntimeError, "null toy 1.*empty pseudo-dataset"):
                rs.nplm_resampling_test(np.arange(1000), np.arange(1000), CONFIG,
                    n_ref=30, n_data=1, n_null=1, n_alternative=0,
                    poisson_fluctuate_n_data=True, seed=0)

    def test_paired_tuning_toys(self):
        with mock_model():
            result = nplm_resampling_hyperparameter_scan(
                np.arange(1000), CONFIG, lambda_values=[1e-6, 1e-5], m_values=[5, 10],
                n_ref=30, n_data=20, n_trials=3, seed=12, poisson_fluctuate_n_data=True,
            )
        rows = result.raw_results
        self.assertEqual(len(rows), 12)
        for offset in (3, 6, 9):
            for left, right in zip(rows[:3], rows[offset:offset + 3]):
                for key in ("t_nplm", "n_data", "model_seed"):
                    self.assertEqual(left[key], right[key])


class PermutationTests(unittest.TestCase):
    def test_seeded_results_match_original_permutation_code(self):
        # Captured from the original permutation implementation with fake_statistic.
        expected = {
            True: ([27219.178, 27394.274, 27333.555, 23725.382],
                   [1805513178, 1531970274, 3976578555, 1627954382]),
            False: ([26362.021, 27394.021, 27333.021, 23725.021], [2529908021] * 4),
        }
        with mock_model():
            for resample, (statistics, seeds) in expected.items():
                result = nplm_permutation_test(
                    np.arange(30), np.arange(20, 40), CONFIG, n_permutations=4,
                    seed=19, resample_nystrom=resample, return_null=True,
                )
                np.testing.assert_array_equal(result.null_statistics, statistics)
                np.testing.assert_array_equal(result.permutation_seeds, seeds)
                self.assertEqual(result.observed_seed, 2529908021)
                self.assertEqual(result.t_obs, 33550.021)
                self.assertEqual(result.p_value, 0.2)
                self.assertEqual(result.z_score, norm.isf(0.2))

    def test_failure_identification_and_rng_restoration(self):
        for bad_value in (np.nan, np.inf, -np.inf, ValueError("optimizer failed")):
            for failure_index, label in ((0, "observed fit"), (2, "permutation 2")):
                calls = []
                def fit(**kwargs):
                    calls.append(kwargs)
                    np.random.seed(3)
                    torch.manual_seed(3)
                    if len(calls) - 1 == failure_index:
                        if isinstance(bad_value, Exception):
                            raise bad_value
                        return bad_value
                    return -2.0  # A finite negative statistic must remain valid.
                np_before = np.random.get_state()
                torch_before = torch.random.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                with mock_model(fit), self.assertRaisesRegex(
                    RuntimeError, rf"{label} failed \(model seed=[0-9]+\)"
                ):
                    nplm_permutation_test(np.arange(30), np.arange(20), CONFIG, n_permutations=4)
                self.assertEqual(len(calls), failure_index + 1)
                np_after = np.random.get_state()
                np.testing.assert_array_equal(np_before[1], np_after[1])
                self.assertEqual(np_before[2:], np_after[2:])
                self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
                if cuda_before is not None:
                    for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
                        self.assertTrue(torch.equal(before, after))

    def test_finite_negative_statistics_are_calibrated(self):
        with mock_model([-2, -3, -2, -1]):
            result = nplm_permutation_test(np.arange(30), np.arange(20), CONFIG, n_permutations=3)
        self.assertEqual(result.p_value, 0.75)
        self.assertEqual(result.n_extreme, 2)


class ExampleTests(unittest.TestCase):
    def test_cli_and_saved_provenance_with_mocked_fits(self):
        from examples import gaussian_1d_generator as example
        with tempfile.TemporaryDirectory() as directory:
            for signal in (0, 5):
                argv = ["gaussian_1d_generator.py", "--n-reference", "40",
                        "--expected-background", "20", "--expected-signal", str(signal),
                        "--n-null", "3", "--n-alt", "3", "--progress-every", "0",
                        "--no-plot", "--output-dir", directory]
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                        mock_model():
                    example.main()
                path = Path(directory) / "gaussian_1d_generator_results.npz"
                with np.load(path, allow_pickle=False) as saved:
                    provenance = json.loads(str(saved["provenance_json"]))
                    self.assertNotEqual(provenance["null"]["seed"], provenance["alternative"]["seed"])
                    self.assertEqual(provenance["alternative"]["model_config"]["NR"], 20)
                    self.assertTrue(np.all(saved["null_signal_counts"] == 0))
                    np.testing.assert_array_equal(
                        saved["alt_data_counts"],
                        saved["alt_background_counts"] + saved["alt_signal_counts"],
                    )
                    if signal == 0:
                        self.assertTrue(np.all(saved["alt_signal_counts"] == 0))


if __name__ == "__main__":
    unittest.main()
