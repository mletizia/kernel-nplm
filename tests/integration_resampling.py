"""Run explicitly in a Falkon-capable environment; no model mocks or skips.

From the repository root: python -m unittest discover -s tests -p 'integration_*.py' -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.gaussian_1d_generator import make_sampler, save_results, RunSummary
from stat_tests import (
    nplm_resampling_null, nplm_resampling_alternative, compare_nplm_results,
)


class FalkonIntegrationTests(unittest.TestCase):
    def test_one_null_reused_with_zero_and_nonzero_signal(self):
        config = {
            "sigma": 0.3, "NR": 80, "M": 20, "lambda": [1e-5],
            "iter": [1000], "cg_tol": 1e-5, "cpu": True,
            "keops": "no", "verbose": 0,
        }
        null = nplm_resampling_null(
            make_sampler(n_reference=400, expected_background=80, expected_signal=0),
            config, n_null=4, seed=123,
        )
        alternatives = {
            name: nplm_resampling_alternative(
                make_sampler(n_reference=400, expected_background=80, expected_signal=signal),
                config, n_alternative=4, seed=seed,
            )
            for name, signal, seed in (("zero", 0, 456), ("signal", 10, 789))
        }
        null_before = null.statistics.copy()
        result = compare_nplm_results(null, alternatives, fit_chi2=True)
        for name, alternative in alternatives.items():
            self.assertTrue(np.all(np.isfinite(alternative.statistics)))
            self.assertTrue(np.all(np.isfinite(result.alternatives[name].p_values)))
            self.assertTrue(np.all(alternative.reference_counts == 400))
        self.assertTrue(np.all(np.isfinite(null.statistics)))
        np.testing.assert_array_equal(null.statistics, null_before)
        self.assertEqual(set(result.alternatives), {"zero", "signal"})

        summary = RunSummary(0, 400, 80, 10, 4, 4, 0.3, 20, 1e-5, 1000, 1e-5, True, "no")
        with tempfile.TemporaryDirectory() as directory:
            path = save_results(output_dir=Path(directory), summary=summary,
                                null_result=null, alt_result=alternatives["signal"])
            with np.load(path, allow_pickle=False) as saved:
                expected = {
                    "t_null", "t_alt", "null_background_counts", "null_signal_counts",
                    "alt_background_counts", "alt_signal_counts", "null_model_seeds",
                    "alt_model_seeds", "summary_json", "provenance_json",
                }
                self.assertTrue(expected.issubset(saved.files))
                provenance = json.loads(str(saved["provenance_json"]))
                self.assertEqual(provenance["null"]["seed"], 123)
                self.assertEqual(provenance["alternative"]["model_config"]["NR"], 80)
        print("Finite Falkon statistics; one null reused for zero and nonzero signal.")


if __name__ == "__main__":
    unittest.main()
