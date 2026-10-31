import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import chronological_split, sampled_train_indices, score


class RepresentationOracleTests(unittest.TestCase):
    def test_chronological_boundary_and_outside_rejected(self):
        times = pd.to_datetime(['2020-03-01', '2020-03-10', '2020-03-11',
                                '2020-03-14'], utc=True).astype('int64').to_numpy()
        train, hold = chronological_split(times, '2020-03')
        np.testing.assert_array_equal(train, [True, True, False, False])
        np.testing.assert_array_equal(hold, ~train)
        with self.assertRaises(ValueError):
            chronological_split(np.append(times, pd.Timestamp('2020-03-16', tz='UTC').value),
                                '2020-03')

    def test_same_deterministic_training_ids_for_both_spaces(self):
        labels = np.array([20] * 12 + [1] * 1000)
        mask = np.ones(len(labels), dtype=bool)
        first = sampled_train_indices(labels, mask, 20)
        second = sampled_train_indices(labels, mask, 20)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int((labels[first] == 20).sum()), 12)
        self.assertEqual(int((labels[first] != 20).sum()), 240)
        self.assertIsNone(sampled_train_indices(labels, mask, 22))

    def test_ap_baseline_is_prevalence(self):
        result = score(np.array([0, 0, 0, 1]), np.ones(4))
        self.assertAlmostEqual(result['ap'], .25)
        self.assertAlmostEqual(result['prevalence'], .25)


if __name__ == '__main__':
    unittest.main()
