import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_complementarity import diagnose


class ComplementarityTests(unittest.TestCase):
    def test_paired_gains_and_source_flips(self):
        y = np.array([0, 1, 2, 0])
        source = np.array([0, 1, 1, 0])
        care = np.array([0, 0, 1, 0])
        rf = np.array([0, 1, 2, 1])
        score, rows = diagnose(y, source, care, rf, [5, 22, 37],
                               {5: 1, 22: 2}, '2020-03', 0)
        self.assertEqual(score['rf_only_correct'], 2)
        self.assertEqual(score['care_only_correct'], 1)
        self.assertEqual(score['added_source_flips'], 1)
        self.assertEqual(score['avoided_source_flips'], 1)
        self.assertEqual(rows[1]['paid_true_support'], 2)
        self.assertEqual(rows[2]['paid_true_support'], 0)


if __name__ == '__main__':
    unittest.main()
