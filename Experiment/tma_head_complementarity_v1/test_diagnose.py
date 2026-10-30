"""Check that a gate keyed by target predictions can expose CARE-suppressed classes."""
import unittest

import numpy as np

from diagnose import diagnose_month


class DiagnosisTests(unittest.TestCase):
    def test_target_predicted_class_can_recover_when_care_suppresses_it(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        target = np.array([0, 0, 1, 1, 2, 2])
        care = np.array([0, 0, 0, 0, 2, 2])
        score, rows = diagnose_month(y, target, care, [10, 20, 30], 0, '2020-03')
        selected = next(row for row in rows if row['target_predicted_device'] == 20)
        self.assertEqual(selected['target_correct'], 2)
        self.assertEqual(selected['care_correct'], 0)
        self.assertTrue(selected['oracle_choose_target'])
        self.assertAlmostEqual(score['oracle_routed_macro_f1'], 1.0)
        self.assertGreater(score['oracle_minus_care_pp'], 0)


if __name__ == '__main__':
    unittest.main()
