import unittest

import numpy as np

from gate import fold_assignments, learn_gate, probability_of_net_gain, route


class GateTests(unittest.TestCase):
    def test_cross_folds_are_balanced_and_deterministic(self):
        a = fold_assignments(1000, 5, 17)
        self.assertTrue(np.array_equal(a, fold_assignments(1000, 5, 17)))
        self.assertEqual(np.bincount(a).tolist(), [200] * 5)

    def test_beta_rule_and_unknown_labels(self):
        self.assertAlmostEqual(probability_of_net_gain(3, 0), .9375)
        self.assertAlmostEqual(probability_of_net_gain(0, 3), .0625)
        # Class 0 gets three verified corrections, no lost correct cases.
        labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1])
        source = np.zeros(len(labels), dtype=int)
        repaired = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        gate, rows = learn_gate(labels, source, repaired, 2)
        self.assertEqual(gate.tolist(), [True, False])
        self.assertEqual(rows[0]['paid_known_support'], 10)
        self.assertEqual((rows[0]['gain_count'], rows[0]['loss_count']), (3, 0))

    def test_route_uses_observable_source_class(self):
        source = np.array([0, 0, 1, 1])
        repaired = np.array([1, 1, 0, 0])
        self.assertEqual(route(source, repaired, [True, False]).tolist(), [1, 1, 1, 1])

    def test_sparse_or_mixed_evidence_keeps_source(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        source = np.zeros(10, dtype=int)
        repaired = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        gate, rows = learn_gate(labels, source, repaired, 2)
        self.assertFalse(gate[0])
        self.assertEqual((rows[0]['gain_count'], rows[0]['loss_count']), (3, 1))


if __name__ == '__main__':
    unittest.main()
