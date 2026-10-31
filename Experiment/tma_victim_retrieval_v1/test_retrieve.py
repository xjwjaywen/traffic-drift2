"""Synthetic checks of nomination, retrieval, and fixed query budgets."""
import unittest

import numpy as np

from retrieve import (candidate_queues, nominate_classes, normalized_features,
                      select_from_queues)


class RetrievalTests(unittest.TestCase):
    def test_nomination_uses_missing_predicted_mass(self):
        anchor_y = np.array([0, 0, 0, 1, 1, 1])
        anchor_pred = np.ones(6, dtype=int)
        pool_pred = np.ones(30, dtype=int)
        focus, rows = nominate_classes(anchor_y, anchor_pred, pool_pred,
                                       [5, 20], max_focus=1)
        self.assertEqual([row['device_id'] for row in focus], [5])
        self.assertEqual(focus[0]['absorbers'], [1])
        self.assertFalse(rows[1]['selected_focus'])

    def test_nearest_retrieves_victim_region_without_pool_labels(self):
        # Candidates 2-6 resemble paid class 0; candidates 7-11 resemble
        # paid true absorber class 1. All have source prediction 1.
        raw = np.array([[2., 0.], [-2., 0.]] +
                       [[2. + .01 * i, .01] for i in range(5)] +
                       [[-2. - .01 * i, -.01] for i in range(5)])
        z = normalized_features(raw)
        pred = np.ones(len(raw), dtype=int)
        anchor = np.array([0, 1])
        paid_y = np.array([0, 1])
        focus = [dict(class_index=0, absorbers=[1])]
        queues = candidate_queues(z, pred, anchor, paid_y, focus,
                                  'nearest', seed=0)
        chosen, info = select_from_queues(queues, anchor, len(raw), 5, seed=0)
        np.testing.assert_array_equal(chosen[:2], anchor)
        self.assertEqual(set(chosen[2:]), set(range(2, 7)))
        self.assertEqual(info['fallback_uniform'], 0)

    def test_region_budget_is_exact_when_focus_absent(self):
        chosen, info = select_from_queues({}, np.array([1, 3]), 12, 5, seed=2)
        self.assertEqual(len(chosen), 7)
        self.assertEqual(len(np.unique(chosen)), 7)
        self.assertEqual(info['fallback_uniform'], 5)


if __name__ == '__main__':
    unittest.main()
