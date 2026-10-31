"""Synthetic check that a real paid-label quota stops concentrated queries."""
import unittest
from unittest.mock import patch

import numpy as np

from experiment import choose_focus, quota_select


class QuotaTests(unittest.TestCase):
    def test_stops_when_true_label_quota_reached(self):
        labels = np.ones(1200, dtype=np.int64) * 20
        labels[:6] = 5
        labels[500:520] = 5
        anchor = np.arange(500)
        queue = {0: np.arange(500, 520)}
        fake_features = np.zeros((1200, 2))
        fake_logits = np.zeros((1200, 2))
        with patch('experiment.select', side_effect=lambda method, x, logits,
                   budget, seed: np.arange(budget)):
            selected, info = quota_select(queue, anchor, labels, [5, 20],
                                          fake_features, fake_logits, 0)
        self.assertEqual(info['targeted_queries'], 9)
        self.assertEqual(info['badge_queries'], 491)
        self.assertEqual(info['true_hits_per_focus']['5'], 9)
        self.assertEqual(int((labels[selected[:509]] == 5).sum()), 15)
        self.assertEqual(len(np.unique(selected)), 1000)

    def test_focus_filters_already_well_labeled_classes(self):
        anchor_y = np.array([0] * 20 + [1] * 5 + [2] * 475)
        anchor_pred = np.array([2] * 25 + [2] * 475)
        pool_pred = np.array([2] * 1000)
        focus, _ = choose_focus(anchor_y, anchor_pred, pool_pred,
                               [5, 20, 30])
        self.assertEqual([row['device_id'] for row in focus], [20])


if __name__ == '__main__':
    unittest.main()
