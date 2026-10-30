"""Small synthetic checks for representative auditing and fixed decisions."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit import (MIN_CLASS_SUPPORT, choose_classes, class_scores, fit_bias,
                   paid_indices)


class FakeReader:
    def __init__(self, root):
        self.root = root

    def file(self, name):
        return self.root / name


class AuditTests(unittest.TestCase):
    def test_paid_prefix_is_verified_and_only_paid_labels_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '2020-03').mkdir()
            paid = pd.DataFrame(dict(record_id=[str(i) for i in range(10)],
                                     device_id=list(range(10))))
            random = paid.copy()
            paid.to_csv(root / '2020-03/hybrid_queries.csv', index=False)
            random.to_csv(root / '2020-03/random_queries.csv', index=False)
            query = dict(ids=np.arange(12, dtype=np.uint64),
                         y=np.arange(12, dtype=np.int64))
            selected, labels, half = paid_indices(
                FakeReader(root), FakeReader(root), '2020-03', query, 10)
            self.assertEqual(half, 5)
            np.testing.assert_array_equal(selected, np.arange(10))
            np.testing.assert_array_equal(labels, np.arange(10))
            random.loc[0, 'record_id'] = '11'
            random.to_csv(root / '2020-03/random_queries.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'uniform anchor'):
                paid_indices(FakeReader(root), FakeReader(root),
                             '2020-03', query, 10)

    def test_bias_only_moves_a_necessary_boundary(self):
        logits = np.tile(np.array([[2., 0.]]), (40, 1))
        labels = np.ones(40, dtype=np.int64)
        bias = fit_bias(logits, labels, 2)
        self.assertEqual((logits + bias).argmax(1).tolist(), [1] * 40)
        with self.assertRaisesRegex(ValueError, 'Invalid or empty'):
            fit_bias(logits, np.full(40, -1), 2)

    def test_class_choice_abstains_on_sparse_paid_class(self):
        labels = np.array([0] * MIN_CLASS_SUPPORT + [1] * (MIN_CLASS_SUPPORT - 1))
        predictions = dict(source=np.zeros(len(labels), dtype=int),
            bias_only=np.zeros(len(labels), dtype=int),
            target_only=labels.copy(), care=np.zeros(len(labels), dtype=int))
        rows = choose_classes(class_scores(labels, predictions, 2), [5, 20])
        self.assertEqual(rows[0]['chosen_method'], 'target_only')
        self.assertEqual(rows[1]['decision_reason'], 'insufficient_uniform_labels')
        self.assertFalse(rows[1]['eligible'])


if __name__ == '__main__':
    unittest.main()
