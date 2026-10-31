"""Synthetic checks for chronological memory diagnosis and paid-label scope."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import audit


class TemporalMemoryAuditTests(unittest.TestCase):
    def test_old_neighbors_transfer_without_inventing_missing_class_evidence(self):
        history_x = np.array([[1., 0.], [0., 1.], [.95, .05]])
        history_y = np.array([0, 1, 0])
        anchor_x = np.array([[.9, .1], [.1, .9], [1., 1.]])
        anchor_y = np.array([0, 1, 2])
        rows = audit.transfer_rows(history_x, history_y, anchor_x,
                                   anchor_y, [5, 20, 37])
        self.assertEqual(rows[0]['top1_old_neighbor_rate'], 1.)
        self.assertEqual(rows[1]['top5_contains_true_rate'], 1.)
        self.assertAlmostEqual(rows[0]['random_memory_top1_rate'], 2 / 3)
        self.assertEqual(rows[2]['anchor_support'], 1)
        self.assertEqual(rows[2]['previous_paid_support'], 0)
        self.assertTrue(np.isnan(rows[2]['top1_old_neighbor_rate']))

    def test_loader_only_requests_query_pool_and_validates_paid_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = np.arange(1001)
            labels = np.arange(1001) % 2
            labels[-1] = 999  # Unqueried class must never enter the output.
            features = np.c_[np.ones(1001), np.arange(1001)]
            hybrid = pd.DataFrame(dict(record_id=ids[:1000].astype(str),
                                       device_id=labels[:1000], known_class=True))
            random = pd.DataFrame(dict(record_id=ids[:1000].astype(str)))
            hybrid.to_csv(root / 'hybrid.csv', index=False)
            random.to_csv(root / 'random.csv', index=False)

            class FakeReader:
                def __init__(self, name):
                    self.name = name

                def file(self, requested):
                    if self.name == 'hybrid' and requested.endswith('hybrid_queries.csv'):
                        return root / 'hybrid.csv'
                    if self.name == 'paper' and requested.endswith('random_queries.csv'):
                        return root / 'random.csv'
                    raise AssertionError(requested)

            class FakeCache:
                def arrays(self, requested, keys):
                    self.requested = requested
                    self.assert_keys = keys
                    return dict(ids=ids, x=features, y=labels)

            cache = FakeCache()
            prep = SimpleNamespace(transform=lambda x: x)
            model = SimpleNamespace(encoder=object())
            with patch.object(audit, 'forward_batches',
                              side_effect=lambda module, x, device: torch.as_tensor(x)):
                paid = audit.load_paid(cache, FakeReader('paper'), FakeReader('hybrid'),
                                       '2020-03', model, prep, 'cpu', [0, 1])
            self.assertEqual(cache.requested, '2020-03_query.npz')
            self.assertEqual(cache.assert_keys, ['ids', 'x', 'y'])
            self.assertEqual(len(paid['labels']), 1000)
            self.assertTrue((paid['labels'] >= 0).all())


if __name__ == '__main__':
    unittest.main()
