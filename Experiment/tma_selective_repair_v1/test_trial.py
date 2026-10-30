"""Synthetic checks; no real TMA data or server access."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import trial
import run as paper
from data import finish, sha
from pilot import run_seed


class PaidQueries:
    def __init__(self, path):
        self.path = path

    def file(self, name):
        assert name == '2020-01/badge_queries.csv'
        return self.path


class TrialTests(unittest.TestCase):
    def test_end_to_end_reuses_paper_run_and_preserves_inputs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            cache = root / 'cache'
            cache.mkdir()
            rng = np.random.default_rng(8)
            roles = ['train', 'val', 'history', 'reference']
            roles += [m + suffix for m in paper.MONTHS for suffix in ('_query', '_eval')]
            for j, role in enumerate(roles):
                y = np.repeat([1, 2], 64)
                if role.endswith(('_query', '_eval')):
                    y = np.r_[y, np.repeat(3, 10)]
                x = rng.normal(y[:, None] * 3, .4, size=(len(y), len(paper.FEATURES))).astype(np.float32)
                ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
                np.savez_compressed(cache / (role + '.npz'), x=x, y=y, ids=ids,
                                    times=np.full(len(y), pd.Timestamp('2020-01-17', tz='UTC').value))
            finish(cache, dict(synthetic=True), [role + '.npz' for role in roles])
            args = argparse.Namespace(device='cpu', epochs=1, steps=2, budget=10,
                                      rf_trees=2, threads=1)
            run_seed(cache, root / 'seed_0', args, 0)
            protocol = dict(paper.PROTOCOL, epochs=1)
            paper_out = paper.run(root, device='cpu', threads=1, protocol=protocol)
            original_hashes = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(trial, 'PROTOCOL', protocol):
                output = trial.run(root, device='cpu', threads=1)
                self.assertEqual(len(pd.read_csv(output / 'summary.csv')), 12)
                self.assertEqual(len(pd.read_csv(output / 'gate_classes.csv')), 8)
                for month in paper.MONTHS:
                    gate = json.loads((output / month / 'gate.json').read_text())
                    self.assertEqual(len(gate['use_repair']), 2)
                    source = np.load(paper_out / month / 'predictions.npz')
                    selected = np.load(output / month / 'predictions.npz')
                    self.assertTrue(np.array_equal(source['badge_care'], selected['badge_care']))
                    self.assertEqual(len(np.load(output / month / 'oof_predictions.npz')['paid_labels']), 10)
                digest = sha(output / 'complete.json')
                trial.run(root, device='cpu', threads=1)
                self.assertEqual(sha(output / 'complete.json'), digest)
            for path, digest in original_hashes.items():
                self.assertEqual(sha(path), digest, str(path))

    def test_only_saved_paid_queries_supply_gate_labels(self):
        ids = np.arange(30, dtype=np.uint64) + np.uint64(2**63)
        labels = np.r_[np.repeat(1, 10), np.repeat(2, 10), np.repeat(3, 10)]
        selected = [0, 10, 20, 21]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'queries.csv'
            pd.DataFrame(dict(record_id=[str(ids[i]) for i in selected],
                              device_id=labels[selected],
                              known_class=[True, True, False, False])).to_csv(path, index=False)
            actual, paid_y = trial.paid_query_indices(
                PaidQueries(path), '2020-01', dict(ids=ids, y=labels), [1, 2], 4)
            self.assertEqual(actual.tolist(), selected)
            self.assertEqual(paid_y.tolist(), [0, 1, -1, -1])

    def test_crossfit_never_trains_on_held_out_paid_queries(self):
        n = 20
        features = torch.from_numpy(np.column_stack([np.arange(n), np.ones(n)]).astype('float32'))
        source = np.zeros(n, dtype=np.int64)
        labels = np.arange(n) % 2
        labels[0] = -1
        model = SimpleNamespace(head=torch.nn.Linear(2, 2))
        seen = []

        def fake_fit(head, train_x, train_y, *args):
            seen.append(set(train_x[:, 0].numpy().astype(int).tolist()))
            self.assertTrue(np.all(train_y >= 0))
            return head, {'train_wall_s': 0.}

        with patch.object(trial, 'fit_paper', side_effect=fake_fit):
            gate, rows, folds, oof, traces = trial.fit_gate(
                model, features, source, np.arange(n), labels,
                torch.zeros((2, 2)), np.array([0, 1]), torch.zeros((2, 2)), 0, 'cpu')
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(traces), 5)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(gate), 2)
        self.assertTrue(np.all(oof >= 0))
        for fold, trained in enumerate(seen):
            self.assertFalse(trained.intersection(set(np.flatnonzero(folds == fold))))
            self.assertEqual(len(trained), int(((folds != fold) & (labels >= 0)).sum()))


if __name__ == '__main__':
    unittest.main()
