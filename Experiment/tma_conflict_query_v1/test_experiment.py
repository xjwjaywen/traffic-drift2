"""Synthetic query selection and end-to-end protocol checks."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import experiment
import run as paper
from data import finish, sha
from pilot import run_seed
from Experiment.tma_hybrid_adaptive_v1 import experiment as hybrid


class ConflictQueryTests(unittest.TestCase):
    def test_active_queries_focus_on_source_target_disagreement(self):
        features = torch.randn(80, 8)
        source = torch.tensor([[3., 0.]] * 80)
        provisional = source.clone()
        provisional[20:60] = torch.tensor([0., 3.])
        selected, info = experiment.conflict_selection(
            features, source, provisional, np.arange(10), 7)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(np.unique(selected)), 20)
        self.assertTrue(np.isin(selected[10:], np.arange(20, 60)).all())
        self.assertEqual(info['selected_disagreements'], 10)

    def test_one_month_same_budget_and_old_artifacts_unchanged(self):
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
                x = rng.normal(y[:, None] * 3, .4,
                               size=(len(y), len(paper.FEATURES))).astype(np.float32)
                ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
                np.savez_compressed(cache / (role + '.npz'), x=x, y=y, ids=ids,
                    times=np.full(len(y), pd.Timestamp('2020-01-17', tz='UTC').value))
            finish(cache, dict(synthetic=True), [role + '.npz' for role in roles])
            args = argparse.Namespace(device='cpu', epochs=1, steps=2,
                                      budget=10, rf_trees=2, threads=1)
            run_seed(cache, root / 'seed_0', args, 0)
            protocol = dict(paper.PROTOCOL, epochs=1)
            paper.run(root, device='cpu', threads=1, protocol=protocol)
            with patch.object(hybrid, 'PROTOCOL', protocol):
                hybrid.run(root, device='cpu', threads=1, months=['2020-03'])
            old = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(experiment, 'PROTOCOL', protocol), \
                    patch.object(experiment, 'MONTHS', ['2020-03']):
                result = experiment.run(root, device='cpu', threads=1)
                summary = pd.read_csv(result / 'summary.csv')
                self.assertEqual(set(summary.method),
                                 {'static_mlp', 'conflict_target', 'conflict_care'})
                queries = pd.read_csv(result / '2020-03/queries.csv')
                self.assertEqual(len(queries), 10)
                self.assertEqual(queries.record_id.nunique(), 10)
                self.assertEqual((queries.stage == 'uniform_anchor').sum(), 5)
                self.assertEqual(len(pd.read_csv(result / 'paired_comparison.csv')), 1)
                marker = sha(result / 'complete.json')
                experiment.run(root, device='cpu', threads=1)
                self.assertEqual(sha(result / 'complete.json'), marker)
            for path, checksum in old.items():
                self.assertEqual(sha(path), checksum, str(path))


if __name__ == '__main__':
    unittest.main()
