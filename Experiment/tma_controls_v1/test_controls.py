"""Small synthetic model/cache fixtures only; never reads real TMA data."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

import controls
from data import finish, sha
from pilot import run_seed


class ControlTests(unittest.TestCase):
    def test_hybrid_budget_and_random_prefix(self):
        torch.manual_seed(4)
        features = torch.randn(120, 64)
        logits = torch.randn(120, 3)
        original = np.random.default_rng(0).choice(120, 10, replace=False)
        a = controls.hybrid_selection(features, logits, original, 0)
        b = controls.hybrid_selection(features, logits, original, 0)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(np.unique(a)), 10)
        np.testing.assert_equal(a[:5], original[:5])
        np.testing.assert_equal(a, b)

    def test_completed_source_reuse_and_new_controls(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); cache = root / 'cache'; cache.mkdir()
            roles = ['train', 'val', 'history', 'reference']
            roles += [m + suffix for m in controls.MONTHS for suffix in ['_query', '_eval']]
            rng = np.random.default_rng(8)
            files = []
            for j, role in enumerate(roles):
                y = np.repeat([1, 2], 64)
                if role.endswith(('_query', '_eval')):
                    y = np.concatenate([y, np.repeat(3, 10)])
                x = rng.normal(y[:, None] * 3, .4, size=(len(y), len(controls.FEATURES))).astype(np.float32)
                x[::5, 0] = np.nan
                ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
                name = role + '.npz'; files.append(name)
                day = role[:7] + '-02' if role.endswith('_query') else '2019-12-17'
                times = np.full(len(y), pd.Timestamp(day, tz='UTC').value, dtype=np.int64)
                np.savez_compressed(cache / name, x=x, y=y, ids=ids, times=times)
            finish(cache, dict(synthetic=True), files)
            args = argparse.Namespace(device='cpu', epochs=1, steps=2, budget=10, rf_trees=2, threads=1)
            run_seed(cache, root / 'seed_0', args, 0)
            original_digest = sha(root / 'seed_0/complete.json')
            source_digest = sha(root / 'seed_0/source.pt')
            out = root / 'controls_v1/seed_0'
            controls.run(root, out, 0, 'cpu', 1)
            result = pd.read_csv(out / 'summary.csv')
            self.assertEqual(len(result), 52)
            new = result[result.method.isin(['random_kd', 'badge_replay', 'badge_kd', 'hybrid_target', 'hybrid_care'])]
            self.assertEqual(len(new), 20)
            self.assertTrue((new.label_queries == 10).all())
            self.assertEqual(sha(root / 'seed_0/complete.json'), original_digest)
            self.assertEqual(sha(root / 'seed_0/source.pt'), source_digest)
            for month in controls.MONTHS:
                kd = json.loads((out / month / 'random_kd_trace.json').read_text())
                base = json.loads((root / 'seed_0' / month / 'random_care_trace.json').read_text())
                self.assertEqual(kd['target_stream_sha256'], base['target_stream_sha256'])
                self.assertEqual(kd['reference_ce_presentations'], 0)
                self.assertEqual(kd['reference_kd_presentations'], base['reference_kd_presentations'])
                replay = json.loads((out / month / 'badge_replay_trace.json').read_text())
                self.assertGreater(replay['reference_ce_presentations'], 0)
                self.assertEqual(replay['reference_kd_presentations'], 0)
                hybrid = pd.read_csv(out / month / 'hybrid_queries.csv', dtype={'record_id': str})
                baseline = pd.read_csv(root / 'seed_0' / month / 'random_queries.csv', dtype={'record_id': str})
                self.assertEqual(hybrid.record_id.iloc[:5].tolist(), baseline.record_id.iloc[:5].tolist())
            finished = sha(out / 'complete.json')
            controls.run(root, out, 0, 'cpu', 1)
            self.assertEqual(sha(out / 'complete.json'), finished)


if __name__ == '__main__':
    unittest.main()
