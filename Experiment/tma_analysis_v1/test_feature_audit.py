"""Synthetic feature-collision and saved-cache fixtures only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import feature_audit as audit
from test_analyze import fixture, mark


class FeatureAuditTests(unittest.TestCase):
    def test_nan_zero_collisions_and_majority_bound(self):
        x = np.array([[0., np.nan], [-0., np.nan], [0., np.nan], [1., 2.]], dtype=np.float32)
        y = np.array([1, 1, 2, 2])
        summary, mixed = audit.collision_stats(x, y)
        self.assertEqual(summary['unique_vectors'], 2)
        self.assertEqual(summary['conflicting_rows'], 3)
        self.assertEqual(summary['empirical_accuracy_ceiling'], .75)
        self.assertEqual(mixed.tolist(), [True, True, True, False])

    def test_timestamp_units_and_clipping_collisions(self):
        value = pd.Timestamp('2020-03-17', tz='UTC').value
        self.assertEqual(audit.timestamps(np.array([value]))[0], 'ns')
        self.assertEqual(audit.timestamps(np.array([value // 1000]))[0], 'us')
        with self.assertRaises(ValueError):
            audit.timestamps(np.array([0]))
        prep = audit.Preprocessor().fit(np.array([[0.], [1.]], dtype=np.float32))
        x = np.array([[1e10], [1e20]], dtype=np.float32)
        self.assertEqual(len(np.unique(audit.keys(x))), 2)
        self.assertEqual(len(np.unique(audit.keys(prep.transform(x)))), 1)

    def test_report_from_completed_cache_without_training(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            cache, run = root / 'cache', root / 'seed_0'
            n = len(audit.FEATURES)
            for path in list(cache.glob('*.npz')):
                with np.load(path) as f:
                    part = {k: f[k] for k in f.files}
                month, role = path.stem.split('_')
                stamp = pd.Timestamp(month + ('-02' if role == 'query' else '-17'), tz='UTC').value
                part.update(x=np.zeros((len(part['y']), n), dtype=np.float32), times=np.full(len(part['y']), stamp, dtype=np.int64))
                np.savez(path, **part)
            hy = np.repeat([1, 2, 3], 60)
            np.savez(cache / 'history.npz', ids=np.arange(len(hy)), y=hy,
                     x=np.zeros((len(hy), n), dtype=np.float32),
                     times=np.full(len(hy), pd.Timestamp('2019-12-17', tz='UTC').value, dtype=np.int64))
            np.savez(run / 'preprocessor.npz', median=np.zeros(n), mean=np.zeros(n), std=np.ones(n))
            config = dict(features=audit.FEATURES)
            (cache / 'config.json').write_text(json.dumps(config))
            mark(cache, config)
            mark(run, dict(cache_sha256=audit.sha(cache / 'complete.json')))
            before = audit.sha(run / 'complete.json')
            out = audit.audit(root)
            modes = json.loads((out / 'dominant_vectors.json').read_text())
            self.assertEqual(len(modes), 4)
            self.assertEqual(modes[0]['labels_at_same_raw16_vector'], {'1': 60, '2': 60})
            self.assertEqual(modes[0]['target_mode_fraction'], 1)
            bounds = pd.read_csv(out / 'collision_bounds.csv')
            target = bounds[bounds.role.str.endswith('_eval')]
            self.assertTrue((target.empirical_accuracy_ceiling == .5).all())
            self.assertEqual(before, audit.sha(run / 'complete.json'))


if __name__ == '__main__':
    unittest.main()
