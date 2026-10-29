"""Artificial saved-output fixtures. No raw TMA data, training or network."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tma_pilot_v1'))
from pilot import evaluate, stats
from data import implementation


def mark(root, signature):
    files = {str(p.relative_to(root)): analyze.sha(p) for p in root.rglob('*')
             if p.is_file() and p.name != 'complete.json'}
    (root / 'complete.json').write_text(json.dumps(dict(signature=signature, artifacts=files)))


def fixture(root):
    cache, run = root / 'cache', root / 'seed_0'
    cache.mkdir(); run.mkdir()
    classes = [1, 2, 3]
    (run / 'run_info.json').write_text(json.dumps(dict(classes=classes)))
    hy = np.repeat([0, 1, 2], 60)
    np.savez(run / 'history_predictions.npz', labels=hy, mlp=hy, rf=hy)
    hist = stats(hy, hy, 3)
    summaries, rows = [], []
    for month_i, month in enumerate(analyze.MONTHS):
        part = run / month; part.mkdir()
        # Third known class absent from the evaluation; one unknown query label.
        ey = np.repeat([0, 1], 60)
        ids = np.arange(120, dtype=np.uint64) + month_i * 1000
        before = np.ones(120, dtype=np.int64)
        after = before.copy(); after[:10] = 0; after[-10:] = 0
        np.savez(cache / (month + '_eval.npz'), ids=ids, y=ey + 1)
        qids = np.array([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64) + month_i * 100
        qy = np.array([1, 2, 99])
        np.savez(cache / (month + '_query.npz'), ids=qids, y=qy)
        for selector in ['random', 'margin', 'badge']:
            pd.DataFrame([dict(record_id=str(qids[i]), device_id=int(qy[i]), known_class=i != 2)
                          for i in [0, 2]]).to_csv(part / (selector + '_queries.csv'), index=False)
        predictions = dict(ids=ids, labels=ey, device_ids=ey + 1)
        for method in analyze.METHODS:
            pred = before if method == 'static_mlp' else ey if method == 'static_rf' else after
            predictions[method] = pred
            baseline = ey if method == 'static_rf' else before
            summary, per_class, cm = evaluate(ey, baseline, pred, hist, classes)
            summaries.append(dict(seed=0, month=month, method=method,
                label_queries=0 if method.startswith('static_') else 2,
                usable_known_labels=0 if method.startswith('static_') else 1, **summary))
            rows.extend(dict(seed=0, month=month, method=method, **r) for r in per_class)
            np.save(part / (method + '_confusion.npy'), cm)
        np.savez(part / 'predictions.npz', **predictions)
    pd.DataFrame(summaries).to_csv(run / 'summary.csv', index=False)
    pd.DataFrame(rows).to_csv(run / 'per_class.csv', index=False)
    mark(cache, {})
    mark(run, dict(cache_sha256=analyze.sha(cache / 'complete.json')))


class AnalysisTests(unittest.TestCase):
    def test_completed_report_and_cache_compatibility(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            source_hashes = implementation()
            before = analyze.sha(root / 'seed_0/complete.json')
            output = analyze.analyze(root)
            details = pd.read_csv(output / 'collapse_details.csv')
            self.assertEqual(set(details.device_id), {1})
            self.assertNotIn('static_rf', set(details.method))
            self.assertEqual(len(details), 28)
            q = pd.read_csv(output / 'query_coverage.csv')
            missing = q[q.device_id == 3]
            self.assertTrue((missing.candidates == 0).all())
            self.assertTrue((missing.queried == 0).all())
            self.assertTrue((q[q.device_id == 99].queried == 1).all())
            self.assertEqual(source_hashes, implementation())
            self.assertEqual(before, analyze.sha(root / 'seed_0/complete.json'))

    def test_reject_modified_and_inconsistent_metrics(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            path = root / 'seed_0/summary.csv'
            table = pd.read_csv(path); table.loc[0, 'macro_f1_present'] = .99
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'changed'):
                analyze.analyze(root)
            mark(root / 'seed_0', dict(cache_sha256=analyze.sha(root / 'cache/complete.json')))
            with self.assertRaisesRegex(ValueError, 'Metric mismatch'):
                analyze.analyze(root)

    def test_query_labels_must_match_saved_pool(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            path = root / 'seed_0/2020-01/random_queries.csv'
            table = pd.read_csv(path, dtype={'record_id': str}); table.loc[0, 'device_id'] = 2
            table.to_csv(path, index=False)
            mark(root / 'seed_0', dict(cache_sha256=analyze.sha(root / 'cache/complete.json')))
            with self.assertRaisesRegex(ValueError, 'Query identity/label'):
                analyze.analyze(root)


if __name__ == '__main__':
    unittest.main()
