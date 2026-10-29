"""Synthetic fixtures only: chronology, no-label selection, metrics and end-to-end repair."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

import data
import pilot


class PilotTests(unittest.TestCase):
    def test_time_boundaries(self):
        times = pd.Series(pd.to_datetime(['1970-01-01', '2019-11-21', '2019-11-22',
            '2019-11-23', '2019-12-14', '2019-12-15', '2019-12-16',
            '2020-01-14', '2020-01-15', '2020-01-16', '2020-05-01'], utc=True))
        masks = data.period_masks(times)
        self.assertEqual(np.flatnonzero(masks['train']).tolist(), [1])
        self.assertEqual(np.flatnonzero(masks['history']).tolist(), [6])
        self.assertEqual(np.flatnonzero(masks['2020-01_query']).tolist(), [7])
        self.assertEqual(np.flatnonzero(masks['2020-01_eval']).tolist(), [9])
        assignments = np.stack(list(masks.values())).sum(0)
        self.assertTrue((assignments <= 1).all())
        self.assertEqual(assignments[[0, 2, 5, 8, 10]].sum(), 0)

    def test_streaming_unique_sampling(self):
        ids = np.random.default_rng(11).integers(0, 2**63, 100, dtype=np.uint64)
        x = np.arange(100, dtype=np.float32)[:, None]
        all_at_once, streamed = data.BottomK(13), data.BottomK(13)
        all_at_once.add(ids, x, x[:, 0], x[:, 0])
        for ix in [slice(0, 40), slice(0, 40), slice(40, 100)]:
            streamed.add(ids[ix], x[ix], x[ix, 0], x[ix, 0])
        self.assertEqual(set(all_at_once.data['ids']), set(streamed.data['ids']))

    def test_absence_not_collapse_unknown_not_known_and_recovery(self):
        classes = [1, 2, 3]
        history = pilot.stats(np.repeat([0, 1, 2], 60), np.repeat([0, 1, 2], 60), 3)
        y = np.array([0] * 60 + [1] * 60 + [-1] * 10)
        before = np.array([1] * 120 + [2] * 10)
        after = np.array([0] * 60 + [1] * 60 + [0] * 10)
        summary, rows, cm = pilot.evaluate(y, before, after, history, classes)
        self.assertEqual(summary['collapse_count'], 1)
        self.assertEqual(summary['recovered_to_recall_0_5'], 1)
        self.assertEqual(summary['unseen_eval_rows'], 10)
        self.assertEqual(summary['accuracy_known'], 1)
        self.assertIsNone(rows[2]['after_recall'])
        self.assertFalse(rows[2]['collapse'])
        self.assertEqual(cm.sum(), 120)

    def test_transform_train_only_and_provenance(self):
        x = np.array([[1., np.nan], [3., np.nan]], dtype=np.float32)
        prep = data.Preprocessor().fit(x)
        med = prep.median.copy()
        result = prep.transform(np.array([[1e9, np.inf]], dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_equal(med, prep.median)
        self.assertEqual(result[0, -1], 1)
        root = Path(__file__).parent
        source = (root / 'existing_care.py').read_text()
        for item in json.loads((root / 'provenance.json').read_text())['functions']:
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == item['name'])
            self.assertEqual(hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(), item['sha256'])

    def test_full_synthetic_pipeline_and_resume(self):
        torch.set_num_threads(1)
        dates = ['2019-11-01', '2019-11-24', '2019-12-01', '2019-12-17']
        dates += [m + d for m in data.MONTHS for d in ['-02', '-17']]
        rng = np.random.default_rng(7)
        rows = []
        for day in dates:
            for c in [1, 2]:
                # Exercise reference fallback for a historically absent device.
                if day == '2019-12-01' and c == 2:
                    continue
                for i in range(70):
                    row = {f: float(rng.normal(c * 10, 1)) for f in data.FEATURES}
                    row.update(deviceId=c, time_start=day + f'T00:{i // 60:02}:{i % 60:02}Z', pr=6)
                    if i % 5 == 0:
                        row['bytes_in'] = np.nan
                    rows.append(row)
        # Extra protocol and timestamp anomalies must never reach training.
        for pr, day in [(1, '2019-11-01'), (6, '1970-01-01')]:
            row = dict(rows[0]); row.update(pr=pr, time_start=day, deviceId=41); rows.append(row)
        body = pd.DataFrame(rows).to_csv(index=False).encode()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            archive = root / 'fixture.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('fixture.csv'); member.size = len(body)
                tar.addfile(member, io.BytesIO(body))
            data.write_json(root / 'fixture.tar.gz.manifest.json', dict(sha256=data.sha(archive)))
            caps = dict(train=60, reference=20, query=100, eval=120)
            data.prepare(archive, root / 'cache', caps)
            data.prepare(archive, root / 'cache', caps)
            train = data.load(root / 'cache', 'train')
            self.assertEqual(set(train['y']), {1, 2})
            args = argparse.Namespace(device='cpu', epochs=1, steps=2, budget=10, rf_trees=2, threads=1)
            pilot.run_seed(root / 'cache', root / 'run', args, 0)
            result = pd.read_csv(root / 'run/summary.csv')
            self.assertEqual(len(result), 32)
            origins = pd.read_csv(root / 'run/reference_ids.csv')
            self.assertEqual((origins.role == 'train_fallback').sum(), 5)
            for month in data.MONTHS:
                traces = [json.loads((root / 'run' / month / (m + '_trace.json')).read_text())
                          for m in ['random_target', 'random_replay', 'random_care']]
                self.assertEqual(len({t['target_stream_sha256'] for t in traces}), 1)
                self.assertEqual(traces[0]['reference_ce_presentations'], 0)
                self.assertGreater(traces[2]['reference_kd_presentations'], 0)
            before = data.sha(root / 'run/summary.csv')
            pilot.run_seed(root / 'cache', root / 'run', args, 0)
            self.assertEqual(before, data.sha(root / 'run/summary.csv'))


if __name__ == '__main__':
    unittest.main()
