"""Synthetic fixtures only: no downloads, real TMA data, or server access."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import run as paper
from data import finish, sha
from pilot import run_seed
import controls


class PaperTests(unittest.TestCase):
    def test_verbatim_original_and_training_trajectory(self):
        torch.set_num_threads(1)
        original_path = paper.EXPERIMENT / 'core_code/scripts/collapse_active_maintenance_tls22.py'
        source = original_path.read_text()
        vendored = (paper.HERE / 'paper_engine.py').read_text()
        provenance = json.loads((paper.HERE / 'provenance.json').read_text())
        env = dict(torch=torch, copy=copy, F=F)
        for name in ['build_head_training_set', 'fit_head']:
            a = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in ast.parse(vendored).body if isinstance(n, ast.FunctionDef) and n.name == name)
            original = ast.get_source_segment(source, a)
            self.assertEqual(original, ast.get_source_segment(vendored, b))
            self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), provenance['functions'][name])
            exec(compile(ast.Module(body=[a], type_ignores=[]), str(original_path), 'exec'), env)
        torch.manual_seed(6)
        head = torch.nn.Linear(4, 3)
        frozen = copy.deepcopy(head.state_dict())
        tx, rx = torch.randn(7, 4), torch.randn(5, 4)
        ty, ry = np.arange(7) % 3, np.arange(5) % 3
        teacher = head(rx).detach()
        protocol = dict(paper.PROTOCOL, epochs=3, batch_size=8)
        for care in [False, True]:
            r = len(ry) if care else 0
            x, y = env['build_head_training_set'](tx, ty, rx, ry, torch.arange(r), 2)
            self.assertTrue(torch.equal(x[:7], tx))
            self.assertTrue(torch.equal(x[7:14], tx))
            self.assertEqual(len(x), 14 + r)
            expected = env['fit_head'](SimpleNamespace(cls_head=head), x, y, .001, 3, 8, .0001,
                torch.device('cpu'), rx if care else None, teacher if care else None, .5 if care else 0., 2., 9)
            actual, trace = paper.fit_paper(head, tx, ty, rx, ry, teacher, care, 9, 'cpu', protocol)
            for key in frozen:
                self.assertTrue(torch.equal(actual.state_dict()[key], expected.state_dict()[key]))
                self.assertTrue(torch.equal(head.state_dict()[key], frozen[key]))
            self.assertEqual(trace['optimizer_steps'], 9 if care else 6)
            self.assertEqual(trace['reference_ce_presentations'], 15 if care else 0)
            self.assertEqual(trace['target_presentations'], 42)
            self.assertEqual(trace['reference_kd_presentations'], 57 if care else 0)
        unchanged, trace = paper.fit_paper(head, tx[:0], ty[:0], rx, ry, teacher, True, 0, 'cpu', protocol)
        self.assertEqual(trace['optimizer_steps'], 0)
        self.assertTrue(torch.equal(unchanged.weight, head.weight))

    def test_metrics_remaining_groups_and_unknowns(self):
        y = np.repeat([0, 1], 60)
        hist = paper.stats(y, y, 2)
        before = np.ones(120, dtype=int)  # class 0 collapses; class 1 is high-recall.
        after = before.copy(); after[:30] = 0
        s, rows, _ = paper.metrics(np.r_[y, -1], np.r_[before, 0], np.r_[after, 1], hist, [10, 11])
        self.assertEqual(s['initial_collapse_count'], 1)
        self.assertEqual(s['remaining_collapse_count'], 0)
        self.assertEqual(s['known_eval_rows'], 120)
        self.assertEqual(s['unseen_eval_rows'], 1)
        self.assertEqual(rows[0]['recovery_status_per_run'], 'recovered')
        self.assertNotIn('collapse_count', s)
        self.assertEqual(s['negative_flip_rate_vs_static_mlp'], 0)
        self.assertAlmostEqual(s['collapse_f1'], 2 * 30 / (60 + 30))
        with self.assertRaises(ValueError):
            paper.metrics(y, before, np.full(120, 9), hist, [10, 11])

    def test_end_to_end_resume_old_results_immutable_and_separate_repair_seed(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); cache = root / 'cache'; cache.mkdir()
            roles = ['train', 'val', 'history', 'reference']
            roles += [m + suffix for m in paper.MONTHS for suffix in ['_query', '_eval']]
            rng = np.random.default_rng(8)
            files = []
            for j, role in enumerate(roles):
                y = np.repeat([1, 2], 64)
                if role.endswith(('_query', '_eval')):
                    y = np.concatenate([y, np.repeat(3, 10)])
                x = rng.normal(y[:, None] * 3, .4, size=(len(y), len(paper.FEATURES))).astype(np.float32)
                x[::5, 0] = np.nan
                ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
                day = (role[:7] + ('-02' if role.endswith('_query') else '-17')) if role.startswith('2020') else '2019-12-17'
                times = np.full(len(y), pd.Timestamp(day, tz='UTC').value, dtype=np.int64)
                name = role + '.npz'; files.append(name)
                np.savez_compressed(cache / name, x=x, y=y, ids=ids, times=times)
            finish(cache, dict(synthetic=True), files)
            args = argparse.Namespace(device='cpu', epochs=1, steps=2, budget=10, rf_trees=2, threads=1)
            run_seed(cache, root / 'seed_0', args, 0)
            controls.run(root, root / 'controls_v1/seed_0', 0, 'cpu', 1)
            originals = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            audit = paper.audit(root)
            self.assertEqual(len(pd.read_csv(audit / 'legacy_metrics_common_groups.csv')), 52)
            out = paper.run(root, device='cpu', threads=1)
            summary = pd.read_csv(out / 'summary.csv')
            self.assertEqual(len(summary), 28)
            self.assertEqual((summary.protocol == 'paper_multiset_epochs').sum(), 20)
            self.assertTrue((summary.loc[summary.protocol == 'paper_multiset_epochs', 'label_queries'] == 10).all())
            for month in paper.MONTHS:
                old = pd.read_csv(root / 'seed_0' / month / 'random_queries.csv', dtype={'record_id': str})
                new = pd.read_csv(out / month / 'random_queries.csv', dtype={'record_id': str})
                pd.testing.assert_frame_equal(old, new)
                trace = json.loads((out / month / 'random_care_trace.json').read_text())
                row = summary[(summary.month == month) & (summary.method == 'random_care')].iloc[0]
                known = int(row.usable_known_labels)
                self.assertEqual(trace['multiset_entries'], 2 * known + 10)
                self.assertAlmostEqual(trace['reference_ce_mixture_weight'], 10 / (2 * known + 10))
                self.assertEqual(trace['target_presentations'], 60 * known)
            self.assertEqual(len(pd.read_csv(out / 'paired_deltas.csv')), 28)
            finished = sha(out / 'complete.json')
            paper.run(root, device='cpu', threads=1)
            self.assertEqual(sha(out / 'complete.json'), finished)
            # Simulate interrupted root finalization: completed months must be reused unchanged.
            monthly = {m: sha(out / m / 'complete.json') for m in paper.MONTHS}
            (out / 'complete.json').unlink()
            paper.run(root, device='cpu', threads=1)
            self.assertEqual(monthly, {m: sha(out / m / 'complete.json') for m in paper.MONTHS})
            other = paper.run(root, repair_seed=1, device='cpu', threads=1)
            self.assertEqual(len(pd.read_csv(other / 'paired_deltas.csv')), 8)
            self.assertFalse((root / 'seed_1').exists())
            agg = paper.aggregate(root, 0, [0, 1])
            self.assertTrue((pd.read_csv(agg / 'summary_mean_sd.csv').macro_f1_present_count == 2).all())
            for path, digest in originals.items():
                self.assertEqual(sha(path), digest, str(path))
            with self.assertRaisesRegex(ValueError, 'Configuration'):
                paper.run(root, device='cpu', threads=1, protocol=dict(paper.PROTOCOL, epochs=1))
            # Corrupt a consumed original input: even a completed output must reject reuse.
            with (cache / '2020-01_query.npz').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'changed completed artifact'):
                paper.run(root, device='cpu', threads=1)


if __name__ == '__main__':
    unittest.main()
