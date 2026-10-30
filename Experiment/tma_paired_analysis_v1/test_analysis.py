"""Synthetic CSV/NPZ fixtures only. No training library, model or dataset required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze_pairs as analysis


def finish(root, signature):
    artifacts = {str(p.relative_to(root)): analysis.sha(p) for p in root.rglob('*')
                 if p.is_file() and p.name != 'complete.json'}
    analysis.write_json(root / 'complete.json', dict(signature=signature, artifacts=artifacts))


def fixture(root, seeds=(0, 1)):
    cache = root / 'cache'; cache.mkdir()
    pilot = root / 'seed_0'; pilot.mkdir()
    class_ids = [5, 22, 26, 37, 20]
    pools = {}
    for j, month in enumerate(analysis.MONTHS):
        y = np.repeat(class_ids, 4)
        y = np.r_[y, 999]
        days = np.r_[np.tile([1, 5, 9, 14], len(class_ids)), 3]
        ids = np.arange(len(y), dtype=np.uint64) + np.uint64(2**63 + j * 1000)
        times = np.array([pd.Timestamp(f'{month}-{day:02d}', tz='UTC').value for day in days])
        pools[month] = dict(ids=ids, y=y, times=times)
        np.savez_compressed(cache / (month + '_query.npz'), **pools[month])
        ey = np.repeat(class_ids, 50)
        et = np.full(len(ey), pd.Timestamp(month + '-18', tz='UTC').value)
        np.savez_compressed(cache / (month + '_eval.npz'), ids=np.arange(len(ey), dtype=np.uint64) +
                            np.uint64(2**63 + 10000 + j * 1000), y=ey, times=et)
    finish(cache, dict(synthetic=True))
    analysis.write_json(pilot / 'run_info.json', dict(classes=class_ids))
    finish(pilot, dict(cache_sha256=analysis.sha(cache / 'complete.json'), budget=5))
    for seed in seeds:
        out = root / 'paper_protocol_v1/source_0' / f'repair_{seed}'
        out.mkdir(parents=True)
        summaries, per_class = [], []
        for month in analysis.MONTHS:
            part = out / month; part.mkdir()
            known = {}
            for selector, offset in [('random', 0), ('badge', 3), ('margin', 17)]:
                indices = np.roll(np.arange(21), -(offset + seed * 3))[:5]
                pool = pools[month]
                known[selector] = int(np.isin(pool['y'][indices], class_ids).sum())
                pd.DataFrame([dict(record_id=str(pool['ids'][i]), device_id=int(pool['y'][i]),
                                   known_class=bool(pool['y'][i] in class_ids)) for i in indices]).to_csv(part / (selector + '_queries.csv'), index=False)
            collapse = np.isin(class_ids, analysis.CASES.get(month, []))
            before = np.where(collapse, 0., .9)
            for method in analysis.METHODS:
                after = np.full(5, .7)
                if method == 'static_mlp':
                    after = before.copy()
                elif not method.startswith('static_'):
                    after[collapse] = (.2 if seed == 0 else .8) if method.endswith('care') else (.8 if seed == 0 else .2)
                meta = dict(source_seed=0, repair_seed=seed, month=month, method=method)
                static = method.startswith('static_')
                summaries.append(dict(**meta, group_definition='static_mlp_history_and_target_recall_v1',
                    label_queries=0 if static else 5, usable_known_labels=0 if static else known[method.split('_')[0]],
                    initial_collapse_count=int(collapse.sum()), remaining_collapse_count=int((collapse & (after < .1)).sum()),
                    macro_f1_present=float(after.mean()), collapse_f1=float(after[collapse].mean()) if collapse.any() else np.nan,
                    high_recall_group_f1=float(after[~collapse].mean()),
                    negative_flip_rate_vs_static_mlp=.1 if method.endswith('care') else .2,
                    new_collapses_among_historically_learned=0))
                for i, device in enumerate(class_ids):
                    per_class.append(dict(**meta, device_id=device, support=50, historical_support=50,
                        historical_recall=1., before_recall=before[i], before_f1=before[i],
                        collapse=bool(collapse[i]), high_recall_group=bool(not collapse[i]),
                        after_recall=after[i], after_f1=after[i]))
        pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
        pd.DataFrame(per_class).to_csv(out / 'per_class.csv', index=False)
        finish(out, dict(source_seed=0, repair_seed=seed, queries='fixture',
                         source_marker=analysis.sha(pilot / 'complete.json'), cache_marker=analysis.sha(cache / 'complete.json'),
                         protocol=dict(epochs=30, batch=64), device='cpu'))


class AnalysisTests(unittest.TestCase):
    def test_pair_directions_missing_and_paired_sd(self):
        meta = dict(month='2020-03', selector='random')
        rows = [analysis.pair_row(dict(meta, seed=0), 'f1', .2, .8),
                analysis.pair_row(dict(meta, seed=1), 'f1', .8, .2)]
        result = analysis.summarize_pairs(pd.DataFrame(rows), ['month', 'selector', 'metric']).iloc[0]
        self.assertEqual((result.wins, result.ties, result.losses), (1, 0, 1))
        self.assertAlmostEqual(result.mean_delta, 0)
        self.assertAlmostEqual(result.sd_delta, np.std([-60, 60], ddof=1))
        nfr = analysis.pair_row(meta, 'nfr', .1, .2, larger=False)
        self.assertEqual(nfr['outcome'], 'win')
        self.assertAlmostEqual(nfr['delta'], -10)
        missing = analysis.pair_row(meta, 'f1', np.nan, np.nan)
        result = analysis.summarize_pairs(pd.DataFrame([missing]), ['month', 'selector', 'metric']).iloc[0]
        self.assertEqual((result.n_valid, result.missing, result.ties), (0, 1, 0))
        self.assertTrue(pd.isna(result.mean_delta))
        self.assertEqual(analysis.pair_row(meta, 'f1', .5, .5 + 1e-14)['outcome'], 'tie')

    def test_time_units_and_temporal_boundary(self):
        instant = pd.Timestamp('2020-03-14T23:59:00Z').value
        for divisor in [1, 1000, 10**6, 10**9]:
            dates = analysis.timestamps(np.array([instant // divisor]), '2020-03', True)
            self.assertEqual(dates[0], pd.Timestamp('2020-03-14T23:59:00Z'))
        with self.assertRaisesRegex(ValueError, 'temporal split'):
            analysis.timestamps(np.array([pd.Timestamp('2020-03-15', tz='UTC').value]), '2020-03', True)

    def test_full_readonly_report_coverage_reuse_and_integrity(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            fixture(root)
            original = {p: analysis.sha(p) for p in root.rglob('*') if p.is_file()}
            out = analysis.analyze(root, seeds=[0, 1])
            pairs = pd.read_csv(out / 'paired_by_seed.csv')
            self.assertEqual(len(pairs), 2 * 4 * 2 * 6)
            wins = pd.read_csv(out / 'paired_wins.csv')
            none = wins[(wins.month == '2020-01') & (wins.metric == 'collapse_f1')]
            self.assertTrue(none.n_valid.eq(0).all())
            self.assertTrue(none.ties.eq(0).all())
            coverage = pd.read_csv(out / 'query_coverage_by_seed.csv')
            self.assertEqual(len(coverage), 2 * 5 * 3)
            case = coverage[(coverage.month == '2020-03') & (coverage.device_id == 5) &
                            (coverage.repair_seed == 0) & (coverage.selector == 'random')].iloc[0]
            self.assertEqual((case.pool_rows, case.queried_rows, case.queried_days, case.available_days), (4, 4, 4, 4))
            self.assertEqual(case.queried_rows_days_8_14, 2)
            self.assertAlmostEqual(case.expected_queries_under_uniform, 5 * 4 / 21)
            self.assertEqual(case.eval_rows, 50)
            daily = pd.read_csv(out / 'query_dates.csv')
            self.assertEqual(len(daily), 2 * 5 * 3 * 14)
            self.assertTrue((daily.queried_rows <= daily.pool_rows).all())
            futures = pd.read_csv(out / 'eval_dates.csv')
            self.assertEqual(futures.eval_rows.sum(), 5 * 50)
            self.assertEqual(len(pd.read_csv(out / 'device_pairs_by_seed.csv')), 2 * 5 * 2 * 2)
            for p, digest in original.items():
                self.assertEqual(analysis.sha(p), digest, str(p))
            finished = analysis.sha(out / 'complete.json')
            analysis.analyze(root, seeds=[0, 1])
            self.assertEqual(analysis.sha(out / 'complete.json'), finished)
            with (root / 'cache/2020-03_query.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'changed completed artifact'):
                analysis.analyze(root, seeds=[0, 1])

    def test_bad_query_labels_and_missing_seed_rejected(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            with self.assertRaises(FileNotFoundError):
                analysis.analyze(root, seeds=[0, 1, 2])
            run = root / 'paper_protocol_v1/source_0/repair_0'
            path = run / '2020-03/random_queries.csv'
            table = pd.read_csv(path, dtype={'record_id': str})
            table.loc[0, 'device_id'] = 22
            table.to_csv(path, index=False)
            signature = json.loads((run / 'complete.json').read_text())['signature']
            finish(run, signature)
            with self.assertRaisesRegex(ValueError, 'Query label differs'):
                analysis.analyze(root, seeds=[0, 1])

    def test_mismatched_groups_and_costs_rejected(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); fixture(root)
            run = root / 'paper_protocol_v1/source_0/repair_0'
            summary = pd.read_csv(run / 'summary.csv')
            classes = pd.read_csv(run / 'per_class.csv')
            classes.loc[classes.method == 'random_care', 'before_f1'] = .123
            with self.assertRaisesRegex(ValueError, 'share fixed'):
                analysis.validate_tables(summary, classes, [5, 22, 26, 37, 20], 0, 0)
            summary.loc[summary.method == 'badge_target', 'usable_known_labels'] = 999
            summary.to_csv(run / 'summary.csv', index=False)
            signature = json.loads((run / 'complete.json').read_text())['signature']
            finish(run, signature)
            with self.assertRaisesRegex(ValueError, 'query costs'):
                analysis.analyze(root, seeds=[0, 1])


if __name__ == '__main__':
    unittest.main()
