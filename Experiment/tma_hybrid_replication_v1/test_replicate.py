"""Check seed pairing and completed-run provenance in aggregation."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from replicate import aggregate, load_runs
from analyze import sha
from run import complete, write_json


class ReplicationTests(unittest.TestCase):
    def make_run(self, root, seed):
        months = ['2020-03', '2020-04']
        paper = root / 'paper_protocol_v1/source_0' / f'repair_{seed}'
        paper.mkdir(parents=True)
        ps = dict(source_seed=0, repair_seed=seed, protocol={'epochs': 30})
        write_json(paper / 'config.json', ps)
        p_rows = []
        h_rows = []
        c_rows = []
        d_rows = []
        for month in months:
            for method, macro in [('badge_care', .60), ('random_care', .61)]:
                p_rows.append(self.row(seed, month, method, macro))
            for method, macro in [('static_mlp', .40), ('hybrid_target', .62),
                                  ('hybrid_care', .63),
                                  ('hybrid_adaptive', .64 if seed == 0 else .62)]:
                h_rows.append(self.row(seed, month, method, macro))
                c_rows.append(dict(month=month, device_id=5, method=method,
                                   after_recall=macro, after_f1=macro))
            d_rows.append(dict(month=month, device_id=5,
                               relax_preservation=seed == 0))
        pd.DataFrame(p_rows).to_csv(paper / 'summary.csv', index=False)
        complete(paper, ps)
        hybrid = root / 'hybrid_adaptive_v1/source_0' / f'repair_{seed}'
        hybrid.mkdir(parents=True)
        hs = dict(source_seed=0, repair_seed=seed, protocol=ps['protocol'],
                  paper_marker=sha(paper / 'complete.json'), months=months,
                  code={'experiment.py': 'frozen'})
        write_json(hybrid / 'config.json', hs)
        pd.DataFrame(h_rows).to_csv(hybrid / 'summary.csv', index=False)
        pd.DataFrame(c_rows).to_csv(hybrid / 'per_class.csv', index=False)
        pd.DataFrame(d_rows).to_csv(hybrid / 'decision_classes.csv', index=False)
        complete(hybrid, hs)

    @staticmethod
    def row(seed, month, method, macro):
        return dict(source_seed=0, repair_seed=seed, month=month, method=method,
                    label_queries=0 if method == 'static_mlp' else 1000,
                    macro_f1_present=macro, collapse_f1=macro,
                    high_recall_group_f1=macro,
                    negative_flip_rate_vs_static_mlp=.01,
                    remaining_collapse_count=1)

    def test_paired_effects_keep_seed_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for seed in (0, 1):
                self.make_run(root, seed)
            out = aggregate(root, 0, [0, 1])
            paired = pd.read_csv(out / 'paired_by_seed.csv')
            rows = paired[(paired.month == '2020-03') &
                          (paired.comparison == 'hybrid_adaptive_minus_hybrid_care')]
            for actual, expected in zip(rows.sort_values('repair_seed').macro_f1_present_pp,
                                        [1.0, -1.0]):
                self.assertAlmostEqual(actual, expected)
            means = pd.read_csv(out / 'paired_mean_sd.csv')
            row = means[(means.month == '2020-03') &
                        (means.comparison == 'hybrid_adaptive_minus_hybrid_care')].iloc[0]
            self.assertAlmostEqual(row.macro_f1_present_pp_mean, 0)
            self.assertEqual(row.positive_count, 1)
            self.assertEqual(row.nonpositive_count, 1)
            self.assertEqual(sha(out / 'complete.json'),
                             sha(aggregate(root, 0, [0, 1]) / 'complete.json'))

    def test_changed_paper_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_run(root, 0)
            paper = root / 'paper_protocol_v1/source_0/repair_0/complete.json'
            paper.write_text(paper.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                load_runs(root, 0, [0])


if __name__ == '__main__':
    unittest.main()
