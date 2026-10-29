"""Synthetic coverage fixtures only; no real dataset access."""
import json
from pathlib import Path
import tempfile
import unittest
import coverage as coverage_report


class CoverageTests(unittest.TestCase):
    def test_absent_rare_unseen_and_returning(self):
        monthly = {('1970-01', 10): 4, ('2019-11', 1): 100,
                   ('2019-11', 2): 2, ('2020-03', 3): 100, ('2020-04', 1): 100}
        weekly = {('2020-W12', 1): 100, ('2020-W12', 2): 2,
                  ('2020-W12', 3): 100, ('2020-W13', 2): 60,
                  ('2020-W15', 1): 80, ('1970-W01', 10): 4}
        r = coverage_report.describe(monthly, weekly)
        self.assertEqual(r['source_ids'], [1, 2])
        self.assertEqual(r['source_below_support_ids'], [2])
        w = {v['period']: v for v in r['weekly']}
        self.assertNotIn('1970-W01', w)
        self.assertEqual(w['2020-W12']['unseen_source_ids'], [3])
        self.assertEqual(w['2020-W12']['present_source_below_support_ids'], [2])
        self.assertEqual(w['2020-W13']['absent_source_ids'], [1])
        self.assertEqual(w['2020-W14']['absent_source_ids'], [1, 2])
        self.assertEqual(w['2020-W14']['rows'], 0)
        self.assertEqual(w['2020-W15']['present_source_at_support_ids'], [1])
        self.assertEqual(r['outside_primary_months'][0]['rows'], 4)

    def test_artifact_integrity_and_count_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'data_audit.json').write_text(json.dumps(dict(valid_time_label_rows=100)))
            (d / 'monthly_class_counts.csv').write_text('month,device_id,rows\n2019-11,1,100\n')
            (d / 'weekly_class_counts.csv').write_text('week,device_id,rows\n2019-W45,1,100\n')
            def marker():
                (d / 'audit_complete.json').write_text(json.dumps(dict(
                    artifacts={n: coverage_report.digest(d / n) for n in coverage_report.INPUTS})))
            marker()
            coverage_report.run(d)
            self.assertTrue((d / 'coverage_v1/device_month_matrix.csv').exists())
            (d / 'weekly_class_counts.csv').write_text('week,device_id,rows\n2019-W45,1,90\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                coverage_report.run(d)
            marker()
            with self.assertRaisesRegex(ValueError, 'disagree'):
                coverage_report.run(d)


if __name__ == '__main__':
    unittest.main()
