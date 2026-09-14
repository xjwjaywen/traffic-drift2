"""Synthetic audit tests with hand-computed collapse and support checks."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_class_audit as audit
import kbs_supplement as kbs
import kbs_sensitivity_report as sensitivity
import kbs_badge_kd_followup as follow
import test_kbs_badge_kd_followup as fixtures


class ClassAuditTests(unittest.TestCase):
    @contextlib.contextmanager
    def study(self, seeds="0,1"):
        with fixtures.BadgeFollowupTests().study(seeds) as (out, args, data):
            follow.followup(args, data)
            yield out

    def test_report_preserves_inputs_and_reconciles_all_methods(self):
        with self.study() as out:
            old = {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in out.rglob("*") if p.is_file()}
            result = audit.analyze(out, [0, 1])
            self.assertEqual(old, {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in old})
            rows = audit.read_csv(result / "all_classes_by_seed.csv")
            self.assertEqual(len(rows), 6 * 2 * 3)
            summaries = audit.read_csv(result / "summary.csv")
            self.assertEqual(len(summaries), 6)
            self.assertTrue(all(r["n_seeds"] == "2" and r["meets_recommended_seed_count"] == "False" for r in summaries))
            self.assertEqual(len(audit.read_csv(result / "badge_class_comparison.csv")), 3)
            baseline = (result / "report.md").read_text()
            audit.analyze(out, [0, 1])
            self.assertEqual(baseline, (result / "report.md").read_text())
            provenance = json.loads((result / "provenance.json").read_text())
            self.assertEqual(provenance["split"], "common")
            self.assertEqual(len(provenance["input_sha256"]), 1 + 12 * 6)
            proc = subprocess.run([sys.executable, "-S", audit.__file__, "--output-dir", str(out),
                                   "--seeds", "0,1"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Class audit saved:", proc.stdout)

    def test_sensitivity_after_followup_adds_18_fits(self):
        fixture_class = fixtures.fixtures.SupplementTests
        original = fixture_class.fixture
        def enough_reference(obj):
            head, x, y, r, ry = original(obj)
            return head, x, y, r.repeat(2, 1), ry.repeat(2)
        with patch.object(fixture_class, "fixture", enough_reference):
            with fixtures.BadgeFollowupTests().study() as (out, args, data):
                follow.followup(args, data)
                old = {p: (p.stat().st_mtime_ns, kbs.file_sha(p))
                       for p in (out / "runs").rglob("*") if p.is_file()}
                args.suite, args.seeds = "sensitivity", "0,1,2"
                with patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fit:
                    kbs.run_study(args, data)
                    self.assertEqual(fit.call_count, 18)
                    kbs.run_study(args, data)
                    self.assertEqual(fit.call_count, 18)
                self.assertEqual(old, {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in old})
                # Legacy summaries include all 5 saved seeds for the two reused
                # baselines. The matched report must use 0-2 for every config.
                self.assertEqual(len(kbs.read_csv(out / "sensitivity_results_by_seed.csv")), 28)
                sensitivity.report(out, [0, 1, 2])
                matched = kbs.read_csv(out / "sensitivity_matched_results_by_seed.csv")
                self.assertEqual(len(matched), 24)
                self.assertEqual({int(r["seed"]) for r in matched}, {0, 1, 2})
                summary = kbs.read_csv(out / "sensitivity_matched_summary.csv")
                self.assertEqual(len(summary), 8)
                self.assertTrue(all(r["n_seeds"] == "3" and r["seed_ids"] == "0,1,2" for r in summary))
                import statistics
                expected = statistics.mean(float(r["common_overall_macro_f1_after"]) for r in matched if r["name"] == "margin_full")
                actual = float(next(r for r in summary if r["name"] == "margin_full")["common_overall_macro_f1_after_mean"])
                self.assertAlmostEqual(expected, actual)
                with self.assertRaisesRegex(ValueError, "not complete"):
                    sensitivity.report(out, [0, 1, 2, 3])

    def test_missing_or_modified_run_rejected_before_report(self):
        with self.study("0") as out:
            with self.assertRaises(FileNotFoundError):
                audit.analyze(out, [0, 1])
            self.assertFalse((out / "class_audit").exists())
            (out / "runs/badge_kd/seed_0/per_class_metrics.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Missing/modified audit input"):
                audit.analyze(out, [0])
            self.assertFalse((out / "class_audit").exists())

    def test_duplicate_rows_rejected_even_with_updated_hash(self):
        with self.study("0") as out:
            run = out / "runs/badge_kd/seed_0"
            rows = audit.read_csv(run / "per_class_metrics.csv")
            rows.append(next(r for r in rows if r["split"] == "common"))
            kbs.write_csv(run / "per_class_metrics.csv", rows)
            done = json.loads((run / "complete.json").read_text())
            kbs.finish(run, done["signature"], list(done["artifacts"]))
            with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
                audit.analyze(out, [0])

    def test_counts_support_and_seed_recurrence(self):
        # Replace synthetic predictions with hand-calculated per-class outcomes.
        with self.study() as out:
            for name in audit.METHODS:
                for seed in [0, 1]:
                    run = out / "runs" / name / f"seed_{seed}"
                    all_rows = audit.read_csv(run / "per_class_metrics.csv")
                    metrics = json.loads((run / "metrics.json").read_text())
                    # Class 0 newly collapses in BADGE KD only; class 1 is an original
                    # collapse, repaired in full but residual in KD. Class 2 has no support.
                    before = [1., 0., 0.]
                    after = [0., 0., 0.] if name == "badge_kd" else [1., 1., 0.]
                    for r in all_rows:
                        if r["split"] == "common":
                            c = int(r["class_id"])
                            r.update(support=10 if c < 2 else 0,
                                     before_f1=before[c], after_f1=after[c],
                                     before_recall=before[c], after_recall=after[c], delta_f1=after[c]-before[c])
                    for group, classes in [("overall", [0, 1, 2]), ("collapse", [1]), ("noncollapse", [0, 2]), ("stable", [2])]:
                        metrics[f"common_{group}_supported_classes"] = sum(c < 2 for c in classes)
                        for when, vals in [("before", before), ("after", after)]:
                            metrics[f"common_{group}_macro_f1_{when}"] = sum(vals[c] for c in classes)/len(classes)
                    kd = name == "badge_kd"
                    metrics.update(common_eval_samples=20, common_noncollapse_new_collapses=int(kd),
                                   common_collapse_residual_count=int(kd), common_noncollapse_degraded_count=int(kd),
                                   common_noncollapse_drop_gt_threshold_count=int(kd), common_noncollapse_worst_delta_f1=-int(kd))
                    kbs.write_csv(run / "per_class_metrics.csv", all_rows)
                    kbs.atomic_json(run / "metrics.json", metrics)
                    done = json.loads((run / "complete.json").read_text())
                    kbs.finish(run, done["signature"], list(done["artifacts"]))
            result = audit.analyze(out, [0, 1])
            recurring = audit.read_csv(result / "class_recurrence.csv")
            kd = {int(r["class_id"]): r for r in recurring if r["method"] == "badge_kd"}
            self.assertEqual(kd[0]["new_collapse_seed_count"], "2")
            self.assertEqual(kd[0]["new_collapse_seeds"], "0,1")
            self.assertEqual(kd[1]["residual_collapse_seed_count"], "2")
            self.assertEqual(kd[2]["new_collapse_seed_count"], "0")
            self.assertEqual(kd[2]["noncollapse_drop_gt_threshold_seed_count"], "0")
            self.assertEqual(kd[2]["support_min"], "0")
            rows = audit.read_csv(result / "all_classes_by_seed.csv")
            self.assertTrue(all(r["worst_noncollapse_rank"] == "" for r in rows if r["class_id"] == "2"))

    def test_metric_disagreement_is_rejected(self):
        with self.study("0") as out:
            run = out / "runs/badge_kd/seed_0"
            metrics = json.loads((run / "metrics.json").read_text())
            metrics["common_noncollapse_new_collapses"] += 1
            kbs.atomic_json(run / "metrics.json", metrics)
            done = json.loads((run / "complete.json").read_text())
            kbs.finish(run, done["signature"], list(done["artifacts"]))
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                audit.analyze(out, [0])


if __name__ == "__main__":
    unittest.main()
