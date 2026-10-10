"""Independent synthetic accounting/provenance tests, not research evidence."""
import contextlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import care_m11_mechanism_diagnostic as audit
import care_cross_month_replication as month
import care_query_seed_robustness as robust
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import export_care_results as result_export
import test_care_cross_month_replication as parent_fixtures


class MechanismDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        # Temporary synthetic heads must actually disagree, so boundary/neighbor
        # pipeline assertions exercise rows rather than pass on an empty probe.
        original = parent_fixtures.fixtures.SmallRepairTests.fixture
        def stronger(obj, path):
            args, info, _, cli = original(obj, path)
            args.steps, args.lr = 12, .04
            return args, info, diag.inputs(args)[1], cli
        with patch.object(parent_fixtures.fixtures.SmallRepairTests, "fixture", stronger):
            ma, pc, _ = parent_fixtures.CrossMonthTests().fixture(root)
        mc = month.inputs(ma, pc)
        month.select(ma, mc)
        month.train(ma, mc)
        month.evaluate(ma, mc)
        cli = ["--source-dir", ma.source_dir, "--source-audit-dir", ma.source_audit_dir,
            "--protection-dir", ma.protection_dir, "--ablation-dir", ma.ablation_dir,
            "--replication-dir", ma.replication_dir, "--source-cache-dir", ma.source_cache_dir,
            "--cache-dir", ma.cache_dir, "--checkpoint", ma.checkpoint,
            "--month-dir", ma.output_dir, "--output-dir", str(root / "mechanism"),
            "--threads", "2", "--probe-cap", "3", "--top-k", "2"]
        args = audit.parser().parse_args(["audit", *cli])
        return args, audit.inputs(args), cli

    def rehash(self, directory, name):
        marker = json.loads((directory / "complete.json").read_text())
        marker["artifacts"][name] = kbs.file_sha(directory / name)
        kbs.atomic_json(directory / "complete.json", marker)

    def test_hand_confusions_damage_partition_and_incoming_false_positives(self):
        y = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1])
        source = np.array([1, 0, 1, 0, 0, 2, 1, 0, 0])
        b = np.array([1, 1, 1, 1, 1, 0, 1, 0, 2])
        c = np.array([0, 0, 2, 0, 0, 2, 2, 0, 2])
        rows, flags = audit.pair_classes(y, source, b, c, 3, [2])
        self.assertEqual({k: int(v.sum()) for k, v in flags.items()}, {
            "extra_source_error_repair": 1, "restored_balanced_damage": 2, "extra_source_correct_damage": 1,
            "lost_balanced_repair": 1, "wrong_to_other_wrong": 2, "both_correct": 1, "both_wrong_unchanged": 1})
        np.testing.assert_array_equal(sum(flags.values()), np.ones(9))
        cms = []
        for prediction in (b, c):
            cm = np.zeros((3, 3), int)
            for truth, pred in zip(y, prediction):
                cm[truth, pred] += 1
            cms.append(cm)
        fb = 2*np.diag(cms[0]) / (cms[0].sum(0) + cms[0].sum(1))
        fc = 2*np.diag(cms[1]) / (cms[1].sum(0) + cms[1].sum(1))
        self.assertAlmostEqual(sum(r["macro_contribution"] for r in rows), (fc-fb).mean())
        for cid, r in enumerate(rows):
            self.assertEqual(r["delta_fp"], (cms[1][:, cid].sum()-cms[1][cid, cid]) - (cms[0][:, cid].sum()-cms[0][cid, cid]))
        # Class 2 gains a true positive and receives two additional false positives.
        self.assertEqual(rows[2]["delta_tp"], 1)
        self.assertEqual(rows[2]["delta_fp"], 2)
        self.assertEqual(sum(r["extra_negative_flips_net"] for r in rows), -1)
        es = audit.pair_edges(y, source, b, c, 3)
        self.assertEqual(sum(e["count"] for e in es), 7)
        for e in es:
            count = ((y == e["true_class"]) & (source == e["source_prediction"]) &
                     (b == e["balanced_prediction"]) & (c == e["coverage_prediction"])).sum()
            self.assertEqual(e["count"], count)

    def test_nested_query_units_and_absent_edges_are_zero(self):
        rows = [{"query_seed": q, "seed": s, "class_id": 7, "delta_f1": d} for q, ds in ((1, [-.1, .3]), (2, [.3, .7])) for s, d in enumerate(ds)]
        _, result = audit.nested(rows, ("class_id",), [1, 2])
        self.assertAlmostEqual(result[0]["delta_f1_mean"], .3)
        self.assertAlmostEqual(result[0]["delta_f1_sd"], statistics.stdev([.1, .5]))
        self.assertEqual(result[0]["delta_f1_positive_query_seeds"], 2)
        self.assertNotIn("query_seed_mean", result[0])
        edges = [{"query_seed": 1, "seed": 0, "true_class": 7, "source_prediction": 1,
                  "balanced_prediction": 1, "coverage_prediction": 7, "kind": "extra_source_error_repair", "count": 12}]
        r = audit.sparse_edge_summary(edges, [1, 2], [0, 1])[0]
        self.assertEqual(r["count_mean"], 3)
        self.assertAlmostEqual(r["count_sd"], statistics.stdev([6, 0]))
        self.assertEqual(r["queries_with_edge"], 1)

    def test_unit_geometry_missing_neighbors_and_exact_boundary_algebra(self):
        x = torch.tensor([[2., 0.], [0., 3.]])
        np.testing.assert_allclose(audit.unit(x), [[1, 0], [0, 1]])
        np.testing.assert_allclose(audit.nearest_squared(audit.unit(x), np.array([[1., 0.]])), [0, 2])
        self.assertIsNone(audit.nearest_squared(audit.unit(x), np.empty((0, 2))))
        b = {"weight": torch.tensor([[1., 0.], [0., 1.]]), "bias": torch.tensor([0., 0.])}
        c = {"weight": torch.tensor([[2., .5], [-1., .5]]), "bias": torch.tensor([.5, -.5])}
        terms = audit.boundary_terms(torch.tensor([[2., 1.], [1., 3.]]), np.array([0, 1]), np.array([1, 0]), b, c)
        np.testing.assert_allclose(terms["gap_shift"], [6, -6])
        np.testing.assert_allclose(terms["feature_term"], [5, -5])
        np.testing.assert_allclose(terms["bias_term"], [1, -1])

    def test_pipeline_no_fit_reference_or_queries_preserves_sources_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            originals = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            real_torch_load = torch.load
            def target_only(path, *a, **kw):
                self.assertNotEqual(Path(path).name, "reference.pt")
                if Path(path).name == "target.pt":
                    self.assertEqual(Path(path).parent.resolve(), Path(args.cache_dir).resolve())
                return real_torch_load(path, *a, **kw)
            with patch.object(torch, "load", side_effect=target_only), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Diagnosis fitted")), \
                 patch.object(month, "collect_target", side_effect=AssertionError("Diagnosis extracted")), \
                 patch.object(robust.ab, "select_supplements", side_effect=AssertionError("Diagnosis queried")):
                audit.audit(args, ctx)
                audit.geometry(args, ctx)
                audit.report(args, ctx)
            # All real synthetic completion layouts, including reused runs and
            # the standalone full audit, must be accepted by the exporter.
            paths = [args.source_dir, args.source_audit_dir, args.protection_dir, args.ablation_dir,
                     args.replication_dir, args.month_dir, args.output_dir]
            for path, (schema, manifest, final) in zip(paths, result_export.STUDIES.values()):
                files, checks = result_export.collect_study(Path(path), schema, manifest, final)
                self.assertTrue(files and checks)
                self.assertTrue(all(p.suffix in result_export.TEXT for p in files.values()))
                if schema == audit.SCHEMA:
                    self.assertNotIn("geometry/probe_rows.csv", files)
                    self.assertFalse(checks["geometry/probe_rows.csv"]["exported"])
            self.assertEqual(originals, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in originals})
            rec = json.loads((ctx["out"] / "audit/reconciliation.json").read_text())
            self.assertEqual(rec["heads_checked"], 22)
            self.assertEqual(rec["prediction_mismatches"], 0)
            classes = audit.read_numeric(ctx["out"] / "audit/class_summary.csv")
            pair = next(r for r in rec["primary_pairs"] if r["comparison"] == "mixed_coverage - mixed_balanced")
            self.assertAlmostEqual(sum(r["macro_contribution_mean"] for r in classes), pair["overall_macro_f1_after_mean"])
            ids = np.load(Path(args.month_dir) / "selection/evaluation_rows.npy")
            qids = {i for c in ctx["choices"] for i in c["rows"]}
            sampled = audit.read_numeric(ctx["out"] / "geometry/probe_rows.csv")
            self.assertGreater(len(sampled), 0)
            self.assertTrue(any(r["paid_d2_difference"] is not None for r in sampled))
            for r in sampled:
                self.assertIn(r["row_index"], ids)
                self.assertNotIn(r["row_index"], qids)
                self.assertAlmostEqual(r["gap_shift"], r["feature_term"] + r["bias_term"])
                if r["paid_d2_difference"] is not None:
                    self.assertAlmostEqual(r["paid_d2_difference"], r["coverage_paid_same_truth_d2"]-r["balanced_paid_same_truth_d2"])
            for r in audit.read_numeric(ctx["out"] / "geometry/probe_classes.csv"):
                self.assertLessEqual(r["sample_count"], 3)
                self.assertLessEqual(r["sample_count"], r["population_count"])
            stamps = {p: p.stat().st_mtime_ns for p in ctx["out"].rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume arrays")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume fit")):
                fresh = audit.inputs(args)
                audit.audit(args, fresh)
                audit.geometry(args, fresh)
                audit.report(args, fresh)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_incomplete_source_stage_gate_and_output_aliases(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            with patch.object(diag, "load_role", side_effect=AssertionError("Premature tensor reads")):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    audit.geometry(args, ctx)
            for path in (args.month_dir, args.cache_dir, args.source_cache_dir, args.checkpoint, args.replication_dir):
                old = args.output_dir
                args.output_dir = path
                with self.assertRaisesRegex(ValueError, "separate"):
                    audit.inputs(args)
                args.output_dir = old
            run = robust.run_path(ctx["mc"], ctx["choices"][0])
            (run / "complete.json").unlink()
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before source complete")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    audit.inputs(args)

    def test_rehashed_source_metrics_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(tmp))
            evaluation = Path(args.month_dir) / "evaluation"
            rows = kbs.read_csv(evaluation / "by_seed.csv")
            rows[0]["overall_macro_f1_after"] = str(float(rows[0]["overall_macro_f1_after"]) + .01)
            kbs.write_csv(evaluation / "by_seed.csv", rows)
            self.rehash(evaluation, "by_seed.csv")
            ctx = audit.inputs(args)
            with self.assertRaisesRegex(ValueError, "metric disagreement"):
                audit.audit(args, ctx)
            self.assertFalse((ctx["out"] / "audit/complete.json").exists())

    def test_rehashed_saved_predictions_must_match_frozen_heads(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            evaluation = Path(args.month_dir) / "evaluation"
            arrays = audit.saved_arrays(ctx)
            key = next(k for k in arrays if "mixed_balanced" in k)
            arrays[key][0] = (arrays[key][0] + 1) % 3
            np.savez_compressed(evaluation / "predictions.npz", **arrays)
            self.rehash(evaluation, "predictions.npz")
            ctx = audit.inputs(args)
            with self.assertRaisesRegex(ValueError, "head/prediction disagreement"):
                audit.audit(args, ctx)

    def test_separate_cpu_cli_stages(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            _, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "audit", "geometry", "report"):
                result = subprocess.run([sys.executable, audit.__file__, stage, *cli], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CARE_M11_MECHANISM_DIAGNOSTIC_COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
