"""Synthetic saved-prediction diagnostics; no CESNET performance claims."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_repair_diagnosis as diagnosis
import kbs_repair_aware as pilot
import kbs_supplement as kbs
import test_kbs_repair_aware as fixtures


class RepairDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def hand_fixture(self):
        y = np.array([0, 0, 1, 1, 2, 2, 2, 0])
        old = np.array([0, 0, 1, 1, 2, 2, 1, 0])
        probe = np.array([1, 0, 0, 1, 1, 2, 2, 0])
        final = np.array([1, 1, 0, 1, 2, 2, 2, 1])
        mask = np.array([True] * 7 + [False])
        counts = np.array([[0, 1, 0], [2, 0, 0], [0, 0, 0]])
        p = {"num_classes": 3, "collapse_classes": [2], "stable_classes": [1],
             "collapse_recall_threshold": .1, "f1_drop_threshold": .05}
        return y, old, probe, final, mask, counts, p

    def fixture(self, root, complete=True):
        original = fixtures.RepairAwareTests().fixture(root)
        sources = pilot.verify_sources(original)
        if complete:
            pilot.select(original, sources)
            pilot.train(original, sources)
            pilot.evaluate(original, sources)
        args = diagnosis.parser().parse_args(["run", "--study-dir", original.study_dir,
            "--cache-dir", original.cache_dir, "--checkpoint", original.checkpoint,
            "--pilot-dir", original.output_dir, "--output-dir", str(root / "diagnosis"), "--threads", "1"])
        return args, original, sources

    def snapshot(self, *folders):
        return {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns)
                for folder in folders for p in Path(folder).rglob("*") if p.is_file()}

    def test_hand_calculated_coverage_enrichment_persistence_and_exposure(self):
        main, cohorts, classes, edges, metrics = diagnosis.diagnostic_metrics(*self.hand_fixture())
        a, nc = main
        self.assertEqual((a["eval_samples"], a["source_correct"], a["flagged_samples"]), (7, 6, 4))
        self.assertEqual((a["final_negative_flips"], a["captured_damage"], a["missed_damage"]), (3, 2, 1))
        for name, expected in {"flag_precision": .5, "damage_coverage": 2/3,
            "enrichment_vs_random": 7/6, "outside_flag_damage_rate": 1/3,
            "probe_negative_persistence": 2/3, "absorber_correct_exposure_share": 4/6,
            "absorber_damage_share": 1, "absorber_damage_rate": .75, "other_class_damage_rate": 0}.items():
            self.assertAlmostEqual(a[name], expected)
        self.assertIsNone(a["absorber_risk_ratio"])
        self.assertEqual(a["probe_negative_resolved"], 1)
        self.assertEqual((nc["eval_samples"], nc["flag_precision"]), (4, 1))
        c = {r["cohort"]: r for r in cohorts}
        self.assertEqual(c["confirmed_relation_flip"]["n"], 2)
        self.assertEqual(c["unconfirmed_relation_flip"]["n"], 2)
        self.assertEqual(c["probe_flip"]["probe_positive"], 1)
        self.assertEqual(classes[0]["negative_flips"], 2)
        self.assertTrue(classes[0]["new_collapse"])
        self.assertEqual(classes[0]["scout_absorber_errors"], 2)
        self.assertEqual(sum(r["negative_flips"] for r in edges), 3)
        edge = next(r for r in edges if r["source_correct_class"] == 0)
        self.assertEqual((edge["captured_by_probe"], edge["confirmed_reverse_scout_errors"]), (1, 2))
        self.assertEqual(metrics["noncollapse_negative_flips"], 3)

    def test_zero_denominators_unsupported_classes_and_valid_seed_counts(self):
        y, old, _, _, mask, counts, p = self.hand_fixture()
        p["num_classes"] = 4
        main, cohorts, classes, edges, _ = diagnosis.diagnostic_metrics(
            y, old, old, old, mask, np.zeros((4, 4), dtype=int), p)
        for row in main:
            for key in ("flag_precision", "damage_coverage", "enrichment_vs_random", "probe_negative_persistence"):
                self.assertIsNone(row[key])
        self.assertEqual(edges, [])
        self.assertIsNone(cohorts[1]["final_negative_fraction"])
        self.assertEqual(classes[3]["support"], 0)
        self.assertIsNone(classes[3]["damage_rate_on_source_correct"])
        s = diagnosis.aggregate([{"method": "a", "seed": i, "x": v}
                                 for i, v in enumerate([None, 1, 3])], ["method"])[0]
        self.assertEqual((s["n_seeds"], s["x_valid_n"], s["x_mean"]), (3, 2, 2))
        self.assertAlmostEqual(s["x_sd"], np.sqrt(2))

    def test_excluded_queries_cannot_affect_metrics_and_cohorts_partition(self):
        inputs = self.hand_fixture()
        before = diagnosis.diagnostic_metrics(*inputs)
        for x, value in zip(inputs[:4], [2, 1, 0, 2]):
            x[-1] = value
        after = diagnosis.diagnostic_metrics(*inputs)
        self.assertEqual(before, after)
        c = {r["cohort"]: r for r in after[1]}
        self.assertEqual(c["all"]["n"], c["probe_flip"]["n"] + c["probe_unchanged"]["n"])
        self.assertEqual(c["probe_flip"]["n"], c["confirmed_relation_flip"]["n"] + c["unconfirmed_relation_flip"]["n"])
        self.assertEqual(sum(r["negative_flips"] for r in after[2]), sum(r["negative_flips"] for r in after[3]))

    def test_end_to_end_cpu_only_reconciliation_resume_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, original, _ = self.fixture(Path(tmp))
            verified = diagnosis.verify_inputs(args)
            before = self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir)
            real_load = torch.load
            def labels_and_logits_only(path, **kwargs):
                self.assertEqual(Path(path).name, "target.pt")
                self.assertEqual(kwargs["map_location"], "cpu")
                self.assertTrue(kwargs["mmap"])
                obj = real_load(path, **kwargs)
                obj["features"] = object()  # unusable even if a caller tries to fit/predict
                return obj
            with patch.object(torch, "load", side_effect=labels_and_logits_only), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("no fitting")), \
                 patch.object(kbs, "predict", side_effect=AssertionError("no inference")), \
                 patch.object(kbs, "device_for", side_effect=AssertionError("no GPU access")):
                diagnosis.analyze(args, verified)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "by_seed.csv")
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(kbs.read_csv(out / "per_class.csv")), 24)
            self.assertEqual(len(kbs.read_csv(out / "cohorts.csv")), 48)
            originals = {(r["seed"], r["method"]): r
                         for r in kbs.read_csv(Path(original.output_dir) / "evaluation/by_seed.csv")}
            for row in rows:
                expected = originals[row["seed"], row["method"]][row["scope"] + "_negative_flips"]
                self.assertEqual(row["final_negative_flips"], expected)
            saved = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("resume must not load tensors")):
                diagnosis.analyze(args, diagnosis.verify_inputs(args))
            self.assertEqual(saved, self.snapshot(out))
            self.assertEqual(before, self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir))

    def test_missing_stages_changed_identity_and_corruption_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, original, sources = self.fixture(Path(tmp), complete=False)
            pilot.select(original, sources)
            with patch.object(torch, "load", side_effect=AssertionError("must not load truth")):
                with self.assertRaises(ValueError): diagnosis.verify_inputs(args)
            pilot.train(original, sources)
            with self.assertRaises(ValueError): diagnosis.verify_inputs(args)
            pilot.evaluate(original, sources)
            verified = diagnosis.verify_inputs(args)
            diagnosis.analyze(args, verified)
            changed = (*verified[:2], {**verified[2], "threads": 2})
            with self.assertRaises(ValueError): diagnosis.analyze(args, changed)
            (Path(args.output_dir) / "report.md").write_text("corrupt")
            with self.assertRaises(ValueError): diagnosis.analyze(args, verified)
            (pilot.selection_dir(Path(args.pilot_dir), 0) / "probe_predictions.npy").write_bytes(b"corrupt")
            with self.assertRaises(ValueError): diagnosis.verify_inputs(args)

    def test_rehashed_semantic_metric_mismatch_and_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            directory = Path(args.pilot_dir) / "evaluation"
            rows = kbs.read_csv(directory / "by_seed.csv")
            next(r for r in rows if r["method"] == "badge")["overall_macro_f1_after"] = .99999
            kbs.write_csv(directory / "by_seed.csv", rows)
            signature = pilot.read_json(directory / "resolved_config.json")
            kbs.finish(directory, signature, pilot.EVAL_FILES)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                diagnosis.analyze(args, diagnosis.verify_inputs(args))
            args.output_dir = str(Path(args.pilot_dir) / "nested")
            with self.assertRaisesRegex(ValueError, "separate sibling"):
                diagnosis.verify_inputs(args)

    def test_separate_cpu_cli_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            common = ["--study-dir", args.study_dir, "--cache-dir", args.cache_dir,
                "--checkpoint", args.checkpoint, "--pilot-dir", args.pilot_dir,
                "--output-dir", args.output_dir, "--threads", "1"]
            for stage in ("preflight", "run"):
                result = subprocess.run([sys.executable, diagnosis.__file__, stage, *common],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(args.output_dir) / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
