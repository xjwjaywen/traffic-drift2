"""Post-hoc diagnosis checks on synthetic tensors, never paper evidence."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_acquisition_diagnosis as diag
import kbs_acquisition_pilot as pilot
import kbs_supplement as kbs
import test_kbs_acquisition_pilot as previous


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        p = previous.AcquisitionTests().fixture(root)
        with contextlib.redirect_stdout(io.StringIO()):
            sources = pilot.load_sources(p)
            pilot.select(p, sources)
            pilot.evaluate(p, sources)
        return diag.parser().parse_args(["run", "--study-dir", p.study_dir, "--cache-dir", p.cache_dir,
            "--checkpoint", p.checkpoint, "--pilot-dir", p.output_dir, "--output-dir", str(root / "diagnosis"),
            "--device", "cpu", "--batch-size", "17", "--threads", "1"])

    def test_hand_calculated_correctness_cells_and_empty_cohort(self):
        y = np.array([0, 0, 1, 1])
        h = np.array([0, 1, 0, 1])
        a = np.array([0, 0, 0, 0])
        confidence = np.array([.99, .99, .99, .8])
        r, classes = diag.stats(y, h, a, confidence, np.arange(4), 3, [1, 2])
        self.assertEqual([r[k] for k in ["both_correct", "head_only_correct", "proto_only_correct", "both_wrong"]], [1, 1, 1, 1])
        self.assertEqual(r["proto_accuracy"], .5)
        self.assertEqual(r["collapse_samples"], 2)
        self.assertEqual(r["collapse_classes_covered"], 1)
        self.assertEqual(r["high_conf_errors"], 2)
        self.assertIsNone(classes[2]["proto_recall"])
        empty, _ = diag.stats(y, h, a, confidence, np.array([], dtype=np.int64), 3, [1])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["proto_accuracy"])

    def test_phase_partition_candidate_exclusion_and_scout_budget(self):
        choice = {"row_indices": [0, 3, 2, 4], "exploration_count": 2, "specialist_count": 1, "fallback_count": 1}
        phases = diag.phases(choice, 6, 4)
        np.testing.assert_array_equal(phases["specialist"], [2])
        np.testing.assert_array_equal(phases["fallback"], [4])
        with self.assertRaises(ValueError):
            diag.phases({**choice, "fallback_count": 2}, 6, 4)
        pool = diag.candidate_pool(np.array([.9, 0, .7, .6, .5, 0], dtype=np.float32), np.zeros(6, dtype=int),
                                   np.array([.8, .2]), np.array([1, 0]), "risk_disagreement", phases["exploration"], 4)
        np.testing.assert_array_equal(pool, [2, 4])
        pairs = diag.scout_pairs(np.array([0, 1, 1, 2]), np.array([1, 1, 2, 0]), np.array([0, 1]), 3)
        self.assertEqual(pairs, [{"true_class": 0, "predicted_absorber": 1, "confirmed_errors": 1}])

    def test_full_pipeline_reconciles_excludes_prototype_rows_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            source_files = [p for folder in [args.study_dir, args.cache_dir, args.pilot_dir]
                            for p in Path(folder).rglob("*") if p.is_file()]
            original = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in source_files}
            with contextlib.redirect_stdout(io.StringIO()):
                diag.run(args)
            out = Path(args.output_dir)
            for seed in (0, 1):
                rows = kbs.read_csv(out / f"seed_{seed}/cohorts.csv")
                by_name = {r["cohort"]: r for r in rows}
                self.assertEqual(int(by_name["reference_heldout"]["n"]), 24 - 15)
                self.assertEqual(float(by_name["reference_heldout"]["proto_accuracy"]), 1.)
                self.assertEqual(float(by_name["target_disagreement"]["proto_accuracy"]), 1.)
                for method in ("disagreement", "risk_disagreement", "shuffled_risk"):
                    for field in ("n", "collapse_samples", "high_conf_errors", "proto_correct"):
                        self.assertEqual(int(by_name[f"{method}/all"][field]), sum(int(by_name[f"{method}/{phase}"][field])
                                         for phase in ("exploration", "specialist", "fallback")))
                self.assertEqual(int(by_name["badge/scout_first_20pct"]["n"]), 2)
                # All cohort rows, including unsupported classes, survive output.
                self.assertEqual(len(kbs.read_csv(out / f"seed_{seed}/per_class.csv")), 3 * len(rows))
            saved = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(torch, "load", side_effect=AssertionError("resume must not load tensors")):
                diag.run(args)
            self.assertEqual(saved, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()})
            self.assertEqual(original, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in source_files})

    def test_corrupt_input_missing_evaluation_and_changed_output_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                diag.run(args)
            (Path(args.output_dir) / "seed_0/cohorts.csv").write_text("broken")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                diag.run(args)
            (Path(args.pilot_dir) / "evaluation/complete.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                diag.verify_inputs(args)
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.output_dir = str(Path(args.pilot_dir) / "bad-output")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                diag.verify_inputs(args)
            args.output_dir = str(Path(tmp) / "safe")
            args.seeds = "9"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                diag.verify_inputs(args)


if __name__ == "__main__":
    unittest.main()
