"""Synthetic scientific-protocol checks; no CESNET results are claimed."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_relation_audit as rel
import kbs_supplement as kbs
import kbs_badge_kd_followup as follow
import test_kbs_badge_kd_followup as previous


class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @contextlib.contextmanager
    def fixture(self):
        with previous.BadgeFollowupTests().study("0,1") as (out, args, data):
            follow.followup(args, data)
            q = rel.parser().parse_args(["freeze", "--study-dir", str(out),
                 "--output-dir", str(out.parent / "relation"), "--seeds", "0,1"])
            yield q

    def reseal(self, directory):
        done = rel.read_json(directory / "complete.json")
        rel.finish(directory, done["signature"], done["artifacts"])

    def test_query_relation_direction_self_loops_and_degree_control(self):
        old, y = [1, 1, 1, 2, 3], [0, 0, 1, 2, 1]
        counts, graph, control = rel.build_relations(old, y, 5, 7)
        self.assertTrue(graph[1, 0])  # true 0 predicted 1 permits prediction 1->0.
        self.assertFalse(graph[0, 1])
        self.assertEqual(counts[1, 0], 2)
        self.assertEqual(np.trace(counts), 2)
        self.assertTrue(np.diag(graph).all())
        self.assertTrue(np.diag(control).all())
        np.testing.assert_array_equal(graph.sum(1), control.sum(1))
        np.testing.assert_array_equal(control, rel.build_relations(old, y, 5, 7)[2])
        # No observed mistakes: all predictions must remain unchanged.
        _, diagonal, _ = rel.build_relations([0, 1], [0, 1], 5, 0)
        np.testing.assert_array_equal(diagonal, np.eye(5, dtype=bool))
        with self.assertRaises(ValueError):
            rel.build_relations([0, 1], [0], 5, 0)
        with self.assertRaises(ValueError):
            rel.build_relations([0, 1], [0, 5], 5, 0)

    def test_hand_calculated_flips_potential_and_all_class_transitions(self):
        counts, graph, _ = rel.build_relations([1, 1, 1, 2, 3], [0, 0, 1, 2, 1], 5, 0)
        y = np.array([0, 0, 1, 2, 3, 3, 0, 2])
        old = np.array([1, 2, 1, 2, 3, 3, 1, 2])
        raw = np.array([0, 0, 0, 0, 3, 1, 2, 2])
        p = {"num_classes": 5, "collapse_classes": [0], "stable_classes": [2],
             "collapse_recall_threshold": .1, "f1_drop_threshold": .05}
        gated, denied = rel.gated_predictions(old, raw, graph)
        np.testing.assert_array_equal(gated, [0, 2, 0, 2, 3, 1, 1, 2])
        np.testing.assert_array_equal(np.flatnonzero(denied), [1, 3, 6])
        m, pcs = rel.metrics(y, old, raw, p)
        g, _ = rel.metrics(y, old, gated, p)
        self.assertEqual((m["positive_flips"], m["negative_flips"], m["wrong_to_different_wrong"]), (2, 3, 1))
        self.assertEqual((g["positive_flips"], g["negative_flips"]), (1, 2))
        self.assertEqual(m["noncollapse_new_collapses"], 1)
        self.assertEqual(pcs[4]["support"], 0)
        self.assertEqual(pcs[4]["new_collapse"], 0)
        pot = rel.relation_potential(y, old, graph, counts, p)
        self.assertEqual(pot["allowed_error_fraction"], 2 / 3)
        self.assertEqual(pot["oracle_accuracy_upper_bound"], 7 / 8)
        self.assertEqual(pot["collapse_micro_recall_upper_bound"], 2 / 3)
        rows = list(rel.transition_rows(y, old, raw, graph, p, pcs))
        self.assertEqual(sum(r["count"] for r in rows), len(y))
        self.assertEqual(sum(r["count"] for r in rows if r["transition"] == "negative_flip"), 3)
        for pred in (raw, gated):
            actual, _ = rel.metrics(y, old, pred, p)
            expected, _, _ = kbs.compare_predictions(y, old, pred, np.ones(len(y), dtype=bool),
                                                   5, [0], [2], .1, .05)
            for group in ("overall", "collapse", "noncollapse", "stable"):
                self.assertAlmostEqual(actual[group + "_macro_f1"], expected[group + "_macro_f1_after"])

    def test_full_audit_matches_saved_metrics_preserves_inputs_and_resumes(self):
        with self.fixture() as args:
            source_files = [p for p in Path(args.study_dir).rglob("*") if p.is_file()]
            before = {p: (rel.audit.sha(p), p.stat().st_mtime_ns) for p in source_files}
            rel.freeze(args)
            rel.evaluate(args)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 7 * 2)
            pcs = kbs.read_csv(out / "evaluation/per_class.csv")
            self.assertEqual(len(pcs), 7 * 2 * 3)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/relation_per_class.csv")), 2 * 2 * 3)
            transitions = kbs.read_csv(out / "evaluation/transitions.csv")
            for seed in ("0", "1"):
                for method in rel.METHODS:
                    self.assertEqual(sum(int(r["count"]) for r in transitions if r["seed"] == seed and r["method"] == method), 18)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 6 * 2)
            for row in rows:
                self.assertEqual(int(row["eval_samples"]), 18)  # 30 minus 6 Margin minus 6 BADGE.
            # Every restricted negative/positive flip must be a subset of its raw flips.
            by = {(r["seed"], r["method"]): r for r in rows}
            for seed in ("0", "1"):
                for method in rel.METHODS:
                    for suffix in ("relation_gate", "degree_control"):
                        for field in ("positive_flips", "negative_flips"):
                            self.assertLessEqual(int(by[(seed, method + "_" + suffix)][field]), int(by[(seed, method)][field]))
            saved = {p: (rel.audit.sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
            with patch.object(rel.np, "load", side_effect=AssertionError("resume must not load arrays")):
                rel.freeze(args)
                rel.evaluate(args)
            self.assertEqual(saved, {p: (rel.audit.sha(p), p.stat().st_mtime_ns) for p in saved})
            self.assertEqual(before, {p: (rel.audit.sha(p), p.stat().st_mtime_ns) for p in source_files})
            # Array artifacts are aligned to original global rows, not silently reindexed.
            with np.load(out / "evaluation/seed_0_gated_predictions.npz") as z:
                np.testing.assert_array_equal(z["row_id"], np.arange(30))
                self.assertEqual(z["common_eval"].sum(), 18)

    def test_freeze_cannot_read_unqueried_labels_or_repaired_predictions(self):
        with self.fixture() as args:
            original = np.load
            class Guard:
                def __init__(self, archive):
                    self.archive = archive
                def __enter__(self):
                    return self
                def __exit__(self, *unused):
                    self.archive.close()
                def __getitem__(self, key):
                    if key not in ("static_pred", "row_id"):
                        raise AssertionError(f"Forbidden freeze member: {key}")
                    return self.archive[key]
            with patch.object(rel.np, "load", side_effect=lambda *a, **kw: Guard(original(*a, **kw))):
                rel.freeze(args)
            # Missing any seed prevents ALL label reads in evaluation.
            (Path(args.output_dir) / "frozen/seed_1.json").unlink()
            with patch.object(rel.np, "load", side_effect=AssertionError("must not read target labels")):
                with self.assertRaises(ValueError):
                    rel.evaluate(args)

    def test_mask_and_query_truth_mismatches_rejected_even_with_refreshed_hashes(self):
        with self.fixture() as args:
            root, _, identity = rel.verify_inputs(args)
            directory = root / "runs/badge_full/seed_0"
            with np.load(directory / "predictions.npz") as z:
                arrays = {k: z[k] for k in z.files}
            arrays["common_eval"][0] = True  # Leaked Margin query.
            np.savez_compressed(directory / "predictions.npz", **arrays)
            self.reseal(directory)
            with self.assertRaisesRegex(ValueError, "exact query union"):
                rel.load_pair(root, identity["study_protocol"], 0)
            arrays["common_eval"][0] = False
            arrays["y_true"][6] = (arrays["y_true"][6] + 1) % 3
            np.savez_compressed(directory / "predictions.npz", **arrays)
            self.reseal(directory)
            with self.assertRaisesRegex(ValueError, "Paid query labels"):
                rel.load_pair(root, identity["study_protocol"], 0)

    def test_corrupt_sources_changed_configuration_and_derived_outputs_fail(self):
        with self.fixture() as args:
            bad = rel.parser().parse_args(["freeze", "--study-dir", args.study_dir,
                    "--output-dir", str(Path(args.study_dir) / "bad"), "--seeds", "0,1"])
            with self.assertRaises(ValueError):
                rel.verify_inputs(bad)
            rel.freeze(args)
            args.seeds = "0"
            with self.assertRaises(ValueError):
                rel.evaluate(args)
            args.seeds = "0,1"
            rel.evaluate(args)
            (Path(args.output_dir) / "evaluation/report.md").write_text("corrupt")
            with self.assertRaises(ValueError):
                rel.evaluate(args)
            (Path(args.study_dir) / "runs/badge_kd/seed_0/predictions.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                rel.verify_inputs(args)

    def test_actual_separate_cli_processes_need_no_torch_or_feature_cache(self):
        with self.fixture() as args:
            script = str(Path(rel.__file__).resolve())
            for command in ("preflight", "freeze", "evaluate", "evaluate"):
                p = subprocess.run([sys.executable, script, command, "--study-dir", args.study_dir,
                                    "--output-dir", args.output_dir, "--seeds", "0,1"],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            # The production module imports only NumPy and the standard-library helper.
            probe = "import sys; sys.path.insert(0, sys.argv[1]); import kbs_relation_audit; assert 'torch' not in sys.modules"
            subprocess.run([sys.executable, "-c", probe, str(Path(script).parent)], check=True)

    def test_aggregate_sample_sd_and_missing_denominator(self):
        rows = [{"method": "m", "seed": 0, "count": 2, "rate": None},
                {"method": "m", "seed": 1, "count": 4, "rate": .5}]
        result = rel.aggregate(rows, "method")[0]
        self.assertEqual(result["count_mean"], 3)
        self.assertAlmostEqual(result["count_sample_sd"], 2 ** .5)
        self.assertEqual(result["rate_n"], 1)
        self.assertIsNone(result["rate_sample_sd"])


if __name__ == "__main__":
    unittest.main()
