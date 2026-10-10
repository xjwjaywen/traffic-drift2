"""Synthetic correctness checks only; never paper evidence."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import small_repair_full_audit as audit
import small_repair_diagnostic as diag
import kbs_supplement as kbs
import test_small_repair_diagnostic as fixtures


class FullAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, finish=True):
        args, info, identity, _ = fixtures.SmallRepairTests().fixture(root)
        diag.select(args, info, identity)
        if finish:
            diag.train(args, info, identity)
            diag.evaluate(args, info, identity)
        cli = ["--source-dir", str(root / "out"), "--cache-dir", args.cache_dir,
               "--checkpoint", args.checkpoint, "--output-dir", str(root / "audit"), "--threads", "1"]
        return audit.parser().parse_args(["run", *cli]), identity, cli

    def test_full_pipeline_preserves_inputs_no_fit_no_query_no_reference_load(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, protocol, _ = self.fixture(root)
            original = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            real_load = torch.load
            def no_reference(path, *pos, **kwargs):
                self.assertNotEqual(Path(path).name, "reference.pt")
                return real_load(path, *pos, **kwargs)
            with patch.object(torch, "load", side_effect=no_reference), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Audit fitted")), \
                 patch.object(diag, "select_indices", side_effect=AssertionError("Audit queried")):
                audit.audit(args)
            self.assertEqual(original, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in original})
            out = root / "audit"
            choices = json.loads((root / "out/selection/queries.json").read_text())
            union = {i for c in choices for i in c["rows"]}
            records = kbs.read_csv(out / "by_seed.csv")
            self.assertEqual(len(records), 12)
            per_class = kbs.read_csv(out / "per_class.csv")
            with np.load(out / "predictions.npz") as p:
                y, old, ids = p["labels"], p["source"], p["row_indices"]
                np.testing.assert_array_equal(ids, [i for i in range(90) if i not in union])
                self.assertGreater(len(ids), 30)
                all_edges = kbs.read_csv(out / "transitions.csv")
                for row in records:
                    meta = (row["seed"], row["selector"], row["engine"])
                    new = p[f"{row['selector']}_seed{row['seed']}_{row['engine']}"]
                    cm = np.zeros((3, 3), dtype=np.int64)
                    for truth, pred in zip(y, new):
                        cm[truth, pred] += 1
                    den = cm.sum(0) + cm.sum(1)
                    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(3), where=den != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
                    neg = (old == y) & (new != y)
                    pos = (old != y) & (new == y)
                    self.assertEqual(int(row["negative_flips"]), int(neg.sum()))
                    for r in per_class:
                        if (r["seed"], r["selector"], r["engine"]) == meta:
                            c = int(r["class_id"])
                            self.assertEqual(int(r["negative_flips"]), int((neg & (y == c)).sum()))
                            self.assertEqual(int(r["positive_flips"]), int((pos & (y == c)).sum()))
                    triplets = {}
                    for truth, before, after in zip(y, old, new):
                        if before != after:
                            key = (int(truth), int(before), int(after))
                            triplets[key] = triplets.get(key, 0) + 1
                    saved = {(int(r["true_class"]), int(r["source_prediction"]), int(r["repaired_prediction"])): int(r["count"])
                        for r in all_edges if (r["seed"], r["selector"], r["engine"]) == meta}
                    self.assertEqual(triplets, saved)
            coverage = kbs.read_csv(out / "query_coverage.csv")
            for c in choices:
                selected = [r for r in coverage if r["selector"] == c["selector"] and int(r["seed"]) == c["seed"]]
                self.assertEqual(sum(int(r["queried_count"]) for r in selected), protocol["budget"])
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertEqual(provenance["new_fits"], 0)
            self.assertEqual(provenance["source_sample_reconciliation"], "passed")
            stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensor")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume loaded array")), \
                 patch.object(audit, "predict_rows", side_effect=AssertionError("Resume predicted")):
                audit.audit(args)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_completion_gate_precedes_truth_and_source_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, _, _ = self.fixture(root, finish=False)
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth accessed early")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    audit.audit(args)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, _, _ = self.fixture(root)
            next((root / "out/runs").rglob("head.pt")).write_bytes(b"CORRUPT")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth accessed early")):
                with self.assertRaisesRegex(ValueError, "Missing/modified"):
                    audit.audit(args)

    def test_rehashed_stale_metrics_and_predictions_fail(self):
        for corrupt in ("by_seed.csv", "predictions.npz"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                args, _, _ = self.fixture(root)
                directory = root / "out/evaluation"
                if corrupt.endswith(".csv"):
                    rows = kbs.read_csv(directory / corrupt)
                    rows[0]["overall_macro_f1_after"] = .987654
                    kbs.write_csv(directory / corrupt, rows)
                else:
                    with np.load(directory / corrupt) as p:
                        data = {k: p[k] for k in p.files}
                    data["random_seed0_linear_care"] = (data["random_seed0_linear_care"] + 1) % 3
                    np.savez_compressed(directory / corrupt, **data)
                marker = json.loads((directory / "complete.json").read_text())
                marker["artifacts"][corrupt] = kbs.file_sha(directory / corrupt)
                kbs.atomic_json(directory / "complete.json", marker)
                with self.assertRaisesRegex(ValueError, "metric disagreement|does not reproduce"):
                    audit.audit(args)
                self.assertFalse((root / "audit/complete.json").exists())

    def test_changed_identity_outputs_and_threads_fail(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, _, _ = self.fixture(root)
            args.threads = 2
            with self.assertRaisesRegex(ValueError, "original CPU thread count"):
                audit.audit(args)
            args.threads = 1
            args.output_dir = args.source_dir
            with self.assertRaisesRegex(ValueError, "separate"):
                audit.audit(args)
            args.output_dir = str(root / "audit")
            Path(args.output_dir).mkdir()
            (Path(args.output_dir) / "other.txt").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                audit.audit(args)
            (Path(args.output_dir) / "other.txt").unlink()
            audit.audit(args)
            args.inference_batch_size = 2
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                audit.audit(args)

    def test_empty_denominators_and_absent_transition_seeds(self):
        self.assertIsNone(audit.ratios(0, 0))
        y, old, new = np.array([0, 0, 2, 2]), np.array([0, 0, 0, 1]), np.array([1, 0, 2, 0])
        groups = kbs.compare_predictions(y, old, new, np.ones(4, dtype=bool), 3, [2], [0], .1, .05)[1]
        rows = audit.class_flips(y, old, new, groups, 3)
        self.assertIsNone(rows[1]["negative_flip_rate"])
        self.assertIsNone(rows[1]["positive_flip_rate"])
        self.assertEqual(rows[2]["wrong_to_other_wrong"], 1)
        self.assertEqual({r["kind"] for r in audit.transitions(y, old, new, 3)}, {"negative", "positive", "wrong_to_other_wrong"})
        edge = {"seed": 0, "selector": "margin", "engine": "linear_care", "true_class": 0,
                "source_prediction": 0, "repaired_prediction": 1, "kind": "negative", "count": 6}
        summary = audit.transition_summary([edge], [0, 1, 2])[0]
        self.assertEqual(summary["count_mean"], 2)
        self.assertEqual(summary["seeds_with_transition"], 1)
        self.assertAlmostEqual(summary["count_sd"], np.std([6, 0, 0], ddof=1))
        values = audit.summarize([{"seed": 0, "group": "x", "rate": None},
                                  {"seed": 1, "group": "x", "rate": .2}], ("group",))[0]
        self.assertEqual(values["rate_valid_seeds"], 1)
        self.assertEqual(values["rate_mean"], .2)

    def test_cli_preflight_and_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, _, cli = self.fixture(root)
            for mode in ("preflight", "run"):
                proc = subprocess.run([sys.executable, audit.__file__, mode, *cli], text=True, capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn('"status": "ready"' if mode == "preflight" else "SMALL_REPAIR_FULL_AUDIT_COMPLETE", proc.stdout)
            self.assertTrue((Path(args.output_dir) / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
