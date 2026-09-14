"""Fixed-pool controls on synthetic tensors only; not research results."""
import contextlib
import copy
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
import kbs_pool_control as control
import kbs_acquisition_pilot as pilot
import kbs_acquisition_diagnosis as diag
import kbs_supplement as kbs
import test_kbs_acquisition_pilot as previous


class PoolControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        p = previous.AcquisitionTests().fixture(root)
        with contextlib.redirect_stdout(io.StringIO()):
            sources = pilot.load_sources(p)
            pilot.select(p, sources)
            pilot.evaluate(p, sources)
        return control.parser().parse_args(["select", "--study-dir", p.study_dir, "--cache-dir", p.cache_dir,
            "--checkpoint", p.checkpoint, "--pilot-dir", p.output_dir, "--output-dir", str(root / "controls")])

    def test_score_distance_probabilities_and_uniform_draws(self):
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [.6, .8]])
        explore = np.random.default_rng(2003).choice(6, 2, replace=False)
        pool = np.array([i for i in range(6) if i not in explore], dtype=np.int64)
        scores = np.array([.1, 1., 3., 9.])
        for policy in (*control.CONTROLS, "original_risk"):
            # Independent probability calculation for each sequential draw.
            rng = np.random.default_rng(2003)
            rng.choice(6, 2, replace=False)
            x = features[pool].numpy()
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            weights = scores if policy in ("pool_score", "original_risk") else np.ones(4)
            available = np.ones(4)
            distance = np.ones(4)
            expected = []
            for step in range(3):
                probability = weights * available * distance
                if probability.sum() <= 1e-15:
                    probability = weights * available
                j = rng.choice(4, p=probability / probability.sum())
                expected.append(pool[j])
                available[j] = 0
                if policy in ("pool_distance", "original_risk"):
                    d = np.maximum(2 - 2 * np.dot(x, x[j]), 0)
                    distance = d if step == 0 else np.minimum(distance, d)
            got = control.sample_pool(features, pool, scores, 3, explore, 0, torch.device("cpu"), policy)
            np.testing.assert_array_equal(got, expected)

    def test_original_hybrid_reconstruction_duplicates_empty_and_exhausted_pool(self):
        for features in (torch.ones(30, 3), torch.randn(30, 3, generator=torch.Generator().manual_seed(4))):
            for scores in (np.zeros(30), np.arange(30, dtype=float) + 1, np.r_[np.ones(2), np.zeros(28)]):
                old = pilot.diverse_query(features, scores, 10, 0, torch.device("cpu"))
                parts = diag.phases(old, 30, 10)
                pool = diag.candidate_pool(scores, np.zeros(30, dtype=int), np.zeros(1),
                    np.zeros(1, dtype=int), "risk_disagreement", parts["exploration"], 10)
                got = control.sample_pool(features, pool, scores[pool], old["specialist_count"],
                    parts["exploration"], 0, torch.device("cpu"), "original_risk")
                np.testing.assert_array_equal(got, parts["specialist"])
                for method in control.CONTROLS:
                    new = control.sample_pool(features, pool, scores[pool], len(got), parts["exploration"], 0, torch.device("cpu"), method)
                    ids = np.concatenate([parts["exploration"], new, parts["fallback"]])
                    self.assertEqual(len(np.unique(ids)), 10)
                    if len(parts["fallback"]):
                        self.assertEqual(set(new), set(pool))
        with self.assertRaises(ValueError):
            control.sample_pool(features, np.array([1]), np.ones(1), 1, np.array([1]), 0, "cpu", "pool_uniform")

    def test_hand_counted_empty_phase_and_summary(self):
        y = np.array([0, 0, 1, 2])
        pred = np.array([0, 1, 0, 2])
        conf = np.array([.99, .99, .99, .8])
        got, class_rows = control.cohort_metrics(y, pred, conf, np.arange(4), [0, 0, 0, 0], 4, [1], [0])
        self.assertEqual((got["n"], got["error_count"], got["high_conf_error_count"], got["collapse_query_count"]), (4, 2, 2, 1))
        self.assertEqual(got["collapse_fraction"], .25)
        self.assertEqual(got["proto_accuracy"], .5)
        self.assertEqual(class_rows[3]["queried"], 0)
        empty, _ = control.cohort_metrics(y, pred, conf, np.array([], dtype=int), [], 4, [1], [0])
        self.assertIsNone(empty["collapse_fraction"])
        self.assertEqual(empty["collapse_query_count"], 0)
        rows = [{"seed": i, "method": "x", "phase": "all", "n": n} for i, n in enumerate([2, 4])]
        summary = control.summarize(rows)[0]
        self.assertEqual(summary["n_mean"], 3)
        self.assertAlmostEqual(summary["n_sd"], np.sqrt(2))

    def test_selection_never_uses_target_labels_and_all_seeds_gate_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                verified = control.verify_inputs(args)
            original_load = torch.load
            def opaque_labels(path, *a, **kw):
                payload = original_load(path, *a, **kw)
                if Path(path).name == "target.pt":
                    payload["labels"] = object()
                return payload
            with contextlib.redirect_stdout(io.StringIO()), patch.object(torch, "load", side_effect=opaque_labels):
                control.select(args, verified)
            complete = Path(args.output_dir) / "seed_1/complete.json"
            complete.unlink()
            with patch.object(torch, "load", side_effect=AssertionError("evaluation must not load labels yet")), self.assertRaisesRegex(ValueError, "Every requested seed"):
                control.evaluate(args, verified)

    def test_pipeline_fairness_reconciliation_union_expectation_resume_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            source_files = {p for folder in (args.study_dir, args.cache_dir, args.pilot_dir) for p in Path(folder).rglob("*") if p.is_file()}
            source_files.add(Path(args.checkpoint))
            before = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in source_files}
            with contextlib.redirect_stdout(io.StringIO()):
                verified = control.verify_inputs(args)
                control.select(args, verified)
                control.evaluate(args, verified)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            for seed in (0, 1):
                selection = pilot.read_json(out / f"seed_{seed}/selection.json")
                pool = pilot.read_json(out / f"seed_{seed}/candidate_pool.json")
                original = verified[1][seed]["choices"]
                fixed = diag.phases(selection["choices"]["original_risk"], 90, 10)
                self.assertEqual(selection["choices"]["original_risk"]["row_indices"], original["risk_disagreement"]["row_indices"])
                for method in control.CONTROLS:
                    parts = diag.phases(selection["choices"][method], 90, 10)
                    np.testing.assert_array_equal(parts["exploration"], fixed["exploration"])
                    np.testing.assert_array_equal(parts["fallback"], fixed["fallback"])
                    self.assertTrue(set(parts["specialist"]).issubset(pool["row_indices"]))
                    phase_rows = {r["phase"]: r for r in rows if int(r["seed"]) == seed and r["method"] == method}
                    for key in ("n", "collapse_query_count", "error_count", "high_conf_error_count", "proto_correct"):
                        self.assertEqual(int(phase_rows["all"][key]), sum(int(phase_rows[p][key]) for p in ("exploration", "specialist", "fallback")))
                union = {i for group in (original, selection["choices"]) for choice in group.values() for i in choice["row_indices"]}
                self.assertEqual(pilot.read_json(out / f"seed_{seed}/common_excluded_ids.json")["row_indices"], sorted(union))
            for row in kbs.read_csv(out / "evaluation/uniform_expectation.csv"):
                expected = int(row["specialist_n"]) * int(row["pool_collapse_count"]) / int(row["pool_n"])
                self.assertAlmostEqual(float(row["analytic_expected_uniform_specialist_collapse"]), expected)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 2 * 3 * 3)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), len(rows) * 3)
            results_before = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(torch, "load", side_effect=AssertionError("no tensor loading on resume")):
                again = control.verify_inputs(args)
                control.select(args, again)
                control.evaluate(args, again)
            self.assertEqual(results_before, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()})
            self.assertEqual(before, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in source_files})

    def test_corruption_changed_runtime_source_or_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.batch_size = 8192
            with self.assertRaisesRegex(ValueError, "original batch_size"):
                control.verify_inputs(args)
            args.batch_size = None
            with contextlib.redirect_stdout(io.StringIO()):
                verified = control.verify_inputs(args)
                control.select(args, verified)
            (Path(args.output_dir) / "seed_0/candidate_pool.json").write_text("broken")
            with self.assertRaises(ValueError):
                control.evaluate(args, verified)
            args.seeds = "0"
            with contextlib.redirect_stdout(io.StringIO()):
                changed = control.verify_inputs(args)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                control.select(args, changed)
            (Path(args.pilot_dir) / "seed_0/selection.json").write_text("broken")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                control.verify_inputs(args)

    def test_changed_original_specialist_order_fails_before_new_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                verified = control.verify_inputs(args)
            reference = torch.load(Path(args.cache_dir) / "reference.pt", weights_only=True)
            target = torch.load(Path(args.cache_dir) / "target.pt", weights_only=True)
            pred, conf, tc = pilot.logits_summary(target["logits"], args.batch_size)
            _, _, rc = pilot.logits_summary(reference["logits"], args.batch_size)
            changed = copy.deepcopy(verified[1][0])
            choice = changed["choices"]["risk_disagreement"]
            first = choice["exploration_count"]
            for key in ("row_indices", "base_score", "inferred_class"):
                choice[key][first], choice[key][first + 1] = choice[key][first + 1], choice[key][first]
            with self.assertRaisesRegex(ValueError, "query order changed"):
                control.select_seed(reference["features"], reference["labels"].numpy(), target["features"],
                    pred, conf, pilot.frequency_risk(rc, tc), 3, 10, 0, changed, "cpu", args.batch_size)

    def test_separate_cli_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            cli = [sys.executable, str(Path(control.__file__))]
            options = [item for name in ("study_dir", "cache_dir", "pilot_dir", "output_dir", "checkpoint")
                       for item in ("--" + name.replace("_", "-"), getattr(args, name))]
            result = subprocess.run(cli + ["select"] + options, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(args.output_dir) / "evaluation").exists())
            result = subprocess.run(cli + ["evaluate"] + options, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
