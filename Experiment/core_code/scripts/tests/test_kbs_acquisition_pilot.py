"""Synthetic acquisition tests; these scores are not research results."""
import inspect
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
import kbs_acquisition_pilot as pilot
import kbs_supplement as kbs


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tensors(self):
        gen = torch.Generator().manual_seed(8)
        ref_labels = torch.arange(3).repeat_interleave(8)
        reference = torch.eye(3)[ref_labels] + .02 * torch.randn(24, 3, generator=gen)
        labels = torch.arange(3).repeat_interleave(30)
        target = torch.eye(3)[labels] + .08 * torch.randn(90, 3, generator=gen)
        logits = 8. * torch.eye(3)[labels]
        logits[:30] = torch.tensor([0., 8., 0.])
        return reference, ref_labels, target, labels, logits

    def fixture(self, root):
        cache, study = root / "cache", root / "study"
        cache.mkdir()
        study.mkdir()
        reference, ry, x, y, logits = self.tensors()
        checkpoint = root / "source.pt"
        checkpoint.write_bytes(b"synthetic checkpoint only")
        torch.save({}, cache / "head.pt")
        torch.save({"features": reference, "labels": ry, "logits": 8. * torch.eye(3)[ry]}, cache / "reference.pt")
        torch.save({"features": x, "labels": y, "logits": logits}, cache / "target.pt")
        spec = {"schema": kbs.SCHEMA, "loader_sha256": kbs.loader_sha()}
        inputs = {"reference": "synthetic-ref", "target": "synthetic-target"}
        info = {"spec": spec, "checkpoint_sha256": kbs.file_sha(checkpoint), "num_classes": 3,
                "periods": {"reference": "fixture-ref", "target": "fixture-target"},
                "sample_counts": {"reference": 24, "target": 90}, "input_stream_sha256": inputs,
                "fingerprint": kbs.digest({"spec": spec, "inputs": inputs}),
                "files": {n: kbs.file_sha(cache / n) for n in ["head.pt", "reference.pt", "target.pt"]}}
        kbs.atomic_json(cache / "manifest.json", info)
        protocol = {"cache": info["fingerprint"], "checkpoint_sha256": info["checkpoint_sha256"],
                    "implementation_sha256": kbs.implementation_sha(), "num_classes": 3,
                    "budget": 10, "collapse_classes": [0], "stable_classes": [2]}
        kbs.atomic_json(study / "study_manifest.json", {"protocol": protocol})
        for seed in (0, 1):
            for method in ("badge", "margin"):
                directory = study / "selections" / method / f"seed_{seed}"
                signature = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
                             "selector": method, "budget": 10, "seed": seed}
                kbs.atomic_json(directory / "resolved_config.json", signature)
                kbs.atomic_json(directory / "selection.json", {"row_indices": list(range(10 + seed, 20 + seed))})
                kbs.finish(directory, signature, ["resolved_config.json", "selection.json"])
        return pilot.parser().parse_args(["select", "--study-dir", str(study), "--cache-dir", str(cache),
                                         "--checkpoint", str(checkpoint), "--output-dir", str(root / "pilot"),
                                         "--device", "cpu", "--seeds", "0,1", "--batch-size", "17", "--threads", "1"])

    def test_reference_ids_match_existing_protocol_and_risk_uses_predictions(self):
        ref, ry, *_ = self.tensors()
        proto, ids = pilot.make_prototypes(ref, ry.numpy(), 3, 0)
        np.testing.assert_array_equal(ids, kbs.replay_indices(ry.numpy(), 3, 5, 0))
        np.testing.assert_array_equal(np.bincount(ry.numpy()[ids]), [5, 5, 5])
        torch.testing.assert_close(proto.norm(dim=1), torch.ones(3))
        # Laplace/Jeffreys smoothing: 1 - (0.5/31.5)/(10.5/31.5).
        risk = pilot.frequency_risk([10, 10, 10], [0, 20, 10])
        self.assertAlmostEqual(risk[0], 1 - .5 / 10.5)
        np.testing.assert_array_equal(risk[1:], [0, 0])

    def test_disagreement_finds_synthetic_confident_errors_without_target_labels(self):
        ref, ry, x, y, logits = self.tensors()
        proto, _ = pilot.make_prototypes(ref, ry.numpy(), 3, 0)
        pred, conf, _ = pilot.logits_summary(logits, 17)
        a, score = pilot.prototype_scores(x, proto, pred, conf, torch.device("cpu"), 17)
        np.testing.assert_array_equal(a, y.numpy())
        self.assertTrue(np.all(score[:30] > 0))
        self.assertTrue(np.all(score[30:] == 0))
        self.assertNotIn("target_labels", inspect.signature(pilot.select_seed).parameters)
        self.assertNotIn("collapse", inspect.signature(pilot.select_seed).parameters)

    def test_fallback_duplicate_features_and_reproducibility(self):
        x = torch.ones(60, 3)
        scores = np.zeros(60)
        a = pilot.diverse_query(x, scores, 10, 0, torch.device("cpu"))
        self.assertEqual(a["specialist_count"], 0)
        self.assertEqual(a["fallback_count"], 5)
        self.assertEqual(len(set(a["row_indices"])), 10)
        scores[:] = 1
        b = pilot.diverse_query(x, scores, 10, 0, torch.device("cpu"))
        c = pilot.diverse_query(x, scores, 10, 0, torch.device("cpu"))
        self.assertEqual(b, c)
        self.assertEqual(a["row_indices"][:5], b["row_indices"][:5])
        self.assertEqual(b["specialist_count"], 5)

    def test_hand_calculated_metrics_and_unsupported_groups(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([1, 1, 1, 0, 2, 1])
        confidence = np.array([.95, .7, .95, .91, .8, .99])
        choice = {"exploration_count": 1, "specialist_count": 2, "inferred_class": [2, 1, 2]}
        metrics, counts = pilot.acquisition_metrics(labels, pred, confidence, [0, 3, 4], 4, [0, 3], [2], choice)
        self.assertEqual(metrics["high_conf_error_count"], 2)
        self.assertAlmostEqual(metrics["high_conf_error_enrichment"], 4 / 3)
        self.assertEqual(metrics["collapse_query_count"], 1)
        self.assertEqual(metrics["collapse_classes_covered"], 1)
        self.assertEqual(metrics["noncollapse_classes_covered"], 2)
        self.assertEqual(metrics["specialist_inferred_accuracy"], 1)
        np.testing.assert_array_equal(counts, [1, 1, 1, 0])
        zero, _ = pilot.acquisition_metrics(labels, labels, confidence, [0, 3], 4, [3], [], {})
        self.assertIsNone(zero["high_conf_error_enrichment"])

    def test_end_to_end_no_label_dependence_resume_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            before = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for folder in [args.cache_dir, args.study_dir]
                      for p in Path(folder).rglob("*") if p.is_file()}
            sources = pilot.load_sources(args)
            original_load = torch.load
            def hidden_labels(path, **kwargs):
                value = original_load(path, **kwargs)
                if Path(path).name == "target.pt":
                    # Labels cannot be indexed, converted to NumPy, iterated or inspected by selection.
                    value["labels"] = object()
                return value
            with patch.object(torch, "load", side_effect=hidden_labels):
                pilot.select(args, sources)
            out = Path(args.output_dir)
            selections_before = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
            pilot.select(args, sources)
            self.assertEqual(selections_before, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()})
            pilot.evaluate(args, sources)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 36)
            for seed in (0, 1):
                chosen = pilot.read_json(out / f"seed_{seed}/selection.json")
                self.assertEqual(sorted(chosen["risk_permutation"]), [0, 1, 2])
                union = set(i for c in chosen["choices"].values() for i in c["row_indices"])
                excluded = pilot.read_json(out / f"seed_{seed}/common_excluded_ids.json")
                self.assertEqual(excluded["row_indices"], sorted(union))
                self.assertEqual(excluded["remaining_count"], 90 - len(union))
                self.assertEqual(chosen["choices"]["badge"]["row_indices"], list(range(10 + seed, 20 + seed)))
            summary = kbs.read_csv(out / "evaluation/summary.csv")
            for record in summary:
                vals = [float(r["high_conf_error_count"]) for r in rows if r["method"] == record["method"]]
                self.assertAlmostEqual(float(record["high_conf_error_count_mean"]), np.mean(vals))
                self.assertAlmostEqual(float(record["high_conf_error_count_sd"]), np.std(vals, ddof=1))
            saved = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
            pilot.evaluate(args, sources)
            self.assertEqual(saved, {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()})
            after = {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns) for folder in [args.cache_dir, args.study_dir]
                     for p in Path(folder).rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            # A fresh output with arbitrary replacement target-label object gives identical query IDs.
            args.output_dir = str(Path(tmp) / "pilot-copy")
            with patch.object(torch, "load", side_effect=hidden_labels):
                pilot.select(args, sources)
            for seed in (0, 1):
                a = pilot.read_json(out / f"seed_{seed}/selection.json")["choices"]
                b = pilot.read_json(Path(args.output_dir) / f"seed_{seed}/selection.json")["choices"]
                for method in pilot.METHODS:
                    self.assertEqual(a[method]["row_indices"], b[method]["row_indices"])

    def test_evaluation_requires_every_seed_before_loading_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            sources = pilot.load_sources(args)
            pilot.select(args, sources)
            (Path(args.output_dir) / "seed_1/complete.json").unlink()
            with patch.object(torch, "load", side_effect=AssertionError("must not load labels")):
                with self.assertRaises(ValueError):
                    pilot.evaluate(args, sources)

    def test_mismatch_corruption_and_duplicate_ids_fail(self):
        for ids in ([1, 1], [-1, 2], [1.5, 2.5]):
            with self.assertRaises(ValueError):
                pilot.check_ids(ids, 10, 2)
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            sources = pilot.load_sources(args)
            pilot.select(args, sources)
            changed = list(sources)
            changed[3] = {**changed[3], "batch_size": 99}
            with self.assertRaises(ValueError):
                pilot.select(args, changed)
            (Path(args.output_dir) / "seed_0/selection.json").write_text("{}")
            with self.assertRaises(ValueError):
                pilot.evaluate(args, sources)
            (Path(args.study_dir) / "selections/badge/seed_0/selection.json").write_text("{}")
            with self.assertRaises(ValueError):
                pilot.load_sources(args)
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.output_dir = str(Path(args.study_dir) / "accidental-output")
            with self.assertRaises(ValueError):
                pilot.load_sources(args)

    def test_cli_separate_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            common = ["--study-dir", args.study_dir, "--cache-dir", args.cache_dir, "--output-dir", args.output_dir,
                      "--checkpoint", args.checkpoint, "--device", "cpu", "--seeds", "0,1", "--threads", "1"]
            for mode in ("select", "evaluate"):
                result = subprocess.run([sys.executable, pilot.__file__, mode, *common], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
