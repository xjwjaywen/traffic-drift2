"""Synthetic correctness tests, never scientific evidence for the proposed method."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_repair_aware as pilot
import kbs_supplement as kbs


class QueryOnlyLabels:
    """Test oracle: any access other than an allowed complete purchase fails."""
    def __init__(self, labels, allowed):
        self.labels, self.allowed, self.accesses = labels, allowed, []

    def __getitem__(self, ids):
        value = tuple(np.asarray(ids).tolist())
        if value not in self.allowed:
            raise AssertionError(f"Unpurchased labels accessed: {value}")
        self.accesses.append(value)
        return self.labels[ids]

    def __array__(self, *args, **kwargs):
        raise AssertionError("Full target truth is not allowed")


class RepairAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        cache, study = root / "cache", root / "study"
        cache.mkdir(); study.mkdir()
        gen = torch.Generator().manual_seed(73)
        head = torch.nn.Linear(4, 3)
        with torch.no_grad():
            head.weight.copy_(torch.randn(3, 4, generator=gen))
            head.bias.zero_()
        reference = torch.randn(30, 4, generator=gen)
        target = torch.randn(120, 4, generator=gen)
        checkpoint = root / "checkpoint.pt"
        torch.save(head.state_dict(), checkpoint)
        torch.save(head.state_dict(), cache / "head.pt")
        for name, x, y in [("reference", reference, torch.arange(30) % 3),
                            ("target", target, torch.arange(120) % 3)]:
            torch.save({"features": x, "logits": head(x).detach(), "labels": y}, cache / f"{name}.pt")
        periods = {"reference": "M-2022-4", "target": "M-2022-12"}
        spec = {"schema": kbs.SCHEMA, "loader_sha256": kbs.loader_sha(), "periods": periods}
        inputs = {"reference": "SYNTHETIC", "target": "SYNTHETIC"}
        info = {"spec": spec, "checkpoint_sha256": kbs.file_sha(checkpoint), "num_classes": 3,
                "periods": periods, "sample_counts": {"reference": 30, "target": 120},
                "input_stream_sha256": inputs, "fingerprint": kbs.digest({"spec": spec, "inputs": inputs}),
                "files": {n: kbs.file_sha(cache / n) for n in ("head.pt", "reference.pt", "target.pt")}}
        kbs.atomic_json(cache / "manifest.json", info)
        p = {"cache": info["fingerprint"], "checkpoint_sha256": info["checkpoint_sha256"],
             "implementation_sha256": kbs.implementation_sha(), "num_classes": 3, "budget": 10,
             "optimizer_steps": 3, "batch_size": 4, "lr": .02, "weight_decay": .0001,
             "target_weight": .7, "replay_weight": .3, "collapse_classes": [0], "stable_classes": [2],
             "collapse_recall_threshold": .1, "f1_drop_threshold": .05}
        kbs.atomic_json(study / "study_manifest.json", {"protocol": p})
        for seed in (0, 1):
            d = study / "selections/badge" / f"seed_{seed}"
            sig = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
                   "selector": "badge", "budget": 10, "seed": seed}
            kbs.atomic_json(d / "resolved_config.json", sig)
            kbs.atomic_json(d / "selection.json", {"row_indices": list(range(seed, seed + 10))})
            kbs.finish(d, sig, ["resolved_config.json", "selection.json"])
        return pilot.parser().parse_args(["select", "--study-dir", str(study), "--cache-dir", str(cache),
            "--checkpoint", str(checkpoint), "--output-dir", str(root / "pilot"), "--device", "cpu",
            "--seeds", "0,1", "--threads", "1", "--batch-size", "19"])

    def snapshot(self, *folders):
        return {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns)
                for folder in folders for p in Path(folder).rglob("*") if p.is_file()}

    def guard(self, allowed):
        original = torch.load
        def load(path, **kwargs):
            obj = original(path, **kwargs)
            if Path(path).name == "target.pt":
                obj["labels"] = QueryOnlyLabels(obj["labels"], allowed)
            return obj
        return load

    def test_relation_direction_and_weighted_query_hand_calculation(self):
        counts = pilot.relation_counts(np.array([1, 1, 2, 2]), np.array([0, 0, 2, 1]), 3)
        np.testing.assert_array_equal(counts, [[0, 2, 0], [0, 0, 1], [0, 0, 0]])
        # A victim 0 -> absorber 1 error makes a later 1 -> 0 flip a protected candidate.
        old = np.array([0] * 8 + [1, 1, 2, 2, 0, 1, 2, 0])
        probe = old.copy(); probe[8:12] = [0, 2, 1, 0]
        rec = pilot.select_queries(np.arange(10), old, probe, counts, 0)
        pool = np.array([8, 9, 10, 11])
        expected = np.random.default_rng(42011).choice(pool, 2, replace=False, p=np.array([3, 1, 2, 1]) / 7)
        np.testing.assert_array_equal(rec["choices"]["flip_relation"]["row_indices"][8:], np.sort(expected))
        self.assertEqual(rec["relation_supported_flip_count"], 2)
        self.assertEqual(rec["choices"]["badge"]["row_indices"], list(range(10)))

    def test_empty_small_pool_zero_relations_uniqueness_and_no_hard_gate(self):
        old = np.zeros(30, dtype=np.int64)
        for flips in ([], [12], list(range(10, 25))):
            probe = old.copy(); probe[flips] = 1
            r = pilot.select_queries(np.arange(10), old, probe, np.zeros((2, 2), dtype=int), 3)
            a, b = (r["choices"][m] for m in ("flip_uniform", "flip_relation"))
            self.assertEqual(a, b)
            self.assertEqual(a["fallback_count"], max(0, 2 - len(flips)))
            for choice in r["choices"].values():
                self.assertEqual(len(set(choice["row_indices"])), 10)
                self.assertEqual(choice["row_indices"][:8], list(range(8)))
        self.assertEqual(r, pilot.select_queries(np.arange(10), old, probe, np.zeros((2, 2), dtype=int), 3))
        # If all flip candidates fit, relation weighting must not change final
        # labels or their training order, nor the global fallback sample.
        probe = old.copy(); probe[[12, 14]] = 1
        weighted = np.array([[0, 0], [100, 0]])
        for flips in ([12], [12, 14]):
            probe = old.copy(); probe[flips] = 1
            r = pilot.select_queries(np.arange(10), old, probe, weighted, 3)
            self.assertEqual(r["choices"]["flip_uniform"], r["choices"]["flip_relation"])

    def test_metrics_hand_count_and_sample_sd(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        old = np.array([1, 1, 1, 1, 2, 2])
        pred = np.array([0, 1, 0, 1, 1, 2])
        p = {"num_classes": 3, "collapse_classes": [0], "stable_classes": [2],
             "collapse_recall_threshold": .1, "f1_drop_threshold": .05}
        m, c = pilot.metrics(y, old, pred, np.array([True, True, True, True, False, True]), p)
        self.assertEqual(m["all_positive_flips"], 1)
        self.assertEqual(m["noncollapse_negative_flips"], 1)
        self.assertAlmostEqual(m["noncollapse_negative_flip_rate_on_old_correct"], 1/3)
        self.assertEqual(c[1]["negative_flips"], 1)
        rows = pilot.aggregate([{"method": "a", "seed": 0, "x": 1}, {"method": "a", "seed": 1, "x": 3}], "method")
        self.assertAlmostEqual(rows[0]["x_sd"], np.sqrt(2))
        self.assertIsNone(pilot.aggregate([{"method": "a", "seed": 0, "x": 1}], "method")[0]["x_sd"])

    def test_end_to_end_budget_gates_pairing_recompute_resume_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            sources = pilot.verify_sources(args)
            before = self.snapshot(args.study_dir, args.cache_dir)
            out = Path(args.output_dir)
            expected_head = torch.load(Path(args.cache_dir) / "head.pt", weights_only=True)
            original_fit = kbs.fit_controlled
            fits = []
            def checked_fit(head, x, y, rx, ry, teacher, **kwargs):
                for k, v in expected_head.items():
                    torch.testing.assert_close(head.state_dict()[k], v)
                self.assertFalse(kwargs["replay_ce"])
                self.assertEqual(kwargs["kd_weight"], .5)
                self.assertEqual(len(rx), 15)
                self.assertEqual(kwargs["steps"], 3)
                saved = copy.deepcopy(head.state_dict())
                result = original_fit(head, x, y, rx, ry, teacher, **kwargs)
                for k, v in saved.items():
                    torch.testing.assert_close(head.state_dict()[k], v)
                fits.append(len(x))
                return result
            scout_sets = {tuple(range(s, s + 8)) for s in (0, 1)}
            with patch.object(torch, "load", side_effect=self.guard(scout_sets)), patch.object(kbs, "fit_controlled", side_effect=checked_fit):
                pilot.select(args, sources)
            self.assertEqual(fits, [8, 8])
            records = pilot.require_selections(out, sources)
            queries = {tuple(c["row_indices"]) for r in records.values() for c in r["choices"].values()}
            with patch.object(torch, "load", side_effect=self.guard(queries)), patch.object(kbs, "fit_controlled", side_effect=checked_fit):
                pilot.train(args, sources)
            self.assertEqual(fits, [8, 8] + [10] * 8)
            pilot.evaluate(args, sources)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 10)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 10)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 30)
            target = torch.load(Path(args.cache_dir) / "target.pt", weights_only=True)
            y, old = target["labels"].numpy(), target["logits"].argmax(1).numpy()
            for seed, record in records.items():
                mask = np.ones(120, dtype=bool); mask[record["common_excluded_indices"]] = False
                for method in pilot.METHODS:
                    saved = next(r for r in rows if int(r["seed"]) == seed and r["method"] == method)
                    pred = np.load(pilot.run_dir(out, seed, method) / "predictions.npy")
                    calculated, _ = pilot.metrics(y, old, pred, mask, sources[1])
                    self.assertAlmostEqual(float(saved["overall_macro_f1_after"]), calculated["overall_macro_f1_after"])
                    self.assertEqual(int(saved["eval_samples"]), int(mask.sum()))
            saved = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("resume should not load tensors")):
                pilot.select(args, sources); pilot.train(args, sources); pilot.evaluate(args, sources)
            self.assertEqual(saved, self.snapshot(out))
            self.assertEqual(before, self.snapshot(args.study_dir, args.cache_dir))

    def test_unqueried_label_changes_cannot_change_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp)); sources = pilot.verify_sources(args)
            pilot.select(args, sources)
            first = pilot.require_selections(Path(args.output_dir), sources)
            original = torch.load
            def swapped(path, **kwargs):
                obj = original(path, **kwargs)
                if Path(path).name == "target.pt":
                    # union of both seeds' scouts is rows 0..8; change every other truth.
                    obj["labels"][9:] = (obj["labels"][9:] + 1) % 3
                return obj
            args.output_dir = str(Path(tmp) / "second")
            with patch.object(torch, "load", side_effect=swapped):
                pilot.select(args, sources)
            self.assertEqual(first, pilot.require_selections(Path(args.output_dir), sources))

    def test_incomplete_stages_block_before_label_load_and_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp)); sources = pilot.verify_sources(args)
            pilot.select(args, sources)
            out = Path(args.output_dir)
            with patch.object(torch, "load", side_effect=AssertionError("must not load truth")):
                with self.assertRaises(ValueError): pilot.evaluate(args, sources)
            (pilot.selection_dir(out, 1) / "complete.json").unlink()
            with patch.object(torch, "load", side_effect=AssertionError("must not load truth")):
                with self.assertRaises(ValueError): pilot.train(args, sources)
            pilot.select(args, sources)
            pilot.train(args, sources)
            path = pilot.run_dir(out, 0, "badge") / "predictions.npy"
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError): pilot.evaluate(args, sources)
            changed = list(sources); changed[3] = {**sources[3], "threads": 2}
            with self.assertRaises(ValueError): pilot.select(args, changed)

    def test_cli_stages_and_bad_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            common = ["--study-dir", args.study_dir, "--cache-dir", args.cache_dir,
                      "--checkpoint", args.checkpoint, "--output-dir", args.output_dir,
                      "--device", "cpu", "--threads", "1", "--seeds", "0,1", "--batch-size", "19"]
            for stage in ("preflight", "select", "train", "evaluate"):
                result = subprocess.run([sys.executable, pilot.__file__, stage, *common], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(args.output_dir) / "evaluation/report.md").exists())
            path = Path(args.study_dir) / "selections/badge/seed_0/selection.json"
            path.write_text("{}")
            with self.assertRaises(ValueError): pilot.verify_sources(args)


if __name__ == "__main__":
    unittest.main()
