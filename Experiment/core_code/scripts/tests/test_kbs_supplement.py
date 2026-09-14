"""CPU regression tests with synthetic inputs; never produce paper evidence."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_supplement as kbs


class SupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(30, 4, generator=gen)
        r = torch.randn(24, 4, generator=gen)
        head = torch.nn.Linear(4, 3)
        with torch.no_grad():
            head.weight.copy_(torch.randn(3, 4, generator=gen))
            head.bias.zero_()
        return head, x, torch.arange(30) % 3, r, torch.arange(24) % 3

    def fit(self, replay_ce=True, kd_weight=.5, seed=0, ref_labels=None):
        head, x, y, r, ry = self.fixture()
        with torch.no_grad():
            logits = head(r).clone()
        original = copy.deepcopy(head.state_dict())
        fitted, trace = kbs.fit_controlled(
            head, x[:9], y[:9], r, ry if ref_labels is None else ref_labels,
            logits, replay_ce=replay_ce, kd_weight=kd_weight,
            temperature=2., target_weight=.7, replay_weight=.3,
            steps=4, batch_size=8, lr=.01, weight_decay=1e-4,
            seed=seed, device=torch.device("cpu"), log_every=0)
        for key in original:
            torch.testing.assert_close(head.state_dict()[key], original[key])
        return fitted, trace

    def test_primary_and_sensitivity_plan(self):
        primary = kbs.make_specs("primary")
        sensitivity = kbs.make_specs("sensitivity")
        self.assertEqual(len(primary), 5)
        self.assertEqual(len(sensitivity), 8)
        self.assertEqual(sum(x["selector"] == "badge" for x in primary), 1)
        self.assertEqual(len({kbs.spec_key(x) for x in sensitivity}), 8)
        self.assertEqual(len(set(map(kbs.spec_key, primary)) &
                             set(map(kbs.spec_key, sensitivity))), 2)

    def test_update_count_and_target_stream_match_all_ablations(self):
        traces = []
        for replay_ce, kd in [(False, 0.), (True, 0.), (False, .5), (True, .5)]:
            _, trace = self.fit(replay_ce, kd)
            traces.append(trace)
            self.assertEqual(trace["optimizer_steps"], 4)
            self.assertEqual(trace["target_presentations"], 32)
            self.assertEqual(trace["reference_ce_presentations"], 32 if replay_ce else 0)
            self.assertEqual(trace["reference_kd_presentations"], 32 if kd else 0)
        self.assertEqual(len({t["target_stream_sha256"] for t in traces}), 1)
        self.assertEqual(len({t["reference_stream_sha256"] for t in traces}), 1)

    def test_kd_only_does_not_use_reference_labels(self):
        a, _ = self.fit(False, .5, ref_labels=torch.zeros(24, dtype=torch.long))
        b, _ = self.fit(False, .5, ref_labels=torch.ones(24, dtype=torch.long))
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name])

    def test_seed_repeatability(self):
        a, _ = self.fit()
        b, _ = self.fit()
        for name in a.state_dict():
            torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name])

    def test_replay_nested_balanced(self):
        labels = np.tile(np.arange(3), 20)
        a = kbs.replay_indices(labels, 3, 2, 0)
        b = kbs.replay_indices(labels, 3, 5, 0)
        self.assertTrue(set(a).issubset(set(b)))
        np.testing.assert_array_equal(np.bincount(labels[b]), [5, 5, 5])
        with self.assertRaises(ValueError):
            kbs.replay_indices(np.array([0, 0, 1]), 3, 2, 0)

    def test_metrics_use_global_confusion_and_paired_mask(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        before = np.array([0, 0, 0, 0, 2, 2])
        after = np.array([1, 0, 1, 1, 2, 2])
        mask = np.array([False, True, True, True, True, True])
        metrics, rows, cm = kbs.compare_predictions(
            labels, before, after, mask, 3, [1], [2], .1, .05)
        self.assertEqual(metrics["eval_samples"], 5)
        self.assertAlmostEqual(rows[0]["before_f1"], .5)
        self.assertAlmostEqual(rows[0]["after_f1"], 1.)
        self.assertAlmostEqual(metrics["collapse_macro_f1_after"], 1.)
        self.assertAlmostEqual(metrics["noncollapse_macro_f1_after"], 1.)
        self.assertEqual(cm["before"].sum(), 5)

    def test_unsupported_classes_not_counted_as_new_collapse(self):
        metrics, rows, _ = kbs.compare_predictions(
            np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
            np.array([True, True]), 3, [1], [0], .1, .05)
        self.assertEqual(metrics["noncollapse_new_collapses"], 0)
        self.assertEqual(metrics["noncollapse_supported_classes"], 1)
        self.assertEqual(rows[2]["support"], 0)

    def test_noncollapse_damage_is_not_hidden_by_stable_subset(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        metrics, _, _ = kbs.compare_predictions(
            labels, labels.copy(), np.array([1, 1, 1, 1, 2, 2]),
            np.ones(6, dtype=bool), 3, [1], [2], .1, .05)
        self.assertEqual(metrics["stable_macro_f1_delta"], 0.)
        self.assertEqual(metrics["noncollapse_degraded_count"], 1)
        self.assertEqual(metrics["noncollapse_drop_gt_threshold_count"], 1)
        self.assertEqual(metrics["noncollapse_new_collapses"], 1)
        self.assertEqual(metrics["noncollapse_macro_f1_delta"], -.5)


    def test_completion_requires_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "metric.txt").write_text("valid")
            kbs.finish(directory, {"protocol": 1}, ["metric.txt"])
            self.assertTrue(kbs.is_complete(directory, {"protocol": 1}))
            with self.assertRaises(ValueError):
                kbs.is_complete(directory, {"protocol": 2})
            (directory / "metric.txt").write_text("changed")
            with self.assertRaises(ValueError):
                kbs.is_complete(directory, {"protocol": 1})

    def test_feature_prepare_cache_reuse_and_invalidation(self):
        import prototype_recalibration_tls22 as proto

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cls_head = torch.nn.Module()
                self.cls_head.fc = torch.nn.Linear(4, 3)

            def forward(self, ppi, flow_stats=None, return_repr=False):
                features = ppi.flatten(1)[:, :4]
                logits = self.cls_head.fc(features)
                return (logits, features) if return_repr else logits

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "data"
            raw.mkdir()
            (raw / "fixture.bin").write_bytes(b"TEST_ONLY")
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"mocked source checkpoint")
            args = kbs.parser().parse_args([
                "prepare", "--cache-dir", str(root / "cache"),
                "--checkpoint", str(checkpoint), "--device", "cpu"])
            spec = {
                "schema": kbs.SCHEMA, "data_config": {"data_dir": str(raw)},
                "checkpoint_sha256": kbs.file_sha(checkpoint),
                "periods": {"reference": args.reference_period, "target": args.target_period},
                "loader_sha256": kbs.loader_sha(), "data_seed": 0}
            batch = {"ppi": torch.arange(48, dtype=torch.float32).reshape(12, 1, 4),
                     "label": torch.arange(12) % 3}
            with patch.object(kbs, "preflight", return_value=({}, spec)), \
                 patch.object(proto, "load_source_model", return_value=(TinyModel(), {}, 3)), \
                 patch.object(proto, "make_test_loader", return_value=([batch], 3)) as loader:
                kbs.prepare(args)
                self.assertEqual(loader.call_count, 2)
                payload = kbs.load_cache(args)
                self.assertEqual(payload["target"]["features"].shape, (12, 4))
                torch.testing.assert_close(payload["head"](payload["target"]["features"]),
                                           payload["target"]["logits"])
                kbs.prepare(args)
                self.assertEqual(loader.call_count, 2)
                (raw / "fixture.bin").write_bytes(b"CHANGED_INPUT")
                with self.assertRaises(ValueError):
                    kbs.prepare(args)
            args.target_period = "M-2022-11"
            with self.assertRaises(ValueError):
                kbs.load_cache(args)

    def test_real_badge_selector_without_target_labels(self):
        head, x, _, _, _ = self.fixture()
        idx = kbs.select_query("badge", x, head(x).detach(), 6, 0)
        self.assertEqual(len(np.unique(idx)), 6)
        self.assertTrue(((idx >= 0) & (idx < len(x))).all())


    def test_synthetic_pipeline_resume_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head, x, y, r, ry = self.fixture()
            r, ry = r.repeat(2, 1), ry.repeat(2)  # Enough reference support for k=10 sensitivity.
            data = {"target": {"features": x, "labels": y.numpy(),
                               "logits": head(x).detach()},
                    "reference": {"features": r, "labels": ry.numpy(),
                                  "logits": head(r).detach()},
                    "head": head, "num_classes": 3,
                    "manifest": {"fingerprint": "SYNTHETIC_TEST_ONLY",
                                 "checkpoint_sha256": "synthetic",
                                 "periods": {"target": "test", "reference": "ref"}}}
            args = kbs.parser().parse_args([
                "run", "--output-dir", str(root / "out"),
                "--device", "cpu", "--seeds", "0",
                "--budget", "6", "--steps", "2",
                "--batch-size", "4", "--collapse-classes", "1",
                "--stable-classes", "2", "--log-every", "0"])
            def selector(name, features, logits, budget, seed):
                return np.arange(budget) if name == "margin" else np.arange(budget, 2*budget)
            with patch.object(kbs, "select_query", side_effect=selector):
                kbs.run_study(args, data)
                files = list((root / "out" / "runs").glob("*/seed_0/metrics.json"))
                self.assertEqual(len(files), 5)
                stamps = {f: f.stat().st_mtime_ns for f in files}
                kbs.run_study(args, data)
                self.assertEqual(stamps, {f: f.stat().st_mtime_ns for f in files})
                args.suite = "sensitivity"
                kbs.run_study(args, data)
                self.assertEqual(stamps, {f: f.stat().st_mtime_ns for f in files})
                self.assertEqual(len(list((root / "out" / "runs").glob("*/seed_0/metrics.json"))), 11)
                self.assertEqual(len(kbs.read_csv(root / "out" / "sensitivity_summary.csv")), 8)
            summaries = list((root / "out").glob("primary_summary.csv"))
            self.assertEqual(len(summaries), 1)
            run = next((root / "out" / "runs").glob("margin_full/seed_0"))
            with np.load(run / "predictions.npz") as pred:
                self.assertFalse(pred["strict_eval"][pred["queried"]].any())
                np.testing.assert_array_equal(
                    pred["common_eval"], ~np.isin(np.arange(30), np.arange(12)))
            per_class = kbs.read_csv(run / "per_class_metrics.csv")
            self.assertEqual(len(per_class), 6)  # 3 classes × 2 paired masks
            self.assertTrue((run / "repaired_head.pt").is_file())
            saved = torch.load(run / "repaired_head.pt", map_location="cpu", weights_only=True)
            self.assertEqual(set(saved["cls_head_state_dict"]), {"fc.weight", "fc.bias"})
            # Changed protocol cannot silently reuse an output directory.
            args.steps = 3
            with self.assertRaises(ValueError):
                kbs.run_study(args, data)


if __name__ == "__main__":
    unittest.main()
