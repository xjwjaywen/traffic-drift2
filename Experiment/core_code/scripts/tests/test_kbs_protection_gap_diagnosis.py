"""Synthetic read-only diagnosis checks; no real experiment evidence."""
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
import kbs_protection_gap_diagnosis as gap
import kbs_learned_protection as learned
import kbs_protection_budget as allocation
import kbs_repair_aware as pilot
import kbs_supplement as kbs
import test_kbs_protection_budget as fixtures


class ProtectionGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            aa, _, _ = fixtures.ProtectionBudgetTests().fixture(root)
            av = allocation.verify_sources(aa)
            allocation.select(aa, av); allocation.train(aa, av); allocation.evaluate(aa, av)
            la = learned.parser().parse_args(["select", "--pilot-dir", aa.pilot_dir,
                "--study-dir", aa.study_dir, "--cache-dir", aa.cache_dir, "--checkpoint", aa.checkpoint,
                "--output-dir", str(root / "learned")])
            lv = learned.verify_sources(la)
            learned.select(la, lv); learned.train(la, lv); learned.evaluate(la, lv)
        args = gap.parser().parse_args(["run", "--learned-dir", la.output_dir,
            "--allocation-dir", aa.output_dir, "--oracle-dir", aa.oracle_dir,
            "--pilot-dir", aa.pilot_dir, "--study-dir", aa.study_dir, "--cache-dir", aa.cache_dir,
            "--checkpoint", aa.checkpoint, "--output-dir", str(root / "gap"), "--threads", "1"])
        return args

    def snapshot(self, *folders):
        return fixtures.ProtectionBudgetTests().snapshot(*folders)

    def test_ce_gradient_matches_autograd_including_bias_and_cancellation(self):
        x = np.array([[1., 2.], [0., -1.], [2., 0.]])
        y = np.array([0, 1, 0])
        w = torch.tensor([[.2, .4], [-.1, .2]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([.1, -.2], dtype=torch.float64, requires_grad=True)
        grads, losses = [], []
        for row, label in zip(x, y):
            loss = torch.nn.functional.cross_entropy(torch.tensor(row)[None] @ w.T+b, torch.tensor([label]))
            g = torch.autograd.grad(loss, (w, b))
            grads.append(np.r_[g[0].detach().numpy().ravel(), g[1].detach().numpy()])
            losses.append(loss.item())
        result = gap.gradient_stats(x, y, {"weight": w.detach(), "bias": b.detach()})
        norms = np.linalg.norm(grads, axis=1)
        self.assertAlmostEqual(result["example_gradient_norm_mean"], norms.mean())
        self.assertAlmostEqual(result["example_gradient_norm_median"], np.median(norms))
        self.assertAlmostEqual(result["mean_gradient_norm"], np.linalg.norm(np.mean(grads, axis=0)))
        self.assertAlmostEqual(result["ce_mean"], np.mean(losses))
        self.assertAlmostEqual(result["gradient_alignment_ratio"], np.linalg.norm(np.mean(grads, axis=0))/norms.mean())
        empty = gap.gradient_stats(x[:0], y[:0], {"weight": w.detach(), "bias": b.detach()})
        self.assertEqual(empty["n"], 0)
        self.assertTrue(all(v is None for k, v in empty.items() if k != "n"))
        for xx, yy in [(x[:, :1], y), (x, y+2), (x*np.nan, y)]:
            with self.assertRaises(ValueError): gap.gradient_stats(xx, yy, {"weight": w.detach(), "bias": b.detach()})

    def test_feature_diversity_zero_vectors_and_small_cohorts(self):
        r = gap.feature_stats(np.array([[1., 0], [1., 0], [0., 1], [0., 0]]))
        self.assertEqual((r["cosine_valid_samples"], r["zero_norm_samples"]), (3, 1))
        self.assertAlmostEqual(r["pair_cosine_mean"], 1/3)
        self.assertAlmostEqual(r["nearest_cosine_mean"], 2/3)
        self.assertIsNone(gap.feature_stats(np.ones((1, 2)))["pair_cosine_mean"])
        self.assertIsNone(gap.feature_stats(np.empty((0, 2)))["feature_norm_mean"])

    def test_hand_counted_overlap_persistence_coverage_and_prefix_audit(self):
        n = 110
        badge = np.arange(100)
        y = np.zeros(n, dtype=int); y[100:105] = [1, 1, 2, 2, 0]
        old, probe, final = y.copy(), y.copy(), y.copy()
        probe[[95, 100, 101, 102]] = 0
        probe[95] = 1
        final[[95, 100, 102, 106]] = 1
        final[100] = 0
        y[106] = old[106] = 2
        mask = np.zeros(n, bool); mask[105:] = True
        ids = np.r_[badge[:95], [95, 100, 101, 102, 104]]
        p = {"num_classes": 3, "collapse_classes": [0]}
        audit, cohorts, cr, pn = gap.tail_diagnostics(ids, badge, y, old, probe, final, mask, p, np.ones((5, 2)))
        self.assertEqual(audit["tail_overlap_badge_training"], 1)
        self.assertEqual(audit["outside_badge_training_n"], 4)
        self.assertEqual(audit["outside_badge_probe_damage_n"], 3)
        self.assertAlmostEqual(audit["outside_badge_probe_damage_persistence"], 2/3)
        self.assertEqual(audit["outside_badge_final_damage_fraction"], .5)
        self.assertEqual(audit["class_covered_noncollapse_badge_damage_fraction"], 1)
        self.assertEqual(cohorts[0]["true_classes"], 3)
        self.assertAlmostEqual(cohorts[0]["effective_true_classes"], 25/9)
        self.assertEqual(sum(r["tail_probe_damage"] for r in cr), int(pn.sum()))
        bad_audit, _, _, _ = gap.tail_diagnostics(badge, badge, y, old, probe, final, mask, p, np.ones((5, 2)))
        self.assertIsNone(bad_audit["outside_badge_probe_damage_persistence"])
        changed = ids.copy(); changed[94] = 99
        a, _, _, _ = gap.tail_diagnostics(changed, badge, y, old, probe, final, mask, p, np.ones((5, 2)))
        self.assertEqual((a["prefix_order_identical"], a["prefix_symmetric_difference"]), (0, 2))
        mask, excluded = gap.common_mask(10, [0, 1, 2], [2, 3, 4])
        self.assertEqual(excluded, [0, 1, 2, 3, 4])
        self.assertEqual(int(mask.sum()), 5)
        with self.assertRaises(ValueError): gap.common_mask(2, [0, 1])

    def test_completed_studies_unified_results_no_training_and_noop_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            paths = [getattr(args, k) for k in ("study_dir", "cache_dir", "pilot_dir", "oracle_dir", "allocation_dir", "learned_dir")]
            before = self.snapshot(*paths)
            original_load = torch.load
            def guarded_load(path, *a, **kw):
                self.assertNotEqual(Path(path).name, "reference.pt")
                self.assertEqual(kw["map_location"], "cpu")
                return original_load(path, *a, **kw)
            with contextlib.ExitStack() as stack:
                for module, name in [(pilot, "fit"), (kbs, "fit_controlled"), (kbs, "predict"),
                                     (kbs, "device_for"), (learned, "fit_ranker"), (learned, "select_queries")]:
                    stack.enter_context(patch.object(module, name, side_effect=AssertionError("No fitting, acquisition, or inference")))
                stack.enter_context(patch.object(torch, "load", side_effect=guarded_load))
                v = gap.verify_inputs(args)
                gap.run(args, v)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "by_seed.csv")
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(kbs.read_csv(out / "paired_by_seed.csv")), 33)
            self.assertEqual(len(kbs.read_csv(out / "per_class.csv")), 72)
            self.assertEqual(len(kbs.read_csv(out / "selection_audit.csv")), 21)
            self.assertEqual(len(kbs.read_csv(out / "gradients.csv")), 126)
            target = original_load(Path(args.cache_dir) / "target.pt", weights_only=True)
            y, old = target["labels"].numpy(), target["logits"].argmax(1).numpy()
            exclusions = pilot.read_json(out / "excluded_ids.json")
            for seed in range(3):
                expected = sorted(set(v[1]["learned"]["records"][seed]["common_excluded_indices"]) |
                                  set(v[1]["allocation"]["records"][seed]["common_excluded_indices"]))
                self.assertEqual(exclusions[str(seed)], expected)
                mask = np.ones(len(y), bool); mask[expected] = False
                for row in [r for r in rows if int(r["seed"]) == seed]:
                    m = row["method"]
                    folder = args.pilot_dir if m == "badge" else args.allocation_dir if m in gap.ORACLES else args.learned_dir
                    pred = old if m == "static" else np.load(pilot.run_dir(Path(folder), seed, m) / "predictions.npy")
                    metric, _ = pilot.metrics(y, old, pred, mask, v[0][1])
                    gap.diagnosis.reconcile(metric, row)
            for r in kbs.read_csv(out / "selection_audit.csv"):
                if r["method"] == "badge":
                    self.assertEqual(r["outside_badge_probe_damage_persistence"], "")
            self.assertEqual(before, self.snapshot(*paths))
            finished = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("no tensor load on resume")), \
                 patch.object(np, "load", side_effect=AssertionError("no array load on resume")):
                gap.run(args, gap.verify_inputs(args))
            self.assertEqual(finished, self.snapshot(out))
            (out / "report.md").write_text("corrupt")
            with self.assertRaises(ValueError): gap.run(args, v)

    def test_stage_gates_seed_output_and_rehashed_old_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            marker = pilot.run_dir(Path(args.learned_dir), 2, learned.METHODS[0]) / "complete.json"
            content = marker.read_bytes(); marker.unlink()
            with patch.object(torch, "load", side_effect=AssertionError("gate before tensor load")):
                with self.assertRaisesRegex(ValueError, "source fits"): gap.verify_inputs(args)
            marker.write_bytes(content)
            for name, value in [("seeds", "0,0"), ("seeds", "9"), ("threads", 0),
                                ("output_dir", str(Path(args.learned_dir)/"nested"))]:
                changed = copy.copy(args); setattr(changed, name, value)
                with self.assertRaises(ValueError): gap.verify_inputs(changed)
            # Updating a completion hash must not conceal a semantically wrong source metric.
            ev = Path(args.learned_dir) / "evaluation"
            records = kbs.read_csv(ev / "by_seed.csv")
            records[0]["overall_macro_f1_after"] = -1
            kbs.write_csv(ev / "by_seed.csv", records)
            signature = pilot.read_json(ev / "resolved_config.json")
            kbs.finish(ev, signature, learned.EVAL_FILES)
            v = gap.verify_inputs(args)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"): gap.run(args, v)
            self.assertFalse((Path(args.output_dir) / "complete.json").exists())

    def test_cpu_cli_separate_preflight_run_and_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            flags = [item for k, value in vars(args).items() if k != "command" for item in ["--"+k.replace("_", "-"), str(value)]]
            # A diagnostic seed subset does not change the frozen source study identities.
            flags += ["--seeds", "0"]
            for mode in ("preflight", "run", "run"):
                result = subprocess.run([sys.executable, gap.__file__, mode, *flags], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(len(kbs.read_csv(Path(args.output_dir)/"by_seed.csv")), 8)


if __name__ == "__main__":
    unittest.main()
