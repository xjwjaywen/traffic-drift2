"""Synthetic label-accounting and learned-acquisition checks, not paper evidence."""
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
import kbs_learned_protection as learned
import kbs_repair_aware as pilot
import kbs_supplement as kbs
import test_kbs_repair_aware as fixtures


class LearnedProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, complete=True):
        pa = fixtures.RepairAwareTests().fixture(root)
        path = Path(pa.study_dir) / "study_manifest.json"
        data = pilot.read_json(path)
        data["protocol"].update(budget=20, optimizer_steps=12, lr=.15)
        kbs.atomic_json(path, data)
        info = pilot.read_json(Path(pa.cache_dir) / "manifest.json")
        for seed in range(3):
            sd = Path(pa.study_dir) / "selections/badge" / f"seed_{seed}"
            sig = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
                   "selector": "badge", "budget": 20, "seed": seed}
            kbs.atomic_json(sd / "resolved_config.json", sig)
            kbs.atomic_json(sd / "selection.json", {"row_indices": list(range(seed, seed+20))})
            kbs.finish(sd, sig, ["resolved_config.json", "selection.json"])
        pa.seeds = "0,1,2"
        sources = pilot.verify_sources(pa)
        if complete:
            pilot.select(pa, sources); pilot.train(pa, sources); pilot.evaluate(pa, sources)
        args = learned.parser().parse_args(["select", "--pilot-dir", pa.output_dir, "--study-dir", pa.study_dir,
            "--cache-dir", pa.cache_dir, "--checkpoint", pa.checkpoint, "--output-dir", str(root / "learned")])
        return args, pa, sources

    def snapshot(self, *paths):
        return fixtures.RepairAwareTests().snapshot(*paths)

    def hand_fixture(self):
        badge = np.arange(100)
        prior = {"scout_indices": list(range(80)), "common_excluded_indices": list(range(100))}
        old = np.zeros(200, dtype=np.int64)
        probe = np.ones(200, dtype=np.int64)
        signals = np.tile([.75, .5, .3, .4], (200, 1))
        labels = np.r_[np.zeros(7, dtype=np.int64), np.ones(8, dtype=np.int64)]
        signals[80:87, 2] = .45
        signals[87:95, 2] = .1
        signals[120:125, 2] = .48
        return badge, prior, old, probe, signals, labels

    def test_ranker_objective_direction_support_constant_and_invalid_data(self):
        x = np.tile([.75, .5, .2, .4], (20, 1)); x[10:, 2] = .45
        y = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
        model = learned.fit_ranker(x, y)
        self.assertEqual(model["status"], "fitted")
        scores = learned.rank_scores(x, model)
        self.assertTrue(np.all(scores[10:] > .5) and np.all(scores[:10] < .5))
        # Independently check the first-order condition of the declared summed-CE objective.
        z = np.column_stack([np.ones(len(y)), (x-model["mean"])/model["scale"]])
        w = np.array(model["weights"])
        grad = z.T @ (scores-y) + np.r_[0, w[1:]]
        np.testing.assert_allclose(grad, 0, atol=1e-7)
        self.assertEqual(model, learned.fit_ranker(x, y))
        model = learned.fit_ranker(np.ones((10, 4)), np.arange(10) % 2)
        np.testing.assert_allclose(learned.rank_scores(np.ones((10, 4)), model), .5)
        for labels in (np.array([], dtype=int), np.zeros(10, int), np.r_[np.ones(4, int), np.zeros(10, int)]):
            m = learned.fit_ranker(np.ones((len(labels), 4)), labels)
            self.assertEqual(m["reason"], "insufficient_binary_support")
            np.testing.assert_array_equal(learned.rank_scores(x, m), x[:, 0])
        for bad_x, bad_y in [(np.full((20, 4), np.nan), y), (x, y+2), (x[:, :3], y)]:
            with self.assertRaises(ValueError): learned.fit_ranker(bad_x, bad_y)
        with patch.dict(learned.SETTINGS, ranker_max_iterations=0):
            self.assertEqual(learned.fit_ranker(x, y)["reason"], "solver_did_not_converge")

    def test_hand_calculated_ranking_paid_partition_ties_and_empty_pool(self):
        badge, prior, old, probe, x, labels = self.hand_fixture()
        r, model = learned.select_queries(badge, prior, old, probe, x, labels, 2, 0)
        self.assertEqual(model["status"], "fitted")
        self.assertEqual(r["fit_indices"], list(range(80, 95)))
        self.assertFalse(set(r["fit_indices"]) & set(prior["scout_indices"]))
        self.assertEqual(r["choices"]["flip_risk50"]["row_indices"][95:], list(range(120, 125)))
        priority = np.random.default_rng(72021).permutation(np.arange(95, 200))
        for name in learned.METHODS[:-1]:
            self.assertEqual(r["choices"][name]["row_indices"][95:], sorted(priority[:5].tolist()))
        union = sorted(set(range(100)) | {i for c in r["choices"].values() for i in c["row_indices"]})
        self.assertEqual(r["common_excluded_indices"], union)
        for c in r["choices"].values():
            self.assertEqual(c["row_indices"][:95], list(range(95)))
            self.assertEqual(len(set(c["row_indices"])), 100)
        # Empty/short pools get the same predeclared fill and never request extra labels.
        for flips in ([], [120], [120, 121, 122]):
            prob = old.copy(); prob[flips] = 1
            short, m = learned.select_queries(badge, prior, old, prob, x, labels, 2, 0)
            self.assertEqual(m["status"], "source_confidence_fallback")
            self.assertEqual(short["choices"]["flip_risk50"], short["choices"]["flip_confidence50"])
            for name in learned.METHODS[1:]:
                c = short["choices"][name]
                self.assertEqual(c["fallback_count"], 5-len(flips))
                self.assertLessEqual(set(flips), set(c["row_indices"]))
                self.assertEqual(len(set(c["row_indices"])), 100)
            if not flips:
                self.assertEqual(len({tuple(c["row_indices"]) for c in short["choices"].values()}), 1)

    def test_signal_math_and_probe_inference_alignment(self):
        a = torch.log(torch.tensor([[.8, .2], [.1, .9]]))
        b = torch.log(torch.tensor([[.3, .7], [.6, .4]]))
        target = {"features": torch.eye(2), "logits": a}
        state = {"weight": b.T, "bias": torch.zeros(2)}
        x, old = learned.infer_signals(target, state, np.array([1, 0]), torch.device("cpu"), 1)
        np.testing.assert_allclose(x, [[.8, .6, .3, .4], [.9, .8, .4, .2]], atol=1e-6)
        np.testing.assert_array_equal(old, [0, 1])
        with self.assertRaisesRegex(ValueError, "Saved probe predictions"):
            learned.infer_signals(target, state, np.array([0, 0]), torch.device("cpu"), 2)

    def test_twelve_fits_query_only_labels_fresh_source_union_metrics_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pa, _ = self.fixture(Path(tmp))
            verified = learned.verify_sources(args)
            before = self.snapshot(args.pilot_dir, args.cache_dir, args.study_dir)
            helper = fixtures.RepairAwareTests()
            fit_allowed = {tuple(learned.parts(verified[0][2][s])[1]) for s in range(3)}
            load_array = np.load
            def no_final(path, *a, **kw):
                self.assertFalse(Path(path).name == "predictions.npy" and "runs" in Path(path).parts)
                return load_array(path, *a, **kw)
            with patch.object(torch, "load", side_effect=helper.guard(fit_allowed)), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("no repair/probe fit during selection")), \
                 patch.object(np, "load", side_effect=no_final):
                learned.select(args, verified)
            records = learned.require_selections(args, verified)
            # Replacing every inaccessible target label cannot change any acquisition.
            other = copy.copy(args); other.output_dir = str(Path(tmp) / "permuted-truth")
            load_torch = torch.load
            all_fit = {i for ids in fit_allowed for i in ids}
            def altered(path, **kw):
                obj = load_torch(path, **kw)
                if Path(path).name == "target.pt":
                    labels = obj["labels"].clone()
                    for i in range(len(labels)):
                        if i not in all_fit: labels[i] = (labels[i]+1) % 3
                    obj["labels"] = fixtures.QueryOnlyLabels(labels, fit_allowed)
                return obj
            with patch.object(torch, "load", side_effect=altered):
                learned.select(other, verified)
            self.assertEqual(records, learned.require_selections(other, verified))
            permitted = {tuple(c["row_indices"]) for r in records.values() for c in r["choices"].values()}
            source = load_torch(Path(args.cache_dir) / "head.pt", weights_only=True)
            real_fit, seen = kbs.fit_controlled, []
            def checked(head, x, y, rx, ry, teacher, **kw):
                for k, v in source.items(): torch.testing.assert_close(head.state_dict()[k], v)
                self.assertEqual((len(x), len(rx), kw["steps"]), (20, 15, 12))
                self.assertEqual((kw["kd_weight"], kw["temperature"], kw["replay_ce"]), (.5, 2., False))
                seen.append(kw["seed"])
                return real_fit(head, x, y, rx, ry, teacher, **kw)
            with patch.object(torch, "load", side_effect=helper.guard(permitted)), patch.object(kbs, "fit_controlled", side_effect=checked):
                learned.train(args, verified)
            self.assertEqual(seen, [0]*4+[1]*4+[2]*4)
            learned.evaluate(args, verified)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 24)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 54)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/acquisition.csv")), 15)
            target = load_torch(Path(args.cache_dir) / "target.pt", weights_only=True)
            y, old = np.asarray(target["labels"]), target["logits"].argmax(1).numpy()
            for seed, r in records.items():
                mask = np.ones(len(y), bool); mask[r["common_excluded_indices"]] = False
                for method in ("badge", *learned.METHODS):
                    rd = pilot.run_dir(Path(args.pilot_dir) if method == "badge" else out, seed, method)
                    pred = np.load(rd / "predictions.npy")
                    metrics, _ = pilot.metrics(y, old, pred, mask, verified[0][1])
                    row = next(v for v in rows if int(v["seed"]) == seed and v["method"] == method)
                    learned.diagnosis.reconcile(metrics, row)
            self.assertEqual(before, self.snapshot(args.pilot_dir, args.cache_dir, args.study_dir))
            saved = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("resume loads no tensor")), \
                 patch.object(np, "load", side_effect=AssertionError("resume loads no array")):
                v = learned.verify_sources(args)
                learned.select(args, v); learned.train(args, v); learned.evaluate(args, v)
            self.assertEqual(saved, self.snapshot(out))
            report = out / "evaluation/report.md"
            report.write_text(report.read_text()+"modified")
            with self.assertRaisesRegex(ValueError, "Missing/modified"):
                learned.evaluate(args, verified)

    def test_missing_stages_changed_identity_and_rehashed_query_semantics_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pa, ps = self.fixture(Path(tmp), complete=False)
            with patch.object(torch, "load", side_effect=AssertionError("no arrays")):
                with self.assertRaises((ValueError, FileNotFoundError)): learned.verify_sources(args)
            pilot.select(pa, ps); pilot.train(pa, ps); pilot.evaluate(pa, ps)
            verified = learned.verify_sources(args); learned.select(args, verified)
            out = Path(args.output_dir)
            sd = pilot.selection_dir(out, 0)
            cp = sd / "complete.json"; data = cp.read_bytes(); cp.unlink()
            with patch.object(torch, "load", side_effect=AssertionError("no arrays")):
                with self.assertRaises(ValueError): learned.train(args, verified)
            cp.write_bytes(data)
            with patch.object(torch, "load", side_effect=AssertionError("no evaluation truth")):
                with self.assertRaisesRegex(ValueError, "All new fits"): learned.evaluate(args, verified)
            changed = (*verified[:2], {**verified[2], "seeds": [0]})
            with self.assertRaises(ValueError): learned.select(args, changed)
            rec = pilot.read_json(sd / "selection.json")
            original = copy.deepcopy(rec)
            rec["common_excluded_indices"].pop()
            kbs.atomic_json(sd / "selection.json", rec)
            kbs.finish(sd, pilot.signature(verified[2], 0, "select"), learned.SELECT_FILES)
            with patch.object(torch, "load", side_effect=AssertionError("no training truth")):
                with self.assertRaisesRegex(ValueError, "Common query"): learned.train(args, verified)
            kbs.atomic_json(sd / "selection.json", original)
            model = pilot.read_json(sd / "ranker.json"); model["positive"] += 1
            kbs.atomic_json(sd / "ranker.json", model)
            kbs.finish(sd, pilot.signature(verified[2], 0, "select"), learned.SELECT_FILES)
            with patch.object(torch, "load", side_effect=AssertionError("no training truth")):
                with self.assertRaisesRegex(ValueError, "declared label-limited"): learned.train(args, verified)
            args.seeds = "9"
            with self.assertRaisesRegex(ValueError, "unique seeds"): learned.verify_sources(args)

    def test_cli_processes_and_source_metric_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            common = [v for flag in ("pilot_dir", "study_dir", "cache_dir", "checkpoint", "output_dir")
                      for v in ("--"+flag.replace("_", "-"), getattr(args, flag))]
            for stage in ("plan", "preflight", "select", "train", "evaluate"):
                result = subprocess.run([sys.executable, learned.__file__, stage, *common], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
                if stage == "plan": self.assertIn('"new_repair_fits": 12', result.stdout)
            # Source scores must be recomputed, not blindly copied after hashing.
            path = Path(args.pilot_dir) / "evaluation/by_seed.csv"
            rows = kbs.read_csv(path); rows[0]["overall_macro_f1_after"] = "-1"
            kbs.write_csv(path, rows)
            cp = path.parent / "complete.json"
            kbs.finish(path.parent, pilot.read_json(cp)["signature"], pilot.EVAL_FILES)
            args.output_dir = str(Path(tmp) / "bad-metrics")
            v = learned.verify_sources(args)
            learned.select(args, v); learned.train(args, v)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"): learned.evaluate(args, v)
            # Incompatible output source nesting is rejected by the shared verifier.
            args.output_dir = str(Path(args.pilot_dir) / "nested")
            with self.assertRaises(ValueError): learned.verify_sources(args)
