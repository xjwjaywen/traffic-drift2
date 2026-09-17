"""Synthetic mechanism-control checks; never CESNET scientific evidence."""
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
import kbs_oracle_protection as oracle
import kbs_repair_aware as pilot
import kbs_supplement as kbs
import test_kbs_repair_aware as fixtures


class OracleProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, complete=True):
        original = fixtures.RepairAwareTests().fixture(root)
        # Synthetic-only source protocol with enough probe damage for a 2-row tail.
        manifest = Path(original.study_dir) / "study_manifest.json"
        p = pilot.read_json(manifest)
        p["protocol"].update(optimizer_steps=12, lr=.15)
        kbs.atomic_json(manifest, p)
        info = pilot.read_json(Path(original.cache_dir) / "manifest.json")
        directory = Path(original.study_dir) / "selections/badge/seed_2"
        sig = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
               "selector": "badge", "budget": 10, "seed": 2}
        kbs.atomic_json(directory / "resolved_config.json", sig)
        kbs.atomic_json(directory / "selection.json", {"row_indices": list(range(2, 12))})
        kbs.finish(directory, sig, ["resolved_config.json", "selection.json"])
        original.seeds = "0,1,2"
        sources = pilot.verify_sources(original)
        if complete:
            pilot.select(original, sources)
            pilot.train(original, sources)
            pilot.evaluate(original, sources)
        args = oracle.parser().parse_args(["select", "--study-dir", original.study_dir,
            "--cache-dir", original.cache_dir, "--checkpoint", original.checkpoint,
            "--pilot-dir", original.output_dir, "--output-dir", str(root / "oracle")])
        return args, original, sources

    def hand_fixture(self):
        record = {"scout_indices": list(range(8)), "choices": {
            m: {"row_indices": list(range(8)) + tail} for m, tail in zip(pilot.METHODS,
                ([8, 9], [20, 21], [24, 25], [26, 27]))}}
        record["common_excluded_indices"] = sorted({i for c in record["choices"].values() for i in c["row_indices"]})
        old = np.zeros(40, dtype=np.int64)
        y = np.arange(40) % 2
        probe = old.copy(); probe[[10, 12, 14, 16, 18]] = 1
        return record, y, old, probe

    def snapshot(self, *folders):
        return {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns)
                for folder in folders for p in Path(folder).rglob("*") if p.is_file()}

    def test_uniform_sampling_oracle_eligibility_accounting_and_exact_union(self):
        previous, y, old, probe = self.hand_fixture()
        record = oracle.select_queries(previous, y, old, probe, 2, 0)
        pools = [np.array([10, 12, 14, 16, 18]), np.arange(8, 40, 2)]
        for method, pool in zip(oracle.METHODS, pools):
            expected = np.sort(np.random.default_rng(52021).choice(pool, 2, replace=False))
            choice = record["choices"][method]
            self.assertEqual(choice["row_indices"], list(range(8)) + expected.tolist())
            self.assertEqual(choice["pool_size"], len(pool))
            self.assertEqual(len(set(choice["row_indices"])), 10)
        self.assertEqual((record["training_rows_per_arm"], record["oracle_truth_rows_inspected"]), (10, 40))
        union = sorted(set(previous["common_excluded_indices"]) |
                       {i for c in record["choices"].values() for i in c["row_indices"]})
        self.assertEqual(record["common_excluded_indices"], union)
        oracle.validate_selection(record, previous, {"budget": 10}, 40)
        self.assertEqual(record, oracle.select_queries(previous, y, old, probe, 2, 0))
        # If both pools coincide, same RNG and tail ordering give identical training rows.
        probe = 1 - old
        r = oracle.select_queries(previous, y, old, probe, 2, 2)
        self.assertEqual(r["choices"][oracle.METHODS[0]], r["choices"][oracle.METHODS[1]])

    def test_insufficient_pools_stop_and_invalid_union_or_labels_rejected(self):
        previous, y, old, probe = self.hand_fixture()
        for changing in ([], [10]):
            probe = old.copy(); probe[changing] = 1
            with self.assertRaisesRegex(ValueError, "Insufficient oracle pool"):
                oracle.select_queries(previous, y, old, probe, 2, 0)
        previous, y, old, probe = self.hand_fixture()
        r = oracle.select_queries(previous, y, old, probe, 2, 0)
        r["common_excluded_indices"].pop()
        with self.assertRaisesRegex(ValueError, "six arms"):
            oracle.validate_selection(r, previous, {"budget": 10}, 40)
        y[9] = 2
        with self.assertRaisesRegex(ValueError, "truth"):
            oracle.select_queries(previous, y, old, probe, 2, 0)

    def test_six_fits_fresh_source_pairing_guarded_labels_recompute_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            verified = oracle.verify_sources(args)
            before = self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir)
            real_np_load = np.load
            def no_final_predictions(path, *a, **kw):
                self.assertNotEqual(Path(path).name, "predictions.npy")
                return real_np_load(path, *a, **kw)
            with patch.object(kbs, "fit_controlled", side_effect=AssertionError("selection must not fit")), \
                 patch.object(kbs, "predict", side_effect=AssertionError("selection must not infer")), \
                 patch.object(np, "load", side_effect=no_final_predictions):
                oracle.select(args, verified)
            records = oracle.require_selections(args, verified)
            allowed = {tuple(c["row_indices"]) for r in records.values() for c in r["choices"].values()}
            source = torch.load(Path(args.cache_dir) / "head.pt", weights_only=True)
            real_fit = kbs.fit_controlled
            fits = []
            def checked_fit(head, x, y, rx, ry, teacher, **kwargs):
                for k, v in source.items():
                    torch.testing.assert_close(head.state_dict()[k], v)
                self.assertEqual((len(x), len(rx), kwargs["steps"]), (10, 15, 12))
                self.assertFalse(kwargs["replay_ce"])
                self.assertEqual((kwargs["kd_weight"], kwargs["temperature"]), (.5, 2.0))
                fits.append(kwargs["seed"])
                return real_fit(head, x, y, rx, ry, teacher, **kwargs)
            helper = fixtures.RepairAwareTests()
            with patch.object(torch, "load", side_effect=helper.guard(allowed)), \
                 patch.object(kbs, "fit_controlled", side_effect=checked_fit):
                oracle.train(args, verified)
            self.assertEqual(fits, [0, 0, 1, 1, 2, 2])
            oracle.evaluate(args, verified)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 21)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 63)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 15)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/selection_audit.csv")), 18)
            y, old = oracle.load_truth_and_source(args)
            for seed, rec in records.items():
                mask = np.ones(len(y), dtype=bool); mask[rec["common_excluded_indices"]] = False
                for method in (*pilot.METHODS, *oracle.METHODS):
                    rd = oracle.run_dir(out, seed, method) if method in oracle.METHODS else pilot.run_dir(Path(args.pilot_dir), seed, method)
                    pred = np.load(rd / "predictions.npy")
                    expected, _ = pilot.metrics(y, old, pred, mask, verified[0][1])
                    row = next(r for r in rows if int(r["seed"]) == seed and r["method"] == method)
                    oracle.diagnosis.reconcile(expected, row)
            audit = kbs.read_csv(out / "evaluation/selection_audit.csv")
            for row in audit:
                if row["method"] in oracle.METHODS:
                    self.assertEqual((row["oracle_truth_rows_inspected"], row["supplement_source_correct"]), ("120", "2"))
                if row["method"] == oracle.METHODS[0]:
                    self.assertEqual(row["supplement_probe_damage"], "2")
            saved = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("resume must not load tensors")):
                verified = oracle.verify_sources(args)
                oracle.select(args, verified); oracle.train(args, verified); oracle.evaluate(args, verified)
            self.assertEqual(saved, self.snapshot(out))
            self.assertEqual(before, self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir))

    def test_missing_stage_and_changed_identity_block_before_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, original, sources = self.fixture(Path(tmp), complete=False)
            pilot.select(original, sources)
            with patch.object(torch, "load", side_effect=AssertionError("no truth access")):
                with self.assertRaises(ValueError): oracle.verify_sources(args)
            pilot.train(original, sources); pilot.evaluate(original, sources)
            verified = oracle.verify_sources(args)
            oracle.select(args, verified)
            cp = oracle.selection_dir(args.output_dir, 2) / "complete.json"
            data = cp.read_bytes(); cp.unlink()
            with patch.object(torch, "load", side_effect=AssertionError("all selections before truth")):
                with self.assertRaises(ValueError): oracle.train(args, verified)
            cp.write_bytes(data)
            with patch.object(torch, "load", side_effect=AssertionError("all fits before evaluation truth")):
                with self.assertRaises(ValueError): oracle.evaluate(args, verified)
            changed = (*verified[:2], {**verified[2], "settings": {**oracle.SETTINGS, "changed": True}})
            with self.assertRaises(ValueError): oracle.select(args, changed)
            args.seeds = "9"
            with self.assertRaisesRegex(ValueError, "unique seeds"):
                oracle.verify_sources(args)

    def test_rehashed_ineligible_query_and_corrupted_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            verified = oracle.verify_sources(args)
            oracle.select(args, verified)
            sd = oracle.selection_dir(args.output_dir, 0)
            record = pilot.read_json(sd / "selection.json")
            y, old = oracle.load_truth_and_source(args)
            m = oracle.METHODS[0]
            ids = record["choices"][m]["row_indices"]
            bad = next(int(i) for i in np.flatnonzero(y != old) if i not in ids)
            ids[-1] = bad; ids[8:] = sorted(ids[8:])
            record["common_excluded_indices"] = sorted(set(verified[1][0]["common_excluded_indices"]) |
                {i for c in record["choices"].values() for i in c["row_indices"]})
            frozen = pilot.read_json(sd / "query_labels.json")
            frozen[m] = y[ids].tolist()
            kbs.atomic_json(sd / "selection.json", record)
            kbs.atomic_json(sd / "query_labels.json", frozen)
            kbs.finish(sd, oracle.signature(verified[2], 0, "select"), oracle.SELECT_FILES)
            with patch.object(kbs, "fit_controlled", side_effect=AssertionError("invalid query must not fit")):
                with self.assertRaisesRegex(ValueError, "oracle condition"):
                    oracle.train(args, verified)
            (sd / "query_labels.json").write_text("corrupt")
            with self.assertRaises(ValueError): oracle.train(args, verified)

    def test_rehashed_old_metrics_rejected_after_common_split_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            directory = Path(args.pilot_dir) / "evaluation"
            rows = kbs.read_csv(directory / "by_seed.csv")
            next(r for r in rows if r["method"] == "badge")["overall_macro_f1_after"] = .9999
            kbs.write_csv(directory / "by_seed.csv", rows)
            kbs.finish(directory, pilot.read_json(directory / "resolved_config.json"), pilot.EVAL_FILES)
            verified = oracle.verify_sources(args)
            oracle.select(args, verified); oracle.train(args, verified)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                oracle.evaluate(args, verified)

    def test_separate_cli_stages_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            common = ["--study-dir", args.study_dir, "--cache-dir", args.cache_dir,
                      "--checkpoint", args.checkpoint, "--pilot-dir", args.pilot_dir,
                      "--output-dir", args.output_dir]
            for mode in ("plan", "preflight", "select", "train", "evaluate"):
                result = subprocess.run([sys.executable, oracle.__file__, mode, *common], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if mode == "plan":
                    self.assertIn('"new_fits": 6', result.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
