"""Synthetic nested-budget controls; no real experiment data or evidence."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_protection_budget as allocation
import kbs_oracle_protection as oracle
import kbs_repair_aware as pilot
import kbs_supplement as kbs
import test_kbs_oracle_protection as oracle_fixtures
import test_kbs_repair_aware as repair_fixtures


class ProtectionBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, oracle_complete=True):
        oa, pa, _ = oracle_fixtures.OracleProtectionTests().fixture(root, complete=False)
        path = Path(oa.study_dir) / "study_manifest.json"
        p = pilot.read_json(path); p["protocol"]["budget"] = 20
        kbs.atomic_json(path, p)
        info = pilot.read_json(Path(oa.cache_dir) / "manifest.json")
        for seed in (0, 1, 2):
            d = Path(oa.study_dir) / "selections/badge" / f"seed_{seed}"
            sig = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
                   "selector": "badge", "budget": 20, "seed": seed}
            kbs.atomic_json(d / "resolved_config.json", sig)
            kbs.atomic_json(d / "selection.json", {"row_indices": list(range(seed, seed + 20))})
            kbs.finish(d, sig, ["resolved_config.json", "selection.json"])
        pv = pilot.verify_sources(pa)
        pilot.select(pa, pv); pilot.train(pa, pv); pilot.evaluate(pa, pv)
        ov = oracle.verify_sources(oa)
        oracle.select(oa, ov)
        if oracle_complete:
            oracle.train(oa, ov); oracle.evaluate(oa, ov)
        args = allocation.parser().parse_args(["select", "--oracle-dir", oa.output_dir,
            "--pilot-dir", pa.output_dir, "--study-dir", oa.study_dir, "--cache-dir", oa.cache_dir,
            "--checkpoint", oa.checkpoint, "--output-dir", str(root / "allocations")])
        return args, oa, ov

    def snapshot(self, *folders):
        return {str(p): (kbs.file_sha(p), p.stat().st_mtime_ns)
                for folder in folders for p in Path(folder).rglob("*") if p.is_file()}

    def hand_fixture(self):
        badge = list(range(20))
        record = {"scout_indices": list(range(16)), "oracle_truth_rows_inspected": 100,
            "choices": {m: {"row_indices": list(range(16)) + tail}
                        for m, tail in zip(oracle.METHODS, ([16, 20, 21, 22], [17, 23, 24, 25]))},
            "common_excluded_indices": list(range(26)) + [30]}
        return badge, record

    def test_hand_calculated_nested_subsets_overlap_order_and_exact_endpoints(self):
        badge, original = self.hand_fixture()
        r = allocation.select_queries(badge, original, 0)
        self.assertEqual(r["common_excluded_indices"], original["common_excluded_indices"])
        self.assertEqual(r["new_truth_rows_inspected_for_allocation"], 0)
        for family in oracle.METHODS:
            tail = np.array(original["choices"][family]["row_indices"][16:])
            perm = np.random.default_rng(62021).permutation(4)
            sets = []
            for pct, count in [(5, 1), (10, 2)]:
                selected = sorted(tail[perm[:count]].tolist())
                filled = [i for i in badge if i not in selected][:20-count]
                c = r["choices"][f"{family}_p{pct:02d}"]
                self.assertEqual(c["protected_indices"], selected)
                self.assertEqual(c["row_indices"], filled + selected)
                self.assertEqual(c["row_indices"][:16], list(range(16)))
                self.assertEqual(len(set(c["row_indices"])), 20)
                self.assertEqual(c["rows_outside_badge_count"], count-c["protected_overlap_badge_count"])
                sets.append(set(selected))
            self.assertLessEqual(sets[0], sets[1])
            self.assertEqual(allocation.mixed_queries(badge, tail, 0, 0)["row_indices"], badge)
            self.assertEqual(allocation.mixed_queries(badge, tail, 4, 0)["row_indices"], original["choices"][family]["row_indices"])
        self.assertEqual(r, allocation.select_queries(badge, original, 0))
        # Identical source tails must give identical allocations for both families.
        original["choices"][oracle.METHODS[1]] = original["choices"][oracle.METHODS[0]]
        r = allocation.select_queries(badge, original, 2)
        for pct in (5, 10):
            self.assertEqual(r["choices"][f"{oracle.METHODS[0]}_p{pct:02d}"]["row_indices"],
                             r["choices"][f"{oracle.METHODS[1]}_p{pct:02d}"]["row_indices"])

    def test_invalid_allocations_duplicate_ids_and_scout_overlap_rejected(self):
        badge, original = self.hand_fixture()
        for tail, size in [([16, 20, 21, 22], 5), ([16, 20, 21, 22], -1),
                           ([16, 16, 21, 22], 1), ([0, 20, 21, 22], 1)]:
            with self.assertRaises(ValueError): allocation.mixed_queries(badge, tail, size, 0)
        with self.assertRaises(ValueError): allocation.select_queries(badge[:10], original, 0)
        original["common_excluded_indices"] = list(range(16))
        with self.assertRaisesRegex(ValueError, "escapes"):
            allocation.select_queries(badge, original, 0)

    def test_twelve_fits_unchanged_endpoints_mask_pairing_and_noop_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            verified = allocation.verify_sources(args)
            before = self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir, args.oracle_dir)
            with patch.object(torch, "load", side_effect=AssertionError("allocation must not load tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("allocation must not read predictions")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("no fit during allocation")):
                allocation.select(args, verified)
            records = allocation.require_selections(args, verified)
            allowed = {tuple(c["row_indices"]) for r in records.values() for c in r["choices"].values()}
            source = torch.load(Path(args.cache_dir) / "head.pt", weights_only=True)
            real_fit = kbs.fit_controlled
            fits = []
            def checked_fit(head, x, y, rx, ry, teacher, **kwargs):
                for k, value in source.items():
                    torch.testing.assert_close(head.state_dict()[k], value)
                self.assertEqual((len(x), len(rx), kwargs["steps"]), (20, 15, 12))
                self.assertEqual((kwargs["replay_ce"], kwargs["kd_weight"], kwargs["temperature"]), (False, .5, 2.0))
                fits.append(kwargs["seed"])
                return real_fit(head, x, y, rx, ry, teacher, **kwargs)
            with patch.object(torch, "load", side_effect=repair_fixtures.RepairAwareTests().guard(allowed)), \
                 patch.object(kbs, "fit_controlled", side_effect=checked_fit):
                allocation.train(args, verified)
            self.assertEqual(fits, [0]*4 + [1]*4 + [2]*4)
            allocation.evaluate(args, verified)
            out = Path(args.output_dir)
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 33)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 99)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/paired_by_seed.csv")), 27)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/curve_by_seed.csv")), 24)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/allocation_audit.csv")), 12)
            originals = {(r["seed"], r["method"]): r for r in kbs.read_csv(Path(args.oracle_dir) / "evaluation/by_seed.csv")}
            for row in rows:
                if row["method"] not in allocation.METHODS:
                    self.assertEqual(row, originals[row["seed"], row["method"]])
            y, old = oracle.load_truth_and_source(args)
            for seed, record in records.items():
                self.assertEqual(record["common_excluded_indices"], verified[1][seed]["common_excluded_indices"])
                mask = np.ones(len(y), dtype=bool); mask[record["common_excluded_indices"]] = False
                for method in allocation.METHODS:
                    pred = np.load(oracle.run_dir(out, seed, method) / "predictions.npy")
                    m, _ = pilot.metrics(y, old, pred, mask, verified[0][0][1])
                    row = next(r for r in rows if int(r["seed"]) == seed and r["method"] == method)
                    allocation.diagnosis.reconcile(m, row)
            saved = self.snapshot(out)
            with patch.object(torch, "load", side_effect=AssertionError("resume should not load tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("resume should not load arrays")):
                verified = allocation.verify_sources(args)
                allocation.select(args, verified); allocation.train(args, verified); allocation.evaluate(args, verified)
            self.assertEqual(saved, self.snapshot(out))
            self.assertEqual(before, self.snapshot(args.study_dir, args.cache_dir, args.pilot_dir, args.oracle_dir))
            (out / "evaluation/report.md").write_text("corrupt")
            with self.assertRaises(ValueError): allocation.evaluate(args, verified)

    def test_stage_gates_source_completion_and_incompatible_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, oa, ov = self.fixture(Path(tmp), oracle_complete=False)
            with patch.object(torch, "load", side_effect=AssertionError("no truth before source completion")):
                with self.assertRaises(ValueError): allocation.verify_sources(args)
            oracle.train(oa, ov); oracle.evaluate(oa, ov)
            verified = allocation.verify_sources(args)
            allocation.select(args, verified)
            cp = oracle.selection_dir(args.output_dir, 2) / "complete.json"
            data = cp.read_bytes(); cp.unlink()
            with patch.object(torch, "load", side_effect=AssertionError("all selections first")):
                with self.assertRaises(ValueError): allocation.train(args, verified)
            cp.write_bytes(data)
            with patch.object(torch, "load", side_effect=AssertionError("all fits first")):
                with self.assertRaises(ValueError): allocation.evaluate(args, verified)
            changed = (*verified[:2], {**verified[2], "seeds": [0, 1]})
            with self.assertRaises(ValueError): allocation.select(args, changed)
            args.output_dir = str(Path(args.oracle_dir) / "nested")
            with self.assertRaisesRegex(ValueError, "separate sibling"):
                allocation.verify_sources(args)

    def test_rehashed_mask_and_frozen_label_changes_rejected_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            verified = allocation.verify_sources(args)
            allocation.select(args, verified)
            sd = oracle.selection_dir(args.output_dir, 0)
            record = pilot.read_json(sd / "selection.json")
            changed = {**record, "common_excluded_indices": record["common_excluded_indices"][:-1]}
            kbs.atomic_json(sd / "selection.json", changed)
            sig = oracle.signature(verified[2], 0, "select")
            kbs.finish(sd, sig, allocation.SELECT_FILES)
            with patch.object(torch, "load", side_effect=AssertionError("no arrays")):
                with self.assertRaisesRegex(ValueError, "nested-subset"):
                    allocation.train(args, verified)
            kbs.atomic_json(sd / "selection.json", record)
            frozen = pilot.read_json(sd / "query_labels.json")
            frozen[allocation.METHODS[0]][0] = (frozen[allocation.METHODS[0]][0] + 1) % 3
            kbs.atomic_json(sd / "query_labels.json", frozen)
            kbs.finish(sd, sig, allocation.SELECT_FILES)
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                allocation.require_selections(args, verified)

    def test_rehashed_source_metric_discrepancy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            directory = Path(args.oracle_dir) / "evaluation"
            rows = kbs.read_csv(directory / "by_seed.csv")
            next(r for r in rows if r["method"] == "badge")["overall_macro_f1_after"] = .9999
            kbs.write_csv(directory / "by_seed.csv", rows)
            kbs.finish(directory, pilot.read_json(directory / "resolved_config.json"), oracle.EVAL_FILES)
            verified = allocation.verify_sources(args)
            allocation.select(args, verified); allocation.train(args, verified)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                allocation.evaluate(args, verified)

    def test_separate_cli_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            common = ["--oracle-dir", args.oracle_dir, "--pilot-dir", args.pilot_dir,
                      "--study-dir", args.study_dir, "--cache-dir", args.cache_dir,
                      "--checkpoint", args.checkpoint, "--output-dir", args.output_dir]
            for mode in ("plan", "preflight", "select", "train", "evaluate"):
                result = subprocess.run([sys.executable, allocation.__file__, mode, *common], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if mode == "plan": self.assertIn('"new_fits": 12', result.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
