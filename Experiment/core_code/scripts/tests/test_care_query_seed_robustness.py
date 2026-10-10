"""Synthetic replication and integrity checks, not evidence of robustness."""
import contextlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import care_query_seed_robustness as robust
import care_mixed_query_ablation as ab
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import test_care_mixed_query_ablation as parent_fixtures
import test_small_repair_diagnostic as fixtures


class QuerySeedRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        aa, ac, _ = parent_fixtures.MixedQueryTests().fixture(root)
        ab.select(aa, ac)
        ab.train(aa, ac)
        ab.evaluate(aa, ac)
        cli = ["--source-dir", aa.source_dir, "--source-audit-dir", aa.source_audit_dir,
            "--protection-dir", aa.protection_dir, "--ablation-dir", aa.output_dir,
            "--cache-dir", aa.cache_dir, "--checkpoint", aa.checkpoint,
            "--output-dir", str(root / "robustness"), "--threads", "1", "--log-every", "0", "--query-seeds", "6174,6175"]
        args = robust.parser().parse_args(["select", *cli])
        return args, robust.inputs(args), cli

    def rehash(self, directory, name):
        marker = json.loads((directory / "complete.json").read_text())
        marker["artifacts"][name] = kbs.file_sha(directory / name)
        kbs.atomic_json(directory / "complete.json", marker)

    def test_hierarchical_paired_summary_excludes_development_and_uses_query_units(self):
        ctx = {"identity": {"development_query_seed": 10, "fresh_query_seeds": [11, 12]}, "protocol": {"seeds": [0, 1]}}
        rows = [{"query_seed": -1, "seed": s, "selector": "margin", **{k: .4 for k in robust.METRICS}} for s in [0, 1]]
        differences = {10: [.8, .8], 11: [-.1, .3], 12: [.3, .7]}
        for qs, deltas in differences.items():
            for seed, delta in enumerate(deltas):
                for selector in robust.SELECTORS:
                    score = .1 + (delta if selector == "mixed_coverage" else 0)
                    rows.append({"query_seed": qs, "seed": seed, "selector": selector, **{k: score for k in robust.METRICS}})
        pairs = robust.paired(rows, ctx)
        _, by_query, _, primary = robust.summarize_fresh(rows, pairs, ctx)
        result = next(r for r in primary if r["comparison"] == "mixed_coverage - mixed_balanced")
        expected_units = [statistics.mean(differences[qs]) for qs in [11, 12]]
        self.assertAlmostEqual(result["overall_macro_f1_after_mean"], statistics.mean(expected_units))
        self.assertAlmostEqual(result["overall_macro_f1_after_sd"], statistics.stdev(expected_units))
        self.assertEqual(result["n_query_seeds"], 2)
        self.assertEqual(result["macro_positive_query_seeds"], 2)
        self.assertEqual(result["negative_flip_reduced_query_seeds"], 0)
        self.assertNotAlmostEqual(result["overall_macro_f1_after_sd"], statistics.stdev([-.1, .3, .3, .7]))
        self.assertEqual({r["query_seed"] for r in by_query}, {10, 11, 12})
        self.assertNotIn("query_seed_mean", result)

    def test_pipeline_only_fresh_fits_paid_labels_shared_quotas_and_preserved_sources(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, ctx, _ = self.fixture(root)
            before = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            real_load = diag.load_role
            def selection_load(base, info, role, keep_labels=True):
                self.assertEqual(role, "target")
                self.assertFalse(keep_labels)
                data = real_load(base, info, role, keep_labels)
                self.assertNotIn("labels", data)
                return data
            with patch.object(diag, "load_role", side_effect=selection_load):
                robust.select(args, ctx)
            choices = robust.selections(ctx)
            self.assertEqual(len(choices), 14)  # (two fresh + one development) x two arms x two train seeds + two Margin
            registries = json.loads((ctx["out"] / "selection/candidate_pools.json").read_text())
            repair_sets = {tuple(r["repair_rows"]) for r in registries.values()}
            self.assertEqual(len(repair_sets), 1)
            for qs in [6173, 6174, 6175]:
                a = next(c for c in choices if c["query_seed"] == qs and c["selector"] == "mixed_balanced")
                b = next(c for c in choices if c["query_seed"] == qs and c["selector"] == "mixed_coverage")
                reg = registries[str(qs)]
                for cid, n in reg["quotas"].items():
                    for c in (a, b):
                        self.assertEqual(len(set(c["supplement_rows"]).intersection(reg["pools"][cid])), n)
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before all fits")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    robust.evaluate(args, ctx)
            allowed = {i for c in choices for i in c["rows"]}
            def training_load(base, info, role, keep_labels=True):
                data = real_load(base, info, role, keep_labels)
                if role == "target":
                    data["labels"] = fixtures.GuardedLabels(data["labels"], allowed)
                return data
            with patch.object(diag, "load_role", side_effect=training_load), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fitted:
                robust.train(args, ctx)
                self.assertEqual(fitted.call_count, 8)  # two fresh queries x two arms x two train seeds
            for c in choices:
                parent = robust.reused_parent(ctx, c)
                if parent:
                    self.assertEqual(kbs.file_sha(robust.run_path(ctx, c) / "head.pt"), parent["head_sha256"])
            robust.evaluate(args, ctx)
            self.assertEqual(before, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in before})
            out = ctx["out"]
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            union = allowed | {i for c in ctx["old_choices"] for i in c["rows"]}
            with np.load(out / "evaluation/predictions.npz") as data:
                np.testing.assert_array_equal(data["row_indices"], [i for i in range(90) if i not in union])
                y, old = data["labels"], data["source"]
                for row in rows:
                    new = data[f"query{row['query_seed']}_{row['selector']}_seed{row['seed']}"]
                    cm = np.zeros((3, 3), dtype=np.int64)
                    for a, b in zip(y, new):
                        cm[a, b] += 1
                    den = cm.sum(0) + cm.sum(1)
                    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(3), where=den != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
                    self.assertEqual(int(row["negative_flips"]), int(((old == y) & (new != y)).sum()))
            exported = kbs.read_csv(out / "evaluation/fresh_paired_summary.csv")
            pairs = kbs.read_csv(out / "evaluation/paired_by_seed.csv")
            for row in exported:
                means = [statistics.mean(float(r["overall_macro_f1_after"]) for r in pairs
                    if int(r["query_seed"]) == qs and r["comparison"] == row["comparison"]) for qs in [6174, 6175]]
                self.assertAlmostEqual(float(row["overall_macro_f1_after_mean"]), statistics.mean(means))
                self.assertAlmostEqual(float(row["overall_macro_f1_after_sd"]), statistics.stdev(means))
                self.assertEqual(int(row["n_query_seeds"]), 2)
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 42)
            stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume loaded arrays")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume fitted")):
                fresh = robust.inputs(args)
                robust.select(args, fresh)
                robust.train(args, fresh)
                robust.evaluate(args, fresh)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_fresh_seed_validation_development_rejection_and_output_aliases(self):
        for value in ("", "1,1", "-1,2"):
            with self.assertRaises(ValueError):
                robust.seed_list(value)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            args.query_seeds = "6173,6174"
            with self.assertRaisesRegex(ValueError, "development query seed"):
                robust.inputs(args)
            args.query_seeds = "6174,6175"
            for path in (args.source_dir, args.source_audit_dir, args.protection_dir, args.ablation_dir, args.cache_dir, args.checkpoint):
                args.output_dir = path
                with self.assertRaisesRegex(ValueError, "separate"):
                    robust.inputs(args)
            args.output_dir = str(ctx["out"])

    def test_missing_parent_corrupt_artifact_and_changed_query_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            robust.select(args, ctx)
            args.query_seeds = "6174,6176"
            changed = robust.inputs(args)
            with self.assertRaisesRegex(ValueError, "code/settings/environment"):
                robust.train(args, changed)
            args.query_seeds = "6174,6175"
            (Path(args.ablation_dir) / "evaluation/complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                robust.inputs(args)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            robust.select(args, ctx)
            robust.train(args, ctx)
            (robust.run_path(ctx, robust.selections(ctx)[0]) / "head.pt").write_bytes(b"CORRUPT")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before integrity checks")):
                with self.assertRaisesRegex(ValueError, "Missing/modified"):
                    robust.evaluate(args, ctx)

    def test_rehashed_bad_query_seed_and_candidate_quota_rejected_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            robust.select(args, ctx)
            path = ctx["out"] / "selection"
            queries = json.loads((path / "queries.json").read_text())
            queries[0]["query_seed"] = 9999
            kbs.atomic_json(path / "queries.json", queries)
            self.rehash(path, "queries.json")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before selection validation")):
                with self.assertRaisesRegex(ValueError, "grid mismatch"):
                    robust.train(args, ctx)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            robust.select(args, ctx)
            path = ctx["out"] / "selection"
            registries = json.loads((path / "candidate_pools.json").read_text())
            reg = registries["6174"]
            cid = next(iter(reg["quotas"]))
            reg["quotas"][cid] += 1
            kbs.atomic_json(path / "candidate_pools.json", registries)
            self.rehash(path, "candidate_pools.json")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before quota validation")):
                with self.assertRaisesRegex(ValueError, "pools/quotas"):
                    robust.train(args, ctx)

    def test_rehashed_stale_trace_and_mismatched_training_stream_rejected(self):
        for key, value in (("source_correct_paid_count", -1), ("reference_stream_sha256", "ALTERED")):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                args, ctx, _ = self.fixture(Path(tmp))
                robust.select(args, ctx)
                robust.train(args, ctx)
                c = next(c for c in robust.selections(ctx) if c["query_seed"] == 6174)
                path = robust.run_path(ctx, c)
                trace = json.loads((path / "trace.json").read_text())
                trace[key] = value
                kbs.atomic_json(path / "trace.json", trace)
                self.rehash(path, "trace.json")
                with self.assertRaisesRegex(ValueError, "paid query trace|training stream/replay"):
                    robust.evaluate(args, ctx)
                self.assertFalse((ctx["out"] / "evaluation/complete.json").exists())

    def test_inventory_reads_metadata_without_tensor_loading_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cache = root / "m11/cache"
            cache.mkdir(parents=True)
            for n in ("head.pt", "reference.pt", "target.pt"):
                (cache / n).write_bytes(b"PRESENCE ONLY, NOT A VALID TENSOR")
            kbs.atomic_json(cache / "manifest.json", {"periods": {"reference": "M-2022-4", "target": "M-2022-11"},
                "num_classes": 178, "files": {n: "NOT HASH VALIDATED" for n in ("head.pt", "reference.pt", "target.pt")}})
            before = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            args = robust.parser().parse_args(["inventory", "--scan-root", str(root)])
            with patch.object(torch, "load", side_effect=AssertionError("Inventory loaded tensors")):
                result = robust.inventory(args)
            self.assertTrue(result["m11_cache_metadata_found"])
            self.assertFalse(result["tensor_hashes_checked"])
            self.assertEqual(before, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in before})

    def test_separate_cli_stages(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "select", "train", "evaluate"):
                p = subprocess.run([sys.executable, robust.__file__, stage, *cli], text=True, capture_output=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                if stage == "evaluate":
                    self.assertIn("CARE_QUERY_SEED_ROBUSTNESS_COMPLETE", p.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
