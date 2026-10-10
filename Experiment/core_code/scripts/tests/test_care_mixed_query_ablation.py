"""Synthetic correctness checks, not empirical evidence for the query policy."""
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
import care_mixed_query_ablation as ab
import care_target_protection as protect
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import test_care_target_protection as parent_fixtures
import test_small_repair_diagnostic as fixtures


class MixedQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        pa, pc, _ = parent_fixtures.TargetProtectionTests().fixture(root)
        protect.select(pa, pc)
        protect.train(pa, pc)
        protect.evaluate(pa, pc)
        cli = ["--source-dir", pa.source_dir, "--source-audit-dir", pa.source_audit_dir,
            "--protection-dir", pa.output_dir, "--cache-dir", pa.cache_dir, "--checkpoint", pa.checkpoint,
            "--output-dir", str(root / "ablation"), "--threads", "1", "--log-every", "0"]
        args = ab.parser().parse_args(["select", *cli])
        return args, ab.inputs(args), cli

    def rehash(self, directory, name):
        marker = json.loads((directory / "complete.json").read_text())
        marker["artifacts"][name] = kbs.file_sha(directory / name)
        kbs.atomic_json(directory / "complete.json", marker)

    def test_queries_fixed_budget_same_pool_quotas_and_geometry_isolation(self):
        gen = torch.Generator().manual_seed(901)
        features = torch.randn(160, 6, generator=gen)
        logits = torch.randn(160, 4, generator=gen)
        queries, reg = ab.select_supplements(features, logits, 24, 12, 8, 6173)
        again, same = ab.select_supplements(features, logits, 24, 12, 8, 6173)
        self.assertEqual((queries, reg), (again, same))
        expected_repair = diag.select_indices("margin", logits, 12, 0).tolist()
        self.assertEqual(reg["repair_rows"], expected_repair)
        predicted = logits.argmax(1).numpy()
        top = torch.topk(logits, 2, dim=1).values
        margin = (top[:, 0] - top[:, 1]).numpy()
        for cid, pool in reg["pools"].items():
            c = int(cid)
            available = np.flatnonzero((predicted == c) & ~np.isin(np.arange(160), expected_repair))
            self.assertTrue(all(margin[i] >= np.median(margin[available]) for i in pool))
            self.assertLessEqual(len(pool), 8)
        pool = {i for v in reg["pools"].values() for i in v}
        for q, c in queries.items():
            self.assertEqual(len(c["rows"]), 24)
            self.assertEqual(len(set(c["rows"])), 24)
            self.assertEqual(c["repair_rows"], expected_repair)
            self.assertEqual(len(c["supplement_rows"]), 12)
            self.assertFalse(set(c["repair_rows"]) & set(c["supplement_rows"]))
            if q != "mixed_random":
                self.assertTrue(set(c["supplement_rows"]).issubset(pool))
        for q in ("mixed_balanced", "mixed_coverage"):
            counts = np.bincount(predicted[queries[q]["supplement_rows"]], minlength=4)
            self.assertEqual(counts.tolist(), [reg["quotas"][str(c)] for c in range(4)])
        # The random and balanced controls cannot depend on feature geometry.
        changed, reg2 = ab.select_supplements(torch.randn(160, 6, generator=gen), logits, 24, 12, 8, 6173)
        self.assertEqual(reg, reg2)
        for q in ("mixed_random", "mixed_pool_random", "mixed_balanced"):
            self.assertEqual(queries[q], changed[q])
        self.assertNotEqual(queries["mixed_coverage"], changed["mixed_coverage"])
        self.assertEqual(queries["mixed_coverage"]["rows"],
            protect.mixed_selection(features, logits, 24, 12, 8, 6173)["rows"])

    def test_pipeline_guards_labels_reuses_heads_preserves_inputs_and_manual_metrics(self):
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
                ab.select(args, ctx)
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before fit completion")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    ab.evaluate(args, ctx)
            choices = ab.selections(ctx)
            allowed = {i for c in choices for i in c["rows"]}
            def training_load(base, info, role, keep_labels=True):
                data = real_load(base, info, role, keep_labels)
                if role == "target":
                    data["labels"] = fixtures.GuardedLabels(data["labels"], allowed)
                return data
            with patch.object(diag, "load_role", side_effect=training_load), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fit:
                ab.train(args, ctx)
                self.assertEqual(fit.call_count, 6)  # three new query arms x two fixture seeds
            for c in choices:
                if c["selector"] in ab.REUSED:
                    parent = ctx["identity"]["reused_parents"][f"{c['selector']}/seed_{c['seed']}"]
                    self.assertEqual(kbs.file_sha(ab.run_path(ctx, c) / "head.pt"), parent["head_sha256"])
            ab.evaluate(args, ctx)
            self.assertEqual(before, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in before})
            out = ctx["out"]
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 10)
            union = allowed | {i for c in ctx["original"] + ctx["parent_choices"] for i in c["rows"]}
            with np.load(out / "evaluation/predictions.npz") as data:
                np.testing.assert_array_equal(data["row_indices"], [i for i in range(90) if i not in union])
                y, old = data["labels"], data["source"]
                for row in rows:
                    new = data[f"{row['selector']}_seed{row['seed']}"]
                    cm = np.zeros((3, 3), dtype=np.int64)
                    for a, b in zip(y, new):
                        cm[a, b] += 1
                    den = cm.sum(0) + cm.sum(1)
                    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(3), where=den != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
                    self.assertEqual(int(row["negative_flips"]), int(((old == y) & (new != y)).sum()))
                    self.assertEqual(int(row["positive_flips"]), int(((old != y) & (new == y)).sum()))
            lookup = {(r["seed"], r["selector"]): r for r in rows}
            for pair in kbs.read_csv(out / "evaluation/paired_by_seed.csv"):
                a, b = pair["comparison"].split(" - ")
                self.assertAlmostEqual(float(pair["overall_macro_f1_after"]),
                    float(lookup[pair["seed"], a]["overall_macro_f1_after"]) -
                    float(lookup[pair["seed"], b]["overall_macro_f1_after"]))
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 30)
            stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume loaded arrays")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume fitted")):
                fresh = ab.inputs(args)
                ab.select(args, fresh)
                ab.train(args, fresh)
                ab.evaluate(args, fresh)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_incomplete_parent_and_output_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            for directory in (args.protection_dir, args.source_dir, args.source_audit_dir, args.cache_dir, args.checkpoint):
                args.output_dir = directory
                with self.assertRaisesRegex(ValueError, "separate"):
                    ab.inputs(args)
            args.output_dir = str(ctx["out"])
            (Path(args.protection_dir) / "evaluation/complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                ab.inputs(args)

    def test_changed_protocol_corrupt_parent_and_corrupt_new_head_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            args.eval_size = 10
            changed = ab.inputs(args)
            with self.assertRaisesRegex(ValueError, "code/settings/environment"):
                ab.train(args, changed)
            args.eval_size = 0
            ab.train(args, ctx)
            path = ab.run_path(ctx, ab.selections(ctx)[0])
            (path / "head.pt").write_bytes(b"CORRUPT")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before integrity checks")):
                with self.assertRaisesRegex(ValueError, "Missing/modified"):
                    ab.evaluate(args, ctx)
            parent = next((Path(args.protection_dir) / "runs").rglob("head.pt"))
            parent.write_bytes(b"CORRUPT PARENT")
            with self.assertRaisesRegex(ValueError, "Missing/modified"):
                ab.inputs(args)

    def test_rehashed_invalid_query_and_quota_rejected_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            path = ctx["out"] / "selection"
            queries = json.loads((path / "queries.json").read_text())
            queries[1]["rows"][0] = -1
            kbs.atomic_json(path / "queries.json", queries)
            self.rehash(path, "queries.json")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before query validation")):
                with self.assertRaisesRegex(ValueError, "Invalid frozen query"):
                    ab.train(args, ctx)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            path = ctx["out"] / "selection"
            pools = json.loads((path / "candidate_pools.json").read_text())
            cid = next(iter(pools["quotas"]))
            pools["quotas"][cid] += 1
            kbs.atomic_json(path / "candidate_pools.json", pools)
            self.rehash(path, "candidate_pools.json")
            with patch.object(diag, "load_role", side_effect=AssertionError("Truth before quota validation")):
                with self.assertRaisesRegex(ValueError, "quotas"):
                    ab.train(args, ctx)

    def test_rehashed_stale_trace_and_changed_reused_head_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            ab.train(args, ctx)
            c = next(c for c in ab.selections(ctx) if c["selector"] == "mixed_random")
            path = ab.run_path(ctx, c)
            trace = json.loads((path / "trace.json").read_text())
            trace["source_correct_paid_count"] += 1
            kbs.atomic_json(path / "trace.json", trace)
            self.rehash(path, "trace.json")
            with self.assertRaisesRegex(ValueError, "Paid query trace"):
                ab.evaluate(args, ctx)
            self.assertFalse((ctx["out"] / "evaluation/complete.json").exists())
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            ab.train(args, ctx)
            c = ab.selections(ctx)[0]
            path = ab.run_path(ctx, c)
            # Re-serialize the same weights: metrics still match, but exact parent identity does not.
            state = torch.load(path / "head.pt", weights_only=True)
            torch.save(state, path / "other-name.pt")
            (path / "head.pt").write_bytes((path / "other-name.pt").read_bytes())
            self.rehash(path, "head.pt")
            with self.assertRaisesRegex(ValueError, "Reused head identity"):
                ab.evaluate(args, ctx)

    def test_rehashed_evaluation_mask_must_exclude_all_queries(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            ab.select(args, ctx)
            ab.train(args, ctx)
            path = ctx["out"] / "selection"
            ids = np.load(path / "evaluation_rows.npy")
            ids[0] = ab.selections(ctx)[0]["rows"][0]
            np.save(path / "evaluation_rows.npy", ids)
            self.rehash(path, "evaluation_rows.npy")
            # Rebind run signatures to test semantic mask validation independently of hashes.
            for c in ab.selections(ctx):
                run = ab.run_path(ctx, c)
                sig = ab.run_signature(ctx, c)
                kbs.atomic_json(run / "resolved_config.json", sig)
                kbs.finish(run, sig, sorted(ab.RUN_ARTIFACTS))
            with self.assertRaisesRegex(ValueError, "Evaluation rows differ"):
                ab.evaluate(args, ctx)

    def test_separate_cli_stages(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "select", "train", "evaluate"):
                run = subprocess.run([sys.executable, ab.__file__, stage, *cli], text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                if stage == "evaluate":
                    self.assertIn("CARE_MIXED_QUERY_ABLATION_COMPLETE", run.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
