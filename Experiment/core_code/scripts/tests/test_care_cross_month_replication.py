"""Synthetic month/cache/protocol checks; never research evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import care_cross_month_replication as month
import care_query_seed_robustness as robust
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import test_care_query_seed_robustness as parent_fixtures
import test_small_repair_diagnostic as fixtures


class SyntheticBackbone(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.cls_head = torch.nn.Module()
        self.cls_head.fc = torch.nn.Linear(4, 3)
        self.cls_head.fc.load_state_dict(state)

    def forward(self, ppi, flow_stats=None, return_repr=False):
        return self.cls_head.fc(ppi), ppi


class CrossMonthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, prepare=True):
        original = fixtures.SmallRepairTests.fixture
        def enriched(obj, path):
            args, info, identity, cli = original(obj, path)
            raw = path / "raw"
            raw.mkdir()
            (raw / "SYNTHETIC.h5").write_bytes(b"NOT CESNET DATA")
            info["spec"].update(data_config={"dataset": "tls22", "data_dir": str(raw),
                "train_period": "M-2022-3", "test_period": "M-2022-5", "batch_size": 256}, data_seed=0,
                checkpoint_sha256=info["checkpoint_sha256"], periods=info["periods"])
            info["input_stream_sha256"] = {r: kbs.digest([r, info["files"][r + ".pt"]]) for r in ("reference", "target")}
            info["data_inventory"] = kbs.data_inventory(raw)
            kbs.atomic_json(Path(args.cache_dir) / "manifest.json", info)
            return args, info, diag.inputs(args)[1], cli
        with patch.object(fixtures.SmallRepairTests, "fixture", enriched):
            ra, _, _ = parent_fixtures.QuerySeedRobustnessTests().fixture(root)
        ra.query_seeds = ",".join(map(str, robust.FRESH_SEEDS))
        pc = robust.inputs(ra)
        robust.select(ra, pc)
        robust.train(ra, pc)
        robust.evaluate(ra, pc)
        cli = ["--source-dir", ra.source_dir, "--source-audit-dir", ra.source_audit_dir,
            "--protection-dir", ra.protection_dir, "--ablation-dir", ra.ablation_dir,
            "--replication-dir", ra.output_dir, "--source-cache-dir", ra.cache_dir,
            "--checkpoint", ra.checkpoint, "--cache-dir", str(root / "m11cache"),
            "--output-dir", str(root / "m11"), "--threads", "1", "--log-every", "0"]
        args = month.parser().parse_args(["prepare", *cli])
        pc = month.parent_context(args)
        if prepare:
            with patch.dict(sys.modules, {"prototype_recalibration_tls22": self.fake_proto(args)}):
                month.prepare(args, pc)
        return args, pc, cli

    def fake_proto(self, args, empty=False, wrong_classes=False, change_raw=False):
        state = torch.load(Path(args.source_cache_dir) / "head.pt", weights_only=True)
        model = SyntheticBackbone(state)
        gen = torch.Generator().manual_seed(987)
        x, y = torch.randn(93, 4, generator=gen), torch.arange(93) % 3
        def loader(cfg, period):
            self.assertEqual(period, "M-2022-11")
            self.assertEqual(cfg["data"]["train_period"], "M-2022-3")
            def batches():
                if empty:
                    return
                for start in range(0, 93, 16):
                    yield {"ppi": x[start:start + 16], "label": y[start:start + 16]}
                if change_raw:
                    (Path(cfg["data"]["data_dir"]) / "SYNTHETIC.h5").write_bytes(b"CHANGED")
            return batches(), 4 if wrong_classes else 3
        return types.SimpleNamespace(load_source_model=lambda *_: (model, {}, 3), make_test_loader=loader)

    def rehash(self, directory, name):
        marker = json.loads((directory / "complete.json").read_text())
        marker["artifacts"][name] = kbs.file_sha(directory / name)
        kbs.atomic_json(directory / "complete.json", marker)

    def test_cache_only_extracts_m11_exact_reference_copy_and_resume_no_tensors(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp), prepare=False)
            original = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            fake = self.fake_proto(args)
            with patch.dict(sys.modules, {"prototype_recalibration_tls22": fake}), \
                 patch.object(fake, "make_test_loader", wraps=fake.make_test_loader) as loader:
                month.prepare(args, pc)
                self.assertEqual(loader.call_count, 1)
            info = month.validate_month_cache(args, pc, month.cache_signature(args, pc))
            self.assertEqual(info["sample_counts"], {"reference": 36, "target": 93})
            for n in ("reference.pt", "head.pt"):
                self.assertEqual(kbs.file_sha(Path(args.cache_dir) / n), pc["info"]["files"][n])
            self.assertEqual(original, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in original})
            stamps = {p: p.stat().st_mtime_ns for p in Path(args.cache_dir).iterdir() if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(month, "collect_target", side_effect=AssertionError("Resume extracted")):
                month.prepare(args, month.parent_context(args))
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_pipeline_paid_labels_new_month_mask_streams_manual_metrics_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp))
            ctx = month.inputs(args, pc)
            original = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            real_load = diag.load_role
            def no_truth(base, info, role, keep_labels=True):
                self.assertEqual(role, "target")
                self.assertFalse(keep_labels)
                return real_load(base, info, role, keep_labels)
            with patch.object(diag, "load_role", side_effect=no_truth):
                month.select(args, ctx)
            choices = month.selections(ctx)
            self.assertEqual(len(choices), 22)  # (5*2 + Margin) * 2 tiny training seeds
            for qs in robust.FRESH_SEEDS:
                a = next(c for c in choices if c["query_seed"] == qs and c["selector"] == "mixed_balanced")
                b = next(c for c in choices if c["query_seed"] == qs and c["selector"] == "mixed_coverage")
                logits = real_load(ctx["base"], ctx["info"], "target", False)["logits"].argmax(1).numpy()
                np.testing.assert_array_equal(np.bincount(logits[a["supplement_rows"]], minlength=3),
                    np.bincount(logits[b["supplement_rows"]], minlength=3))
            with patch.object(diag, "load_role", side_effect=AssertionError("Premature held-out truth")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    month.evaluate(args, ctx)
            allowed = {r for c in choices for r in c["rows"]}
            def paid_only(base, info, role, keep_labels=True):
                data = real_load(base, info, role, keep_labels)
                if role == "target":
                    data["labels"] = fixtures.GuardedLabels(data["labels"], allowed)
                return data
            with patch.object(diag, "load_role", side_effect=paid_only), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fit:
                month.train(args, ctx)
                self.assertEqual(fit.call_count, len(choices))
            month.evaluate(args, ctx)
            self.assertEqual(original, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in original})
            rows = kbs.read_csv(ctx["out"] / "evaluation/by_seed.csv")
            with np.load(ctx["out"] / "evaluation/predictions.npz") as data:
                ids = [i for i in range(93) if i not in allowed]
                np.testing.assert_array_equal(data["row_indices"], ids)
                # M12 has 90 rows and a different query union; its IDs never enter M11 exclusion.
                self.assertTrue(any(i >= 90 for i in ids))
                old_union = {i for c in robust.selections(pc) + pc["old_choices"] for i in c["rows"]}
                self.assertTrue(set(ids) & old_union)
                y, before = data["labels"], data["source"]
                for row in rows:
                    after = data[f"query{row['query_seed']}_{row['selector']}_seed{row['seed']}"]
                    cm = np.zeros((3, 3), dtype=np.int64)
                    for a, b in zip(y, after):
                        cm[a, b] += 1
                    den = cm.sum(0) + cm.sum(1)
                    f1 = np.divide(2*np.diag(cm), den, out=np.zeros(3), where=den != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
                    self.assertEqual(int(row["negative_flips"]), int(((before == y) & (after != y)).sum()))
                _, source_classes, _ = kbs.compare_predictions(y, before, before, np.ones(len(y), bool), 3, [1], [2], .1, .05)
                expected_low = [r["class_id"] for r in source_classes if r["support"] > 0 and r["before_f1"] < .1]
                groups = json.loads((ctx["out"] / "evaluation/evaluation_groups.json").read_text())
                self.assertEqual(groups["m11_low_f1_classes"], expected_low)
            pairs = kbs.read_csv(ctx["out"] / "evaluation/paired_by_seed.csv")
            for row in kbs.read_csv(ctx["out"] / "evaluation/fresh_paired_summary.csv"):
                units = [statistics.mean(float(r["overall_macro_f1_after"]) for r in pairs
                    if int(r["query_seed"]) == qs and r["comparison"] == row["comparison"]) for qs in robust.FRESH_SEEDS]
                self.assertAlmostEqual(float(row["overall_macro_f1_after_mean"]), statistics.mean(units))
                self.assertAlmostEqual(float(row["overall_macro_f1_after_sd"]), statistics.stdev(units))
                self.assertEqual(int(row["n_query_seeds"]), 5)
            stamps = {p: p.stat().st_mtime_ns for p in ctx["out"].rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume loaded arrays")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume trained")):
                fresh = month.inputs(args, month.parent_context(args))
                month.select(args, fresh)
                month.train(args, fresh)
                month.evaluate(args, fresh)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_empty_wrong_class_and_changed_raw_leave_cache_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp), prepare=False)
            for name, kwargs, expected in (("empty", {"empty": True}, "no samples"),
                ("wrongclass", {"wrong_classes": True}, "class universes"), ("changed", {"change_raw": True}, "Raw data changed")):
                args.cache_dir = str(Path(tmp) / name)
                with patch.dict(sys.modules, {"prototype_recalibration_tls22": self.fake_proto(args, **kwargs)}):
                    with self.assertRaisesRegex(ValueError, expected):
                        month.prepare(args, pc)
                self.assertFalse((Path(args.cache_dir) / "manifest.json").exists())

    def test_wrong_backbone_head_and_modified_completed_tensor_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp), prepare=False)
            fake = self.fake_proto(args)
            model, _, _ = fake.load_source_model()
            with torch.no_grad():
                model.cls_head.fc.bias.add_(1)
            with patch.dict(sys.modules, {"prototype_recalibration_tls22": fake}):
                with self.assertRaisesRegex(ValueError, "backbone head"):
                    month.prepare(args, pc)
            self.assertFalse((Path(args.cache_dir) / "manifest.json").exists())
            # Same registered interrupted cache can restart with the verified source.
            with patch.dict(sys.modules, {"prototype_recalibration_tls22": self.fake_proto(args)}):
                month.prepare(args, pc)
            tensor = Path(args.cache_dir) / "target.pt"
            with tensor.open("ab") as stream:
                stream.write(b"CORRUPTED")
            with self.assertRaisesRegex(ValueError, "modified cache file"):
                month.inputs(args, pc)

    def test_paths_missing_parent_and_corrupt_cache_fail(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp))
            for name in ("source_dir", "source_cache_dir", "replication_dir", "checkpoint"):
                bad = copy.deepcopy(args)
                bad.output_dir = getattr(args, name)
                with self.assertRaisesRegex(ValueError, "separate"):
                    month.parent_context(bad)
            bad = copy.deepcopy(args)
            bad.cache_dir = args.output_dir
            with self.assertRaisesRegex(ValueError, "separate"):
                month.parent_context(bad)
            parent_marker = Path(args.replication_dir) / "evaluation/complete.json"
            parent_marker.rename(parent_marker.with_suffix(".bak"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                month.parent_context(args)
            parent_marker.with_suffix(".bak").rename(parent_marker)
            info = json.loads((Path(args.cache_dir) / "manifest.json").read_text())
            info["input_stream_sha256"]["reference"] = "changed"
            kbs.atomic_json(Path(args.cache_dir) / "manifest.json", info)
            with self.assertRaisesRegex(ValueError, "provenance"):
                month.inputs(args, pc)

    def test_rehashed_quota_mask_and_stream_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, pc, _ = self.fixture(Path(tmp))
            ctx = month.inputs(args, pc)
            month.select(args, ctx)
            choices = month.selections(ctx)
            directory = ctx["out"] / "selection"
            original = (directory / "candidate_pools.json").read_text()
            regs = json.loads(original)
            first = regs[str(robust.FRESH_SEEDS[0])]
            cid = next(iter(first["quotas"]))
            first["quotas"][cid] += 1
            kbs.atomic_json(directory / "candidate_pools.json", regs)
            self.rehash(directory, "candidate_pools.json")
            with self.assertRaisesRegex(ValueError, "quota"):
                month.selections(ctx)
            (directory / "candidate_pools.json").write_text(original)
            self.rehash(directory, "candidate_pools.json")
            month.train(args, ctx)
            ids = np.load(directory / "evaluation_rows.npy")
            np.save(directory / "evaluation_rows.npy", ids[:-1])
            self.rehash(directory, "evaluation_rows.npy")
            # Update run signatures after deliberately altering the selection completion hash.
            for c in choices:
                path = robust.run_path(ctx, c)
                sig = robust.run_signature(ctx, c)
                kbs.atomic_json(path / "resolved_config.json", sig)
                kbs.finish(path, sig, sorted(robust.ab.RUN_ARTIFACTS))
            with self.assertRaisesRegex(ValueError, "evaluation mask"):
                month.evaluate(args, ctx)
            np.save(directory / "evaluation_rows.npy", ids)
            self.rehash(directory, "evaluation_rows.npy")
            for c in choices:
                path = robust.run_path(ctx, c)
                sig = robust.run_signature(ctx, c)
                kbs.atomic_json(path / "resolved_config.json", sig)
                kbs.finish(path, sig, sorted(robust.ab.RUN_ARTIFACTS))
            path = robust.run_path(ctx, choices[0])
            trace = json.loads((path / "trace.json").read_text())
            trace["reference_stream_sha256"] = "changed"
            kbs.atomic_json(path / "trace.json", trace)
            self.rehash(path, "trace.json")
            with self.assertRaisesRegex(ValueError, "stream"):
                month.evaluate(args, ctx)

    def test_nested_summary_and_no_low_f1_group(self):
        ctx = {"identity": {"fresh_query_seeds": [1, 2]}, "protocol": {"seeds": [0, 1]}}
        rows = [{"query_seed": -1, "seed": s, "selector": "margin", **{k: .1 for k in month.METRICS}} for s in [0, 1]]
        for qs, differences in ((1, [-.1, .3]), (2, [.3, .7])):
            for seed, difference in enumerate(differences):
                for arm in robust.SELECTORS:
                    rows.append({"query_seed": qs, "seed": seed, "selector": arm,
                        **{k: .2 + (difference if arm == "mixed_coverage" else 0) for k in month.METRICS}})
        for row in rows:
            row["m11_low_f1_macro_after"] = row["m11_low_f1_macro_delta"] = None
        _, _, _, primary = month.summaries(rows, month.paired(rows, ctx), ctx)
        r = primary[0]
        self.assertAlmostEqual(r["overall_macro_f1_after_mean"], .3)
        self.assertAlmostEqual(r["overall_macro_f1_after_sd"], statistics.stdev([.1, .5]))
        self.assertNotIn("m11_low_f1_macro_after_mean", r)
        self.assertNotIn("query_seed_mean", r)
        self.assertEqual(r["n_query_seeds"], 2)

    def test_separate_cli_stages_use_prepared_cache(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            _, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "select", "train", "evaluate"):
                result = subprocess.run([sys.executable, month.__file__, stage, *cli], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CARE_M11_REPLICATION_COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
