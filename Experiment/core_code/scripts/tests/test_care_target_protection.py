"""Synthetic mechanism checks only, not research evidence."""
import contextlib
import copy
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
import care_target_protection as protect
import small_repair_diagnostic as diag
import small_repair_full_audit as full
import kbs_supplement as kbs
import test_small_repair_diagnostic as fixtures


class TargetProtectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        old, info, protocol, _ = fixtures.SmallRepairTests().fixture(root)
        diag.select(old, info, protocol)
        diag.train(old, info, protocol)
        diag.evaluate(old, info, protocol)
        full_args = full.parser().parse_args(["run", "--source-dir", old.output_dir, "--cache-dir", old.cache_dir,
            "--checkpoint", old.checkpoint, "--output-dir", str(root / "audit"), "--threads", "1"])
        full.audit(full_args)
        cli = ["--source-dir", old.output_dir, "--source-audit-dir", full_args.output_dir,
            "--cache-dir", old.cache_dir, "--checkpoint", old.checkpoint,
            "--output-dir", str(root / "protect"), "--threads", "1", "--protection-budget", "2",
            "--candidate-cap", "8", "--log-every", "0"]
        args = protect.parser().parse_args(["select", *cli])
        return args, protect.inputs(args), cli

    def test_all_rival_feasible_segment_matches_independent_linear_bound(self):
        source = torch.nn.Linear(1, 3)
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[2.], [0.], [-1.]]))
            source.bias.zero_()
        anchors = protect.AnchorConstraints(source, torch.ones(1, 1), np.array([0]), .25, 1., 1e-6)
        candidate = copy.deepcopy(source)
        with torch.no_grad():
            candidate.weight[2] = 10.
        alpha, _, violation = anchors.accept(candidate, source.state_dict())
        # The third class, not the source runner-up, supplies the limiting inequality.
        expected = (3 - (.25 - .5e-6)) / (3 - (-8))
        self.assertAlmostEqual(alpha, expected, places=6)
        self.assertLessEqual(violation, 1e-6)
        self.assertEqual(anchors.negative_flips(candidate), 0)
        self.assertEqual(anchors.violation(source), 0)
        tied = protect.AnchorConstraints(source, torch.zeros(1, 1), np.array([0]), .25, 1., 1e-6)
        self.assertEqual(len(tied.y), 0)
        self.assertEqual(tied.excluded_weak_correct, 1)

    def test_empty_constraints_reproduce_legacy_care_and_matched_streams(self):
        torch.manual_seed(111)
        source = torch.nn.Linear(4, 3)
        x, rx = torch.randn(12, 4), torch.randn(9, 4)
        y = (source(x).argmax(1).numpy() + 1) % 3
        ry, teacher = np.arange(9) % 3, source(rx).detach()
        p = dict(lr=.01, weight_decay=.0001, steps=10, batch_size=4,
                 target_weight=.7, replay_weight=.3, kd_weight=.5, temperature=2.)
        a, ta = kbs.fit_controlled(source, x, y, rx, ry, teacher,
            replay_ce=True, seed=0, device=diag.CPU, log_every=0, **p)
        anchors = protect.AnchorConstraints(source, x, y, .25, 1., 1e-6)
        self.assertEqual(len(anchors.y), 0)
        b, tb = protect.fit_constrained(source, x, y, rx, ry, teacher, p, 0, anchors, 0)
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
        for k in ("target_stream_sha256", "reference_stream_sha256", "target_presentations", "reference_ce_presentations", "mean_losses"):
            self.assertEqual(ta[k], tb[k])
        self.assertEqual(tb["accepted_step_scale_mean"], 1)

    def test_label_free_selection_budget_partitions_strata_and_repeatability(self):
        gen = torch.Generator().manual_seed(3)
        x = torch.randn(90, 4, generator=gen)
        logits = torch.randn(90, 3, generator=gen)
        a = protect.mixed_selection(x, logits, 12, 6, 8, 6173)
        b = protect.mixed_selection(x, logits, 12, 6, 8, 6173)
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 12)
        self.assertEqual(len(set(a["rows"])), 12)
        self.assertEqual(set(a["repair_rows"]) & set(a["protection_rows"]), set())
        np.testing.assert_array_equal(a["repair_rows"], diag.select_indices("margin", logits, 6, 0))
        self.assertEqual(a["source_predicted_protection_classes"], 3)
        for c in range(3):
            candidates = np.flatnonzero((logits.argmax(1).numpy() == c) & ~np.isin(np.arange(90), a["repair_rows"]))
            top = torch.topk(logits, 2, dim=1).values
            margin = (top[:, 0] - top[:, 1]).numpy()
            selected = [i for i in a["protection_rows"] if int(logits[i].argmax()) == c]
            self.assertTrue(all(margin[i] >= np.median(margin[candidates]) for i in selected))

    def test_pipeline_only_paid_labels_six_new_fits_all_old_files_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, ctx, _ = self.fixture(root)
            before = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            protect.select(args, ctx)
            with patch.object(diag, "load_role", side_effect=AssertionError("Held-out truth read before fits")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    protect.evaluate(args, ctx)
            choices = protect.selections(Path(args.output_dir), ctx[-1])
            allowed = {i for c in choices for i in c["rows"]}
            real_load = diag.load_role
            def guarded(base, info, role, keep_labels=True):
                data = real_load(base, info, role, keep_labels)
                if role == "target":
                    data["labels"] = fixtures.GuardedLabels(data["labels"], allowed)
                return data
            with patch.object(diag, "load_role", side_effect=guarded), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as legacy_fit, \
                 patch.object(protect, "fit_constrained", wraps=protect.fit_constrained) as constrained_fit:
                protect.train(args, ctx)
                self.assertEqual(legacy_fit.call_count, 2)  # two mixed CARE fits, Margin reused
                self.assertEqual(constrained_fit.call_count, 4)
            protect.evaluate(args, ctx)
            self.assertEqual(before, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in before})
            out = Path(args.output_dir)
            runs = [json.loads(p.read_text()) for p in (out / "runs").rglob("trace.json")]
            self.assertEqual(len(runs), 12)
            for p in (out / "runs").rglob("constrained/trace.json"):
                t = json.loads(p.read_text())
                self.assertEqual(t["anchor_negative_flips"], 0)
                self.assertLessEqual(t["max_accepted_constraint_violation"], 1e-6)
            original = ctx[5]
            union = allowed | {i for c in original for i in c["rows"]}
            with np.load(out / "evaluation/predictions.npz") as data:
                np.testing.assert_array_equal(data["row_indices"], [i for i in range(90) if i not in union])
                y, old = data["labels"], data["source"]
                for row in kbs.read_csv(out / "evaluation/by_seed.csv"):
                    new = data[f"{row['selector']}_seed{row['seed']}_{row['engine']}"]
                    self.assertEqual(int(row["negative_flips"]), int(((old == y) & (new != y)).sum()))
                    cm = np.zeros((3, 3))
                    for t, p in zip(y, new):
                        cm[t, p] += 1
                    den = cm.sum(0) + cm.sum(1)
                    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(3), where=den != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
            stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(np, "load", side_effect=AssertionError("Resume loaded arrays")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume fitted")):
                protect.select(args, ctx)
                protect.train(args, ctx)
                protect.evaluate(args, ctx)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})

    def test_missing_source_audit_corruption_changed_protocol_and_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, ctx, _ = self.fixture(root)
            args.source_audit_dir = str(root / "missing")
            with self.assertRaisesRegex(ValueError, "full-target source audit"):
                protect.inputs(args)
            args.source_audit_dir = str(root / "audit")
            args.output_dir = args.source_audit_dir
            with self.assertRaisesRegex(ValueError, "separate"):
                protect.inputs(args)
            args.output_dir = str(root / "protect")
            protect.select(args, ctx)
            changed = list(ctx)
            changed[-1] = {**ctx[-1], "margin_fraction": .5}
            with self.assertRaisesRegex(ValueError, "code/settings/environment"):
                protect.train(args, changed)
            protect.train(args, ctx)
            head = next((Path(args.output_dir) / "runs").rglob("head.pt"))
            head.write_bytes(b"CORRUPT")
            with self.assertRaisesRegex(ValueError, "Missing/modified"):
                protect.evaluate(args, ctx)

    def test_separate_cli_stages(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "select", "train", "evaluate"):
                p = subprocess.run([sys.executable, protect.__file__, stage, *cli], text=True, capture_output=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                if stage == "evaluate":
                    self.assertIn("CARE_TARGET_PROTECTION_COMPLETE", p.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").is_file())

    def test_rehashed_invalid_queries_and_stale_anchor_trace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            protect.select(args, ctx)
            directory = Path(args.output_dir) / "selection"
            choices = json.loads((directory / "queries.json").read_text())
            choices[0]["rows"][0] = -1
            choices[0]["repair_rows"] = list(choices[0]["rows"])
            kbs.atomic_json(directory / "queries.json", choices)
            marker = json.loads((directory / "complete.json").read_text())
            marker["artifacts"]["queries.json"] = kbs.file_sha(directory / "queries.json")
            kbs.atomic_json(directory / "complete.json", marker)
            with patch.object(diag, "load_role", side_effect=AssertionError("Read truth before rejecting invalid queries")):
                with self.assertRaisesRegex(ValueError, "Invalid frozen"):
                    protect.train(args, ctx)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, ctx, _ = self.fixture(Path(tmp))
            protect.select(args, ctx)
            protect.train(args, ctx)
            directory = Path(args.output_dir) / "runs/margin/seed_0/constrained"
            trace = json.loads((directory / "trace.json").read_text())
            trace["anchor_negative_flips"] += 1
            kbs.atomic_json(directory / "trace.json", trace)
            marker = json.loads((directory / "complete.json").read_text())
            marker["artifacts"]["trace.json"] = kbs.file_sha(directory / "trace.json")
            kbs.atomic_json(directory / "complete.json", marker)
            with self.assertRaisesRegex(ValueError, "Anchor trace disagrees"):
                protect.evaluate(args, ctx)
            self.assertFalse((Path(args.output_dir) / "evaluation/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
