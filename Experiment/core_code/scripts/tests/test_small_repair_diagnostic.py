"""Synthetic correctness tests only: these scores are not research evidence."""
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
import small_repair_diagnostic as diag
import kbs_supplement as kbs


class GuardedLabels:
    def __init__(self, labels, allowed):
        self.labels, self.allowed = labels, set(map(int, allowed))

    def __getitem__(self, rows):
        if not set(map(int, rows)).issubset(self.allowed):
            raise AssertionError("Unqueried labels accessed")
        return self.labels[rows]

    def __array__(self, *args, **kwargs):
        raise AssertionError("Full label conversion")


class SmallRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        gen = torch.Generator().manual_seed(71)
        head = torch.nn.Linear(4, 3)
        with torch.no_grad():
            head.weight.copy_(torch.randn(3, 4, generator=gen))
            head.bias.zero_()
        cache = root / "cache"
        cache.mkdir()
        checkpoint = root / "source.pt"
        checkpoint.write_bytes(b"SYNTHETIC SOURCE CHECKPOINT, NOT PAPER DATA")
        torch.save(head.state_dict(), cache / "head.pt")
        counts = {"target": 90, "reference": 36}
        for role, n in counts.items():
            x = torch.randn(n, 4, generator=gen)
            torch.save({"features": x, "labels": torch.arange(n) % 3,
                        "logits": head(x).detach()}, cache / f"{role}.pt")
        info = {"spec": {"schema": kbs.SCHEMA, "loader_sha256": kbs.loader_sha()},
                "periods": {"reference": "M-2022-4", "target": "M-2022-12"},
                "num_classes": 3, "fingerprint": "SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE",
                "sample_counts": counts, "checkpoint_sha256": kbs.file_sha(checkpoint),
                "files": {n: kbs.file_sha(cache / n) for n in ("head.pt", "reference.pt", "target.pt")}}
        kbs.atomic_json(cache / "manifest.json", info)
        cli = ["--cache-dir", str(cache), "--checkpoint", str(checkpoint),
               "--output-dir", str(root / "out"), "--seeds", "0,1", "--threads", "1",
               "--expected-classes", "3", "--expected-features", "4", "--collapse-classes", "1",
               "--stable-classes", "2", "--budget", "6", "--steps", "3", "--batch-size", "4",
               "--replay-per-class", "2", "--hidden", "5", "--eval-size", "30", "--log-every", "0"]
        args = diag.parser().parse_args(["select", *cli])
        args.seeds, args.selectors = [0, 1], ["random", "margin"]
        _, identity = diag.inputs(args)
        return args, info, identity, cli

    def test_selection_is_label_free_ties_stable_and_eval_excludes_all_queries(self):
        logits = torch.zeros(30, 3)
        margin = diag.select_indices("margin", logits, 5, 0)
        np.testing.assert_array_equal(margin, np.arange(5))
        np.testing.assert_array_equal(margin, diag.select_indices("margin", logits, 5, 99))
        random = diag.select_indices("random", logits, 5, 0)
        np.testing.assert_array_equal(random, diag.select_indices("random", logits, 5, 0))
        choices = [{"rows": margin.tolist()}, {"rows": random.tolist()}, {"rows": [25, 26]}]
        rows = diag.evaluation_indices(30, choices, 10, 1729)
        self.assertFalse(set(rows) & set(np.concatenate([margin, random, [25, 26]])))
        np.testing.assert_array_equal(rows, diag.evaluation_indices(30, choices, 10, 1729))
        self.assertEqual(len(diag.evaluation_indices(30, choices, 0, 1729)),
                         30 - len(set(np.concatenate([margin, random, [25, 26]]))))

    def test_residual_initial_predictions_exact_and_source_unchanged(self):
        torch.manual_seed(7)
        source, x = torch.nn.Linear(4, 3), torch.randn(12, 4)
        original = copy.deepcopy(source.state_dict())
        a, b = diag.ResidualHead(source, 5, 0), diag.ResidualHead(source, 5, 0)
        torch.testing.assert_close(a(x), source(x), rtol=0, atol=0)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            a.base.weight.add_(1)
        for key, value in original.items():
            torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)

    def test_guarded_query_labels_shared_streams_and_protection_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, info, identity, _ = self.fixture(Path(tmp))
            source = diag.head_from_cache(args)
            target, ref = diag.load_role(args, info, "target"), diag.load_role(args, info, "reference")
            idx, ridx = np.arange(6), np.arange(6)
            target["labels"] = GuardedLabels(target["labels"], idx)
            ref["labels"] = GuardedLabels(ref["labels"], ridx)
            before = copy.deepcopy(source.state_dict())
            traces = []
            for engine in diag.ENGINES:
                fitted, trace = diag.fit_one(args, identity, source, target, ref,
                    {"rows": idx.tolist(), "seed": 0}, ridx, engine)
                self.assertTrue(all(torch.isfinite(p).all() for p in fitted.parameters()))
                traces.append(trace)
                self.assertEqual(trace["reference_ce_presentations"], 0 if engine == "linear_ce" else 12)
                self.assertEqual(trace["reference_kd_presentations"], 0 if engine == "linear_ce" else 12)
            self.assertEqual(len({t["target_stream_sha256"] for t in traces}), 1)
            self.assertEqual(len({t["reference_stream_sha256"] for t in traces}), 1)
            self.assertGreater(traces[2]["trainable_parameters"], traces[1]["trainable_parameters"])
            for key, value in before.items():
                torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)

    def test_pipeline_input_integrity_resume_and_independent_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args, info, identity, _ = self.fixture(root)
            original = {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in (root / "cache").iterdir()}
            diag.select(args, info, identity)
            with patch.object(diag, "load_role", side_effect=AssertionError("Loaded labels too early")):
                with self.assertRaisesRegex(ValueError, "Fit incomplete"):
                    diag.evaluate(args, info, identity)
            diag.train(args, info, identity)
            diag.evaluate(args, info, identity)
            out = root / "out"
            rows = kbs.read_csv(out / "evaluation/by_seed.csv")
            self.assertEqual(len(rows), 12)
            with np.load(out / "evaluation/predictions.npz") as p:
                y, old, ids = p["labels"], p["source"], p["row_indices"]
                for row in rows:
                    new = p[f"{row['selector']}_seed{row['seed']}_{row['engine']}"]
                    # Independent confusion/metric and flip arithmetic.
                    cm = np.zeros((3, 3), dtype=np.int64)
                    for truth, pred in zip(y, new):
                        cm[truth, pred] += 1
                    support, predicted = cm.sum(1), cm.sum(0)
                    f1 = np.divide(2 * np.diag(cm), support + predicted,
                                   out=np.zeros(3, dtype=float), where=(support + predicted) != 0)
                    self.assertAlmostEqual(float(row["overall_macro_f1_after"]), f1.mean())
                    correct = old == y
                    self.assertEqual(int(row["negative_flips"]), int((correct & (new != y)).sum()))
                    self.assertAlmostEqual(float(row["negative_flip_rate"]), (correct & (new != y)).sum() / correct.sum())
                choices = json.loads((out / "selection/queries.json").read_text())
                self.assertFalse(set(ids) & {i for c in choices for i in c["rows"]})
            self.assertEqual(len(kbs.read_csv(out / "evaluation/per_class.csv")), 36)
            stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
            with patch.object(torch, "load", side_effect=AssertionError("Resume loaded tensors")), \
                 patch.object(kbs, "fit_controlled", side_effect=AssertionError("Resume fitted")):
                diag.select(args, info, identity)
                diag.train(args, info, identity)
                diag.evaluate(args, info, identity)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})
            self.assertEqual(original, {p: (kbs.file_sha(p), p.stat().st_mtime_ns) for p in original})
            changed = {**identity, "steps": 4}
            with self.assertRaisesRegex(ValueError, "settings/code/environment"):
                diag.train(args, info, changed)
            damaged = next((out / "runs").rglob("head.pt"))
            damaged.write_bytes(b"CORRUPTED")
            with self.assertRaisesRegex(ValueError, "Missing/modified"):
                diag.evaluate(args, info, identity)

    def test_missing_cache_hash_checkpoint_periods_and_output_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, info, _, _ = self.fixture(Path(tmp))
            args.target_period = "M-2022-11"
            with self.assertRaisesRegex(ValueError, "periods"):
                diag.inputs(args)
            args.target_period = "M-2022-12"
            args.output_dir = args.cache_dir
            with self.assertRaisesRegex(ValueError, "separate"):
                diag.inputs(args)
            args.output_dir = str(Path(tmp) / "out")
            Path(args.checkpoint).write_bytes(b"CHANGED CHECKPOINT")
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                diag.inputs(args)
            (Path(args.cache_dir) / "target.pt").write_bytes(b"CHANGED CACHE")
            with self.assertRaisesRegex(ValueError, "Missing/modified"):
                diag.inputs(args)

    def test_server_local_engine_edit_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, _, _ = self.fixture(Path(tmp))
            real_sha = kbs.file_sha
            def changed_engine(path):
                return "different" if Path(path) == Path(kbs.__file__) else real_sha(path)
            with patch.object(kbs, "file_sha", side_effect=changed_engine):
                with self.assertRaisesRegex(ValueError, "differs from the tested engine"):
                    diag.inputs(args)

    def test_separate_cli_processes_and_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, _, cli = self.fixture(Path(tmp))
            for stage in ("preflight", "select", "train", "evaluate"):
                proc = subprocess.run([sys.executable, diag.__file__, stage, *cli], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                if stage == "preflight":
                    self.assertIn('"status": "ready"', proc.stdout)
                if stage == "evaluate":
                    self.assertIn("SMALL_REPAIR_COMPLETE", proc.stdout)
            self.assertTrue((Path(args.output_dir) / "evaluation/complete.json").is_file())

    def test_preflight_rejects_insufficient_reference_and_unknown_output(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, info, identity, cli = self.fixture(Path(tmp))
            out = Path(args.output_dir)
            out.mkdir()
            (out / "other-results.txt").write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "Unregistered files"):
                diag.select(args, info, identity)
            self.assertEqual((out / "other-results.txt").read_text(), "preserve me")
            args.output_dir = str(Path(tmp) / "new-out")
            cli[cli.index("--output-dir") + 1] = args.output_dir
            ref_path = Path(args.cache_dir) / "reference.pt"
            reference = torch.load(ref_path, weights_only=True)
            reference["labels"].zero_()
            torch.save(reference, ref_path)
            info["files"]["reference.pt"] = kbs.file_sha(ref_path)
            kbs.atomic_json(Path(args.cache_dir) / "manifest.json", info)
            proc = subprocess.run([sys.executable, diag.__file__, "preflight", *cli], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("Insufficient historical reference", proc.stderr)
            self.assertNotIn('"status": "ready"', proc.stdout)


if __name__ == "__main__":
    unittest.main()
