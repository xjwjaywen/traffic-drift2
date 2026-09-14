"""Synthetic continuation checks; no scores from these tests are paper evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_supplement as kbs
import kbs_badge_kd_followup as follow
import test_kbs_supplement as fixtures


class BadgeFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @contextlib.contextmanager
    def study(self, seeds="0,1,2,3,4"):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            head, x, y, r, ry = fixtures.SupplementTests().fixture()
            data = {"target": {"features": x, "labels": y.numpy(), "logits": head(x).detach()},
                    "reference": {"features": r, "labels": ry.numpy(), "logits": head(r).detach()},
                    "head": head, "num_classes": 3,
                    "manifest": {"fingerprint": "SYNTHETIC_TEST_ONLY", "checkpoint_sha256": "synthetic"}}
            args = kbs.parser().parse_args([
                "run", "--output-dir", str(out), "--device", "cpu", "--seeds", seeds,
                "--budget", "6", "--steps", "2", "--batch-size", "4", "--threads", "1",
                "--collapse-classes", "1", "--stable-classes", "2", "--log-every", "0"])
            def select(name, features, logits, budget, seed):
                return np.arange(budget) if name == "margin" else np.arange(budget, 2 * budget)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(kbs, "select_query", side_effect=select):
                kbs.run_study(args, data)
                yield out, args, data

    def test_only_reference_ce_differs(self):
        full, kd = follow.specs()
        self.assertEqual(full, next(s for s in kbs.make_specs("primary") if s["name"] == "badge_full"))
        self.assertEqual({k for k in full if full[k] != kd[k]}, {"name", "replay_ce"})
        self.assertFalse(kd["replay_ce"])
        self.assertEqual(kd["kd_weight"], .5)
        self.assertEqual(kd["selector"], "badge")

    def test_five_new_fits_preserve_25_runs_and_resume(self):
        with self.study() as (out, args, data):
            old = {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in out.rglob("*") if p.is_file()}
            engine_sha = kbs.implementation_sha()
            original_registry = kbs.make_specs
            with patch.object(kbs, "select_query", side_effect=AssertionError("must reuse selections")), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fit:
                follow.followup(args, data)
                self.assertEqual(fit.call_count, 5)
                self.assertTrue(all(not c.kwargs["replay_ce"] and c.kwargs["kd_weight"] == .5
                                    for c in fit.call_args_list))
                follow.followup(args, data)
                self.assertEqual(fit.call_count, 5)
            self.assertEqual(engine_sha, kbs.implementation_sha())
            self.assertIs(kbs.make_specs, original_registry)
            self.assertEqual(old, {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in old})
            self.assertEqual(len(list((out / "runs").glob("*/seed_*/complete.json"))), 30)
            rows = kbs.read_csv(out / "badge_kd_results_by_seed.csv")
            self.assertEqual(len(rows), 10)
            summaries = kbs.read_csv(out / "badge_kd_summary.csv")
            self.assertTrue(all(r["n_seeds"] == "5" and r["recommended_n_seeds"] == "5"
                                and r["meets_recommended_seed_count"] == "True" for r in summaries))
            pairs = kbs.read_csv(out / "badge_kd_paired_by_seed.csv")
            self.assertEqual(len(pairs), 5)
            self.assertTrue(all(r["pairing_verified"] == "True" for r in pairs))
            for seed in range(5):
                a, b = [next(r for r in rows if r["name"] == n and r["seed"] == str(seed))
                        for n in ["badge_full", "badge_kd"]]
                self.assertAlmostEqual(float(pairs[seed]["common_overall_macro_f1_after_difference"]),
                                       float(b["common_overall_macro_f1_after"]) -
                                       float(a["common_overall_macro_f1_after"]))

    def test_missing_baseline_stops_before_training(self):
        with self.study("0") as (out, args, data):
            args.seeds = "0,1"
            with patch.object(kbs, "fit_controlled", side_effect=AssertionError("must fail first")):
                with self.assertRaisesRegex(ValueError, "completed run is required"):
                    follow.followup(args, data)
            self.assertFalse((out / "runs" / "badge_kd").exists())

    def test_changed_protocol_or_selection_is_rejected(self):
        with self.study("0") as (out, args, data):
            changed = copy.copy(args)
            changed.steps += 1
            with self.assertRaisesRegex(ValueError, "Study protocol changed"):
                follow.followup(changed, data)
            (out / "selections" / "badge" / "seed_0" / "selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Missing/modified artifact"):
                follow.followup(args, data)

    def test_unpaired_ids_rejected_even_with_valid_completion_hash(self):
        with self.study("0") as (out, args, data):
            follow.followup(args, data)
            run = out / "runs" / "badge_kd" / "seed_0"
            rows = kbs.read_csv(run / "replay_ids.csv")
            rows.reverse()
            kbs.write_csv(run / "replay_ids.csv", rows)
            complete = json.loads((run / "complete.json").read_text())
            kbs.finish(run, complete["signature"], list(complete["artifacts"]))
            with self.assertRaisesRegex(ValueError, "Unpaired replay_ids.csv"):
                follow.summarize(out)

    def test_partial_summary_and_changed_extension(self):
        with self.study("0") as (out, args, data):
            follow.followup(args, data)
            summary = kbs.read_csv(out / "badge_kd_summary.csv")
            self.assertTrue(all(r["n_seeds"] == "1" and r["meets_recommended_seed_count"] == "False"
                                and r["common_overall_macro_f1_after_sample_sd"] == "" for r in summary))
            path = out / "badge_kd_extension_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["identity"]["followup_sha256"] = "changed"
            kbs.atomic_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "Follow-up implementation changed"):
                follow.followup(args, data)


if __name__ == "__main__":
    unittest.main()
