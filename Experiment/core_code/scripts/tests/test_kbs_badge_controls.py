"""Synthetic protocol checks only; these scores are not CESNET evidence."""
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kbs_supplement as kbs
import kbs_badge_kd_followup as follow
import kbs_badge_controls as controls
import test_kbs_badge_kd_followup as fixtures


class BadgeControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @contextlib.contextmanager
    def study(self, seeds="0"):
        with fixtures.BadgeFollowupTests().study(seeds) as (out, args, data):
            follow.followup(args, data)
            yield out, args, data

    def resign(self, run):
        done = json.loads((run / "complete.json").read_text())
        kbs.finish(run, done["signature"], list(done["artifacts"]))

    def test_factorial_changes_only_loss_switches_and_restores_registry(self):
        ft, replay, kd, full = controls.specs()
        self.assertEqual([full, kd], follow.specs())
        self.assertEqual([(s["replay_ce"], s["kd_weight"]) for s in controls.specs()],
                         [(False, 0.), (True, 0.), (False, .5), (True, .5)])
        for spec in [ft, replay, kd]:
            self.assertLessEqual({k for k in spec if spec[k] != full[k]}, {"name", "replay_ce", "kd_weight"})
        original = (kbs.make_specs, kbs.summarize)
        with self.assertRaises(RuntimeError), controls.registry([0]):
            self.assertEqual(kbs.make_specs(controls.SUITE), controls.specs())
            raise RuntimeError("test cleanup")
        self.assertEqual(original, (kbs.make_specs, kbs.summarize))

    def test_sample_sd_and_paired_difference_aggregation(self):
        row = controls.aggregate([{"common_delta_difference": -2.}, {"common_delta_difference": 4.}],
                                 {"comparison": "synthetic"})
        self.assertEqual(row["n_seeds"], 2)
        self.assertEqual(row["common_delta_difference_mean"], 1.)
        self.assertAlmostEqual(row["common_delta_difference_sample_sd"], 18. ** .5)

    def test_ten_new_fits_preserve_all_old_files_and_resume_without_cache(self):
        with self.study("0,1,2,3,4") as (out, args, data):
            old = {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in out.rglob("*") if p.is_file()}
            head = copy.deepcopy(data["head"].state_dict())
            engine = kbs.implementation_sha()
            with patch.object(kbs, "select_query", side_effect=AssertionError("must reuse")), \
                 patch.object(kbs, "fit_controlled", wraps=kbs.fit_controlled) as fit:
                controls.run(args, data)
                self.assertEqual(fit.call_count, 10)
                self.assertTrue(all(c.kwargs["kd_weight"] == 0 for c in fit.call_args_list))
                self.assertEqual(sum(c.kwargs["replay_ce"] for c in fit.call_args_list), 5)
                with patch.object(kbs, "load_cache", side_effect=AssertionError("no cache on resume")), \
                     patch.object(kbs, "prepare", side_effect=AssertionError("no raw dataset on resume")):
                    controls.run(args)
                self.assertEqual(fit.call_count, 10)
            self.assertEqual(old, {p: (p.stat().st_mtime_ns, kbs.file_sha(p)) for p in old})
            self.assertEqual(engine, kbs.implementation_sha())
            self.assertTrue(all(torch.equal(head[k], data["head"].state_dict()[k]) for k in head))
            self.assertEqual(len(list((out / "runs").glob("*/seed_*/complete.json"))), 40)
            summaries = kbs.read_csv(out / "badge_controls_summary.csv")
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(r["n_seeds"] == "5" for r in summaries))
            pairs = kbs.read_csv(out / "badge_controls_paired_by_seed.csv")
            self.assertEqual(len(pairs), 25)
            rows = kbs.read_csv(out / "badge_controls_results_by_seed.csv")
            for pair in pairs:
                a, b = pair["comparison"].split(" minus ")
                values = {r["name"]: r for r in rows if r["seed"] == pair["seed"]}
                for key in ["common_overall_macro_f1_after", "common_negative_flips", "common_noncollapse_new_collapses"]:
                    self.assertAlmostEqual(float(pair[key + "_difference"]), float(values[a][key]) - float(values[b][key]))
            classes = kbs.read_csv(out / "badge_controls_per_class.csv")
            self.assertEqual(len(classes), 5 * 4 * 2 * 3)
            status = json.loads((out / "badge_controls_status.json").read_text())
            self.assertTrue(status["all_requested_complete"])
            for name, digest in status["artifacts"].items():
                self.assertEqual(kbs.file_sha(out / name), digest)

    def test_matched_seeds_only_and_summary_without_tensors(self):
        with self.study("0,1") as (out, args, data):
            args.seeds = "0"
            controls.run(args, data)
            with patch.object(kbs.torch, "load", side_effect=AssertionError("no tensor loading")):
                controls.summarize(out, [0, 1])
            status = json.loads((out / "badge_controls_status.json").read_text())
            self.assertFalse(status["all_requested_complete"])
            self.assertEqual(status["completed_seeds"], [0])
            summaries = kbs.read_csv(out / "badge_controls_summary.csv")
            self.assertTrue(all(r["n_seeds"] == "1" for r in summaries))
            self.assertTrue(all(r["common_overall_macro_f1_after_sample_sd"] == "" for r in summaries))
            rows = kbs.read_csv(out / "badge_controls_results_by_seed.csv")
            self.assertEqual({r["seed"] for r in rows}, {"0"})
            (out / "runs/badge_replay/seed_0/complete.json").unlink()
            controls.summarize(out, [0, 1])
            self.assertTrue(all(r["n_seeds"] == "0" for r in kbs.read_csv(out / "badge_controls_summary.csv")))

    def test_missing_kd_or_cache_stops_before_fitting(self):
        with fixtures.BadgeFollowupTests().study("0") as (out, args, data):
            with patch.object(kbs, "fit_controlled", side_effect=AssertionError("must fail first")):
                with self.assertRaisesRegex(ValueError, "completed run is required"):
                    controls.run(args, data)
            self.assertFalse((out / "runs/badge_ft_only").exists())
            follow.followup(args, data)
            args.cache_dir = str(out / "nonexistent-cache")
            with patch.object(kbs, "prepare", side_effect=AssertionError("must not extract")):
                with self.assertRaisesRegex(ValueError, "existing primary feature cache"):
                    controls.run(args)

    def test_changed_protocol_identity_or_corrupt_run_rejected(self):
        with self.study() as (out, args, data):
            changed = copy.copy(args)
            changed.steps += 1
            with patch.object(kbs, "fit_controlled", side_effect=AssertionError("must fail first")):
                with self.assertRaisesRegex(ValueError, "Study protocol changed"):
                    controls.run(changed, data)
            controls.run(args, data)
            manifest = out / "badge_controls_extension_manifest.json"
            original = manifest.read_text()
            info = json.loads(original)
            info["identity"]["runner_sha256"] = "modified"
            kbs.atomic_json(manifest, info)
            with self.assertRaisesRegex(ValueError, "implementation/protocol changed"):
                controls.run(args, data)
            manifest.write_text(original)
            (out / "runs/badge_ft_only/seed_0/metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Missing/modified artifact"):
                controls.run(args, data)

    def test_rehashed_semantic_mismatches_are_not_accepted(self):
        with self.study() as (out, args, data):
            controls.run(args, data)
            run = out / "runs/badge_ft_only/seed_0"
            cases = [
                ("replay_ids.csv", "Unpaired replay_ids.csv", lambda p: kbs.write_csv(p, list(reversed(kbs.read_csv(p))))),
                ("training_trace.json", "Incorrect training reference_kd_presentations",
                 lambda p: kbs.atomic_json(p, {**json.loads(p.read_text()), "reference_kd_presentations": 8})),
                ("metrics.json", "Metrics disagree",
                 lambda p: kbs.atomic_json(p, {**json.loads(p.read_text()), "common_overall_macro_f1_after": .1234567})),
            ]
            for name, error, edit in cases:
                with self.subTest(file=name):
                    path = run / name
                    original = path.read_bytes()
                    edit(path)
                    self.resign(run)
                    with self.assertRaisesRegex(ValueError, error):
                        controls.summarize(out, [0])
                    path.write_bytes(original)
                    self.resign(run)
            path = run / "predictions.npz"
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["common_eval"][0] = ~arrays["common_eval"][0]
            np.savez_compressed(path, **arrays)
            self.resign(run)
            with self.assertRaisesRegex(ValueError, "Unpaired predictions/evaluation"):
                controls.summarize(out, [0])

    def test_global_exclusion_is_checked(self):
        with self.study() as (out, args, data):
            controls.run(args, data)
            for spec in controls.specs():
                run = controls.run_dir(out, spec, 0)
                path = run / "predictions.npz"
                with np.load(path) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                arrays["common_eval"][0] = True
                np.savez_compressed(path, **arrays)
                self.resign(run)
            with self.assertRaisesRegex(ValueError, "Incorrect exclusion common_eval"):
                controls.summarize(out, [0])

    def test_cli_plan_and_summary_in_separate_process(self):
        script = Path(controls.__file__)
        plan = subprocess.run([sys.executable, str(script), "plan"], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(plan.stdout)["maximum_new_fits"], 10)
        bad = subprocess.run([sys.executable, str(script), "plan", "--seeds", "0,0"], capture_output=True)
        self.assertEqual(bad.returncode, 2)
        with self.study() as (out, args, data):
            controls.run(args, data)
            result = subprocess.run([sys.executable, str(script), "summarize", "--output-dir", str(out),
                                     "--seeds", "0", "--cache-dir", str(out / "absent")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1/1 seeds x 4", result.stdout)


if __name__ == "__main__":
    unittest.main()
