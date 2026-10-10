"""Integrity/export and isolated Git publication checks with synthetic local remotes."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_care_results as ex

NAME = "care_m11_replication_v1"
SCHEMA = ex.STUDIES[NAME][0]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


class ExportTests(unittest.TestCase):
    def fixture(self, tmp):
        repo, remote = tmp / "dirty server repo", tmp / "remote.git"
        repo.mkdir()
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.name", "Synthetic Test")
        git(repo, "config", "user.email", "synthetic@example.invalid")
        (repo / "code.py").write_text("original\n")
        (repo / "other.py").write_text("original\n")
        (repo / ".gitignore").write_text("Experiment/core_code/outputs/\n")
        git(repo, "add", ".")
        git(repo, "commit", "-m", "Synthetic source")
        git(tmp, "init", "--bare", str(remote))
        git(repo, "remote", "add", "origin", str(remote))
        git(repo, "push", "origin", "HEAD:refs/heads/codex/small-repair-diagnosis")
        outputs = repo / "Experiment/core_code/outputs"
        root = outputs / NAME
        protocol = {"schema": SCHEMA, "SYNTHETIC": True}
        ex.write(root / "manifest.json", {"protocol": protocol})
        selection = {"protocol": protocol, "stage": "select"}
        self.stage(root / "selection", selection, {"queries.json": "[]\n", "evaluation_rows.npy": "SYNTHETIC BINARY\n"})
        runpath = "runs/query_seed_1/mixed_coverage/seed_0/care"
        rs = {"protocol": protocol, "selection_sha256": ex.sha(root / "selection/complete.json")}
        self.stage(root / runpath, rs, {"head.pt": "SYNTHETIC HEAD\n", "trace.json": '{"steps": 1}\n'})
        es = {"protocol": protocol, "stage": "evaluate", "selection_sha256": rs["selection_sha256"],
              "runs": {runpath: ex.sha(root / runpath / "complete.json")}}
        self.stage(root / "evaluation", es, {"report.md": "# 原始报告\n\n未修改。\n", "per_class.csv": "class_id,f1\n1,0.5\n",
                    "predictions.npz": "SYNTHETIC PREDICTIONS\n"})
        (root / "unlisted_secrets.txt").write_text("must never export")
        args = ex.parser().parse_args(["publish", "--repo-dir", str(repo), "--outputs-dir", str(outputs)])
        return args, repo, root

    def stage(self, directory, signature, contents):
        ex.write(directory / "resolved_config.json", signature)
        for name, value in contents.items():
            (directory / name).write_text(value)
        artifacts = {n: ex.sha(directory / n) for n in ("resolved_config.json", *contents)}
        ex.write(directory / "complete.json", {"signature": signature, "artifacts": artifacts})

    def snapshot(self, root):
        return {p: (ex.sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def test_exact_subset_hashes_zip_missing_skips_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, root = self.fixture(Path(tmp))
            before = self.snapshot(root)
            snapshot, _ = ex.export(args)
            bundle = snapshot / "bundle" / ex.DEST
            actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
            self.assertIn(f"{NAME}/evaluation/report.md", actual)
            self.assertTrue(all(Path(p).suffix in ex.TEXT for p in actual))
            self.assertFalse(any("unlisted" in p for p in actual))
            self.assertEqual(before, self.snapshot(root))
            self.assertEqual((root / "evaluation/report.md").read_bytes(), (bundle / NAME / "evaluation/report.md").read_bytes())
            info = ex.read(bundle / "export_manifest.json")
            self.assertEqual(info["studies"][NAME]["artifacts"]["evaluation/predictions.npz"]["exported"], False)
            self.assertIn("care_m11_mechanism_diagnostic_v1", info["skipped"])
            with zipfile.ZipFile(next(snapshot.glob("*.zip"))) as z:
                self.assertEqual(set(z.namelist()), {str(ex.DEST / p) for p in actual})

    def test_corrupt_binary_missing_run_protocol_and_escaping_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, root = self.fixture(Path(tmp))
            binary = root / "evaluation/predictions.npz"
            old = binary.read_bytes()
            binary.write_bytes(b"CORRUPT")
            with self.assertRaisesRegex(ValueError, "Hash disagreement"):
                ex.export(args)
            binary.write_bytes(old)
            run = root / "runs/query_seed_1/mixed_coverage/seed_0/care/complete.json"
            run.unlink()
            with self.assertRaisesRegex(ValueError, "Missing completed artifact"):
                ex.export(args)
            for path in ("../raw.h5", "/absolute/file", "evaluation/../../other"):
                with self.assertRaises(ValueError):
                    ex.inside(root, path)

    def test_symlink_limits_no_finished_studies_and_explicit_missing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, _, root = self.fixture(Path(tmp))
            with patch.object(ex, "MAX_FILE", 1):
                with self.assertRaisesRegex(ValueError, "exceeds 25 MiB"):
                    ex.export(args)
            with patch.object(ex, "MAX_TOTAL", 1):
                with self.assertRaisesRegex(ValueError, "exceeds 100 MiB"):
                    ex.export(args)
            path = root / "evaluation/report.md"
            content = path.read_text()
            path.unlink()
            elsewhere = Path(tmp) / "report.md"
            elsewhere.write_text(content)
            path.symlink_to(elsewhere)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                ex.export(args)
            args.studies = "care_m11_mechanism_diagnostic_v1"
            with self.assertRaisesRegex(ValueError, "incomplete/missing"):
                ex.export(args)
            args.studies = "auto"
            (root / "evaluation/complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "No completed"):
                ex.export(args)

    def test_push_isolated_branch_preserves_dirty_checkout_index_and_main(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, repo, _ = self.fixture(Path(tmp))
            (repo / "code.py").write_text("staged change\n")
            git(repo, "add", "code.py")
            (repo / "other.py").write_text("unstaged change\n")
            state = (git(repo, "status", "--porcelain"), git(repo, "diff", "--cached"),
                     git(repo, "diff"), git(repo, "rev-parse", "HEAD"))
            snapshot, tag = ex.export(args)
            ex.publish(args, snapshot, tag)
            self.assertEqual(state, (git(repo, "status", "--porcelain"), git(repo, "diff", "--cached"),
                                    git(repo, "diff"), git(repo, "rev-parse", "HEAD")))
            self.assertEqual(git(repo, "branch", "--show-current"), "main")
            saved = ex.read(snapshot / "publish_state.json")
            self.assertEqual(ex.git(repo, "ls-remote", "origin", f"refs/heads/{saved['branch']}", network=True).split()[0], saved["commit"])
            changed = git(Path(saved["worktree"]), "diff", "--name-only", saved["base"], saved["commit"]).splitlines()
            self.assertTrue(changed and all(p.startswith(str(ex.DEST) + "/") for p in changed))
            self.assertEqual(git(repo, "ls-remote", "origin", "refs/heads/main"), "")

    def test_push_failure_preserves_zip_worktree_and_commit(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            args, _, _ = self.fixture(Path(tmp))
            snapshot, tag = ex.export(args)
            real = ex.git
            def fail_push(repo, *argv, **kw):
                if argv[0] == "push":
                    raise subprocess.CalledProcessError(128, ["git", "push"])
                return real(repo, *argv, **kw)
            with patch.object(ex, "git", side_effect=fail_push):
                with self.assertRaises(subprocess.CalledProcessError):
                    ex.publish(args, snapshot, tag)
            saved = ex.read(snapshot / "publish_state.json")
            self.assertEqual(git(Path(saved["worktree"]), "rev-parse", "HEAD"), saved["commit"])
            self.assertTrue(next(snapshot.glob("*.zip")).is_file())


if __name__ == "__main__":
    unittest.main()
