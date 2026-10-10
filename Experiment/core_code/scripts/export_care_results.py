#!/usr/bin/env python3
"""Export completed CARE text results; publish in an isolated Git worktree.

Standard library only. No training, tensors, raw data or cache access.
Completion hashes attest file integrity, not correctness of research conclusions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
import zipfile

CORE = Path(__file__).resolve().parents[1]
DEST = Path("Experiment/results/care_development")
SCHEMA = "care-result-export-v1"
# Name -> protocol schema, root manifest, final stage.
STUDIES = {
    "small_repair_diagnostic_v1": ("small-repair-diagnostic-v1", "diagnostic_manifest.json", "evaluation"),
    "small_repair_full_audit_v1": ("small-repair-full-audit-v1", None, ""),
    "care_target_protection_v1": ("care-target-protection-v1", "manifest.json", "evaluation"),
    "care_mixed_query_ablation_v1": ("care-mixed-query-ablation-v1", "manifest.json", "evaluation"),
    "care_query_seed_robustness_v1": ("care-query-seed-robustness-v1", "manifest.json", "evaluation"),
    "care_m11_replication_v1": ("care-m11-replication-v1", "manifest.json", "evaluation"),
    "care_m11_mechanism_diagnostic_v1": ("care-m11-mechanism-diagnostic-v1", "manifest.json", "report"),
}
TEXT = {".json", ".csv", ".md"}
OMIT = {"geometry/probe_rows.csv"}  # Row-level probes can exceed the Git text budget.
MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def inside(root, relative):
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe relative artifact: {relative}")
    p = root.joinpath(*rel.parts)
    for item in (p, *p.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise ValueError(f"Symlink in result artifact: {item}")
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes study: {relative}")
    return p


def collect_study(root, schema, manifest_name, final_stage):
    """Follow the final stage's local completion references, never external caches."""
    files, checked, visited = {}, {}, set()

    def record(p, expected=None):
        if not p.is_file():
            raise ValueError(f"Missing completed artifact: {p}")
        actual = sha(p)
        if expected is not None and actual != expected:
            raise ValueError(f"Hash disagreement: {p}")
        rel = p.relative_to(root).as_posix()
        if rel in checked and checked[rel]["sha256"] != actual:
            raise ValueError(f"Artifact changed while exporting: {p}")
        exported = p.suffix in TEXT and rel not in OMIT
        checked[rel] = {"sha256": actual, "bytes": p.stat().st_size, "exported": exported}
        if exported:
            if p.stat().st_size > MAX_FILE:
                raise ValueError(f"Text artifact exceeds 25 MiB: {p}")
            # Reject binary masquerading as text, without changing original bytes.
            p.read_text(encoding="utf-8")
            files[rel] = p
        return actual

    protocol = None
    if manifest_name:
        manifest = inside(root, manifest_name)
        record(manifest)
        protocol = read(manifest)["protocol"]
        if protocol.get("schema") != schema:
            raise ValueError(f"Unexpected study schema: {root}")

    def visit(stage, expected=None):
        marker = inside(root, str(Path(stage) / "complete.json"))
        record(marker, expected)
        if stage in visited:
            return
        visited.add(stage)
        info = read(marker)
        signature, artifacts = info["signature"], info["artifacts"]
        if protocol is not None and signature.get("protocol") != protocol:
            raise ValueError(f"Stage/root protocol disagreement: {marker}")
        if not artifacts or "resolved_config.json" not in artifacts:
            raise ValueError(f"Invalid completion marker: {marker}")
        for name, checksum in sorted(artifacts.items()):
            p = inside(root, str(Path(stage) / name))
            record(p, checksum)
        if read(inside(root, str(Path(stage) / "resolved_config.json"))) != signature:
            raise ValueError(f"Resolved signature disagreement: {marker}")
        # References belong to this study. Parent-study hashes remain provenance
        # and are not followed to arbitrary paths or raw/cache directories.
        for name, key in (("selection", "selection_sha256"), ("audit", "audit_sha256"),
                          ("geometry", "geometry_sha256")):
            if key in signature:
                visit(name, signature[key])
        for run, checksum in sorted(signature.get("runs", {}).items()):
            if not str(run).startswith("runs/"):
                raise ValueError(f"Unexpected local run reference: {run}")
            visit(run, checksum)

    visit(final_stage)
    final = read(inside(root, str(Path(final_stage) / "complete.json")))
    if "report.md" not in final["artifacts"]:
        raise ValueError(f"Completed report absent: {root}")
    if protocol is None and final["signature"].get("schema") != schema:
        raise ValueError(f"Unexpected standalone study schema: {root}")
    return files, checked


def git(repo, *args, network=False, capture=True):
    env = os.environ.copy()
    command = ["git"]
    if network:
        for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(name, None)
        env["GIT_TERMINAL_PROMPT"] = "0"
        command += ["-c", "http.proxy=", "-c", "http.https://github.com.proxy=",
                    "-c", "remote.origin.proxy="]
    result = subprocess.run([*command, "-C", str(repo), *args], env=env, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def export(args):
    repo = Path(args.repo_dir).resolve()
    outputs = Path(args.outputs_dir).absolute()
    selected = list(STUDIES) if args.studies == "auto" else args.studies.split(",")
    if not selected or len(selected) != len(set(selected)) or any(s not in STUDIES for s in selected):
        raise ValueError("--studies must be auto or distinct known study names")
    all_files, inventory, skipped = {}, {}, []
    for name in selected:
        schema, manifest, final = STUDIES[name]
        root = outputs / name
        if not (root / final / "complete.json").is_file():
            if args.studies != "auto":
                raise ValueError(f"Requested study incomplete/missing: {name}")
            skipped.append(name)
            print(f"Skip unfinished/missing: {name}", flush=True)
            continue
        print(f"Verify completed artifacts: {name} (disk I/O only)", flush=True)
        files, checks = collect_study(root, schema, manifest, final)
        all_files.update({f"{name}/{rel}": p for rel, p in files.items()})
        inventory[name] = {"schema": schema, "source_dir": str(root), "artifacts": checks}
    if not inventory:
        raise ValueError("No completed CARE studies to export")
    total = sum(p.stat().st_size for p in all_files.values())
    if total > MAX_TOTAL:
        raise ValueError("Text export exceeds 100 MiB; use --studies to select fewer studies")
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    snapshot = outputs / "care_result_exports" / tag
    bundle = snapshot / "bundle" / DEST
    bundle.mkdir(parents=True)
    for rel, source in sorted(all_files.items()):
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        study, relative = rel.split("/", 1)
        expected = inventory[study]["artifacts"][relative]["sha256"]
        if sha(source) != expected or sha(target) != expected:
            raise ValueError(f"Source changed while copying: {source}")
    metadata = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": git(repo, "rev-parse", "HEAD"),
        "source_git_branch": git(repo, "branch", "--show-current"),
        "exporter_sha256": sha(__file__), "exported_bytes": total,
        "exported_files": len(all_files), "studies": inventory, "skipped": skipped,
        "verification": "completed local stage artifact SHA256 and resolved signatures; no semantic recomputation",
        "excluded": "raw data, features, checkpoints, prediction arrays, row-level geometry/probe_rows.csv and all non-text artifacts"}
    write(bundle / "export_manifest.json", metadata)
    (bundle / "README.md").write_text(
        "# Server CARE development results\n\n"
        "Original UTF-8 reports, CSVs and JSON protocols copied without numerical changes.\n"
        "See export_manifest.json for source paths, byte hashes, completed studies and exclusions.\n"
        "Binary artifacts and row-level geometry/probe_rows.csv appear in records only as provenance.\n"
        "This text subset cannot resume experiments; full arrays/heads/caches remain on the server.\n"
        "Completion hashes establish integrity; development results are not novelty or causal evidence.\n")
    archive = snapshot / f"care-results-{tag}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(snapshot / "bundle"))
    print(f"CARE_RESULTS_EXPORTED: {snapshot}", flush=True)
    print(f"ZIP: {archive}", flush=True)
    print(f"Text files: {len(all_files)}, bytes: {total}; binary artifacts stay on server", flush=True)
    return snapshot, tag


def publish(args, snapshot, tag):
    repo = Path(args.repo_dir).resolve()
    # Fetch and create a fresh branch; never change the server checkout's branch,
    # index or modified code. Do not reset, clean, force-push or push main.
    git(repo, "fetch", "origin", args.code_branch, network=True, capture=False)
    base = git(repo, "rev-parse", "FETCH_HEAD")
    branch = f"codex/server-results-{tag}"
    worktree = snapshot / "worktree"
    git(repo, "worktree", "add", "--quiet", "-b", branch, str(worktree), base, capture=False)
    write(snapshot / "publish_state.json", {"branch": branch, "worktree": str(worktree), "base": base})
    print(f"Upload worktree: {worktree}\nResults branch: {branch}", flush=True)
    dest = worktree / DEST
    if dest.exists():
        raise ValueError(f"Code branch already contains {DEST}; refusing to overwrite it")
    shutil.copytree(snapshot / "bundle" / DEST, dest)
    git(worktree, "add", "--", DEST.as_posix())
    git(worktree, "commit", "--quiet", "-m", "Record completed server CARE development results", capture=False)
    commit = git(worktree, "rev-parse", "HEAD")
    write(snapshot / "publish_state.json", {"branch": branch, "worktree": str(worktree), "base": base, "commit": commit})
    print(f"Results commit: {commit}\nResults branch: {branch}", flush=True)
    try:
        git(worktree, "push", "origin", f"{commit}:refs/heads/{branch}", network=True, capture=False)
        remote = git(worktree, "ls-remote", "origin", f"refs/heads/{branch}", network=True)
        if not remote or remote.split()[0] != commit:
            raise ValueError("Remote result commit was not confirmed")
    except (subprocess.CalledProcessError, ValueError):
        print(f"Upload unconfirmed; preserved results commit and worktree: {worktree}", file=sys.stderr)
        print("After configuring your own Git authentication, retry push from this worktree; "
              "or transfer the ZIP to your Mac. Do not send tokens/passwords in chat.", file=sys.stderr)
        raise
    print(f"CARE_RESULTS_PUSHED: {branch} {commit}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("export", "publish"))
    p.add_argument("--repo-dir", default=str(CORE.parents[1]))
    p.add_argument("--outputs-dir", default=str(CORE / "outputs"))
    p.add_argument("--studies", default="auto", help="auto skips unfinished studies; or comma-separated known names")
    p.add_argument("--code-branch", default="codex/small-repair-diagnosis")
    return p


def main():
    args = parser().parse_args()
    if args.code_branch != "codex/small-repair-diagnosis":
        raise ValueError("Use the designated code branch as the isolated results base")
    snapshot, tag = export(args)
    if args.command == "publish":
        try:
            publish(args, snapshot, tag)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"CARE_RESULTS_UPLOAD_NOT_CONFIRMED: {snapshot}\n{e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as e:
        print(f"CARE_RESULTS_EXPORT_FAILED: {e}", file=sys.stderr)
        sys.exit(1)
