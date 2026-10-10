#!/usr/bin/env python3
"""Locate CARE inputs using only the standard library; never load model tensors."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ENTRYPOINTS = {
    "collapse_active_maintenance_tls22.py", "kbs_supplement.py",
    "kbs_repair_aware.py", "run_kbs_supplement.sh", "run_kbs_repair_aware.sh",
    "train.py",
}
CONFIGS = {"train_tls22_cnn.yaml", "eval_tls22.yaml"}
CACHE_FILES = {"target.pt", "reference.pt", "head.pt"}
PRUNE = {".git", "__pycache__", "node_modules", "venv", "envs",
         "miniconda3", "anaconda3", "miniforge3", ".cache", ".Trash"}
PACKAGES = ["torch", "numpy", "scipy", "scikit-learn", "PyYAML",
            "cesnet-datazoo", "torchcde", "cvxpy", "osqp"]


def describe(path):
    path = Path(path)
    try:
        stat = path.stat()
        result = {"path": str(path), "bytes": stat.st_size,
                  "GiB": round(stat.st_size / 1024 ** 3, 4)}
        if path.is_symlink():
            result["symlink_target"] = os.path.realpath(path)
        return result
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}


def expected_care_inputs(root):
    candidates = [root / "traffic-drift2/Experiment/core_code", root / "Experiment/core_code", root]
    core = next((path for path in candidates if (path / "configs/eval_tls22.yaml").is_file()
                 or (path / "scripts/kbs_supplement.py").is_file()), None)
    if core is None:
        return {"core": None, "files": {}, "all_present": False}
    relative_paths = {
        "TLS dataset": "data/tls22/S/CESNET-TLS-Year22-S.h5",
        "Source checkpoint": "outputs/tls22_cnn/best_model.pt",
        "Target features": "outputs/kbs_supplement_v1/cache/target.pt",
        "Reference features": "outputs/kbs_supplement_v1/cache/reference.pt",
        "Source head": "outputs/kbs_supplement_v1/cache/head.pt",
        "Cache manifest": "outputs/kbs_supplement_v1/cache/manifest.json",
    }
    files = {}
    for label, relative in relative_paths.items():
        path = core / relative
        files[label] = {**describe(path), "present": path.is_file()}
    return {"core": str(core), "files": files,
            "all_present": all(item["present"] for item in files.values())}


def print_expected(inputs):
    print("\nDEFAULT CARE INPUTS (presence only; hashes/content NOT validated)")
    if inputs["core"] is None:
        print("  CARE core directory not identified from this root.")
        return
    for label, item in inputs["files"].items():
        status = "PRESENT" if item["present"] else "MISSING"
        print(" ", status, label + ":", item["path"], "->", item.get("GiB", "?"), "GiB")
    print("  All six default inputs present:", inputs["all_present"])


def environment(root):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(root)
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"), "packages": packages,
            "cpu_count": os.cpu_count(), "disk_free_GiB": round(disk.free / 1024 ** 3, 2)}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        memory = {}
        for line in meminfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                memory[key + "_GiB"] = round(int(value.split()[0]) / 1024 ** 2, 2)
        info["memory"] = memory
    if shutil.which("nvidia-smi"):
        try:
            query = subprocess.run([
                "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
            info["gpu_status"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            info["gpu_status"] = str(exc)
    return info


def scan(root, max_depth, max_entries, seconds):
    started = time.monotonic()
    result = {"scripts": [], "configs": [], "datasets": [], "checkpoints": [],
              "cache_groups": [], "symlinks": [], "errors": [],
              "entries_visited": 0, "truncated": False}
    groups = {}
    counts = {name: 0 for name in ["scripts", "configs", "datasets", "checkpoints", "symlinks"]}

    def add(category, value):
        counts[category] += 1
        if len(result[category]) < 250:
            result[category].append(value)

    def failed(exc):
        if len(result["errors"]) < 10:
            result["errors"].append(str(exc))

    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=failed):
        base = Path(directory)
        depth = len(base.relative_to(root).parts)
        dirs.sort()
        files.sort()
        result["entries_visited"] += len(dirs) + len(files)
        if result["entries_visited"] > max_entries or time.monotonic() - started > seconds:
            result["truncated"] = True
            break
        for name in dirs:
            path = base / name
            if path.is_symlink():
                add("symlinks", {"path": str(path), "target": os.path.realpath(path),
                                 "target_exists": path.exists()})
        dirs[:] = [name for name in dirs if name not in PRUNE and not name.startswith(".venv")
                   and not (base / name).is_symlink() and depth < max_depth]
        for name in files:
            path = base / name
            if name in ENTRYPOINTS:
                add("scripts", describe(path))
            elif name in CONFIGS:
                item = describe(path)
                try:
                    for line in path.read_text().splitlines():
                        if line.strip().startswith("data_dir:"):
                            item["configured_data_dir"] = line.split(":", 1)[1].strip()
                except (OSError, UnicodeError) as exc:
                    item["config_read_error"] = str(exc)
                add("configs", item)
            elif name.endswith((".h5", ".hdf5", ".parquet", ".csv.xz")):
                add("datasets", describe(path))
            elif name in CACHE_FILES:
                groups.setdefault(str(base), {})[name] = describe(path)
            elif name == "best_model.pt" or name.endswith((".pth", ".ckpt")):
                add("checkpoints", describe(path))
    for directory, files in sorted(groups.items()):
        manifest = Path(directory) / "manifest.json"
        group = {"directory": directory, "files": files,
                 "manifest_present": manifest.is_file(),
                 "all_expected_files_present": all(name in files and "error" not in files[name]
                                                   for name in CACHE_FILES) and manifest.is_file()}
        if manifest.is_file():
            try:
                if manifest.stat().st_size <= 1024 ** 2:
                    data = json.loads(manifest.read_text())
                    if isinstance(data, dict):
                        group["manifest_metadata"] = {key: data[key] for key in
                            ["schema", "num_classes", "periods", "checkpoint_sha256"] if key in data}
            except (OSError, UnicodeError, ValueError) as exc:
                group["manifest_read_error"] = str(exc)
        result["cache_groups"].append(group)
    result["counts"] = counts
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["max_depth"] = max_depth
    return result


def print_report(report):
    env, scan_result = report["environment"], report["scan"]
    print("CARE SERVER INVENTORY (paths and metadata; cache compatibility NOT validated)")
    print("Root:", report["root"])
    print("Python:", env["executable"], "version", env["python"])
    print("Conda:", env["conda_env"], "prefix:", env["conda_prefix"])
    print("Packages:", ", ".join(name + "=" + (version or "MISSING") for name, version in env["packages"].items()))
    print("CPU cores:", env["cpu_count"], "disk free GiB:", env["disk_free_GiB"], "memory:", env.get("memory", "unavailable"))
    if scan_result is None:
        print_expected(report["expected_care_inputs"])
        return
    print("Top-level folders:", ", ".join(report["top_level_folders"]))
    for category in ["scripts", "configs", "datasets", "checkpoints", "symlinks"]:
        items = scan_result[category]
        print("\n" + category.upper() + " (found " + str(scan_result["counts"][category]) + "; showing up to 12)")
        for item in items[:12]:
            detail = item.get("target") or item.get("configured_data_dir") or str(item.get("GiB", "?")) + " GiB"
            print(" ", item["path"], "->", detail)
        if not items:
            print("  No matches within scan limits.")
    candidates = [group for group in scan_result["cache_groups"]
                  if "target.pt" in group["files"] or "reference.pt" in group["files"]
                  or "cache" in Path(group["directory"]).parts]
    candidates.sort(key=lambda group: (not group["all_expected_files_present"], group["directory"]))
    print("\nFEATURE CACHE GROUPS (found " + str(len(candidates)) + "; showing up to 12)")
    for group in candidates[:12]:
        present = ",".join(sorted(group["files"]))
        print(" ", group["directory"], "files:", present, "manifest:", group["manifest_present"],
              "all expected files present:", group["all_expected_files_present"])
        if group.get("manifest_metadata"):
            print("  metadata:", json.dumps(group["manifest_metadata"], ensure_ascii=False))
    if not candidates:
        print("  No feature cache candidates within scan limits; standalone trained heads are excluded.")
    print_expected(report["expected_care_inputs"])
    if "gpu_status" in env:
        print("\nGPU (index,name,total memory,used memory,utilization):\n" + env["gpu_status"])
    print("\nScan seconds:", scan_result["elapsed_seconds"], "depth:", scan_result["max_depth"],
          "entries:", scan_result["entries_visited"], "truncated:", scan_result["truncated"])
    for error in scan_result["errors"]:
        print("Scan access error:", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/xjw")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-entries", type=int, default=150000)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--care-only", action="store_true", help="Check the six default CARE inputs without a recursive scan")
    parser.add_argument("--output", type=Path, help="Optional JSON report; refuses to overwrite an existing file")
    args = parser.parse_args()
    root = Path(args.root).expanduser().absolute()
    if not root.is_dir():
        parser.error("Root directory does not exist: " + str(root))
    if args.max_depth < 0 or args.max_entries <= 0 or args.seconds <= 0:
        parser.error("Depth must be nonnegative; entry/time limits must be positive")
    if args.output and args.output.exists():
        parser.error("Report already exists; choose a new --output filename")
    try:
        top = sorted(p.name for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        parser.error(str(exc))
    report = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "root": str(root), "top_level_folders": top, "environment": environment(root),
              "expected_care_inputs": expected_care_inputs(root),
              "scan": None if args.care_only else scan(root, args.max_depth, args.max_entries, args.seconds)}
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print("\nJSON report:", args.output.absolute())


if __name__ == "__main__":
    main()
