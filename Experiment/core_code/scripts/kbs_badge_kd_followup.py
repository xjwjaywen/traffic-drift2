#!/usr/bin/env python3
"""Add paired BADGE FT+KD runs to a completed controlled-v1 study."""
import contextlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

import kbs_supplement as kbs

SUITE = "badge_kd"
BASE_SPECS = kbs.make_specs


def specs(suite=SUITE):
    if suite != SUITE:
        return BASE_SPECS(suite)
    full = next(s for s in BASE_SPECS("primary") if s["name"] == "badge_full")
    return [full, {**full, "name": "badge_kd", "replay_ce": False}]


def verified_run(out, protocol, spec, seed):
    directory = out / "runs" / spec["name"] / f"seed_{seed}"
    if not kbs.is_complete(directory, {"protocol": protocol, "spec": spec, "seed": seed}):
        raise ValueError(f"A verified completed run is required: {directory}")
    return directory


def verify_baselines(out, seeds):
    protocol = json.loads((out / "study_manifest.json").read_text())["protocol"]
    if protocol["implementation_sha256"] != kbs.implementation_sha():
        raise ValueError("The original training implementation changed; this follow-up cannot reuse the study.")
    for seed in seeds:
        baseline = verified_run(out, protocol, specs()[0], seed)
        for selector in ["margin", "badge"]:
            directory = out / "selections" / selector / f"seed_{seed}"
            signature = {"cache": protocol["cache"], "implementation": kbs.implementation_sha(),
                         "selector": selector, "budget": protocol["budget"], "seed": seed}
            if not kbs.is_complete(directory, signature):
                raise ValueError(f"Existing verified query selection is required: {directory}")
            if selector == "badge":
                indices = json.loads((directory / "selection.json").read_text())["row_indices"]
                if indices != [int(r["row_index"]) for r in kbs.read_csv(baseline / "query_ids.csv")]:
                    raise ValueError(f"Saved BADGE selection differs from baseline query IDs: seed {seed}")
    return protocol


def paired_metrics(out, protocol, seed):
    full, kd = [verified_run(out, protocol, s, seed) for s in specs()]
    for filename in ["query_ids.csv", "replay_ids.csv"]:
        # The CE-use flag intentionally differs, but IDs, order and labels must match.
        columns = ["sample_id", "row_index", "label"]
        ids = [[tuple(row[c] for c in columns) for row in kbs.read_csv(d / filename)]
               for d in [full, kd]]
        if ids[0] != ids[1]:
            raise ValueError(f"Unpaired {filename}: seed {seed}")
    traces = [json.loads((d / "training_trace.json").read_text()) for d in [full, kd]]
    for field in ["optimizer_steps", "target_presentations", "reference_kd_presentations",
                  "target_stream_sha256", "reference_stream_sha256"]:
        if traces[0][field] != traces[1][field]:
            raise ValueError(f"Unpaired training {field}: seed {seed}")
    if traces[1]["reference_ce_presentations"] != 0:
        raise ValueError(f"BADGE FT+KD unexpectedly used reference CE: seed {seed}")
    with np.load(full / "predictions.npz") as a, np.load(kd / "predictions.npz") as b:
        for field in ["row_id", "y_true", "static_pred", "queried", "strict_eval", "common_eval"]:
            if not np.array_equal(a[field], b[field]):
                raise ValueError(f"Unpaired predictions/evaluation {field}: seed {seed}")
    metrics = [json.loads((d / "metrics.json").read_text()) for d in [full, kd]]
    row = {"seed": seed, "pairing_verified": True, "difference": "badge_kd minus badge_full"}
    for field in metrics[0]:
        if field.startswith(("common_", "strict_")):
            if metrics[0][field] is not None and metrics[1][field] is not None:
                row[field + "_difference"] = metrics[1][field] - metrics[0][field]
    return metrics, row


def summarize(out, suite=SUITE):
    if suite != SUITE:
        raise ValueError(f"Unexpected follow-up suite: {suite}")
    out = Path(out)
    seeds = sorted(int(p.parent.name.removeprefix("seed_"))
                   for p in (out / "runs" / "badge_kd").glob("seed_*/complete.json"))
    protocol = verify_baselines(out, seeds)
    selected = {s["name"]: [] for s in specs()}
    pairs = []
    for seed in seeds:
        metrics, pair = paired_metrics(out, protocol, seed)
        for metric in metrics:
            selected[metric["name"]].append(metric)
        pairs.append(pair)
    rows, summaries = [], []
    for spec in specs():
        values = selected[spec["name"]]
        rows.extend(values)
        item = {**spec, "n_seeds": len(values), "recommended_n_seeds": 5,
                "meets_recommended_seed_count": len(values) >= 5}
        if values:
            for field in values[0]:
                if field.startswith(("common_", "strict_")) or field == "train_wall_s":
                    numbers = [float(r[field]) for r in values if r.get(field) is not None]
                    if numbers:
                        item[field + "_mean"] = statistics.mean(numbers)
                        item[field + "_sample_sd"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        summaries.append(item)
    # Only matched, verified seeds enter these new tables. Primary tables stay untouched.
    kbs.write_csv(out / f"{SUITE}_results_by_seed.csv", rows)
    kbs.write_csv(out / f"{SUITE}_summary.csv", summaries)
    kbs.write_csv(out / f"{SUITE}_paired_by_seed.csv", pairs)
    print(f"Verified BADGE pairs: {len(pairs)}/5 recommended seeds", flush=True)


@contextlib.contextmanager
def followup_registry():
    """Register only specs/reporting, leaving the hash-pinned numerical engine intact.

    The v1 runner resolves these two module callbacks. Scope their replacement to
    this single-process follow-up; do not override hashing, fitting or selection.
    """
    original_specs, original_summary = kbs.make_specs, kbs.summarize
    kbs.make_specs, kbs.summarize = specs, summarize
    try:
        yield
    finally:
        kbs.make_specs, kbs.summarize = original_specs, original_summary


def followup(args, data=None):
    out = Path(args.output_dir).resolve()
    if not (out / "study_manifest.json").is_file():
        raise ValueError("Run primary first, or point OUTPUT_DIR to the completed primary study.")
    print("Validating completed BADGE baselines and saved query selections...", flush=True)
    with kbs.file_lock(out / ".badge_kd.lock"):
        with kbs.file_lock(out / ".lock"):
            verify_baselines(out, kbs.int_list(args.seeds))
            identity = {"followup_sha256": kbs.file_sha(Path(__file__)),
                        "implementation_sha256": kbs.implementation_sha(), "specs": specs()}
            path = out / "badge_kd_extension_manifest.json"
            if path.exists():
                if json.loads(path.read_text())["identity"] != identity:
                    raise ValueError("Follow-up implementation changed; preserve this extension's completed outputs.")
            else:
                kbs.atomic_json(path, {"identity": identity, "environment": kbs.runtime_info()})
        if data is None:
            if not (Path(args.cache_dir) / "manifest.json").is_file():
                raise ValueError("The existing primary feature cache is required; set CACHE_DIR to it.")
            print("Validating the existing feature cache; no feature extraction is planned...", flush=True)
            kbs.prepare(args)  # Validates inputs and reuses the existing cache.
            data = kbs.load_cache(args)
        with followup_registry():
            args.suite = SUITE
            kbs.run_study(args, data)


def main():
    args = kbs.parser().parse_args()
    if args.command not in ["plan", "run", "summarize"] or args.suite != "primary":
        raise ValueError("Use this follow-up with plan, run or summarize; omit --suite.")
    args.seeds = args.seeds if args.seeds is not None else "0,1,2,3,4"
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    for name in ["budget", "steps", "batch_size", "threads"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.lr <= 0 or args.weight_decay < 0 or args.log_every < 0:
        raise ValueError("Invalid optimizer/logging settings.")
    os.chdir(kbs.CORE)
    if args.command == "plan":
        print(json.dumps({"suite": SUITE, "seeds": seeds, "configs": specs(),
                          "required_completed_baselines": len(seeds),
                          "maximum_new_fits": len(seeds), "optimizer_steps_per_fit": args.steps,
                          "note": "Reuses verified badge_full runs and saved Margin/BADGE queries."}, indent=2))
    elif args.command == "run":
        followup(args)
    else:
        with kbs.file_lock(Path(args.output_dir) / ".lock"):
            summarize(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
