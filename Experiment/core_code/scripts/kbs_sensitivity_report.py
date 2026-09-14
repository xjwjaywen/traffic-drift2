#!/usr/bin/env python3
"""Report sensitivity configurations on the same requested repair seeds."""
import json
from pathlib import Path
import statistics
import sys

import kbs_supplement as kbs


def report(output_dir, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    out = Path(output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        protocol = json.loads((out / "study_manifest.json").read_text())["protocol"]
        rows, summaries = [], []
        for spec in kbs.make_specs("sensitivity"):
            selected = []
            for seed in seeds:
                run = out / "runs" / spec["name"] / f"seed_{seed}"
                signature = {"protocol": protocol, "spec": spec, "seed": seed}
                if not kbs.is_complete(run, signature):
                    raise ValueError(f"Requested sensitivity run is not complete: {run}")
                metrics = json.loads((run / "metrics.json").read_text())
                if any(metrics.get(k) != v for k, v in {**spec, "seed": seed}.items()):
                    raise ValueError(f"Metrics identity mismatch: {run}")
                selected.append(metrics)
            rows.extend(selected)
            item = {**spec, "seed_ids": ",".join(map(str, seeds)), "n_seeds": len(seeds),
                    "recommended_n_seeds": 3, "meets_recommended_seed_count": len(seeds) >= 3}
            for field in selected[0]:
                if field.startswith(("common_", "strict_")) or field == "train_wall_s":
                    values = [float(r[field]) for r in selected if r.get(field) is not None]
                    if values:
                        item[field + "_mean"] = statistics.mean(values)
                        item[field + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summaries.append(item)
        kbs.write_csv(out / "sensitivity_matched_results_by_seed.csv", rows)
        kbs.write_csv(out / "sensitivity_matched_summary.csv", summaries)
        print(f"Matched sensitivity: {len(summaries)} configurations x {len(seeds)} seeds = {len(rows)} runs", flush=True)
        print(f"Results: {out / 'sensitivity_matched_summary.csv'}", flush=True)


def main():
    args = kbs.parser().parse_args(["summarize", "--suite", "sensitivity", *sys.argv[1:]])
    if args.suite != "sensitivity":
        raise ValueError("This report requires the sensitivity suite.")
    report(args.output_dir, kbs.int_list(args.seeds if args.seeds is not None else "0,1,2"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
