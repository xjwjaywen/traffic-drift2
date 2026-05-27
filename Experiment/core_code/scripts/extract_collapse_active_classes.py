"""
Extract class lists for collapse-aware active maintenance.

The active-maintenance experiments need dataset-specific class groups:
  - collapse classes: final-period classes below the recall threshold
  - absorber classes: top predicted targets for those collapsed classes
  - stable classes: high-recall, low-delta classes used for reporting/replay

Usage from Experiment/core_code/:
    python scripts/extract_collapse_active_classes.py \
      --collapse-csv outputs/per_class_collapse_quic22/collapse_classes.csv \
      --selected-stable-csv outputs/class_conditional_drift_quic22/selected_stable_classes.csv

By default the script prints shell assignments that can be eval'ed by a runner:
    COLLAPSE_CLASSES="..."
    ABSORBER_CLASSES="..."
    STABLE_CLASSES="..."
"""
import argparse
import csv
import json
import os


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def as_int(row, key, default=None):
    value = row.get(key)
    if value in {None, ""}:
        return default
    return int(float(value))


def unique_ordered(values):
    seen = set()
    out = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def comma(values):
    return ",".join(str(int(v)) for v in values)


def final_collapsed_rows(rows, recall_threshold, min_support):
    out = []
    for row in rows:
        support = as_int(row, "final_support", 0)
        recall = as_float(row, "final_recall", 1.0)
        if support >= min_support and recall < recall_threshold:
            out.append(row)
    return sorted(
        out,
        key=lambda r: (
            as_float(r, "final_recall", 1.0),
            -as_int(r, "final_support", 0),
            as_int(r, "class_id", 0),
        ),
    )


def stable_from_selected(path, top_k):
    if not path or not os.path.exists(path):
        return []
    rows = read_csv(path)
    classes = [as_int(row, "class_id") for row in rows[:top_k]]
    return [c for c in classes if c is not None]


def stable_from_collapse_rows(
    rows,
    top_k,
    min_support,
    min_recall,
    max_abs_delta_recall,
):
    candidates = []
    for row in rows:
        class_id = as_int(row, "class_id")
        support = as_int(row, "final_support", 0)
        recall = as_float(row, "final_recall", 0.0)
        delta = as_float(row, "delta_final_recall_from_ref", 999.0)
        first_collapse = row.get("first_collapse_period") or ""
        if (
            class_id is not None
            and support >= min_support
            and first_collapse == ""
            and recall >= min_recall
            and abs(delta) <= max_abs_delta_recall
        ):
            candidates.append(row)

    if not candidates:
        candidates = [
            row for row in rows
            if as_int(row, "final_support", 0) >= min_support
            and (row.get("first_collapse_period") or "") == ""
        ]

    candidates = sorted(
        candidates,
        key=lambda r: (
            abs(as_float(r, "delta_final_recall_from_ref", 999.0)),
            -as_float(r, "final_recall", 0.0),
            -as_int(r, "final_support", 0),
        ),
    )
    return [as_int(row, "class_id") for row in candidates[:top_k]]


def build_groups(args):
    rows = read_csv(args.collapse_csv)
    collapsed = final_collapsed_rows(
        rows,
        args.collapse_recall_threshold,
        args.min_support,
    )
    collapse_classes = [as_int(row, "class_id") for row in collapsed]
    absorber_classes = unique_ordered(
        as_int(row, "final_top_confusion_target") for row in collapsed
    )
    stable_classes = stable_from_selected(args.selected_stable_csv, args.stable_top_k)
    if not stable_classes:
        stable_classes = stable_from_collapse_rows(
            rows,
            args.stable_top_k,
            args.min_support,
            args.stable_min_recall,
            args.stable_max_abs_delta_recall,
        )
    return {
        "collapse_classes": [c for c in collapse_classes if c is not None],
        "absorber_classes": [c for c in absorber_classes if c is not None],
        "stable_classes": [c for c in stable_classes if c is not None],
    }


def print_shell(groups):
    print(f'COLLAPSE_CLASSES="{comma(groups["collapse_classes"])}"')
    print(f'ABSORBER_CLASSES="{comma(groups["absorber_classes"])}"')
    print(f'STABLE_CLASSES="{comma(groups["stable_classes"])}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collapse-csv", required=True)
    parser.add_argument("--selected-stable-csv", default=None)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--stable-top-k", type=int, default=20)
    parser.add_argument("--stable-min-recall", type=float, default=0.8)
    parser.add_argument("--stable-max-abs-delta-recall", type=float, default=0.05)
    parser.add_argument("--format", choices=["shell", "json"], default="shell")
    args = parser.parse_args()

    groups = build_groups(args)
    if args.format == "json":
        print(json.dumps(groups, indent=2))
    else:
        print_shell(groups)


if __name__ == "__main__":
    main()
