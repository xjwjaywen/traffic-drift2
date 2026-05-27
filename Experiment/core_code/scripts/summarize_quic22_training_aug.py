"""
Compare baseline QUIC22 sequential static results with a training-augmentation run.
"""
import argparse
import json
import os


def load_static_periods(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    static = results.get("static")
    if static is None:
        raise KeyError(f"No static results in {path}")
    periods = static.get("periods")
    if periods is None:
        raise KeyError(f"No static.periods results in {path}")
    return periods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="outputs/eval_quic22/results_sequential.json")
    parser.add_argument("--aug-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline = load_static_periods(args.baseline_results)
    aug = load_static_periods(args.aug_results)
    os.makedirs(args.output_dir, exist_ok=True)

    periods = [period for period in baseline if period in aug]
    lines = [
        "# QUIC22 Training-Time Channel Augmentation Summary",
        "",
        "| period | baseline macro-F1 | augmented macro-F1 | delta macro-F1 | baseline acc | augmented acc | delta acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for period in periods:
        b = baseline[period]
        a = aug[period]
        row = {
            "period": period,
            "baseline_macro_f1": float(b["macro_f1"]),
            "aug_macro_f1": float(a["macro_f1"]),
            "delta_macro_f1": float(a["macro_f1"] - b["macro_f1"]),
            "baseline_accuracy": float(b["accuracy"]),
            "aug_accuracy": float(a["accuracy"]),
            "delta_accuracy": float(a["accuracy"] - b["accuracy"]),
        }
        rows.append(row)
        lines.append(
            f"| {period} | {row['baseline_macro_f1']:.4f} | {row['aug_macro_f1']:.4f} | "
            f"{row['delta_macro_f1']:+.4f} | {row['baseline_accuracy']:.4f} | "
            f"{row['aug_accuracy']:.4f} | {row['delta_accuracy']:+.4f} |"
        )

    if rows:
        mean_delta = sum(row["delta_macro_f1"] for row in rows) / len(rows)
        final = rows[-1]
        lines += [
            "",
            f"- Mean delta macro-F1: `{mean_delta:+.4f}`",
            f"- Final-period delta macro-F1: `{final['delta_macro_f1']:+.4f}`",
        ]

    report_path = os.path.join(args.output_dir, "quic22_training_aug_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    csv_path = os.path.join(args.output_dir, "quic22_training_aug_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in row.keys()) + "\n")

    print(f"Saved report: {report_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()

