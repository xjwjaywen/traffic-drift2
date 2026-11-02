"""Read-only query-to-development audit of RF-to-target proposals.

This compares the already completed paid OOF feasibility report with frozen
full-RF development predictions. Development labels were examined before
this audit was designed. Do not use its donor-edge outcomes to choose a
route and call that route independently validated. Final labels/features
stay unopened.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import read_development_labels
from f1_feasibility import macro_f1
from rf_disagreement_probe import compare, verified


HERE = Path(__file__).resolve().parent


def development_edges(y, static, rf, target, seed):
    mask = (rf == target) & (static != rf)
    present = np.unique(y)
    routed = np.where(mask, rf, static)
    macro_delta = 100 * (macro_f1(y, routed, present) -
                         macro_f1(y, static, present))
    all_counts = compare(mask, y, static, rf)
    edges = []
    for donor in np.unique(static[mask]):
        edge = mask & (static == donor)
        counts = compare(edge, y, static, rf)
        edge_route = np.where(edge, rf, static)
        edges.append(dict(seed=seed, source_predicted_class=int(donor),
                          **counts, target_true=int(np.sum(edge & (y == target))),
                          source_true=int(np.sum(edge & (y == donor))),
                          edge_macro_delta_pp=100 * (
                              macro_f1(y, edge_route, present) -
                              macro_f1(y, static, present))))
    return dict(seed=seed, curated_rows=len(y), proposal_rows=all_counts["rows"],
                rf_wins=all_counts["rf_wins"],
                source_wins=all_counts["source_wins"],
                both_wrong=all_counts["both_wrong"],
                all_route_dev_macro_delta_pp=macro_delta), edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", default="home07")
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--paid-report", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_f1_feasibility_v1/home07"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_transport_audit_v1"))
    args = parser.parse_args()
    if not 0 <= args.target_class < 24:
        parser.error("Target class outside 0..23")
    output = args.output.resolve() / args.home
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new and empty: {output}")
    paid = args.paid_report.resolve(strict=True)
    paid_config = json.loads((paid / "config.json").read_text())
    if (paid_config["home"] != args.home or
            paid_config["target_class"] != args.target_class):
        raise ValueError("Paid report does not match this audit")
    paid_summary = pd.read_csv(paid / "summary.csv")
    paid_edges = pd.read_csv(paid / "edges.csv")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())[
        "archive_sha256"]
    if sha256(archive) != expected or paid_config["archive_sha256"] != expected:
        raise ValueError("Archive differs from paid report or timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = (official.loc[official.home == args.home]
               .set_index("mac").device_type.to_dict())
    y_all = read_development_labels(archive, members.loc[args.home, "member"],
                                    args.home, windows, mapping)
    selected = y_all >= 0
    y = y_all[selected].astype(np.int64)
    base = args.baseline.resolve(strict=True) / args.home
    static_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
    if len(static_all) != len(y_all):
        raise ValueError("Frozen source prediction length differs")
    static = np.asarray(static_all[selected], dtype=np.int64)
    summaries, edges = [], []
    for seed in range(5):
        rf_all = np.load(verified(base / f"seed_{seed}", "frozen_rf.npy"),
                         mmap_mode="r")
        if len(rf_all) != len(y_all):
            raise ValueError("Frozen RF prediction length differs")
        rf = np.asarray(rf_all[selected], dtype=np.int64)
        row, edge_rows = development_edges(y, static, rf, args.target_class,
                                            seed)
        summaries.append(row)
        edges.extend(edge_rows)
        print("seed", seed, "development audit complete", flush=True)
    dev_summary = pd.DataFrame(summaries)
    dev_edges = pd.DataFrame(edges)
    if sorted(paid_summary.seed.tolist()) != list(range(5)):
        raise ValueError("Paid summary seeds differ")
    joined = paid_summary[["seed", "paid_rows", "proposal_rows", "delta_pp",
                           "row_boot_low_pp", "day_boot_low_pp"]].merge(
        dev_summary, on="seed", suffixes=("_paid", "_dev"), validate="one_to_one")
    joined["paid_proposal_rate"] = (joined.proposal_rows_paid /
                                    joined.paid_rows)
    joined["dev_proposal_rate"] = (joined.proposal_rows_dev /
                                   joined.curated_rows)
    paid_edge = paid_edges[["seed", "source_predicted_class", "rows",
                            "rf_wins", "source_wins", "target_true",
                            "source_true", "edge_macro_delta_pp"]]
    dev_edge = dev_edges[["seed", "source_predicted_class", "rows",
                          "rf_wins", "source_wins", "target_true",
                          "source_true", "edge_macro_delta_pp"]]
    edge_joined = paid_edge.merge(
        dev_edge, on=["seed", "source_predicted_class"], how="outer",
        suffixes=("_paid", "_dev"), validate="one_to_one").fillna(0)
    edge_joined["paid_target_fraction"] = np.divide(
        edge_joined.target_true_paid, edge_joined.rows_paid,
        out=np.full(len(edge_joined), np.nan), where=edge_joined.rows_paid > 0)
    edge_joined["dev_target_fraction"] = np.divide(
        edge_joined.target_true_dev, edge_joined.rows_dev,
        out=np.full(len(edge_joined), np.nan), where=edge_joined.rows_dev > 0)
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "summary.csv", index=False)
    edge_joined.to_csv(output / "edges.csv", index=False)
    report = "\n".join([
        "Exploratory read-only query-to-development RF target-route audit.",
        "Paid predictions are OOF RF trained on 800 labels plus references;",
        "development predictions are full RF trained on 1000 labels plus references.",
        "Thus changes mix temporal transport and RF training-size effects.",
        "Development outcomes were viewed before this audit; no final labels/features decoded.",
        "Seeds overlap and are not independent homes.",
        "Fixed all-RF-to-target route (percentage points):",
        joined[["seed", "paid_proposal_rate", "dev_proposal_rate",
                "delta_pp", "all_route_dev_macro_delta_pp", "rf_wins",
                "source_wins"]].round(4).to_string(index=False),
        "Donor-edge proposal counts and target fractions:",
        edge_joined[["seed", "source_predicted_class", "rows_paid",
                     "rows_dev", "paid_target_fraction", "dev_target_fraction",
                     "edge_macro_delta_pp_paid", "edge_macro_delta_pp_dev"]]
        .sort_values(["seed", "rows_dev"], ascending=[True, False])
        .round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
