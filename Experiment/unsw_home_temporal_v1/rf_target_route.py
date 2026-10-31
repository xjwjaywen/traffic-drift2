"""Exploratory paid-evidence RF-target route, with an explicit freeze boundary.

Run `prepare` before `evaluate`. Preparation uses only the existing paid
query labels to decide whether to accept RF-to-target disagreements for each
seed. Evaluation reads development labels only after gate.json exists.
The target class was chosen after seeing development outcomes. This is not
an independent validation or a novelty claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import metrics, read_development_labels
from rf_disagreement_probe import run as oof_audit, verified


HERE = Path(__file__).resolve().parent
RULE = "accept all RF-to-target disagreements iff OOF RF wins > source wins"


def prepare(args):
    output = args.output.resolve() / args.home
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Gate output must be new and empty: {output}")
    comparisons = oof_audit(args)
    gate = dict(home=args.home, target_class=args.target_class,
                source_class_probe=args.source_class, rule=RULE,
                paid_labels_per_seed=1000, folds=5,
                code_sha256={name: sha256(HERE / name) for name in
                             ("rf_disagreement_probe.py", "rf_target_route.py")},
                decisions=[dict(item, accept_rf=item["rf_wins"] > item["source_wins"])
                           for item in comparisons],
                development_labels_read=False, final_window_read=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print("GATE FROZEN:", output / "gate.json", flush=True)
    print("Accepted seeds:", [d["seed"] for d in gate["decisions"]
                              if d["accept_rf"]], flush=True)


def evaluate(args):
    output = args.output.resolve() / args.home
    gate_file = output / "gate.json"
    gate = json.loads(gate_file.read_text())
    if (gate["home"] != args.home or gate["target_class"] != args.target_class
            or gate["source_class_probe"] != args.source_class
            or gate["rule"] != RULE or len(gate["decisions"]) != 5):
        raise ValueError("Frozen gate differs from evaluation arguments")
    code_sha = {name: sha256(HERE / name) for name in
                ("rf_disagreement_probe.py", "rf_target_route.py")}
    if gate["code_sha256"] != code_sha:
        raise ValueError("Code changed after gate freeze")
    if (output / "REPORT.txt").exists():
        raise ValueError("Development route already evaluated")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        raise ValueError("Archive differs from reviewed timestamp audit")
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    official = pd.read_csv(manifest / "official_devices.csv")
    mapping = (official.loc[official.home == args.home]
               .set_index("mac").device_type.to_dict())
    y = read_development_labels(archive, members.loc[args.home, "member"],
                                args.home, windows, mapping)
    base = args.baseline.resolve(strict=True) / args.home
    static = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
    if len(static) != len(y):
        raise ValueError("Frozen source prediction length differs")
    rows, classes = [], []
    row, cls = metrics(y, static, static, args.home, -1, "static_mlp")
    rows.append(row)
    classes.extend(cls)
    for decision in gate["decisions"]:
        seed = int(decision["seed"])
        rf = np.load(verified(base / f"seed_{seed}", "frozen_rf.npy"),
                     mmap_mode="r")
        if len(rf) != len(y):
            raise ValueError("Frozen RF prediction length differs")
        routed = np.where((rf == args.target_class) & decision["accept_rf"],
                          rf, static)
        for method, pred in (("rf", rf), ("routed", routed)):
            row, cls = metrics(y, static, pred, args.home, seed, method)
            rows.append(row)
            classes.extend(cls)
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(classes)
    summary.to_csv(output / "summary.csv", index=False)
    detail.to_csv(output / "per_class.csv", index=False)
    focus = detail.loc[detail.device_type.isin([6, 8, 9, 10])]
    report = "\n".join([
        "Exploratory class-9 RF-target route; gate frozen from paid OOF labels.",
        "The target class and rule were chosen after viewing development results.",
        "Five seeds overlap; no final-window row was read.",
        "Whole-model development metrics:",
        summary[["seed", "method", "macro_f1_present",
                 "negative_flip_rate_vs_static_mlp"]].round(5).to_string(index=False),
        "Focus-class development F1 means:",
        focus.groupby(["device_type", "method"])["f1"].mean().round(5).to_string(),
        "Paid OOF gate decisions:",
        pd.DataFrame(gate["decisions"])[["seed", "rows", "rf_wins",
                                          "source_wins", "net_wins",
                                          "accept_rf"]].to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--home", default="home07")
    parser.add_argument("--source-class", type=int, default=6)
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--prepared", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_prepared_v1"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--archive", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal/all.zip"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_rf_target_route_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA unavailable; use --device cpu")
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
