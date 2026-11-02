"""Simulate whether 1000 later labels can resolve paired macro-F1 decisions.

This uses already examined development labels as a replay oracle, separate
from the earlier 1000 training labels per seed. All model predictions were
frozen before development labels were read. Row bootstrap intervals here are
descriptive, not temporal or finite-sample safety certificates. Final-window
rows are never decoded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import read_development_labels
from f1_feasibility import macro_f1
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
CLASSES = np.arange(24)


def boot_macro(weights, truth_onehot, pred):
    predicted = (pred[:, None] == CLASSES).astype(np.float32)
    tp = weights @ (truth_onehot * predicted)
    support = weights @ truth_onehot
    predicted_count = weights @ predicted
    scores = np.divide(2 * tp, support + predicted_count,
                       out=np.zeros_like(tp), where=(support + predicted_count) > 0)
    return scores.mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--audit-labels", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_f1_audit_feasibility_v1"))
    args = parser.parse_args()
    if args.audit_labels < 100 or args.bootstrap_draws < 100:
        parser.error("Audit labels and bootstrap draws must each be at least 100")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new and empty: {output}")
    archive = args.archive.resolve(strict=True)
    manifest = HERE / "metadata"
    expected = json.loads((manifest / "audit.json").read_text())["archive_sha256"]
    if sha256(archive) != expected:
        parser.error("Archive differs from reviewed timestamp audit")
    windows = pd.read_csv(manifest / "windows.csv").set_index(["home", "split"])
    members = pd.read_csv(manifest / "members.csv").set_index("home")
    official = pd.read_csv(manifest / "official_devices.csv")
    base_root = args.baseline.resolve(strict=True)
    homes = [args.home] if args.home else PRIMARY
    rows = []
    for home_index, home in enumerate(homes):
        mapping = (official.loc[official.home == home]
                   .set_index("mac").device_type.to_dict())
        y_all = read_development_labels(archive, members.loc[home, "member"],
                                        home, windows, mapping)
        selected = y_all >= 0
        y = y_all[selected].astype(np.int64)
        if len(y) < args.audit_labels:
            raise ValueError(f"Too few curated development rows: {home}")
        base = base_root / home
        static_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(static_all) != len(y_all):
            raise ValueError(f"Frozen source prediction length differs: {home}")
        static = np.asarray(static_all[selected], dtype=np.int64)
        full_static = macro_f1(y, static, CLASSES)
        full_present = int(len(np.unique(y)))
        for seed in range(5):
            rng = np.random.default_rng(660000 + 100 * home_index + seed)
            sampled = rng.choice(len(y), args.audit_labels, replace=False)
            y_sample = y[sampled]
            static_sample = static[sampled]
            truth = (y_sample[:, None] == CLASSES).astype(np.float32)
            weights = rng.multinomial(args.audit_labels,
                                      np.full(args.audit_labels,
                                              1 / args.audit_labels),
                                      size=args.bootstrap_draws).astype(np.float32)
            boot_static = boot_macro(weights, truth, static_sample)
            sample_static = macro_f1(y_sample, static_sample, CLASSES)
            for method, filename in (("care", "care.npy"),
                                     ("frozen_rf", "frozen_rf.npy")):
                candidate_all = np.load(verified(base / f"seed_{seed}", filename),
                                        mmap_mode="r")
                if len(candidate_all) != len(y_all):
                    raise ValueError(f"Frozen prediction length differs: {home} {method}")
                candidate = np.asarray(candidate_all[selected], dtype=np.int64)
                full_delta = 100 * (macro_f1(y, candidate, CLASSES) -
                                    full_static)
                sample_delta = 100 * (
                    macro_f1(y_sample, candidate[sampled], CLASSES) -
                    sample_static)
                boot_delta = 100 * (boot_macro(weights, truth,
                                               candidate[sampled]) - boot_static)
                low, median, high = np.quantile(boot_delta, [.025, .5, .975])
                decision = ("positive" if low > 0 else
                            "negative" if high < 0 else "unresolved")
                rows.append(dict(
                    home=home, seed=seed, method=method,
                    development_rows=len(y), audit_labels=args.audit_labels,
                    full_present_classes=full_present,
                    sampled_present_classes=int(len(np.unique(y_sample))),
                    full_macro_delta_pp=full_delta,
                    sampled_macro_delta_pp=sample_delta,
                    bootstrap_low_pp=float(low),
                    bootstrap_median_pp=float(median),
                    bootstrap_high_pp=float(high),
                    interval_contains_full_delta=bool(low <= full_delta <= high),
                    audit_decision=decision,
                    decision_matches_full_sign=bool(
                        (decision == "positive" and full_delta > 0) or
                        (decision == "negative" and full_delta < 0))
                    if decision != "unresolved" else None))
            print(home, "seed", seed, "paired F1 audit complete", flush=True)
    result = pd.DataFrame(rows)
    summary = (result.groupby(["home", "method"], as_index=False)
               .agg(mean_full_delta_pp=("full_macro_delta_pp", "mean"),
                    mean_sampled_delta_pp=("sampled_macro_delta_pp", "mean"),
                    resolved_seeds=("audit_decision",
                                    lambda x: int((x != "unresolved").sum())),
                    interval_contains_full_seeds=("interval_contains_full_delta",
                                                  "sum"),
                    mean_sampled_present_classes=("sampled_present_classes", "mean")))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Exploratory information-feasibility replay on already examined UNSW development rows.",
        f"Each seed uses {args.audit_labels} additional uniformly sampled development labels, separate from 1000 earlier training labels.",
        "Paired source-versus-CARE/RF macro F1 uses the fixed 24-class inventory; absent classes score zero.",
        "Intervals are ordinary row-bootstrap descriptions; they ignore temporal dependence and model-fitting uncertainty.",
        "The full development labels are used only as a replay oracle to measure coverage and actual direction.",
        "Overlapping seeds are not independent homes; no final-window label or feature decoded.",
        "Per-home summary:",
        summary.round(4).to_string(index=False),
        "Across-home means by method:",
        summary.groupby("method")[["mean_full_delta_pp", "resolved_seeds",
                                   "interval_contains_full_seeds",
                                   "mean_sampled_present_classes"]]
        .mean().round(4).to_string(),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "config.json").write_text(json.dumps(dict(
        homes=homes, archive_sha256=expected,
        audit_labels=args.audit_labels, bootstrap_draws=args.bootstrap_draws,
        implementation_sha256=sha256(Path(__file__)),
        final_window_read=False), indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
