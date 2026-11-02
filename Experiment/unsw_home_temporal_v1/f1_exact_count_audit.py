"""Paired closed-set F1 audit: exact prediction marginals versus plug-in.

Exploratory replay on already examined development rows. The official 24-type
inventory defines the same curated population as the existing model control.
Frozen source and CARE predictions determine both query designs; labels are
used only to score the drawn rows and to report full-development oracle error.
No final-window row is decoded. Bootstrap intervals are descriptive only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import read_development_labels
from f1_feasibility import macro_f1
from f1_stratified_audit import allocate
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent
CLASSES = np.arange(24)


def draw_sample(strata, mass, allocation, draws, rng):
    indices = []
    groups = []
    for group, n in enumerate(allocation):
        chosen = rng.choice(np.flatnonzero(strata == group), int(n), replace=False)
        indices.append(chosen)
        groups.extend([group] * len(chosen))
    indices = np.concatenate(indices)
    groups = np.asarray(groups, dtype=np.int64)
    expansion = mass[groups] / allocation[groups]
    boot_weights = np.zeros((draws, len(indices)), dtype=np.float64)
    start = 0
    for group, n in enumerate(allocation):
        stop = start + n
        boot_weights[:, start:stop] = rng.multinomial(
            int(n), np.full(n, 1 / n), size=draws
        ) * (mass[group] / n)
        start = stop
    return indices, expansion, boot_weights


def paired_delta(weights, y, source, candidate, p0, p1, exact_counts):
    onehot = (y[:, None] == CLASSES).astype(np.float64)
    source_onehot = (source[:, None] == CLASSES).astype(np.float64)
    candidate_onehot = (candidate[:, None] == CLASSES).astype(np.float64)
    support = weights @ onehot
    tp0 = weights @ (onehot * source_onehot)
    tp1 = weights @ (onehot * candidate_onehot)
    if exact_counts:
        pred0, pred1 = p0[None, :], p1[None, :]
    else:
        pred0 = weights @ source_onehot
        pred1 = weights @ candidate_onehot
    f0 = np.divide(2 * tp0, support + pred0,
                   out=np.zeros_like(tp0), where=(support + pred0) > 0)
    f1 = np.divide(2 * tp1, support + pred1,
                   out=np.zeros_like(tp1), where=(support + pred1) > 0)
    return 100 * (f1 - f0).mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--audit-labels", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_f1_exact_count_audit_v1"))
    args = parser.parse_args()
    if args.audit_labels < 100 or args.bootstrap_draws < 100:
        parser.error("Audit labels and bootstrap draws must each be at least 100")
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
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
                                        home, windows, mapping).astype(np.int64)
        curated = y_all >= 0
        y = y_all[curated]
        if len(y) < args.audit_labels:
            raise ValueError(f"Too few curated development rows: {home}")
        base = base_root / home
        source_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(source_all) != len(y_all):
            raise ValueError(f"Frozen source prediction length differs: {home}")
        source = np.asarray(source_all[curated], dtype=np.int64)
        p0 = np.bincount(source, minlength=24).astype(np.float64)
        full_source = macro_f1(y, source, CLASSES)
        for seed in range(5):
            candidate_all = np.load(verified(base / f"seed_{seed}", "care.npy"),
                                    mmap_mode="r")
            if len(candidate_all) != len(y_all):
                raise ValueError(f"Frozen CARE prediction length differs: {home} {seed}")
            candidate = np.asarray(candidate_all[curated], dtype=np.int64)
            p1 = np.bincount(candidate, minlength=24).astype(np.float64)
            full_delta = 100 * (macro_f1(y, candidate, CLASSES) - full_source)
            for design in ("uniform", "agreement_disagreement"):
                if design == "uniform":
                    strata = np.zeros(len(y), dtype=np.int64)
                else:
                    strata = (source != candidate).astype(np.int64)
                _, strata, mass = np.unique(strata, return_inverse=True,
                                            return_counts=True)
                allocation = allocate(mass, args.audit_labels)
                rng = np.random.default_rng(
                    880000 + 100 * home_index + seed +
                    (0 if design == "uniform" else 10000))
                ids, expansion, boot_weights = draw_sample(
                    strata, mass, allocation, args.bootstrap_draws, rng)
                sample_y, sample_source, sample_candidate = (
                    y[ids], source[ids], candidate[ids])
                for estimator in ("sample_prediction_counts", "exact_prediction_counts"):
                    exact = estimator == "exact_prediction_counts"
                    point = paired_delta(expansion[None, :], sample_y,
                                         sample_source, sample_candidate,
                                         p0, p1, exact)[0]
                    boot = paired_delta(boot_weights, sample_y, sample_source,
                                        sample_candidate, p0, p1, exact)
                    low, median, high = np.quantile(boot, [.025, .5, .975])
                    decision = ("positive" if low > 0 else
                                "negative" if high < 0 else "unresolved")
                    rows.append(dict(home=home, seed=seed, design=design,
                                     estimator=estimator,
                                     full_macro_delta_pp=full_delta,
                                     sampled_delta_pp=float(point),
                                     bootstrap_low_pp=float(low),
                                     bootstrap_median_pp=float(median),
                                     bootstrap_high_pp=float(high),
                                     decision=decision,
                                     interval_contains_full=bool(low <= full_delta <= high),
                                     decision_matches_full_sign=(
                                         bool((decision == "positive" and full_delta > 0) or
                                              (decision == "negative" and full_delta < 0))
                                         if decision != "unresolved" else None),
                                     sampled_present_classes=len(np.unique(sample_y)),
                                     sampled_disagreements=int(np.sum(
                                         sample_source != sample_candidate))))
            print(home, "seed", seed, "paired audit complete", flush=True)
    result = pd.DataFrame(rows)
    summary = (result.groupby(["home", "design", "estimator"], as_index=False)
               .agg(mean_full_delta_pp=("full_macro_delta_pp", "mean"),
                    mean_sampled_delta_pp=("sampled_delta_pp", "mean"),
                    resolved_seeds=("decision", lambda x: int((x != "unresolved").sum())),
                    wrong_resolved_seeds=("decision_matches_full_sign",
                                          lambda x: int((x == False).sum())),
                    interval_contains_full_seeds=("interval_contains_full", "sum"),
                    mean_present_classes=("sampled_present_classes", "mean"),
                    mean_disagreements=("sampled_disagreements", "mean")))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "per_home.csv", index=False)
    means = (summary.groupby(["design", "estimator"])
             [["mean_sampled_delta_pp", "resolved_seeds", "wrong_resolved_seeds",
               "interval_contains_full_seeds", "mean_present_classes",
               "mean_disagreements"]].mean())
    report = "\n".join([
        "Exploratory paired CARE-versus-source macro-F1 audit on the curated 24-class development population.",
        "Same 1000 additional paid labels within each design and same labels for both estimators.",
        "Prediction marginals are either HT-estimated from the sample or counted exactly from frozen predictions.",
        "Agreement/disagreement stratification and label allocation use only frozen predictions.",
        "The ordinary within-stratum row bootstrap is descriptive, not a temporal safety certificate.",
        "Five seeds overlap within each home; development outcomes were already examined; final window untouched.",
        "Across-home means (homes are independent units):",
        means.round(4).to_string(),
        "Per-home results:",
        summary.round(4).to_string(index=False),
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
