"""Compare uniform and prediction-pair-stratified 1000-label F1 audits.

This is an exploratory replay on already examined development labels. The
sampling design uses only frozen source/CARE predictions and their group
counts, although the replay oracle loads all development labels. Ordinary
within-stratum bootstrap intervals are descriptive, not safety certificates.
No final-window row is decoded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import sha256
from evaluate import read_development_labels
from f1_audit_feasibility import CLASSES, boot_macro
from f1_feasibility import macro_f1
from predict import PRIMARY
from rf_disagreement_probe import verified

HERE = Path(__file__).resolve().parent


def make_strata(source, candidate):
    """Keep common prediction pairs; merge tiny pairs by source class."""
    pair = source * 24 + candidate
    counts = np.bincount(pair, minlength=24 * 24)
    # At most 400 common pairs plus 24 pooled source groups fit the audit floor.
    threshold = max(100, int(np.ceil(len(pair) / 400)))
    common = counts[pair] >= threshold
    # Rare pairs are pooled by source prediction. All keys use predictions only.
    key = np.where(common, pair, 24 * 24 + source)
    _, inverse, mass = np.unique(key, return_inverse=True, return_counts=True)
    return inverse, mass


def allocate(mass, budget):
    """Two per stratum, then half mass and half square-root-mass allocation."""
    groups = len(mass)
    if groups * 2 > budget:
        raise ValueError(f"Too many strata for two-per-stratum floor: {groups}")
    allotment = np.minimum(mass, 2).astype(np.int64)
    remaining = budget - int(allotment.sum())
    while remaining:
        capacity = mass - allotment
        eligible = capacity > 0
        if not eligible.any():
            raise ValueError("Audit budget exceeds population")
        m = mass[eligible].astype(np.float64)
        score = .5 * m / m.sum() + .5 * np.sqrt(m) / np.sqrt(m).sum()
        desired = remaining * score
        add = np.minimum(np.floor(desired).astype(np.int64), capacity[eligible])
        if not add.any():
            order = np.argsort(-desired)
            available = np.flatnonzero(eligible)
            for local in order[:remaining]:
                allotment[available[local]] += 1
            break
        allotment[eligible] += add
        remaining -= int(add.sum())
    if allotment.sum() != budget or np.any(allotment > mass):
        raise AssertionError("Invalid stratum allocation")
    return allotment


def sampled_audit(y, source, candidate, strata, mass, allotment, draws, rng):
    ids = []
    sample_groups = []
    for group, n in enumerate(allotment):
        selected = rng.choice(np.flatnonzero(strata == group), int(n), replace=False)
        ids.append(selected)
        sample_groups.extend([group] * len(selected))
    ids = np.concatenate(ids)
    sample_groups = np.asarray(sample_groups)
    expansion = mass[sample_groups] / allotment[sample_groups]
    sampled_y, sampled_source, sampled_candidate = y[ids], source[ids], candidate[ids]
    truth = (sampled_y[:, None] == CLASSES).astype(np.float32)
    boot_weights = np.zeros((draws, len(ids)), dtype=np.float32)
    start = 0
    for group, n in enumerate(allotment):
        stop = start + n
        boot_weights[:, start:stop] = rng.multinomial(
            int(n), np.full(n, 1 / n), size=draws
        ) * (mass[group] / n)
        start = stop
    boot_delta = 100 * (
        boot_macro(boot_weights, truth, sampled_candidate) -
        boot_macro(boot_weights, truth, sampled_source)
    )
    curated = sampled_y >= 0
    estimate = 100 * (
        macro_f1(sampled_y[curated], sampled_candidate[curated],
                 CLASSES, expansion[curated]) -
        macro_f1(sampled_y[curated], sampled_source[curated],
                 CLASSES, expansion[curated])
    )
    low, median, high = np.quantile(boot_delta, [.025, .5, .975])
    return dict(sampled_delta_pp=estimate, bootstrap_low_pp=float(low),
                bootstrap_median_pp=float(median), bootstrap_high_pp=float(high),
                sampled_present_classes=len(np.unique(sampled_y[curated])),
                sampled_curated_rows=int(curated.sum()),
                sampled_disagreements=int(np.sum(sampled_source != sampled_candidate)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--home", choices=PRIMARY)
    parser.add_argument("--audit-labels", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--baseline", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_home_temporal_predictions_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_f1_stratified_audit_v1"))
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
        # The archive reader stores labels as int8; macro_f1 encodes y*24+pred.
        # Promote before arithmetic so class IDs >= 6 cannot overflow int8.
        y_all = read_development_labels(archive, members.loc[home, "member"],
                                        home, windows, mapping).astype(np.int64)
        selected = y_all >= 0
        if len(y_all) < args.audit_labels or not selected.any():
            raise ValueError(f"Too few development rows: {home}")
        base = base_root / home
        source_all = np.load(verified(base, "static_mlp.npy"), mmap_mode="r")
        if len(source_all) != len(y_all):
            raise ValueError(f"Frozen source prediction length differs: {home}")
        source = np.asarray(source_all, dtype=np.int64)
        source_f1 = macro_f1(y_all[selected], source[selected], CLASSES)
        for seed in range(5):
            candidate_all = np.load(verified(base / f"seed_{seed}", "care.npy"),
                                    mmap_mode="r")
            if len(candidate_all) != len(y_all):
                raise ValueError(f"Frozen CARE prediction length differs: {home} {seed}")
            candidate = np.asarray(candidate_all, dtype=np.int64)
            full_delta = 100 * (macro_f1(y_all[selected], candidate[selected],
                                        CLASSES) - source_f1)
            pair_strata, pair_mass = make_strata(source, candidate)
            for design in ("uniform", "prediction_pair_stratified"):
                if design == "uniform":
                    strata = np.zeros(len(y_all), dtype=np.int64)
                    mass = np.array([len(y_all)], dtype=np.int64)
                    allotment = np.array([args.audit_labels], dtype=np.int64)
                else:
                    strata, mass = pair_strata, pair_mass
                    allotment = allocate(mass, args.audit_labels)
                rng = np.random.default_rng(
                    770000 + 100 * home_index + seed +
                    (0 if design == "uniform" else 10000))
                result = sampled_audit(y_all, source, candidate, strata, mass,
                                       allotment, args.bootstrap_draws, rng)
                low, high = result["bootstrap_low_pp"], result["bootstrap_high_pp"]
                decision = ("positive" if low > 0 else
                            "negative" if high < 0 else "unresolved")
                rows.append(dict(home=home, seed=seed, design=design,
                                 full_macro_delta_pp=full_delta,
                                 groups=len(mass), decision=decision,
                                 interval_contains_full=bool(low <= full_delta <= high),
                                 decision_matches_full_sign=(
                                     bool((decision == "positive" and full_delta > 0) or
                                          (decision == "negative" and full_delta < 0))
                                     if decision != "unresolved" else None),
                                 **result))
            print(home, "seed", seed, "uniform and stratified complete", flush=True)
    results = pd.DataFrame(rows)
    summary = (results.groupby(["home", "design"], as_index=False)
               .agg(mean_full_delta_pp=("full_macro_delta_pp", "mean"),
                    mean_sampled_delta_pp=("sampled_delta_pp", "mean"),
                    resolved_seeds=("decision", lambda x: int((x != "unresolved").sum())),
                    wrong_resolved_seeds=("decision_matches_full_sign",
                                          lambda x: int((x == False).sum())),
                    interval_contains_full_seeds=("interval_contains_full", "sum"),
                    mean_sampled_present_classes=("sampled_present_classes", "mean"),
                    mean_sampled_curated_rows=("sampled_curated_rows", "mean"),
                    mean_sampled_disagreements=("sampled_disagreements", "mean")))
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "per_seed.csv", index=False)
    summary.to_csv(output / "per_home.csv", index=False)
    report = "\n".join([
        "Exploratory 1000-label paired macro-F1 audit replay; CARE versus frozen source.",
        "Development labels are already examined; sampling code uses predictions only, not true labels.",
        "All development rows are eligible; out-of-inventory rows consume labels but not F1 counts.",
        "Prediction-pair strata and allocations use unlabeled frozen predictions only.",
        "Pairs below max(100, ceiling(N/400)) rows are merged by source class; at most 424 strata.",
        "The audit allocates two labels per stratum when that stratum has at least two rows.",
        "Remaining labels use half mass / half square-root-mass allocation; HT expansion recovers each stratum mass.",
        "Both designs use the fixed 24-class macro F1 and the same 1000 additional labels.",
        "Intervals are descriptive within-stratum row bootstraps, not valid temporal or simultaneous safety bounds.",
        "Five seeds per home overlap; final-window data untouched.",
        "Per-home results:",
        summary.round(4).to_string(index=False),
        "Across-home means by design (each home one unit):",
        summary.groupby("design")[list(summary.columns[2:])].mean().round(4).to_string(),
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
