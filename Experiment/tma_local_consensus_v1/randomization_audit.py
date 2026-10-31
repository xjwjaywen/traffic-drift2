"""Post-hoc matched randomization audit of the local-consensus control.

Evaluation labels enter this diagnostic. Its tail areas are descriptive only:
the TMA months and the three-neighbor rule were chosen after earlier results.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for folder in ("tma_pilot_v1", "tma_analysis_v1", "tma_paper_v1"):
    sys.path.insert(0, str(HERE.parent / folder))
from analyze import Reader
from pilot import stats
from run import check_predictions, context, metrics


def score(y, static, pred, historical, classes):
    """Return the three decision metrics using the paper's group definition."""
    n = len(classes)
    before = stats(y, static, n)
    after = stats(y, pred, n)
    present = after["support"] > 0
    learned = ((before["support"] >= 50) & (historical["support"] >= 50)
               & (historical["recall"] >= .8))
    collapsed = learned & (before["recall"] < .1)
    previously_correct = (y >= 0) & (static == y)
    return np.array([
        after["f1"][present].mean(),
        after["f1"][collapsed].mean() if collapsed.any() else np.nan,
        (pred[previously_correct] != y[previously_correct]).mean(),
    ])


def pair_groups(care, nearest, accept):
    groups = []
    for source, destination in np.unique(
            np.stack((care[accept], nearest[accept]), axis=1), axis=0):
        pair = (care == source) & (nearest == destination)
        groups.append((np.flatnonzero(pair), int((accept & pair).sum())))
    return groups


def run(root, source_seed, seeds, draws, random_seed):
    if draws < 20:
        raise ValueError("At least 20 random draws are needed")
    source, cache, classes, _ = context(root, source_seed)
    consensus = Reader(root / "local_consensus_v1" / f"source_{source_seed}")
    nearest = Reader(root / "nearest_anchor_v1" / f"source_{source_seed}")
    history = source.arrays("history_predictions.npz", ["labels", "mlp"])
    historical = stats(history["labels"], history["mlp"], len(classes))
    rng = np.random.default_rng(random_seed)
    records = []
    aggregate = {}
    for month in ("2020-03", "2020-04"):
        target = cache.arrays(month + "_eval.npz", ["ids", "y"])
        for seed in seeds:
            hybrid = Reader(root / "hybrid_adaptive_v1" /
                            f"source_{source_seed}" / f"repair_{seed}")
            base = hybrid.arrays(month + "/predictions.npz",
                ["ids", "labels", "device_ids", "static_mlp", "hybrid_care"])
            y = check_predictions(base, target, classes)
            care = base["hybrid_care"]
            nn = np.load(nearest.file(month + f"/predictions_seed_{seed}.npy"),
                         allow_pickle=False)
            saved = consensus.arrays(month + f"/predictions_seed_{seed}.npz",
                                     ["consensus", "accept", "random_matched"])
            accept = saved["accept"]
            if len(nn) != len(care) or not np.array_equal(
                    saved["consensus"], np.where(accept, nn, care)):
                raise ValueError("Saved consensus predictions do not match mask")
            observed = score(y, base["static_mlp"], saved["consensus"],
                             historical, classes)
            expected, _, _ = metrics(y, base["static_mlp"],
                                     saved["consensus"], historical, classes)
            check = np.array([expected[k] for k in (
                "macro_f1_present", "collapse_f1",
                "negative_flip_rate_vs_static_mlp")], dtype=float)
            if not np.allclose(observed, check, equal_nan=True):
                raise ValueError("Diagnostic metrics do not match paper metrics")
            groups = pair_groups(care, nn, accept)
            random_scores = np.empty((draws, 3))
            for draw in range(draws):
                selected = np.zeros(len(care), dtype=bool)
                for candidates, count in groups:
                    selected[rng.choice(candidates, count, replace=False)] = True
                pred = care.copy()
                pred[selected] = nn[selected]
                random_scores[draw] = score(y, base["static_mlp"], pred,
                                             historical, classes)
            aggregate.setdefault(month, []).append((observed, random_scores))
            for j, name in enumerate(("macro_f1_present", "collapse_f1",
                                      "negative_flip_rate_vs_static_mlp")):
                values = random_scores[:, j]
                records.append(dict(month=month, repair_seed=seed, metric=name,
                    consensus=float(observed[j]), random_mean=float(values.mean()),
                    random_p05=float(np.quantile(values, .05)),
                    random_p95=float(np.quantile(values, .95)),
                    consensus_minus_random_mean=float(observed[j] - values.mean()),
                    fraction_random_at_least_consensus=float(
                        (values >= observed[j]).mean())))
            print(month, "seed", seed, "audited", draws, "matched draws", flush=True)
    frame = pd.DataFrame(records)
    summary_rows = []
    for month, results in aggregate.items():
        observed_mean = np.mean([item[0] for item in results], axis=0)
        random_means = np.mean([item[1] for item in results], axis=0)
        for j, name in enumerate(("macro_f1_present", "collapse_f1",
                                  "negative_flip_rate_vs_static_mlp")):
            values = random_means[:, j]
            summary_rows.append(dict(month=month, metric=name,
                consensus=float(observed_mean[j]), random_mean=float(values.mean()),
                random_p05=float(np.quantile(values, .05)),
                random_p95=float(np.quantile(values, .95)),
                consensus_minus_random_mean=float(observed_mean[j] - values.mean()),
                fraction_random_at_least_consensus=float(
                    (values >= observed_mean[j]).mean())))
    grouped = pd.DataFrame(summary_rows)
    out = root / "local_consensus_v1" / f"source_{source_seed}" / "randomization_audit"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_seed.csv", index=False)
    grouped.to_csv(out / "summary.csv", index=False)
    report = "\n".join([
        "Post-hoc matched randomization audit; future labels used for evaluation.",
        "Random draws match accepted counts within every CARE-to-1-NN prediction pair.",
        f"Draws per seed={draws}; random seed={random_seed}.",
        "Tail fractions are descriptive, not confirmatory p-values or novelty evidence.",
        "Across-seed mean with paired draw distribution:", grouped.to_string(index=False),
        "Per-seed results:", frame.to_string(index=False),
    ]) + "\n"
    (out / "REPORT.txt").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/pilot_v1"))
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--repair-seeds", default="0,1,2,3,4")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20261031)
    args = parser.parse_args()
    seeds = tuple(sorted(set(int(s) for s in args.repair_seeds.split(","))))
    run(args.root.resolve(strict=True), args.source_seed, seeds, args.draws,
        args.random_seed)


if __name__ == "__main__":
    main()
