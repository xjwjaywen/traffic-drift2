"""Fixed target-neighbor consensus control for CARE versus 1-NN disagreements.

This is a predefined feasibility control on already examined TMA months, not
a novel method or a blind test. No future labels inform the gate.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ("tma_pilot_v1", "tma_analysis_v1", "tma_paper_v1",
               "tma_selective_repair_v1", "tma_local_geometry_v1"):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader
from data import finish, sha, validate_complete, write_csv, write_json
from pilot import forward_batches, mapped, stats
from probe import paid_rows
from run import check_predictions, context, metrics
from trial import load_source

FOCUS = {"2020-03": (5, 22, 26, 37), "2020-04": (20,)}
K = 3


def consensus_mask(eval_features, paid_features, paid_y, nearest_pred,
                   care_pred, device):
    """Accept 1-NN only if three nearest paid target anchors share its class."""
    known = paid_y >= 0
    if int(known.sum()) < K:
        return np.zeros(len(eval_features), dtype=bool)
    support = F.normalize(torch.as_tensor(paid_features[known], dtype=torch.float32,
                                          device=device), dim=1)
    targets = F.normalize(torch.as_tensor(eval_features, dtype=torch.float32,
                                          device=device), dim=1)
    labels = torch.as_tensor(paid_y[known], dtype=torch.long, device=device)
    votes = []
    with torch.no_grad():
        for chunk in targets.split(4096):
            chosen = (chunk @ support.T).topk(K, dim=1).indices
            top_labels = labels[chosen]
            votes.append(((top_labels == top_labels[:, :1]).all(dim=1)
                          & (top_labels[:, 0] == torch.as_tensor(
                              nearest_pred[len(votes) * 4096:len(votes) * 4096 + len(chunk)],
                              device=device))).cpu().numpy())
    supported = np.concatenate(votes)
    return supported & (nearest_pred != care_pred)


def run(root, source_seed=0, repair_seeds=(0, 1, 2, 3, 4),
        device="cuda", threads=8):
    reader, cache, classes, source_signature = context(root, source_seed)
    nearest = Reader(root / "nearest_anchor_v1" / f"source_{source_seed}")
    hybrids = {seed: Reader(root / "hybrid_adaptive_v1" /
                            f"source_{source_seed}" / f"repair_{seed}")
               for seed in repair_seeds}
    signature = dict(schema=1, source_seed=source_seed,
        repair_seeds=list(repair_seeds), source_signature=source_signature,
        source_marker=sha(reader.root / "complete.json"),
        cache_marker=sha(cache.root / "complete.json"),
        nearest_marker=sha(nearest.root / "complete.json"),
        hybrid_markers={str(s): sha(hybrids[s].root / "complete.json")
                        for s in repair_seeds},
        code_sha256=sha(HERE / "experiment.py"),
        rule="top_3_paid_target_neighbors_unanimous_and_match_1nn",
        random_control="same_override_count_uniform_over_1nn_care_disagreements",
        device=device, threads=threads, months=list(FOCUS))
    output = root / "local_consensus_v1" / f"source_{source_seed}"
    if output.is_symlink() or (output.exists() and any(output.iterdir())
                              and not (output / "config.json").exists()):
        raise ValueError(f"Unrecognized output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if validate_complete(output, signature):
        print("LOCAL CONSENSUS COMPLETE (verified):", output, flush=True)
        return output
    write_json(output / "config.json", signature)
    torch.set_num_threads(threads)
    model, prep = load_source(reader, classes)
    history = reader.arrays("history_predictions.npz", ["labels", "mlp"])
    historical = stats(history["labels"], history["mlp"], len(classes))
    summary_rows, focus_rows, coverage_rows = [], [], []
    for month, focus_classes in FOCUS.items():
        query = cache.arrays(month + "_query.npz", ["ids", "x"])
        future_x = cache.arrays(month + "_eval.npz", ["x"])["x"]
        qfeatures = forward_batches(model.encoder,
                                    prep.transform(query["x"]), device).numpy()
        efeatures = forward_batches(model.encoder,
                                    prep.transform(future_x), device).numpy()
        month_out = output / month
        month_out.mkdir(exist_ok=True)
        for seed in repair_seeds:
            indices, raw_y = paid_rows(hybrids[seed], month, query["ids"], 1000)
            paid_y = mapped(raw_y, classes)
            saved = hybrids[seed].arrays(month + "/predictions.npz",
                                         ["hybrid_care"])
            care = saved["hybrid_care"]
            nn = np.load(nearest.file(month + f"/predictions_seed_{seed}.npy"),
                         allow_pickle=False)
            if len(care) != len(nn) or len(nn) != len(efeatures):
                raise ValueError("Prediction length mismatch")
            accept = consensus_mask(efeatures, qfeatures[indices], paid_y,
                                    nn, care, torch.device(device))
            disagreement = np.flatnonzero(nn != care)
            rng = np.random.default_rng(20261031 + seed + int(month[-2:]) * 100)
            random_accept = np.zeros(len(nn), dtype=bool)
            random_accept[rng.choice(disagreement, int(accept.sum()),
                                     replace=False)] = True
            consensus_pred, random_pred = care.copy(), care.copy()
            consensus_pred[accept] = nn[accept]
            random_pred[random_accept] = nn[random_accept]
            np.savez_compressed(month_out / f"predictions_seed_{seed}.npz",
                                consensus=consensus_pred, random_matched=random_pred,
                                accept=accept, random_accept=random_accept)
            coverage_rows.append(dict(month=month, repair_seed=seed,
                disagreements=len(disagreement), accepted=int(accept.sum()),
                accepted_fraction_of_disagreements=float(accept.sum() / len(disagreement))
                if len(disagreement) else 0.))
            print(month, "seed", seed, "accepted", int(accept.sum()), "/",
                  len(disagreement), "disagreements; predictions fixed", flush=True)
        # The candidate and matched random-control predictions have all been
        # saved before future evaluation labels are opened by this script.
        target = cache.arrays(month + "_eval.npz", ["ids", "y"])
        for seed in repair_seeds:
            base = hybrids[seed].arrays(month + "/predictions.npz",
                ["ids", "labels", "device_ids", "static_mlp", "hybrid_care"])
            y = check_predictions(base, target, classes)
            nn = np.load(nearest.file(month + f"/predictions_seed_{seed}.npy"),
                         allow_pickle=False)
            with np.load(month_out / f"predictions_seed_{seed}.npz",
                         allow_pickle=False) as data:
                candidates = {name: data[name] for name in
                              ("consensus", "random_matched")}
            for name, pred in (("hybrid_care", base["hybrid_care"]),
                               ("nearest_anchor", nn), *candidates.items()):
                summary, per_class, _ = metrics(y, base["static_mlp"], pred,
                                                 historical, classes)
                summary_rows.append(dict(month=month, repair_seed=seed,
                                         method=name, **summary))
                for focus in focus_classes:
                    row = next(row for row in per_class if row["device_id"] == focus)
                    focus_rows.append(dict(month=month, repair_seed=seed,
                                           method=name, **row))
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "focus_classes.csv", focus_rows)
    write_csv(output / "coverage.csv", coverage_rows)
    grouped = pd.DataFrame(summary_rows).groupby(["month", "method"])[
        ["macro_f1_present", "collapse_f1", "high_recall_group_f1",
         "negative_flip_rate_vs_static_mlp"]].mean().reset_index()
    focus = pd.DataFrame(focus_rows).groupby(["month", "device_id", "method"])[
        "after_f1"].mean().reset_index()
    report = "\n".join([
        "Exploratory target-paid 3-neighbor consensus control, fixed before this run.",
        "No extra labels; same 1000 paid queries and existing CARE/1-NN predictions.",
        "Random control accepts the same number of 1-NN/CARE disagreements.",
        "All TMA months were examined earlier; this is not independent validation or a novelty claim.",
        "Whole-model means:", grouped.to_string(index=False),
        "Focus-class F1 means:", focus.to_string(index=False),
        "Accepted disagreement counts:", pd.DataFrame(coverage_rows).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    artifacts = [str(p.relative_to(output)) for p in output.rglob("*")
                 if p.is_file() and p.name not in ("config.json", "complete.json")]
    finish(output, signature, artifacts)
    print(report, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/pilot_v1"))
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--repair-seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    seeds = tuple(sorted(set(int(value) for value in args.repair_seeds.split(","))))
    if not seeds or args.threads < 1:
        parser.error("Require repair seeds and positive threads")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    run(args.root.resolve(strict=True), args.source_seed, seeds,
        args.device, args.threads)


if __name__ == "__main__":
    main()
