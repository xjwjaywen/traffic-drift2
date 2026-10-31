"""Fixed paid-label nearest-neighbor ranking probe on examined TMA months.

This is an exploratory binary geometry diagnostic, not a deployable repair.
Only the 1000 previously paid hybrid labels per seed act as target anchors.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for folder in ("tma_pilot_v1", "tma_analysis_v1", "tma_paper_v1",
               "tma_selective_repair_v1"):
    sys.path.insert(0, str(EXPERIMENT / folder))
from analyze import Reader
from pilot import forward_batches
from run import context
from trial import load_source

FOCUS = {"2020-03": (5, 22, 26, 37), "2020-04": (20,)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paid_rows(hybrid, month, query_ids, budget):
    table = pd.read_csv(hybrid.file(month + "/hybrid_queries.csv"),
                        dtype={"record_id": str})
    if len(table) != budget or table.record_id.duplicated().any():
        raise ValueError(f"Unexpected paid budget or duplicate ID: {month}")
    lookup = {str(key): i for i, key in enumerate(query_ids)}
    if len(lookup) != len(query_ids):
        raise ValueError("Query cache has duplicate IDs")
    try:
        indices = np.asarray([lookup[key] for key in table.record_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("Paid query ID absent from cache") from exc
    return indices, table.device_id.to_numpy(dtype=np.int64)


def local_scores(eval_features, paid_features, labels, focus, device):
    positive = labels == focus
    if not positive.any() or positive.all():
        return None
    support = F.normalize(torch.as_tensor(paid_features, dtype=torch.float32,
                                          device=device), dim=1)
    future = F.normalize(torch.as_tensor(eval_features, dtype=torch.float32,
                                         device=device), dim=1)
    pos_mask = torch.as_tensor(positive, device=device)
    results = []
    with torch.no_grad():
        for chunk in future.split(4096):
            similarity = chunk @ support.T
            nearest_pos = similarity[:, pos_mask].max(1).values
            nearest_neg = similarity[:, ~pos_mask].max(1).values
            results.append(torch.stack((nearest_pos, nearest_pos - nearest_neg),
                                       dim=1).cpu().numpy())
    return np.concatenate(results)


def run(root, source_seed, repair_seeds, device, threads):
    reader, cache, classes, _ = context(root, source_seed)
    torch.set_num_threads(threads)
    source, preprocessor = load_source(reader, classes)
    mapping = {device_id: i for i, device_id in enumerate(classes)}
    output = root / "local_geometry_probe_v1" / f"source_{source_seed}"
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Output directory must be new and empty: {output}")
    rows = []
    for month, focus_classes in FOCUS.items():
        query = cache.arrays(month + "_query.npz", ["ids", "x"])
        # Evaluation labels are opened only after every ranking score for the
        # month has been computed from paid labels and frozen features.
        future_x = cache.arrays(month + "_eval.npz", ["x"])["x"]
        qfeatures = forward_batches(source.encoder,
                                    preprocessor.transform(query["x"]), device).numpy()
        efeatures = forward_batches(source.encoder,
                                    preprocessor.transform(future_x), device).numpy()
        source_logits = forward_batches(source.head, efeatures, device)
        source_probs = source_logits.softmax(1).numpy()
        frozen_scores = []
        for seed in repair_seeds:
            hybrid = Reader(root / "hybrid_adaptive_v1" / f"source_{source_seed}" /
                            f"repair_{seed}")
            indices, paid_y = paid_rows(hybrid, month, query["ids"], 1000)
            for focus in focus_classes:
                if focus not in mapping:
                    raise ValueError(f"Focus class absent from source: {focus}")
                scores = local_scores(efeatures, qfeatures[indices], paid_y,
                                      focus, torch.device(device))
                frozen_scores.append((seed, focus, int((paid_y == focus).sum()), scores))
                print(month, "seed", seed, "class", focus,
                      "paid positives", int((paid_y == focus).sum()), flush=True)
        future_y = cache.arrays(month + "_eval.npz", ["y"])["y"]
        for seed, focus, paid_positives, scores in frozen_scores:
            truth = future_y == focus
            common = dict(month=month, repair_seed=seed, device_id=focus,
                          paid_positives=paid_positives,
                          future_positives=int(truth.sum()), future_rows=len(truth),
                          prevalence=float(truth.mean()),
                          source_ap=float(average_precision_score(
                              truth, source_probs[:, mapping[focus]])))
            if scores is None:
                rows.append(dict(**common, nearest_positive_ap=None,
                                 positive_minus_negative_ap=None))
            else:
                rows.append(dict(**common,
                    nearest_positive_ap=float(average_precision_score(truth, scores[:, 0])),
                    positive_minus_negative_ap=float(average_precision_score(truth, scores[:, 1]))))
    output.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "scores.csv", index=False)
    mean = table.groupby(["month", "device_id"])[
        ["paid_positives", "prevalence", "source_ap", "nearest_positive_ap",
         "positive_minus_negative_ap"]].mean().reset_index()
    report = "\n".join([
        "TMA exploratory paid-label geometry probe; not a new method result.",
        "Focus classes and months were previously examined. No independent validation.",
        "Only 1000 existing paid hybrid labels per seed are used as target anchors.",
        "Scores are fixed nearest-positive cosine similarity and positive-minus-negative margin.",
        "Future labels are used only for AP evaluation; AP does not establish multiclass F1 or a usable threshold.",
        "A class with zero paid positives has no target-anchor score.",
        "Across-seed means:", mean.to_string(index=False),
        "See scores.csv for all seeds and source-probability AP reference.",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    (output / "manifest.txt").write_text(
        f"code_sha256={sha256(Path(__file__))}\n"
        f"source_complete_sha256={sha256(reader.root / 'complete.json')}\n"
        f"cache_complete_sha256={sha256(cache.root / 'complete.json')}\n")
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
        parser.error("Require at least one repair seed and positive threads")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    run(args.root.resolve(strict=True), args.source_seed, seeds,
        args.device, args.threads)


if __name__ == "__main__":
    main()
