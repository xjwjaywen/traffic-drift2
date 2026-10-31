"""Matched-budget frozen-feature 1-NN control on examined TMA months.

The classifier is a standard control, not a proposed new CARE method.
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
from run import check_predictions, context, metrics, output_directory
from trial import load_references, load_source

FOCUS = {"2020-03": (5, 22, 26, 37), "2020-04": (20,)}


def nearest_anchor(features, support_features, support_labels, device):
    """Cosine 1-NN, with a stable support order for exact ties."""
    if len(support_features) != len(support_labels) or not len(support_labels):
        raise ValueError("Support features/labels mismatch")
    support = F.normalize(torch.as_tensor(support_features, dtype=torch.float32,
                                          device=device), dim=1)
    targets = F.normalize(torch.as_tensor(features, dtype=torch.float32,
                                          device=device), dim=1)
    label_tensor = torch.as_tensor(support_labels, dtype=torch.long, device=device)
    outputs = []
    with torch.no_grad():
        for chunk in targets.split(4096):
            chosen = (chunk @ support.T).argmax(1)
            outputs.append(label_tensor[chosen].cpu().numpy())
    return np.concatenate(outputs).astype(np.int64)


def run(root, source_seed=0, repair_seeds=(0, 1, 2, 3, 4),
        device="cuda", threads=8):
    reader, cache, classes, source_signature = context(root, source_seed)
    nonlinear = Reader(root / "nonlinear_head_control_v1" / f"source_{source_seed}")
    papers, hybrids = {}, {}
    for seed in repair_seeds:
        papers[seed] = Reader(output_directory(root, source_seed) / f"repair_{seed}")
        hybrids[seed] = Reader(root / "hybrid_adaptive_v1" / f"source_{source_seed}" /
                               f"repair_{seed}")
    signature = dict(schema=1, source_seed=source_seed,
        repair_seeds=list(repair_seeds), source_signature=source_signature,
        source_marker=sha(reader.root / "complete.json"),
        cache_marker=sha(cache.root / "complete.json"),
        nonlinear_marker=sha(nonlinear.root / "complete.json"),
        paper_markers={str(s): sha(papers[s].root / "complete.json") for s in repair_seeds},
        hybrid_markers={str(s): sha(hybrids[s].root / "complete.json") for s in repair_seeds},
        code_sha256=sha(HERE / "experiment.py"),
        probe_code_sha256=sha(EXPERIMENT / "tma_local_geometry_v1/probe.py"),
        method="cosine_1nn_on_paid_hybrid_labels_plus_5_source_refs_per_class",
        device=device, threads=threads, months=list(FOCUS))
    output = root / "nearest_anchor_v1" / f"source_{source_seed}"
    if output.is_symlink() or (output.exists() and any(output.iterdir())
                              and not (output / "config.json").exists()):
        raise ValueError(f"Unrecognized output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if validate_complete(output, signature):
        print("NEAREST ANCHOR COMPLETE (verified):", output, flush=True)
        return output
    write_json(output / "config.json", signature)
    torch.set_num_threads(threads)
    model, prep = load_source(reader, classes)
    history = reader.arrays("history_predictions.npz", ["labels", "mlp"])
    historical = stats(history["labels"], history["mlp"], len(classes))
    summary_rows, class_rows = [], []
    for month, focus_classes in FOCUS.items():
        query = cache.arrays(month + "_query.npz", ["ids", "x"])
        future_x = cache.arrays(month + "_eval.npz", ["x"])["x"]
        query_features = forward_batches(model.encoder,
                                         prep.transform(query["x"]), device).numpy()
        future_features = forward_batches(model.encoder,
                                          prep.transform(future_x), device).numpy()
        month_output = output / month
        month_output.mkdir(exist_ok=True)
        for seed in repair_seeds:
            indices, paid_raw = paid_rows(hybrids[seed], month, query["ids"], 1000)
            paid_y = mapped(paid_raw, classes)
            ref_x, ref_y = load_references(reader, cache, papers[seed], classes,
                                           source_seed, seed)
            ref_features = forward_batches(model.encoder,
                                           prep.transform(ref_x), device).numpy()
            usable = paid_y >= 0
            support_x = np.concatenate((query_features[indices[usable]], ref_features))
            support_y = np.concatenate((paid_y[usable], ref_y))
            pred = nearest_anchor(future_features, support_x, support_y,
                                  torch.device(device))
            np.save(month_output / f"predictions_seed_{seed}.npy", pred)
            print(month, "seed", seed, "1-NN predictions fixed; paid", len(indices),
                  "known", int(usable.sum()), "references", len(ref_y), flush=True)
        # Evaluate only after all new predictions for this month have been fixed.
        target = cache.arrays(month + "_eval.npz", ["ids", "y"])
        for seed in repair_seeds:
            base = hybrids[seed].arrays(month + "/predictions.npz",
                ["ids", "labels", "device_ids", "static_mlp", "hybrid_care"])
            y = check_predictions(base, target, classes)
            rf = nonlinear.arrays(month + f"/repair_{seed}/predictions.npz",
                                  ["frozen_rf"])["frozen_rf"]
            nearest = np.load(month_output / f"predictions_seed_{seed}.npy",
                              allow_pickle=False)
            if not (len(y) == len(rf) == len(nearest)):
                raise ValueError("Evaluation prediction length mismatch")
            for name, pred in (("static_mlp", base["static_mlp"]),
                               ("hybrid_care", base["hybrid_care"]),
                               ("frozen_rf", rf), ("nearest_anchor", nearest)):
                summary, per_class, _ = metrics(y, base["static_mlp"], pred,
                                                 historical, classes)
                summary_rows.append(dict(month=month, repair_seed=seed,
                                         method=name, **summary))
                for focus in focus_classes:
                    row = next(row for row in per_class if row["device_id"] == focus)
                    class_rows.append(dict(month=month, repair_seed=seed,
                                           method=name, **row))
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "focus_classes.csv", class_rows)
    grouped = pd.DataFrame(summary_rows).groupby(["month", "method"])[
        ["macro_f1_present", "collapse_f1", "high_recall_group_f1",
         "negative_flip_rate_vs_static_mlp"]].mean().reset_index()
    focus = pd.DataFrame(class_rows).groupby(["month", "device_id", "method"])[
        "after_f1"].mean().reset_index()
    report = "\n".join([
        "Exploratory frozen-feature cosine 1-NN control on the same 1000 paid hybrid labels.",
        "All methods use identical paid query identities; 1-NN also uses the same 5/class source references as CARE and RF.",
        "TMA months were previously examined; this is not an independent test or a new method.",
        "Whole-model means:", grouped.to_string(index=False),
        "Focus-class F1 means:", focus.to_string(index=False),
        "A positive binary AP from the geometry probe does not guarantee good multiclass F1.",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    names = [str(p.relative_to(output)) for p in output.rglob("*")
             if p.is_file() and p.name not in ("config.json", "complete.json")]
    finish(output, signature, names)
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
