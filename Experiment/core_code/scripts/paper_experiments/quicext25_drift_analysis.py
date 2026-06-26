"""
QUICEXT-25 drift mechanism analysis.

Step 1: Confusion matrix to discover victim→absorber pairs
Step 2: Fisher ratio at reference vs target to classify mechanism
Step 3: Output CSV + summary for paper

Usage from Experiment/core_code/:
    python scripts/paper_experiments/quicext25_drift_analysis.py \
      --config configs/eval_quicext25.yaml \
      --checkpoint outputs/quicext25_cnn/best_model.pt \
      --output-dir outputs/quicext25_drift_analysis
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

import prototype_recalibration_tls22 as proto

COLLAPSE_CLASSES = [17, 21, 23, 25]


def find_absorber(labels, preds, victim, num_classes):
    """Find the class that absorbs most of victim's predictions."""
    mask = labels == victim
    if mask.sum() == 0:
        return None, 0, 0.0
    victim_preds = preds[mask]
    n_total = int(mask.sum())
    correct = int((victim_preds == victim).sum())
    recall = correct / n_total

    counts = np.zeros(num_classes, dtype=int)
    for p in victim_preds:
        if p != victim:
            counts[p] += 1
    absorber = int(np.argmax(counts))
    absorber_count = int(counts[absorber])
    absorber_frac = absorber_count / n_total
    return absorber, absorber_count, absorber_frac, recall, n_total


def compute_fisher_ratio(features, labels, cls_a, cls_b):
    """Fisher discriminant ratio between two classes."""
    mask_a = labels == cls_a
    mask_b = labels == cls_b
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a < 5 or n_b < 5:
        return None, n_a, n_b
    feat_a = features[mask_a].numpy()
    feat_b = features[mask_b].numpy()
    mu_a = feat_a.mean(axis=0)
    mu_b = feat_b.mean(axis=0)
    spread_a = np.mean(np.linalg.norm(feat_a - mu_a, axis=1))
    spread_b = np.mean(np.linalg.norm(feat_b - mu_b, axis=1))
    dist = np.linalg.norm(mu_a - mu_b)
    return float(dist / (spread_a + spread_b + 1e-8)), n_a, n_b


def compute_margin(logits, labels, cls):
    """Mean margin (top-2 logit gap) for samples of a given class."""
    mask = labels == cls
    if mask.sum() < 5:
        return None
    cls_logits = logits[mask]
    top2 = torch.topk(cls_logits, 2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    return float(np.mean(margin))


def compute_absorber_confidence(logits, labels, victim, absorber):
    """Mean softmax probability of absorber class for victim samples."""
    import torch.nn.functional as F
    mask = labels == victim
    if mask.sum() < 5:
        return None
    probs = F.softmax(logits[mask], dim=1)
    return float(probs[:, absorber].mean())


def main():
    parser = argparse.ArgumentParser(description="QUICEXT-25 drift analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2024-6")
    parser.add_argument("--target-period", default="M-2025-5")
    parser.add_argument("--collapse-classes", default=None,
                        help="Comma-separated collapse class IDs (default: auto-detect)")
    parser.add_argument("--recall-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    # --- Collect reference and target outputs ---
    print(f"=== QUICEXT-25 Drift Mechanism Analysis ===")
    print(f"Reference: {args.reference_period}, Target: {args.target_period}")
    print(f"Num classes: {num_classes}")

    ref_loader, _ = proto.make_test_loader(eval_cfg, args.reference_period)
    ref_out = proto.collect_outputs(model, ref_loader, device,
                                    desc=f"Ref {args.reference_period}")

    tgt_loader, _ = proto.make_test_loader(eval_cfg, args.target_period)
    tgt_out = proto.collect_outputs(model, tgt_loader, device,
                                    desc=f"Tgt {args.target_period}")

    ref_preds = ref_out["logits"].argmax(dim=1).numpy()
    tgt_preds = tgt_out["logits"].argmax(dim=1).numpy()
    ref_labels = np.asarray(ref_out["labels"])
    tgt_labels = np.asarray(tgt_out["labels"])

    # --- Step 1: Identify collapse classes and absorbers ---
    if args.collapse_classes:
        collapse_classes = [int(c) for c in args.collapse_classes.split(",")]
    else:
        collapse_classes = []
        for c in range(num_classes):
            mask = tgt_labels == c
            if mask.sum() < 10:
                continue
            recall = (tgt_preds[mask] == c).sum() / mask.sum()
            if recall < args.recall_threshold:
                collapse_classes.append(c)

    print(f"\n--- Step 1: Collapse Detection (target {args.target_period}) ---")
    print(f"Collapse classes (recall < {args.recall_threshold}): {collapse_classes}")

    print(f"\n--- Step 2: Absorber Identification (confusion matrix) ---")
    print(f"{'Victim':>6} {'#Samples':>8} {'Recall':>7} {'Absorber':>8} "
          f"{'#Absorbed':>9} {'AbsFrac':>8}")
    print("-" * 55)

    pairs = []
    for victim in collapse_classes:
        result = find_absorber(tgt_labels, tgt_preds, victim, num_classes)
        if result is None:
            print(f"  {victim:>4d}  -- no samples --")
            continue
        absorber, abs_count, abs_frac, recall, n_total = result
        pairs.append((victim, absorber))
        print(f"  {victim:>4d} {n_total:>8d} {recall:>7.3f} {absorber:>8d} "
              f"{abs_count:>9d} {abs_frac:>8.3f}")

    # Also show reference period recall for comparison
    print(f"\n--- Reference period recall ({args.reference_period}) ---")
    for victim, absorber in pairs:
        mask = ref_labels == victim
        if mask.sum() == 0:
            print(f"  {victim:>4d}  -- no samples in ref --")
            continue
        ref_recall = (ref_preds[mask] == victim).sum() / mask.sum()
        print(f"  {victim:>4d}  recall={float(ref_recall):.3f}  (n={int(mask.sum())})")

    # --- Step 3: Fisher ratio analysis ---
    print(f"\n--- Step 3: Fisher Discriminant Ratio ---")
    print(f"{'Victim':>6} {'Abs':>4} {'J_ref':>8} {'J_tgt':>8} {'ΔJ%':>8} "
          f"{'Mechanism':>12} {'MarginRef':>9} {'MarginTgt':>9} "
          f"{'AbsConf_r':>9} {'AbsConf_t':>9}")
    print("-" * 100)

    rows = []
    for victim, absorber in pairs:
        fisher_ref, nv_ref, na_ref = compute_fisher_ratio(
            ref_out["features"], ref_out["labels"], victim, absorber)
        fisher_tgt, nv_tgt, na_tgt = compute_fisher_ratio(
            tgt_out["features"], tgt_out["labels"], victim, absorber)

        delta_fisher = None
        if fisher_ref and fisher_tgt and fisher_ref > 0:
            delta_fisher = (fisher_tgt - fisher_ref) / fisher_ref * 100

        if delta_fisher is not None:
            if delta_fisher < -10:
                mechanism = "feature"
            elif delta_fisher > 20:
                mechanism = "boundary"
            else:
                mechanism = "ambiguous"
        else:
            mechanism = "unknown"

        margin_ref = compute_margin(ref_out["logits"], ref_out["labels"], victim)
        margin_tgt = compute_margin(tgt_out["logits"], tgt_out["labels"], victim)
        abs_conf_ref = compute_absorber_confidence(
            ref_out["logits"], ref_out["labels"], victim, absorber)
        abs_conf_tgt = compute_absorber_confidence(
            tgt_out["logits"], tgt_out["labels"], victim, absorber)

        row = {
            "victim": victim,
            "absorber": absorber,
            "mechanism": mechanism,
            "fisher_ref": fisher_ref,
            "fisher_tgt": fisher_tgt,
            "delta_fisher_pct": delta_fisher,
            "n_victim_ref": nv_ref,
            "n_victim_tgt": nv_tgt,
            "n_absorber_ref": na_ref,
            "n_absorber_tgt": na_tgt,
            "margin_ref": margin_ref,
            "margin_tgt": margin_tgt,
            "margin_drop": (margin_ref - margin_tgt) if margin_ref and margin_tgt else None,
            "absorber_conf_ref": abs_conf_ref,
            "absorber_conf_tgt": abs_conf_tgt,
            "absorber_conf_rise": (abs_conf_tgt - abs_conf_ref) if abs_conf_ref and abs_conf_tgt else None,
        }
        rows.append(row)

        def _f(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(f"  {victim:>4d} {absorber:>4d} {_f(fisher_ref):>8} {_f(fisher_tgt):>8} "
              f"{_f(delta_fisher, '+.1f'):>7}% {mechanism:>12} "
              f"{_f(margin_ref):>9} {_f(margin_tgt):>9} "
              f"{_f(abs_conf_ref, '.4f'):>9} {_f(abs_conf_tgt, '.4f'):>9}")

    # --- Step 4: Per-class recall at multiple time points ---
    print(f"\n--- Step 4: Collapse Timeline (all test periods) ---")
    test_periods = eval_cfg.get("data", {}).get("test_periods", [])
    if test_periods:
        timeline = {}
        for period in test_periods:
            try:
                p_loader, _ = proto.make_test_loader(eval_cfg, period)
                p_out = proto.collect_outputs(model, p_loader, device, desc=f"Timeline {period}")
                p_preds = p_out["logits"].argmax(dim=1).numpy()
                p_labels = np.asarray(p_out["labels"])
                period_recalls = {}
                for victim, _ in pairs:
                    mask = p_labels == victim
                    if mask.sum() > 0:
                        period_recalls[victim] = float((p_preds[mask] == victim).sum() / mask.sum())
                    else:
                        period_recalls[victim] = None
                timeline[period] = period_recalls
            except Exception as e:
                print(f"  {period}: error — {e}")

        header = f"{'Period':<12}" + "".join(f"  cls_{v:>3d}" for v, _ in pairs)
        print(header)
        print("-" * len(header))
        for period in test_periods:
            if period in timeline:
                vals = "".join(
                    f"  {timeline[period].get(v, 'N/A'):>7.3f}"
                    if timeline[period].get(v) is not None else "      N/A"
                    for v, _ in pairs
                )
                print(f"{period:<12}{vals}")

    # --- Save outputs ---
    csv_path = os.path.join(args.output_dir, "quicext25_drift_analysis.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")

    summary = {
        "dataset": "CESNET-QUICEXT-25",
        "reference_period": args.reference_period,
        "target_period": args.target_period,
        "num_classes": num_classes,
        "collapse_classes": collapse_classes,
        "pairs": [(v, a) for v, a in pairs],
        "mechanisms": {str(r["victim"]): r["mechanism"] for r in rows},
    }
    import json
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print(f"\n=== Analysis Complete ===")
    print(f"Pairs: {pairs}")
    mechs = [r['mechanism'] for r in rows]
    for m in set(mechs):
        print(f"  {m}: {mechs.count(m)} classes")


if __name__ == "__main__":
    main()
