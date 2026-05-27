"""
Collapse-aware active maintenance MVP for TLS-Year22.

This is a diagnostic active-label experiment, not zero-label TTA. It asks:
given a small label budget in a target period, does querying high-risk samples
around known absorber predictions repair collapse classes better than random
labels?

Usage from Experiment/core_code/:
    python scripts/collapse_active_maintenance_tls22.py \
      --config configs/eval_tls22.yaml \
      --checkpoint outputs/tls22_cnn/best_model.pt \
      --reference-period M-2022-4 \
      --target-period M-2022-12 \
      --output-dir outputs/collapse_active_maintenance_tls22_m12
"""
import argparse
import copy
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import prototype_recalibration_tls22 as proto


DEFAULT_COLLAPSE_CLASSES = [56, 163, 174, 48, 38, 69, 104, 47, 66, 10, 109, 26]
DEFAULT_STABLE_CLASSES = [
    8, 15, 44, 57, 59, 62, 64, 76, 94, 98,
    99, 107, 113, 119, 128, 130, 131, 132, 144, 145,
]
DEFAULT_ABSORBER_CLASSES = [96, 46, 2, 14, 45, 105, 5, 71, 156, 13]
REPLAY_MODES = {"none", "stable", "all", "absorber", "collapse", "stable_absorber"}


def parse_int_list(value, default=None):
    if value is None or str(value).strip() == "":
        return list(default or [])
    return [int(x) for x in str(value).replace(",", " ").split()]


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def period_slug(period):
    return period.lower().replace("m-2022-", "m").replace("-", "_")


def top2_margin(logits):
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def prediction_signals(logits):
    probs = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    margin = top2_margin(logits)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    preds = probs.argmax(dim=1)
    return probs, confidence, margin, entropy, preds


def prototype_distance_signals(features, prototypes, valid_mask):
    feat_n = F.normalize(features, dim=1)
    proto_n = F.normalize(prototypes, dim=1)
    sims = feat_n @ proto_n.t()
    if valid_mask is not None:
        sims[:, ~valid_mask.to(sims.device)] = -1e9
    nearest_proto = sims.argmax(dim=1)
    nearest_distance = 1.0 - sims.max(dim=1).values
    return nearest_distance.cpu(), nearest_proto.cpu()


def normalized_score(values):
    values = values.float()
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    return ((values - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0)


def replay_class_set(mode, num_classes, collapse_classes, stable_classes, absorber_classes):
    if mode == "none":
        return []
    if mode == "stable":
        return [c for c in stable_classes if 0 <= c < num_classes]
    if mode == "absorber":
        return [c for c in absorber_classes if 0 <= c < num_classes]
    if mode == "collapse":
        return [c for c in collapse_classes if 0 <= c < num_classes]
    if mode == "stable_absorber":
        return sorted({
            c for c in list(stable_classes) + list(absorber_classes)
            if 0 <= c < num_classes
        })
    if mode == "all":
        return list(range(num_classes))
    raise ValueError(f"Unknown replay mode: {mode}")


def sample_replay_indices(labels, classes, per_class, seed):
    if per_class <= 0 or not classes:
        return torch.empty(0, dtype=torch.long)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    sampled = []
    for c in classes:
        candidates = torch.nonzero(labels_t == int(c), as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue
        k = min(int(per_class), candidates.numel())
        perm = torch.randperm(candidates.numel(), generator=gen)[:k]
        sampled.append(candidates[perm])
    if not sampled:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(sampled)


def build_head_training_set(
    target_features,
    target_labels,
    ref_features,
    ref_labels,
    replay_idx,
    target_repeat,
):
    target_repeat = max(1, int(target_repeat))
    feature_parts = [target_features] * target_repeat
    label_parts = [torch.as_tensor(target_labels, dtype=torch.long)] * target_repeat
    if replay_idx.numel() > 0:
        feature_parts.append(ref_features[replay_idx])
        label_parts.append(torch.as_tensor(ref_labels[replay_idx.numpy()], dtype=torch.long))
    return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0)


def select_indices(
    strategy,
    logits,
    labels,
    budget,
    num_classes,
    collapse_classes,
    absorber_classes,
    seed,
    nearest_distance=None,
    nearest_proto=None,
):
    n = logits.shape[0]
    budget = min(int(budget), n)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    _, confidence, margin, entropy, preds = prediction_signals(logits)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    def random_from(candidates, k):
        if candidates.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long)
        k = min(k, candidates.numel())
        perm = torch.randperm(candidates.numel(), generator=gen)[:k]
        return candidates[perm]

    def balanced_topk_by_pred(candidates, score, k, descending=False):
        if candidates.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long)
        grouped = []
        for cls in torch.unique(preds[candidates]).tolist():
            cls_candidates = candidates[preds[candidates] == int(cls)]
            if cls_candidates.numel() == 0:
                continue
            order = torch.argsort(score[cls_candidates], descending=descending)
            grouped.append(cls_candidates[order])
        if not grouped:
            return torch.empty(0, dtype=torch.long)
        selected = []
        positions = [0 for _ in grouped]
        while len(selected) < k:
            progressed = False
            for group_idx, group in enumerate(grouped):
                pos = positions[group_idx]
                if pos >= group.numel():
                    continue
                selected.append(group[pos])
                positions[group_idx] += 1
                progressed = True
                if len(selected) >= k:
                    break
            if not progressed:
                break
        if not selected:
            return torch.empty(0, dtype=torch.long)
        return torch.stack(selected).long()

    if strategy == "random":
        return torch.randperm(n, generator=gen)[:budget]

    if strategy == "entropy":
        return torch.argsort(entropy, descending=True)[:budget]

    if strategy == "margin":
        return torch.argsort(margin)[:budget]

    if strategy == "confidence_low":
        return torch.argsort(confidence)[:budget]

    if strategy in {
        "absorber_random",
        "absorber_margin",
        "absorber_margin_balanced",
        "absorber_entropy",
        "absorber_distance",
        "absorber_proto_disagree",
        "hybrid_risk",
    }:
        absorber = torch.zeros(num_classes, dtype=torch.bool)
        absorber[torch.tensor([c for c in absorber_classes if 0 <= c < num_classes])] = True
        candidates = torch.nonzero(absorber[preds], as_tuple=False).squeeze(1)
        if strategy == "absorber_random":
            selected = random_from(candidates, budget)
        elif strategy == "absorber_margin_balanced":
            selected = balanced_topk_by_pred(candidates, margin, budget, descending=False)
        elif strategy == "absorber_margin":
            k = min(budget, candidates.numel())
            order = torch.argsort(margin[candidates])[:k] if k > 0 else torch.empty(0, dtype=torch.long)
            selected = candidates[order] if k > 0 else torch.empty(0, dtype=torch.long)
        elif strategy == "absorber_entropy":
            k = min(budget, candidates.numel())
            order = torch.argsort(entropy[candidates], descending=True)[:k] if k > 0 else torch.empty(0, dtype=torch.long)
            selected = candidates[order] if k > 0 else torch.empty(0, dtype=torch.long)
        elif strategy in {"absorber_distance", "absorber_proto_disagree", "hybrid_risk"}:
            if nearest_distance is None or nearest_proto is None:
                raise ValueError(f"{strategy} requires prototype distance signals")
            nearest_distance = nearest_distance.cpu()
            nearest_proto = nearest_proto.cpu()
            if strategy == "absorber_proto_disagree":
                candidates = candidates[nearest_proto[candidates] != preds[candidates].cpu()]
                score = nearest_distance
            elif strategy == "hybrid_risk":
                distance_score = normalized_score(nearest_distance)
                entropy_score = normalized_score(entropy.cpu())
                low_margin_score = 1.0 - normalized_score(margin.cpu())
                disagree_score = (nearest_proto != preds.cpu()).float()
                score = distance_score + entropy_score + low_margin_score + disagree_score
            else:
                score = nearest_distance
            k = min(budget, candidates.numel())
            order = torch.argsort(score[candidates], descending=True)[:k] if k > 0 else torch.empty(0, dtype=torch.long)
            selected = candidates[order] if k > 0 else torch.empty(0, dtype=torch.long)
    elif strategy == "oracle_collapse_random":
        collapse = torch.zeros(num_classes, dtype=torch.bool)
        collapse[torch.tensor([c for c in collapse_classes if 0 <= c < num_classes])] = True
        candidates = torch.nonzero(collapse[labels_t], as_tuple=False).squeeze(1)
        selected = random_from(candidates, budget)
    elif strategy == "oracle_collapse_margin":
        collapse = torch.zeros(num_classes, dtype=torch.bool)
        collapse[torch.tensor([c for c in collapse_classes if 0 <= c < num_classes])] = True
        candidates = torch.nonzero(collapse[labels_t], as_tuple=False).squeeze(1)
        k = min(budget, candidates.numel())
        order = torch.argsort(margin[candidates])[:k] if k > 0 else torch.empty(0, dtype=torch.long)
        selected = candidates[order] if k > 0 else torch.empty(0, dtype=torch.long)
    else:
        raise ValueError(f"Unknown active strategy: {strategy}")

    if selected.numel() < budget:
        used = torch.zeros(n, dtype=torch.bool)
        used[selected] = True
        topup = random_from(torch.nonzero(~used, as_tuple=False).squeeze(1), budget - selected.numel())
        selected = torch.cat([selected, topup])
    return selected[:budget]


def fit_head(
    model,
    train_features,
    train_labels,
    lr,
    epochs,
    batch_size,
    weight_decay,
    device,
    distill_features=None,
    distill_logits=None,
    distill_weight=0.0,
    distill_temperature=2.0,
):
    head = copy.deepcopy(model.cls_head).to(device)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    x = train_features.to(device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    n = x.shape[0]
    use_distill = (
        distill_weight > 0.0
        and distill_features is not None
        and distill_logits is not None
        and distill_features.shape[0] > 0
    )
    if use_distill:
        dx = distill_features.to(device)
        teacher_logits = distill_logits.to(device)
        dn = dx.shape[0]
        temp = float(distill_temperature)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = F.cross_entropy(head(x[idx]), y[idx])
            if use_distill:
                didx = torch.randint(0, dn, (idx.numel(),), device=device)
                student_log_probs = F.log_softmax(head(dx[didx]) / temp, dim=1)
                teacher_probs = F.softmax(teacher_logits[didx] / temp, dim=1)
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temp ** 2)
                loss = loss + float(distill_weight) * distill_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def predict_head(head, features, device, chunk_size=8192):
    preds = []
    head.eval()
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size].to(device)
        preds.append(head(chunk).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def collapse_counts(report, collapse_classes, recall_threshold, severe_threshold):
    collapsed = 0
    severe = 0
    for c in collapse_classes:
        item = report.get(str(c), {})
        recall = float(item.get("recall", 0.0))
        if recall < recall_threshold:
            collapsed += 1
        if recall < severe_threshold:
            severe += 1
    return collapsed, severe


def summarize(labels, preds, collapse_classes, stable_classes, thresholds):
    row = proto.summarize_predictions(labels, preds, collapse_classes, stable_classes)
    metrics = proto.compute_metrics(labels, preds)
    collapsed, severe = collapse_counts(
        metrics["classification_report"],
        collapse_classes,
        thresholds["collapse"],
        thresholds["severe"],
    )
    row["collapsed_count"] = collapsed
    row["severe_collapsed_count"] = severe
    return row, metrics["classification_report"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-period", default="M-2022-4")
    parser.add_argument("--target-period", default="M-2022-12")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="50,100,200,500,1000")
    parser.add_argument(
        "--strategies",
        default=(
            "random,entropy,margin,absorber_random,absorber_margin,"
            "absorber_distance,absorber_proto_disagree,hybrid_risk,"
            "oracle_collapse_random"
        ),
    )
    parser.add_argument("--collapse-classes", default=None)
    parser.add_argument("--stable-classes", default=None)
    parser.add_argument("--absorber-classes", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--replay-mode",
        choices=sorted(REPLAY_MODES),
        default="none",
        help="Reference-period labeled replay set used while fitting the target head.",
    )
    parser.add_argument(
        "--replay-per-class",
        type=int,
        default=0,
        help="Number of reference samples per replay class. Zero disables replay.",
    )
    parser.add_argument(
        "--target-repeat",
        type=int,
        default=1,
        help="Repeat selected target labels during head fitting to balance replay.",
    )
    parser.add_argument(
        "--replay-distill-weight",
        type=float,
        default=0.0,
        help="KL distillation weight on replay samples against the frozen source head.",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Temperature for replay distillation.",
    )
    parser.add_argument("--min-prototype-support", type=int, default=1)
    parser.add_argument("--collapse-recall-threshold", type=float, default=0.1)
    parser.add_argument("--severe-recall-threshold", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, num_classes = proto.load_source_model(args.checkpoint, device)
    eval_cfg = proto.load_config(args.config)
    eval_cfg["data"]["num_classes"] = num_classes

    collapse_classes = parse_int_list(args.collapse_classes, DEFAULT_COLLAPSE_CLASSES)
    stable_classes = parse_int_list(args.stable_classes, DEFAULT_STABLE_CLASSES)
    absorber_classes = parse_int_list(args.absorber_classes, DEFAULT_ABSORBER_CLASSES)
    budgets = parse_int_list(args.budgets)
    strategies = [s for s in args.strategies.replace(",", " ").split() if s]
    thresholds = {
        "collapse": args.collapse_recall_threshold,
        "severe": args.severe_recall_threshold,
    }

    print(f"Using device: {device}")
    print(f"Reference period: {args.reference_period}")
    print(f"Target period: {args.target_period}")
    print(f"Num classes: {num_classes}")
    print(f"Collapse classes: {collapse_classes}")
    print(f"Absorber classes: {absorber_classes}")
    print(
        f"Replay: mode={args.replay_mode} per_class={args.replay_per_class} "
        f"target_repeat={args.target_repeat} "
        f"distill_weight={args.replay_distill_weight}"
    )

    ref_loader, ref_classes = proto.make_test_loader(eval_cfg, args.reference_period)
    if ref_classes != num_classes:
        print(f"WARNING: reference loader classes={ref_classes}, model classes={num_classes}")
    ref_outputs = proto.collect_outputs(
        model, ref_loader, device, desc=f"Reference {args.reference_period}"
    )
    prototypes, proto_support, valid_mask = proto.build_prototypes(
        ref_outputs["features"],
        ref_outputs["labels"],
        num_classes,
        args.min_prototype_support,
    )
    print(
        f"Built reference prototypes for {int(valid_mask.sum())}/{num_classes} classes "
        f"(min_support={args.min_prototype_support})."
    )
    replay_classes = replay_class_set(
        args.replay_mode,
        num_classes,
        collapse_classes,
        stable_classes,
        absorber_classes,
    )
    replay_idx = sample_replay_indices(
        ref_outputs["labels"],
        replay_classes,
        args.replay_per_class,
        args.seed + 10007,
    )
    if replay_idx.numel() > 0:
        print(
            f"Using {int(replay_idx.numel())} replay samples from "
            f"{len(replay_classes)} reference classes."
        )
    replay_features = ref_outputs["features"][replay_idx] if replay_idx.numel() > 0 else None
    replay_logits = None
    if replay_features is not None and args.replay_distill_weight > 0.0:
        with torch.no_grad():
            model.cls_head.eval()
            replay_logits = model.cls_head(replay_features.to(device)).cpu()

    loader, loader_classes = proto.make_test_loader(eval_cfg, args.target_period)
    if loader_classes != num_classes:
        print(f"WARNING: loader classes={loader_classes}, model classes={num_classes}")
    outputs = proto.collect_outputs(
        model, loader, device, desc=f"Collect {args.target_period}"
    )
    features = outputs["features"]
    logits = outputs["logits"]
    labels = outputs["labels"]
    static_preds = logits.argmax(dim=1).numpy()
    nearest_distance, nearest_proto = prototype_distance_signals(features, prototypes, valid_mask)

    rows = []
    per_class_rows = []
    selected_rows = []

    static_summary, static_report = summarize(
        labels, static_preds, collapse_classes, stable_classes, thresholds
    )
    rows.append({
        "method": "static",
        "strategy": "",
        "budget": 0,
        "replay_mode": args.replay_mode,
        "replay_per_class": args.replay_per_class,
        "target_repeat": args.target_repeat,
        "replay_distill_weight": args.replay_distill_weight,
        "distill_temperature": args.distill_temperature,
        "replay_samples": int(replay_idx.numel()),
        "selected_collapse_labels": 0,
        "selected_absorber_preds": 0,
        **static_summary,
    })

    for strategy in strategies:
        for budget in budgets:
            idx = select_indices(
                strategy,
                logits,
                labels,
                budget,
                num_classes,
                collapse_classes,
                absorber_classes,
                args.seed,
                nearest_distance=nearest_distance,
                nearest_proto=nearest_proto,
            )
            selected_labels = labels[idx.numpy()]
            selected_preds = static_preds[idx.numpy()]
            selected_distance = nearest_distance[idx].numpy()
            train_features, train_labels = build_head_training_set(
                features[idx],
                selected_labels,
                ref_outputs["features"],
                ref_outputs["labels"],
                replay_idx,
                args.target_repeat,
            )
            head = fit_head(
                model,
                train_features,
                train_labels,
                args.ft_lr,
                args.ft_epochs,
                args.ft_batch_size,
                args.ft_weight_decay,
                device,
                distill_features=replay_features,
                distill_logits=replay_logits,
                distill_weight=args.replay_distill_weight,
                distill_temperature=args.distill_temperature,
            )
            preds = predict_head(head, features, device)
            summary, report = summarize(labels, preds, collapse_classes, stable_classes, thresholds)

            selected_collapse = int(np.isin(selected_labels, collapse_classes).sum())
            selected_absorber = int(np.isin(selected_preds, absorber_classes).sum())
            rows.append({
                "method": "active_head_ft",
                "strategy": strategy,
                "budget": int(idx.numel()),
                "replay_mode": args.replay_mode,
                "replay_per_class": args.replay_per_class,
                "target_repeat": args.target_repeat,
                "replay_distill_weight": args.replay_distill_weight,
                "distill_temperature": args.distill_temperature,
                "replay_samples": int(replay_idx.numel()),
                "head_train_samples": int(train_features.shape[0]),
                "selected_collapse_labels": selected_collapse,
                "selected_absorber_preds": selected_absorber,
                "selected_mean_proto_distance": float(selected_distance.mean()) if selected_distance.size else "",
                **summary,
            })
            for c in collapse_classes:
                item = report.get(str(c), {})
                per_class_rows.append({
                    "strategy": strategy,
                    "budget": int(idx.numel()),
                    "class_id": c,
                    "support": int(item.get("support", 0)),
                    "recall": float(item.get("recall", 0.0)),
                    "f1": float(item.get("f1-score", 0.0)),
                })
            for c in range(num_classes):
                count = int((selected_labels == c).sum())
                if count:
                    selected_rows.append({
                        "strategy": strategy,
                        "budget": int(idx.numel()),
                        "class_id": c,
                        "selected_count": count,
                    })
            print(
                f"{strategy:<22} budget={int(idx.numel()):4d} "
                f"macro={summary['overall_macro_f1']:.4f} "
                f"collapse={summary['bad_macro_f1']:.4f} "
                f"stable={summary['stable_macro_f1']:.4f} "
                f"collapsed={summary['collapsed_count']} "
                f"sel_collapse={selected_collapse} "
                f"replay={int(replay_idx.numel())} "
                f"distill={args.replay_distill_weight:g}"
            )

    write_csv(os.path.join(args.output_dir, "results_by_budget.csv"), rows)
    write_csv(
        os.path.join(args.output_dir, f"per_collapse_class_{period_slug(args.target_period)}.csv"),
        per_class_rows,
    )
    write_csv(os.path.join(args.output_dir, "selected_class_counts.csv"), selected_rows)
    summary = {
        "target_period": args.target_period,
        "reference_period": args.reference_period,
        "num_classes": num_classes,
        "collapse_classes": collapse_classes,
        "stable_classes": stable_classes,
        "absorber_classes": absorber_classes,
        "budgets": budgets,
        "strategies": strategies,
        "seed": args.seed,
        "replay_mode": args.replay_mode,
        "replay_per_class": args.replay_per_class,
        "target_repeat": args.target_repeat,
        "replay_distill_weight": args.replay_distill_weight,
        "distill_temperature": args.distill_temperature,
        "replay_samples": int(replay_idx.numel()),
        "replay_classes": replay_classes,
        "min_prototype_support": args.min_prototype_support,
        "num_valid_prototypes": int(valid_mask.sum()),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best = max(
        [r for r in rows if r["method"] != "static"],
        key=lambda r: (r["bad_macro_f1"], r["overall_macro_f1"]),
    )
    print("\n=== Collapse Active Maintenance Summary ===")
    print(
        f"static macro={static_summary['overall_macro_f1']:.4f} "
        f"collapse={static_summary['bad_macro_f1']:.4f} "
        f"stable={static_summary['stable_macro_f1']:.4f} "
        f"collapsed={static_summary['collapsed_count']}"
    )
    print(
        f"best strategy={best['strategy']} budget={best['budget']} "
        f"macro={best['overall_macro_f1']:.4f} "
        f"collapse={best['bad_macro_f1']:.4f} "
        f"stable={best['stable_macro_f1']:.4f} "
        f"collapsed={best['collapsed_count']}"
    )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
