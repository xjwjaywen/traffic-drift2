#!/usr/bin/env python3
"""Exploratory acquisition audit. Selection has no target-label argument; no training."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

import kbs_supplement as kbs

METHODS = ("badge", "margin", "random", "disagreement", "risk_disagreement", "shuffled_risk")
SETTINGS = {
    "schema": "care-acquisition-pilot-v1",
    "reference_per_class": 5,
    "prototype": "mean of unit-normalized reference features, then unit-normalize",
    "prototype_temperature": 0.1,
    "risk": "positive predicted-frequency drop with Jeffreys smoothing 0.5",
    "risk_strength": 2.0,
    "exploration_fraction": 0.5,
    "candidate_multiplier": 20,
    "specialist_score": "q(prototype_argmax) * p(head_argmax) on disagreements",
    "diversity": "score-weighted k-means++ on unit features in a capped candidate pool",
    "high_confidence_threshold": 0.9,
    "status": "exploratory; no repair or novelty claim; target labels only for retrospective evaluation",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def check_ids(ids, n, count):
    raw = np.asarray(ids)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("Query IDs must be a one-dimensional integer array")
    if len(raw) != count or len(np.unique(raw)) != count or np.any((raw < 0) | (raw >= n)):
        raise ValueError("Query IDs have the wrong budget, duplicates, or out-of-range rows")
    return raw.astype(np.int64)


def completed(directory, signature, expected):
    path = Path(directory) / "complete.json"
    if path.exists() and set(read_json(path).get("artifacts", {})) != set(expected):
        raise ValueError(f"Incomplete artifact manifest: {path}")
    return kbs.is_complete(directory, signature)


def load_sources(args):
    """Verify immutable cache and saved selections. No target labels are interpreted."""
    study, cache = Path(args.study_dir).resolve(), Path(args.cache_dir).resolve()
    out = Path(args.output_dir).resolve()
    if out == study or study in out.parents or out in study.parents or out == cache or cache in out.parents or out in cache.parents:
        raise ValueError("Pilot output must be separate from the original study/cache (use a sibling directory)")
    info = read_json(cache / "manifest.json")
    protocol = read_json(study / "study_manifest.json")["protocol"]
    if info["spec"]["schema"] != kbs.SCHEMA or info["spec"]["loader_sha256"] != kbs.loader_sha():
        raise ValueError("Cache schema/loader identity mismatch")
    if set(info["files"]) != {"head.pt", "reference.pt", "target.pt"}:
        raise ValueError("Cache does not contain the complete expected file inventory")
    if info["fingerprint"] != kbs.digest({"spec": info["spec"], "inputs": info["input_stream_sha256"]}):
        raise ValueError("Invalid cache fingerprint")
    if protocol["cache"] != info["fingerprint"] or protocol["checkpoint_sha256"] != info["checkpoint_sha256"]:
        raise ValueError("Study and cache are not paired")
    if protocol["implementation_sha256"] != kbs.implementation_sha():
        raise ValueError("The original controlled-v1 implementation has changed")
    if kbs.file_sha(args.checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("Source checkpoint differs from cached checkpoint")
    if protocol["num_classes"] != info["num_classes"]:
        raise ValueError("Class universe differs between cache and study")
    print("Verifying existing cache hashes (no extraction or training)...", flush=True)
    kbs.validate_cache_files(cache, info)
    n, budget = info["sample_counts"]["target"], protocol["budget"]
    if not 1 < budget < n:
        raise ValueError("Budget must be greater than one and smaller than the pool")
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32 - 20000 for s in seeds):
        raise ValueError("Use unique nonnegative seeds within the NumPy seed range")
    saved, hashes = {}, {}
    for seed in seeds:
        saved[seed] = {}
        for method in ("badge", "margin"):
            directory = study / "selections" / method / f"seed_{seed}"
            signature = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
                         "selector": method, "budget": budget, "seed": seed}
            if not completed(directory, signature, ["resolved_config.json", "selection.json"]):
                raise ValueError(f"Missing verified existing selection: {directory}")
            saved[seed][method] = check_ids(read_json(directory / "selection.json")["row_indices"], n, budget)
            hashes[f"{method}/{seed}"] = kbs.file_sha(directory / "selection.json")
    identity = {"settings": SETTINGS, "methods": list(METHODS), "seeds": seeds,
                "budget": budget, "cache_fingerprint": info["fingerprint"],
                "cache_manifest_sha256": kbs.file_sha(cache / "manifest.json"),
                "study_manifest_sha256": kbs.file_sha(study / "study_manifest.json"),
                "baseline_selection_sha256": hashes,
                "pilot_sha256": kbs.file_sha(Path(__file__)),
                "engine_sha256": kbs.implementation_sha(), "device": args.device,
                "batch_size": args.batch_size, "threads": args.threads,
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    return info, protocol, saved, identity


def logits_summary(logits, batch_size):
    pred, confidence = [], []
    counts = np.zeros(logits.shape[1], dtype=np.int64)
    for start in range(0, len(logits), batch_size):
        block = logits[start:start + batch_size].float()
        if not torch.isfinite(block).all():
            raise ValueError("Non-finite cached logits")
        p = block.argmax(1).numpy()
        pred.append(p)
        confidence.append(block.softmax(1).max(1).values.numpy())
        counts += np.bincount(p, minlength=logits.shape[1])
    return np.concatenate(pred), np.concatenate(confidence), counts


def frequency_risk(reference_counts, target_counts):
    a, b = np.asarray(reference_counts, dtype=float), np.asarray(target_counts, dtype=float)
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Invalid predicted-class counts")
    pr, pt = (a + .5) / (a.sum() + .5 * len(a)), (b + .5) / (b.sum() + .5 * len(b))
    return np.clip(1.0 - pt / pr, 0., 1.)


def make_prototypes(reference_features, reference_labels, classes, seed):
    ids = kbs.replay_indices(reference_labels, classes, SETTINGS["reference_per_class"], seed)
    x = reference_features[torch.from_numpy(ids)].float()
    if not torch.isfinite(x).all() or torch.any(x.norm(dim=1) <= 1e-12):
        raise ValueError("Invalid/zero reference feature used for a prototype")
    x = F.normalize(x, dim=1).reshape(classes, SETTINGS["reference_per_class"], -1).mean(1)
    if torch.any(x.norm(dim=1) <= 1e-12):
        raise ValueError("Degenerate class prototype")
    return F.normalize(x, dim=1), ids


@torch.no_grad()
def prototype_scores(features, prototypes, head_pred, head_conf, device, batch_size):
    """No labels or label-derived collapse groups accepted by this API."""
    proto = prototypes.to(device)
    inferred, score = [], []
    for start in range(0, len(features), batch_size):
        x = features[start:start + batch_size].to(device=device, dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite target features")
        valid = x.norm(dim=1) > 1e-12
        q = (F.normalize(x, dim=1) @ proto.T / SETTINGS["prototype_temperature"]).softmax(1)
        conf, a = q.max(1)
        a, conf, valid = a.cpu().numpy(), conf.cpu().numpy(), valid.cpu().numpy()
        stop = start + len(a)
        base = conf * head_conf[start:stop] * (a != head_pred[start:stop]) * valid
        inferred.append(a)
        score.append(base)
    return np.concatenate(inferred), np.concatenate(score)


@torch.no_grad()
def diverse_query(features, scores, budget, seed, device):
    """Shared random exploration; score-weighted geometric diversity for the remainder."""
    n = len(features)
    rng = np.random.default_rng(seed + 2003)
    explore_n = int(budget * SETTINGS["exploration_fraction"])
    explore = rng.choice(n, explore_n, replace=False)
    allowed = np.ones(n, dtype=bool)
    allowed[explore] = False
    candidates = np.flatnonzero(allowed & (scores > 0))
    # Stable row-ID tie breaking and the same cap in all ablation arms.
    ordered = np.lexsort((candidates, -scores[candidates]))
    cap = SETTINGS["candidate_multiplier"] * budget
    pool = candidates[ordered[:cap]]
    slots = min(budget - explore_n, len(pool))
    picked = []
    if slots:
        x = F.normalize(features[torch.from_numpy(pool)].to(device=device, dtype=torch.float32), dim=1)
        weights = scores[pool].astype(np.float64)
        available = np.ones(len(pool), dtype=bool)
        min_distance = np.ones(len(pool), dtype=np.float64)
        for step in range(slots):
            mass = weights * min_distance * available
            if mass.sum() <= 1e-15:
                mass = weights * available  # Duplicated/identical features still yield unique IDs.
            j = int(rng.choice(len(pool), p=mass / mass.sum()))
            picked.append(int(pool[j]))
            available[j] = False
            distance = (2.0 - 2.0 * (x @ x[j])).clamp_min(0).cpu().numpy()
            min_distance = distance.astype(np.float64) if step == 0 else np.minimum(min_distance, distance)
    chosen = np.concatenate([explore, np.asarray(picked, dtype=np.int64)])
    allowed[chosen] = False
    missing = budget - len(chosen)
    fallback = rng.choice(np.flatnonzero(allowed), missing, replace=False) if missing else np.array([], dtype=np.int64)
    ids = check_ids(np.concatenate([chosen, fallback]), n, budget)
    return {"row_indices": ids.tolist(), "exploration_count": explore_n,
            "specialist_count": slots, "fallback_count": missing,
            "positive_candidates": len(candidates), "capped_candidates": len(pool)}


def select_seed(reference_features, reference_labels, target_features, head_pred, head_conf,
                risk, classes, budget, seed, saved, device, batch_size):
    prototypes, ref_ids = make_prototypes(reference_features, reference_labels, classes, seed)
    inferred, base = prototype_scores(target_features, prototypes, head_pred, head_conf, device, batch_size)
    permutation = np.random.default_rng(seed + 3001).permutation(classes)
    scores = {"disagreement": base,
              "risk_disagreement": base * (1 + SETTINGS["risk_strength"] * risk[inferred]),
              "shuffled_risk": base * (1 + SETTINGS["risk_strength"] * risk[permutation][inferred])}
    choices = {m: {"row_indices": saved[m].tolist()} for m in ("badge", "margin")}
    choices["random"] = {"row_indices": np.random.default_rng(seed + 4001).choice(len(target_features), budget, replace=False).tolist()}
    for method, score in scores.items():
        started = time.perf_counter()
        choices[method] = diverse_query(target_features, score, budget, seed, device)
        choices[method]["selection_wall_s"] = time.perf_counter() - started
        ids = np.asarray(choices[method]["row_indices"], dtype=np.int64)
        choices[method]["inferred_class"] = inferred[ids].tolist()
        choices[method]["base_score"] = base[ids].tolist()
        print(f"seed {seed} {method}: {choices[method]['specialist_count']} specialist + "
              f"{choices[method]['exploration_count']} exploration + {choices[method]['fallback_count']} fallback", flush=True)
    for value in choices.values():
        check_ids(value["row_indices"], len(target_features), budget)
    return {"seed": seed, "reference_row_indices": ref_ids.tolist(), "risk_permutation": permutation.tolist(),
            "risk": risk.tolist(), "choices": choices}


def seed_signature(identity, seed):
    return {"pilot": kbs.digest(identity), "seed": seed}


def select(args, sources):
    info, _, saved, identity = sources
    out = Path(args.output_dir)
    manifest = out / "pilot_manifest.json"
    if manifest.exists() and read_json(manifest)["protocol"] != identity:
        raise ValueError("Pilot protocol/inputs changed. Use a new PILOT_OUTPUT_DIR")
    if not manifest.exists():
        kbs.atomic_json(manifest, {"protocol": identity, "environment": kbs.runtime_info(),
                                   "source_periods": info["periods"]})
    pending = [s for s in identity["seeds"] if not completed(out / f"seed_{s}", seed_signature(identity, s),
                                                            ["resolved_config.json", "selection.json", "common_excluded_ids.json"])]
    if not pending:
        print("All label-free selections verified; nothing to recompute.", flush=True)
        return
    device = kbs.device_for(args.device)
    print("Loading existing cached tensors...", flush=True)
    reference = torch.load(Path(args.cache_dir) / "reference.pt", map_location="cpu", weights_only=True)
    target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True)
    # Existing .pt bundles include labels. Discard them immediately; selector APIs cannot receive them.
    del target["labels"]
    classes, n = info["num_classes"], info["sample_counts"]["target"]
    if len(target["features"]) != n or target["logits"].shape != (n, classes):
        raise ValueError("Target tensor shapes differ from manifest")
    if len(reference["features"]) != info["sample_counts"]["reference"] or reference["logits"].shape != (len(reference["features"]), classes):
        raise ValueError("Reference tensor shapes differ from manifest")
    if reference["features"].shape[1] != target["features"].shape[1]:
        raise ValueError("Reference/target feature dimensions differ")
    _, _, ref_counts = logits_summary(reference["logits"], args.batch_size)
    head_pred, head_conf, target_counts = logits_summary(target["logits"], args.batch_size)
    risk = frequency_risk(ref_counts, target_counts)
    for seed in pending:
        kbs.seed_all(seed)
        print(f"Selecting seed {seed}; target labels discarded, no model fitting...", flush=True)
        directory = out / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        signature = seed_signature(identity, seed)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        result = select_seed(reference["features"], reference["labels"].numpy(), target["features"],
                             head_pred, head_conf, risk, classes, identity["budget"], seed, saved[seed], device, args.batch_size)
        result["reference_predicted_counts"], result["target_predicted_counts"] = ref_counts.tolist(), target_counts.tolist()
        result["cache_fingerprint"] = info["fingerprint"]
        kbs.atomic_json(directory / "selection.json", result)
        union = sorted(set(i for v in result["choices"].values() for i in v["row_indices"]))
        kbs.atomic_json(directory / "common_excluded_ids.json", {"cache_fingerprint": info["fingerprint"],
                                                               "row_indices": union, "remaining_count": n - len(union)})
        kbs.finish(directory, signature, ["resolved_config.json", "selection.json", "common_excluded_ids.json"])
    print("All selections committed. Target-label evaluation is a separate process.", flush=True)


def acquisition_metrics(labels, pred, confidence, ids, classes, collapse, stable, choice):
    """Retrospective audit only; no outputs from this function feed acquisition."""
    ids = check_ids(ids, len(labels), len(ids))
    hc = confidence >= SETTINGS["high_confidence_threshold"]
    wrong = pred != labels
    victim = np.isin(labels, collapse)
    counts = np.bincount(labels[ids], minlength=classes)
    high_wrong = hc & wrong
    hc_count = int(high_wrong[ids].sum())
    collapse_count = int(victim[ids].sum())
    pool_rate = float(high_wrong.mean())
    result = {"budget": len(ids), "error_count": int(wrong[ids].sum()),
              "error_fraction": float(wrong[ids].mean()), "high_conf_error_count": hc_count,
              "high_conf_error_fraction": hc_count / len(ids),
              "high_conf_error_enrichment": hc_count / len(ids) / pool_rate if pool_rate else None,
              "high_conf_error_retrieval_recall": hc_count / int(high_wrong.sum()) if high_wrong.any() else None,
              "collapse_query_count": collapse_count, "collapse_high_conf_error_count": int((victim & high_wrong)[ids].sum()),
              "collapse_classes_covered": int((counts[collapse] > 0).sum()),
              "collapse_classes_with_at_least_5": int((counts[collapse] >= 5).sum()),
              "all_classes_covered": int((counts > 0).sum()),
              "noncollapse_classes_covered": int((counts[np.setdiff1d(np.arange(classes), collapse)] > 0).sum()),
              "stable_classes_covered": int((counts[stable] > 0).sum()),
              "specialist_count": choice.get("specialist_count", 0),
              "specialist_inferred_correct_count": None, "specialist_inferred_accuracy": None}
    if "inferred_class" in choice:
        start, count = choice["exploration_count"], choice["specialist_count"]
        a = np.asarray(choice["inferred_class"])[start:start + count]
        hits = int((a == labels[ids[start:start + count]]).sum())
        result["specialist_inferred_correct_count"] = hits
        result["specialist_inferred_accuracy"] = hits / count if count else None
    return result, counts


def summarize_rows(rows):
    result = []
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        item = {"method": method, "n_seeds": len(group), "seed_ids": ",".join(str(r["seed"]) for r in group)}
        for key in group[0]:
            if key in ("method", "seed"):
                continue
            values = [r[key] for r in group if r[key] is not None]
            item[key + "_mean"] = statistics.mean(values) if values else None
            item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(item)
    return result


def evaluate(args, sources):
    info, protocol, _, identity = sources
    out = Path(args.output_dir)
    if read_json(out / "pilot_manifest.json")["protocol"] != identity:
        raise ValueError("Selection/evaluation protocol mismatch")
    selections = {}
    for seed in identity["seeds"]:
        directory = out / f"seed_{seed}"
        if not completed(directory, seed_signature(identity, seed),
                         ["resolved_config.json", "selection.json", "common_excluded_ids.json"]):
            raise ValueError(f"Every requested seed must finish selection before evaluation: {seed}")
        selections[seed] = read_json(directory / "selection.json")
    # Evaluation can first access target labels only after ALL requested selections are verified.
    signature = {"pilot": kbs.digest(identity), "selection_sha256": {
        str(s): kbs.file_sha(out / f"seed_{s}" / "selection.json") for s in identity["seeds"]}}
    directory = out / "evaluation"
    artifacts = ["resolved_config.json", "summary.csv", "by_seed.csv", "per_class.csv", "paired_by_seed.csv", "report.md", "pool.json"]
    if completed(directory, signature, artifacts):
        print(f"Verified existing report: {directory / 'report.md'}", flush=True)
        return
    target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True)
    labels = target["labels"].numpy()
    classes = info["num_classes"]
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) != info["sample_counts"]["target"] or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid evaluation labels")
    pred, conf, _ = logits_summary(target["logits"], args.batch_size)
    support = np.bincount(labels, minlength=classes)
    collapse, stable = protocol["collapse_classes"], protocol["stable_classes"]
    if any(c < 0 or c >= classes for c in collapse + stable) or len(set(collapse)) != len(collapse):
        raise ValueError("Invalid evaluation-only class groups")
    rows, per_class, pairs = [], [], []
    for seed, result in selections.items():
        for method in METHODS:
            choice = result["choices"][method]
            ids = check_ids(choice["row_indices"], len(labels), identity["budget"])
            metrics, counts = acquisition_metrics(labels, pred, conf, ids, classes, collapse, stable, choice)
            rows.append({"method": method, "seed": seed, **metrics})
            for c in range(classes):
                class_ids = ids[labels[ids] == c]
                per_class.append({"method": method, "seed": seed, "class_id": c,
                                  "pool_support": int(support[c]), "queried": int(counts[c]),
                                  "queried_errors": int((pred[class_ids] != c).sum()),
                                  "queried_high_conf_errors": int(((pred[class_ids] != c) & (conf[class_ids] >= SETTINGS['high_confidence_threshold'])).sum()),
                                  "is_original_collapse": c in collapse, "is_original_stable": c in stable})
        matched = {r["method"]: r for r in rows if r["seed"] == seed}
        for control in ("badge", "disagreement", "shuffled_risk"):
            pairs.append({"seed": seed, "comparison": f"risk_disagreement minus {control}", **{
                key: matched["risk_disagreement"][key] - matched[control][key]
                for key in matched[control] if key not in ("method", "seed") and
                matched[control][key] is not None and matched["risk_disagreement"][key] is not None}})
    summary = summarize_rows(rows)
    directory.mkdir(parents=True, exist_ok=True)
    kbs.atomic_json(directory / "resolved_config.json", signature)
    kbs.write_csv(directory / "summary.csv", summary)
    kbs.write_csv(directory / "by_seed.csv", rows)
    kbs.write_csv(directory / "per_class.csv", per_class)
    kbs.write_csv(directory / "paired_by_seed.csv", pairs)
    kbs.atomic_json(directory / "pool.json", {
        "target_samples": len(labels), "class_support": support.tolist(),
        "high_confidence_threshold": SETTINGS["high_confidence_threshold"],
        "high_conf_errors": int(((pred != labels) & (conf >= SETTINGS['high_confidence_threshold'])).sum()),
        "collapse_groups_from_original_study_evaluation_only": collapse,
        "cache_fingerprint": info["fingerprint"]})
    lines = ["# 选样探索结果（尚未训练修复模型）", "",
             "高置信错误：源模型 softmax ≥ 0.9 且预测错误。以下标签仅用于事后评估。",
             "每个方法预算相同；新方法一半随机探索、一半分歧选样。原型每类仅用原协议的 5 个参考样本。",
             "风险仅为预测频率下降代理，并非原稿完整五信号监测器；真实流量占比变化也可能触发风险。", "",
             "| 方法 | 高置信错误数 | 崩溃类查询数 | 崩溃类覆盖数 | 全类覆盖数 | 专项推断正确率 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        accuracy = r["specialist_inferred_accuracy_mean"]
        lines.append(f"| {r['method']} | {r['high_conf_error_count_mean']:.1f} | {r['collapse_query_count_mean']:.1f} | "
                     f"{r['collapse_classes_covered_mean']:.1f} | {r['all_classes_covered_mean']:.1f} | "
                     + (f"{accuracy:.3f}" if accuracy is not None else "—") + " |")
    lines += ["", "均值/样本标准差见 summary.csv，逐种子差值见 paired_by_seed.csv，所有类别见 per_class.csv。",
              "先比较 risk_disagreement 与 disagreement、shuffled_risk，再与 BADGE 比较；不根据最好的单个种子下结论。",
              "Margin 是确定性已存选样，跨种子重复不是独立证据。原型/随机选样种子也不是独立训练模型或独立月份。",
              "查到更多错分样本不代表修复后 F1 更高，也不能推断没有新崩溃。此报告不做显著性或成功/失败自动判定。",
              "M12 已用于提出此方案，属于开发性探索；若继续，冻结方案后使用未用于选方案/调参的数据验证。",
              "未来修复比较须用每个 seed 的 common_excluded_ids.json 排除所有方法查询并重算基线评估，不能直接拼旧汇总。",
              "没有目标标签参与选样；现有 .pt 文件捆绑标签，选择进程加载后立即丢弃，评价进程在全部选样完成后才解释标签。", ""]
    (directory / "report.md").write_text("\n".join(lines))
    kbs.finish(directory, signature, artifacts)
    print("\n".join(lines), flush=True)
    print(f"Results: {directory / 'summary.csv'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["plan", "preflight", "select", "evaluate"])
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_acquisition_pilot_v1"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--threads", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.mode == "plan":
        print(json.dumps({"settings": SETTINGS, "methods": METHODS, "arguments": vars(args),
                          "budget": "inherited from verified original study (normally 1000)"}, indent=2, ensure_ascii=False))
        return
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = load_sources(args)
    if args.mode == "preflight":
        kbs.device_for(args.device)
        print(json.dumps(sources[3], indent=2, ensure_ascii=False))
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with kbs.file_lock(out / ".lock"):
        (select if args.mode == "select" else evaluate)(args, sources)


if __name__ == "__main__":
    main()
