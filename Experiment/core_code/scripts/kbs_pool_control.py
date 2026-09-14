#!/usr/bin/env python3
"""Fixed risk-pool sampling controls. Exploratory selection and separate label audit; no training."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

import kbs_acquisition_diagnosis as diag
import kbs_acquisition_pilot as pilot
import kbs_supplement as kbs

CONTROLS = ("pool_uniform", "pool_score", "pool_distance")
METHODS = ("badge", "random", "original_risk", *CONTROLS)
SETTINGS = {
    "schema": "care-fixed-risk-pool-v1",
    "pool": "reconstruct original risk_disagreement cap, scores and exploration unchanged",
    "sampling": "without replacement; uniform, score-only, distance-only; saved score*distance control",
    "rng": "seed+2003, consume and verify original exploration draw, then one choice per specialist",
    "fallback": "reuse original fallback; permitted only when specialist exhausts the pool",
    "status": "M12 development diagnostic; no repair, generalization or novelty claim",
}
SEED_FILES = ["resolved_config.json", "selection.json", "candidate_pool.json", "common_excluded_ids.json"]
EVAL_FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "per_class.csv",
              "paired_by_seed.csv", "uniform_expectation.csv", "report.md"]


def verify_inputs(args):
    original = pilot.read_json(Path(args.pilot_dir) / "pilot_manifest.json")["protocol"]
    # Use the original numerical settings, not a different device/batch reconstruction.
    for key in ("device", "batch_size", "threads"):
        value = getattr(args, key)
        if value is not None and value != original[key]:
            raise ValueError(f"Fixed-pool reconstruction requires original {key}={original[key]}")
        setattr(args, key, original[key])
    sources, choices, verified = diag.verify_inputs(args)
    identity = {"settings": SETTINGS, "methods": list(METHODS), "seeds": verified["seeds"],
                "original_pilot": verified["pilot"], "original_selection_sha256": verified["selection_sha256"],
                "original_evaluation_sha256": verified["evaluation_sha256"],
                "implementation_sha256": kbs.file_sha(Path(__file__)),
                "diagnosis_sha256": kbs.file_sha(Path(diag.__file__)),
                "device": args.device, "batch_size": args.batch_size, "threads": args.threads,
                "budget": original["budget"]}
    return sources, choices, identity


def signature(identity, seed):
    return {"pool_control": kbs.digest(identity), "seed": seed}


@torch.no_grad()
def sample_pool(features, pool, weights, count, exploration, seed, device, policy):
    """No target labels or collapse IDs accepted. All policies use the identical fixed pool."""
    n = len(features)
    pool = pilot.check_ids(pool, n, len(pool))
    exploration = pilot.check_ids(exploration, n, len(exploration))
    weights = np.asarray(weights, dtype=np.float64)
    if policy not in (*CONTROLS, "original_risk") or not 0 <= count <= len(pool):
        raise ValueError("Invalid sampling policy/count")
    if weights.shape != pool.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Pool scores must be positive and finite")
    if np.intersect1d(pool, exploration).size:
        raise ValueError("Pool intersects shared exploration")
    rng = np.random.default_rng(seed + 2003)
    if not np.array_equal(rng.choice(n, len(exploration), replace=False), exploration):
        raise ValueError("Shared exploration differs from original RNG draw")
    distance_policy = policy in ("pool_distance", "original_risk")
    if distance_policy and count:
        x = features[torch.from_numpy(pool)].to(device=device, dtype=torch.float32)
        if not torch.isfinite(x).all() or torch.any(x.norm(dim=1) <= 1e-12):
            raise ValueError("Invalid fixed-pool features")
        x = F.normalize(x, dim=1)
    base = weights if policy in ("pool_score", "original_risk") else np.ones(len(pool))
    available = np.ones(len(pool), dtype=bool)
    distance = np.ones(len(pool))
    picked = []
    for step in range(count):
        mass = base * distance * available
        if mass.sum() <= 1e-15:
            mass = base * available
        j = int(rng.choice(len(pool), p=mass / mass.sum()))
        picked.append(int(pool[j]))
        available[j] = False
        if distance_policy:
            new_distance = (2.0 - 2.0 * (x @ x[j])).clamp_min(0).cpu().numpy().astype(np.float64)
            distance = new_distance if step == 0 else np.minimum(distance, new_distance)
    return pilot.check_ids(np.asarray(picked, dtype=np.int64), n, count)


def select_seed(reference_features, reference_labels, target_features, head_pred, head_conf,
                risk, classes, budget, seed, original, device, batch_size):
    prototypes, ref_ids = pilot.make_prototypes(reference_features, reference_labels, classes, seed)
    if ref_ids.tolist() != original["reference_row_indices"]:
        raise ValueError("Reference prototype rows changed")
    np.testing.assert_allclose(risk, original["risk"], rtol=0, atol=1e-12)
    inferred, base = pilot.prototype_scores(target_features, prototypes, head_pred, head_conf, device, batch_size)
    old = original["choices"]["risk_disagreement"]
    parts = diag.phases(old, len(target_features), budget)
    if not np.array_equal(inferred[parts["all"]], old["inferred_class"]):
        raise ValueError("Prototype decisions changed")
    np.testing.assert_allclose(base[parts["all"]], old["base_score"], rtol=1e-4, atol=1e-6)
    pool = diag.candidate_pool(base, inferred, risk, np.asarray(original["risk_permutation"]),
                               "risk_disagreement", parts["exploration"], budget)
    slots = min(budget - len(parts["exploration"]), len(pool))
    if len(pool) != old["capped_candidates"] or len(parts["specialist"]) != slots:
        raise ValueError("Reconstructed pool size or specialist budget changed")
    if not np.isin(parts["specialist"], pool).all() or np.intersect1d(pool, parts["fallback"]).size:
        raise ValueError("Original specialist/fallback does not match reconstructed pool")
    if len(parts["fallback"]) and slots != len(pool):
        raise ValueError("Fallback allowed only after exhausting the fixed pool")
    scores = base[pool] * (1 + pilot.SETTINGS["risk_strength"] * risk[inferred[pool]])
    replayed = sample_pool(target_features, pool, scores, slots, parts["exploration"], seed, device, "original_risk")
    if not np.array_equal(replayed, parts["specialist"]):
        raise ValueError("Reconstructed original score*distance query order changed; controls cannot proceed")
    choices = {"badge": dict(original["choices"]["badge"]),
               "random": dict(original["choices"]["random"]), "original_risk": dict(old)}
    for method in CONTROLS:
        started = time.perf_counter()
        picked = sample_pool(target_features, pool, scores, slots, parts["exploration"], seed, device, method)
        ids = np.concatenate([parts["exploration"], picked, parts["fallback"]])
        choices[method] = {"row_indices": ids.tolist(), "exploration_count": len(parts["exploration"]),
                           "specialist_count": slots, "fallback_count": len(parts["fallback"]),
                           "selection_wall_s": time.perf_counter() - started}
        print(f"seed {seed}: {method} selected {slots} specialists from the same {len(pool)} rows", flush=True)
    for choice in choices.values():
        ids = pilot.check_ids(choice["row_indices"], len(target_features), budget)
        choice["inferred_class"] = inferred[ids].tolist()
    return {"seed": seed, "reference_row_indices": ref_ids.tolist(), "choices": choices}, {
        "row_indices": pool.tolist(), "score": scores.tolist(), "inferred_class": inferred[pool].tolist()}


def select(args, verified):
    sources, originals, identity = verified
    info = sources[0]
    out = Path(args.output_dir)
    manifest = out / "pool_control_manifest.json"
    if manifest.exists() and pilot.read_json(manifest)["protocol"] != identity:
        raise ValueError("Pool control protocol changed; use a new POOL_OUTPUT_DIR")
    if not manifest.exists():
        kbs.atomic_json(manifest, {"protocol": identity, "environment": kbs.runtime_info()})
    pending = [s for s in identity["seeds"] if not pilot.completed(out / f"seed_{s}", signature(identity, s), SEED_FILES)]
    if not pending:
        print("All fixed-pool selections verified; nothing to recompute.", flush=True)
        return
    reference = torch.load(Path(args.cache_dir) / "reference.pt", map_location="cpu", weights_only=True)
    target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True)
    # Legacy cache bundles labels. No semantic access: drop them before selecting any new row.
    del target["labels"]
    classes, n = info["num_classes"], info["sample_counts"]["target"]
    if target["logits"].shape != (n, classes) or len(target["features"]) != n:
        raise ValueError("Target shapes differ from original manifest")
    _, _, reference_counts = pilot.logits_summary(reference["logits"], args.batch_size)
    pred, conf, target_counts = pilot.logits_summary(target["logits"], args.batch_size)
    risk = pilot.frequency_risk(reference_counts, target_counts)
    device = kbs.device_for(args.device)
    for seed in pending:
        directory = out / f"seed_{seed}"
        kbs.atomic_json(directory / "resolved_config.json", signature(identity, seed))
        result, pool = select_seed(reference["features"], reference["labels"].numpy(), target["features"],
            pred, conf, risk, classes, identity["budget"], seed, originals[seed], device, args.batch_size)
        kbs.atomic_json(directory / "selection.json", result)
        kbs.atomic_json(directory / "candidate_pool.json", pool)
        # All six original policies plus all new controls, for any FUTURE repair comparison.
        union = sorted({i for group in (originals[seed]["choices"], result["choices"])
                        for choice in group.values() for i in choice["row_indices"]})
        kbs.atomic_json(directory / "common_excluded_ids.json", {"cache_fingerprint": info["fingerprint"],
            "row_indices": union, "remaining_count": n - len(union),
            "scope": "union of all six original pilot methods and all three new controls"})
        kbs.finish(directory, signature(identity, seed), SEED_FILES)
    print("All fixed-pool selections committed. Run evaluation in a separate process.", flush=True)


def cohort_metrics(y, pred, confidence, ids, inferred, classes, collapse, stable):
    """Evaluation only; empty phases and every class remain explicit."""
    n = len(ids)
    yy = y[ids]
    error = pred[ids] != yy
    high_error = error & (confidence[ids] >= pilot.SETTINGS["high_confidence_threshold"])
    victim = np.isin(yy, collapse)
    counts = np.bincount(yy, minlength=classes)
    row = {"n": n, "error_count": int(error.sum()), "error_fraction": float(error.mean()) if n else None,
           "high_conf_error_count": int(high_error.sum()),
           "collapse_query_count": int(victim.sum()), "collapse_fraction": float(victim.mean()) if n else None,
           "collapse_high_conf_error_count": int((victim & high_error).sum()),
           "collapse_classes_covered": int((counts[collapse] > 0).sum()),
           "collapse_classes_with_at_least_5": int((counts[collapse] >= 5).sum()),
           "all_classes_covered": int((counts > 0).sum()),
           "stable_classes_covered": int((counts[stable] > 0).sum()),
           "proto_correct": int((np.asarray(inferred) == yy).sum()),
           "proto_accuracy": float((np.asarray(inferred) == yy).mean()) if n else None}
    per_class = [{"class_id": c, "queried": int(counts[c]),
                  "queried_errors": int(error[yy == c].sum()),
                  "queried_high_conf_errors": int(high_error[yy == c].sum()),
                  "is_original_collapse": c in collapse, "is_original_stable": c in stable}
                 for c in range(classes)]
    return row, per_class


def summarize(rows):
    summary = []
    for method, phase in dict.fromkeys((r["method"], r["phase"]) for r in rows):
        group = [r for r in rows if (r["method"], r["phase"]) == (method, phase)]
        item = {"method": method, "phase": phase, "n_seeds": len(group),
                "seed_ids": ",".join(str(r["seed"]) for r in group)}
        for key in group[0]:
            if key in ("method", "phase", "seed"):
                continue
            values = [r[key] for r in group if r[key] is not None]
            item[key + "_mean"] = statistics.mean(values) if values else None
            item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(item)
    return summary


def evaluate(args, verified):
    sources, _, identity = verified
    info, protocol = sources[:2]
    out = Path(args.output_dir)
    if pilot.read_json(out / "pool_control_manifest.json")["protocol"] != identity:
        raise ValueError("Selection/evaluation protocol mismatch")
    selections, pools, hashes = {}, {}, {}
    for seed in identity["seeds"]:
        directory = out / f"seed_{seed}"
        if not pilot.completed(directory, signature(identity, seed), SEED_FILES):
            raise ValueError(f"Every requested seed must finish selection before evaluation: {seed}")
        selections[seed] = pilot.read_json(directory / "selection.json")
        pools[seed] = pilot.read_json(directory / "candidate_pool.json")
        hashes[str(seed)] = kbs.file_sha(directory / "complete.json")
    evaluation_signature = {"pool_control": kbs.digest(identity), "completed_seeds_sha256": hashes}
    directory = out / "evaluation"
    if pilot.completed(directory, evaluation_signature, EVAL_FILES):
        print(f"Verified existing report: {directory / 'report.md'}", flush=True)
        return
    # First evaluation access to target labels, after all requested seeds have been verified.
    target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True)
    y = target["labels"].numpy()
    classes, n = info["num_classes"], info["sample_counts"]["target"]
    if y.ndim != 1 or y.dtype.kind not in "iu" or len(y) != n or np.any((y < 0) | (y >= classes)):
        raise ValueError("Invalid evaluation labels")
    pred, conf, _ = pilot.logits_summary(target["logits"], args.batch_size)
    collapse, stable = protocol["collapse_classes"], protocol["stable_classes"]
    original_rows = {(int(r["seed"]), r["method"]): r
                     for r in kbs.read_csv(Path(args.pilot_dir) / "evaluation/by_seed.csv")}
    rows, per_class, expectations, pairs = [], [], [], []
    for seed in identity["seeds"]:
        selected = selections[seed]
        for method in METHODS:
            choice = selected["choices"][method]
            all_ids = pilot.check_ids(choice["row_indices"], n, identity["budget"])
            inferred = np.asarray(choice["inferred_class"], dtype=np.int64)
            if inferred.shape != all_ids.shape or np.any((inferred < 0) | (inferred >= classes)):
                raise ValueError("Invalid saved prototype decisions")
            lookup = dict(zip(all_ids, inferred))
            for phase, ids in diag.phases(choice, n, identity["budget"]).items():
                metrics, class_rows = cohort_metrics(y, pred, conf, ids, [lookup[i] for i in ids], classes, collapse, stable)
                rows.append({"seed": seed, "method": method, "phase": phase, **metrics})
                per_class.extend({"seed": seed, "method": method, "phase": phase, **r} for r in class_rows)
                if phase == "all" and method in ("badge", "random", "original_risk"):
                    old = original_rows[seed, "risk_disagreement" if method == "original_risk" else method]
                    for key in ("error_count", "high_conf_error_count", "collapse_query_count",
                                "collapse_classes_covered", "all_classes_covered"):
                        if metrics[key] != int(old[key]):
                            raise ValueError(f"Original evaluation does not reconcile: {seed}/{method}/{key}")
        pool_ids = pilot.check_ids(pools[seed]["row_indices"], n, len(pools[seed]["row_indices"]))
        pool_stats, class_rows = cohort_metrics(y, pred, conf, pool_ids, pools[seed]["inferred_class"], classes, collapse, stable)
        rows.append({"seed": seed, "method": "fixed_risk_pool", "phase": "pool", **pool_stats})
        per_class.extend({"seed": seed, "method": "fixed_risk_pool", "phase": "pool", **r} for r in class_rows)
        parts = diag.phases(selected["choices"]["original_risk"], n, identity["budget"])
        # Hypergeometric mean conditional on each fixed pool, not a measured result or training CI.
        expected = len(parts["specialist"]) * pool_stats["collapse_query_count"] / len(pool_ids) if len(pool_ids) else 0.
        fixed = int(np.isin(y[np.concatenate([parts["exploration"], parts["fallback"]])], collapse).sum())
        expectations.append({"seed": seed, "pool_n": len(pool_ids),
            "pool_collapse_count": pool_stats["collapse_query_count"], "specialist_n": len(parts["specialist"]),
            "analytic_expected_uniform_specialist_collapse": expected,
            "analytic_expected_uniform_total_collapse": expected + fixed})
        matched = {(r["method"], r["phase"]): r for r in rows if r["seed"] == seed}
        for method in CONTROLS:
            for control, phase in (("original_risk", "all"), ("original_risk", "specialist"), ("badge", "all")):
                a, b = matched[method, phase], matched[control, phase]
                pairs.append({"seed": seed, "comparison": f"{method} minus {control}", "phase": phase, **{
                    key: a[key] - b[key] for key in a if key not in ("seed", "method", "phase")
                    and a[key] is not None and b[key] is not None}})
    summary = summarize(rows)
    kbs.atomic_json(directory / "resolved_config.json", evaluation_signature)
    for name, content in (("by_seed", rows), ("summary", summary), ("per_class", per_class),
                          ("paired_by_seed", pairs), ("uniform_expectation", expectations)):
        kbs.write_csv(directory / f"{name}.csv", content)
    lines = ["# 固定候选池抽样对照（未训练修复模型）", "",
        "四个候选方案共享原 risk_disagreement 候选池、分数、500/1000 随机探索比例和总预算；只改变专项抽样规则。",
        "original_risk 复用原分数×距离选样，pool_uniform 均匀抽样，pool_score 仅分数，pool_distance 仅距离。",
        "BADGE/random 复用原全预算查询，仅作为外部参照，不共享候选池。目标标签只在全部选样完成后用于本报告。", "",
        "| 方法 | 总崩溃类查询数 | 崩溃类覆盖数 | 高置信错误数 | 全类覆盖数 | 专项崩溃类查询数 | 专项崩溃类比例 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r = next(x for x in summary if x["method"] == method and x["phase"] == "all")
        s = next((x for x in summary if x["method"] == method and x["phase"] == "specialist"), None)
        special = f"{s['collapse_query_count_mean']:.1f} | {s['collapse_fraction_mean']:.3%}" if s and s["collapse_fraction_mean"] is not None else "— | —"
        lines.append(f"| {method} | {r['collapse_query_count_mean']:.1f} | {r['collapse_classes_covered_mean']:.1f} | "
                     f"{r['high_conf_error_count_mean']:.1f} | {r['all_classes_covered_mean']:.1f} | {special} |")
    expected = statistics.mean(r["analytic_expected_uniform_specialist_collapse"] for r in expectations)
    lines += ["", f"固定候选池均匀专项抽样的崩溃类数量解析期望：{expected:.2f}（逐种子条件期望再求平均，不是实测结果）。",
        "逐种子差值见 paired_by_seed.csv；均值和样本标准差见 summary.csv；全部类别见 per_class.csv。", "",
        "判断：对比 uniform/score/distance 与原分数×距离，区分池内权重和几何抽样的影响；交互作用也可能存在。",
        "若均匀池抽样恢复崩溃样本但仍不及 BADGE，说明抽样损失可修复，尚未证明替代 BADGE 有价值。",
        "若三种对照都无稳定收益，停止这一分支，不继续在 M12 上搜索权重/阈值；另行评估基于已确认错误的反馈方案。",
        "不按最好种子选择赢家，不做自动显著性/成功判定。原型/抽样种子不是独立训练模型或独立月份。",
        "风险仍是预测频率下降代理，不是完整监测器；均匀池对照也仍依赖原型决定的候选池。",
        "这是已看过 M12 结果后的开发性探索，不能当作独立验证、修复 F1、无新增崩溃或创新性证据。",
        "未来若训练修复模型，必须排除本次 common_excluded_ids.json 中原六方法与新增三方法的查询并集，重算基线。",
        "冻结候选方法后，需在未用于选方案/调参的数据上验证。", ""]
    (directory / "report.md").write_text("\n".join(lines))
    kbs.finish(directory, evaluation_signature, EVAL_FILES)
    print("\n".join(lines), flush=True)
    print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["plan", "preflight", "select", "evaluate"])
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--pilot-dir", default=str(kbs.CORE / "outputs/kbs_acquisition_pilot_v1"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_pool_control_v1"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--seeds", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    return p


def main():
    args = parser().parse_args()
    if args.mode == "plan":
        print(json.dumps({"settings": SETTINGS, "methods": METHODS, "arguments": vars(args),
                          "runtime": "inherit original pilot device/batch/threads; overrides must match"}, indent=2, ensure_ascii=False))
        return
    verified = verify_inputs(args)
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    kbs.device_for(args.device)
    if args.mode == "preflight":
        print(json.dumps(verified[2], indent=2, ensure_ascii=False))
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with kbs.file_lock(out / ".lock"):
        (select if args.mode == "select" else evaluate)(args, verified)


if __name__ == "__main__":
    main()
