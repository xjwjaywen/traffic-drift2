#!/usr/bin/env python3
"""Post-hoc failure localization for the frozen acquisition pilot. No new queries/training."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

import kbs_acquisition_pilot as pilot
import kbs_supplement as kbs

SEED_FILES = ["resolved_config.json", "cohorts.csv", "per_class.csv", "scout_pairs.json"]
EVAL_FILES = ["resolved_config.json", "summary.csv", "by_seed.csv", "per_class.csv", "paired_by_seed.csv", "report.md", "pool.json"]


def verify_inputs(args):
    original = pilot.read_json(Path(args.pilot_dir) / "pilot_manifest.json")["protocol"]
    # Verify using the ORIGINAL pilot runtime options. Diagnosis has its own options/identity.
    original_args = argparse.Namespace(study_dir=args.study_dir, cache_dir=args.cache_dir,
        checkpoint=args.checkpoint, output_dir=args.pilot_dir, seeds=",".join(map(str, original["seeds"])),
        device=original["device"], batch_size=original["batch_size"], threads=original["threads"])
    sources = pilot.load_sources(original_args)
    if sources[3] != original:
        raise ValueError("Original pilot identity changed; restore its code/environment/inputs")
    out = Path(args.output_dir).resolve()
    for source in (args.pilot_dir, args.study_dir, args.cache_dir):
        root = Path(source).resolve()
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("Diagnosis output must be separate from all source directories")
    seeds = original["seeds"] if args.seeds is None else kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(original["seeds"]):
        raise ValueError("Diagnosis seeds must be a nonempty unique subset of completed pilot seeds")
    choices, hashes = {}, {}
    for seed in original["seeds"]:
        directory = Path(args.pilot_dir) / f"seed_{seed}"
        if not pilot.completed(directory, pilot.seed_signature(original, seed),
                               ["resolved_config.json", "selection.json", "common_excluded_ids.json"]):
            raise ValueError(f"Missing complete pilot seed {seed}")
        hashes[str(seed)] = kbs.file_sha(directory / "selection.json")
        choices[seed] = pilot.read_json(directory / "selection.json")
    evaluation = Path(args.pilot_dir) / "evaluation"
    if not pilot.completed(evaluation, {"pilot": kbs.digest(original), "selection_sha256": hashes}, EVAL_FILES):
        raise ValueError("Run the original pilot evaluation before diagnosis")
    identity = {"schema": "care-acquisition-diagnosis-v1", "pilot": kbs.digest(original),
                "evaluation_sha256": kbs.file_sha(evaluation / "complete.json"),
                "selection_sha256": hashes, "seeds": seeds,
                "diagnosis_sha256": kbs.file_sha(Path(__file__)), "device": args.device,
                "batch_size": args.batch_size, "threads": args.threads,
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "scout_fraction": .2, "status": "post-hoc diagnostic only; no new query policy"}
    return sources, choices, identity


def stats(y, head, proto, confidence, rows, classes, collapse):
    """Counts and rates use the supplied cohort only, including empty cohorts."""
    yy, hh, aa, cc = y[rows], head[rows], proto[rows], confidence[rows]
    n = len(yy)
    hs = kbs.class_stats(kbs.confusion(yy, hh, classes))
    ps = kbs.class_stats(kbs.confusion(yy, aa, classes))
    support = hs[3]
    hc, pc = hh == yy, aa == yy
    high_error = (~hc) & (cc >= pilot.SETTINGS["high_confidence_threshold"])
    victim = np.isin(yy, collapse)
    counts = {"n": n, "head_correct": int(hc.sum()), "proto_correct": int(pc.sum()),
              "both_correct": int((hc & pc).sum()), "head_only_correct": int((hc & ~pc).sum()),
              "proto_only_correct": int((~hc & pc).sum()), "both_wrong": int((~hc & ~pc).sum()),
              "high_conf_errors": int(high_error.sum()), "collapse_samples": int(victim.sum()),
              "collapse_high_conf_errors": int((victim & high_error).sum()),
              "classes_covered": int((support > 0).sum()),
              "collapse_classes_covered": int((support[collapse] > 0).sum())}
    counts.update({"head_accuracy": counts["head_correct"] / n if n else None,
                   "proto_accuracy": counts["proto_correct"] / n if n else None,
                   "high_conf_error_fraction": counts["high_conf_errors"] / n if n else None,
                   "collapse_fraction": counts["collapse_samples"] / n if n else None,
                   "head_supported_macro_recall": float(hs[1][support > 0].mean()) if n else None,
                   "proto_supported_macro_recall": float(ps[1][support > 0].mean()) if n else None})
    per_class = [{"class_id": c, "true_support": int(support[c]),
                  "head_predicted": int(hs[4][c]), "proto_predicted": int(ps[4][c]),
                  "head_recall": float(hs[1][c]) if support[c] else None,
                  "proto_recall": float(ps[1][c]) if support[c] else None,
                  "is_original_collapse": int(c in collapse)} for c in range(classes)]
    return counts, per_class


def phases(choice, n, budget):
    ids = pilot.check_ids(choice["row_indices"], n, budget)
    parts = {"all": ids}
    if "specialist_count" in choice:
        e, s, f = [choice[k] for k in ("exploration_count", "specialist_count", "fallback_count")]
        if any(not isinstance(v, int) or v < 0 for v in [e, s, f]) or e + s + f != budget:
            raise ValueError("Invalid stored selection phase sizes")
        parts.update(exploration=ids[:e], specialist=ids[e:e+s], fallback=ids[e+s:])
    return parts


def candidate_pool(base, inferred, risk, permutation, method, exploration, budget):
    factor = np.ones(len(base))
    if method == "risk_disagreement":
        factor += pilot.SETTINGS["risk_strength"] * risk[inferred]
    elif method == "shuffled_risk":
        factor += pilot.SETTINGS["risk_strength"] * risk[permutation][inferred]
    # Preserve v1 dtype for plain disagreement to reproduce its ranking exactly.
    score = base if method == "disagreement" else base * factor
    allowed = score > 0
    allowed[exploration] = False
    ids = np.flatnonzero(allowed)
    order = np.lexsort((ids, -score[ids]))
    return ids[order[:pilot.SETTINGS["candidate_multiplier"] * budget]]


def scout_pairs(y, head, ids, classes):
    """Only first 20% of saved BADGE IDs supply confirmed errors; no retrieval yet."""
    cm = kbs.confusion(y[ids], head[ids], classes)
    return [{"true_class": int(a), "predicted_absorber": int(b), "confirmed_errors": int(cm[a, b])}
            for a, b in zip(*np.nonzero(cm)) if a != b]


def diagnose_seed(args, sources, selected, seed, reference, target, summaries):
    info, protocol, _, original = sources
    classes, budget = info["num_classes"], original["budget"]
    yref, y = reference["labels"].numpy(), target["labels"].numpy()
    rpred, rconf, rcounts, pred, conf, tcounts = summaries
    prototypes, ids = pilot.make_prototypes(reference["features"], yref, classes, seed)
    if ids.tolist() != selected["reference_row_indices"]:
        raise ValueError("Reference prototype IDs differ from the frozen selection")
    device = kbs.device_for(args.device)
    print(f"seed {seed}: checking reference geometry excluding {len(ids)} prototype rows...", flush=True)
    rproto, _ = pilot.prototype_scores(reference["features"], prototypes, rpred, rconf, device, args.batch_size)
    print(f"seed {seed}: reconstructing target candidate funnel...", flush=True)
    proto, base = pilot.prototype_scores(target["features"], prototypes, pred, conf, device, args.batch_size)
    risk = pilot.frequency_risk(rcounts, tcounts)
    np.testing.assert_allclose(risk, selected["risk"], rtol=0, atol=1e-12)
    permutation = np.asarray(selected["risk_permutation"], dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(classes)):
        raise ValueError("Invalid saved risk permutation")
    collapse = protocol["collapse_classes"]
    rows, per_class = [], []
    def add(name, indices, role="target"):
        yy, hh, aa, cc = (yref, rpred, rproto, rconf) if role == "reference" else (y, pred, proto, conf)
        metrics, class_rows = stats(yy, hh, aa, cc, indices, classes, collapse)
        rows.append({"seed": seed, "cohort": name, **metrics})
        per_class.extend({"seed": seed, "cohort": name, **r} for r in class_rows)
    reference_mask = np.ones(len(yref), dtype=bool)
    reference_mask[ids] = False
    add("reference_heldout", np.flatnonzero(reference_mask), "reference")
    add("target_all", np.arange(len(y)))
    add("target_disagreement", np.flatnonzero(base > 0))
    add("target_high_conf_disagreement", np.flatnonzero((base > 0) & (conf >= .9)))
    for method in pilot.METHODS:
        choice = selected["choices"][method]
        split = phases(choice, len(y), budget)
        for phase, indices in split.items():
            add(f"{method}/{phase}", indices)
        if "specialist" in split:
            query = split["all"]
            if not np.array_equal(proto[query], choice["inferred_class"]):
                raise ValueError("Recomputed prototype decisions differ from saved decisions")
            np.testing.assert_allclose(base[query], choice["base_score"], rtol=1e-4, atol=1e-6)
            pool = candidate_pool(base, proto, risk, permutation, method, split["exploration"], budget)
            if len(pool) != choice["capped_candidates"] or not np.isin(split["specialist"], pool).all():
                raise ValueError("Reconstructed candidate pool differs from frozen selection")
            add(f"{method}/candidate_pool", pool)
    scout_n = max(1, int(.2 * budget))
    scout = np.asarray(selected["choices"]["badge"]["row_indices"][:scout_n], dtype=np.int64)
    add("badge/scout_first_20pct", scout)
    pairs = scout_pairs(y, pred, scout, classes)
    return rows, per_class, {"seed": seed, "labels_consumed": scout_n, "pairs": pairs,
                             "confirmed_error_count": sum(r["confirmed_errors"] for r in pairs),
                             "distinct_error_pairs": len(pairs)}


def reconcile(rows, saved_rows, seed):
    expected = {r["method"]: r for r in saved_rows if int(r["seed"]) == seed}
    mapping = {"n": "budget", "high_conf_errors": "high_conf_error_count",
               "collapse_samples": "collapse_query_count", "classes_covered": "all_classes_covered",
               "collapse_classes_covered": "collapse_classes_covered"}
    for method in pilot.METHODS:
        actual = next(r for r in rows if r["cohort"] == f"{method}/all")
        for key, old in mapping.items():
            if actual[key] != int(expected[method][old]):
                raise ValueError(f"Diagnosis disagrees with original evaluation: {seed}/{method}/{key}")
        if method in ("disagreement", "risk_disagreement", "shuffled_risk"):
            specialist = next(r for r in rows if r["cohort"] == f"{method}/specialist")
            if specialist["proto_correct"] != int(expected[method]["specialist_inferred_correct_count"]):
                raise ValueError("Specialist correctness differs from original evaluation")


def aggregate(out, identity):
    rows, scout = [], []
    for seed in identity["seeds"]:
        directory = out / f"seed_{seed}"
        if not pilot.completed(directory, {"diagnosis": kbs.digest(identity), "seed": seed}, SEED_FILES):
            raise ValueError(f"Incomplete diagnosis seed {seed}")
        for row in kbs.read_csv(directory / "cohorts.csv"):
            rows.append({k: (v if k == "cohort" else float(v) if v else None) for k, v in row.items()})
        payload = pilot.read_json(directory / "scout_pairs.json")
        scout.append({k: v for k, v in payload.items() if k != "pairs"})
    summary = []
    for name in dict.fromkeys(r["cohort"] for r in rows):
        group = [r for r in rows if r["cohort"] == name]
        item = {"cohort": name, "n_seeds": len(group)}
        for key in group[0]:
            if key in ("seed", "cohort"):
                continue
            vals = [r[key] for r in group if r[key] is not None]
            item[key + "_mean"] = statistics.mean(vals) if vals else None
            item[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        summary.append(item)
    kbs.write_csv(out / "summary.csv", summary)
    kbs.write_csv(out / "scout_summary.csv", scout)
    key_cohorts = ["reference_heldout", "target_all", "target_disagreement", "target_high_conf_disagreement",
                   "badge/all", "badge/scout_first_20pct", "random/all",
                   "disagreement/candidate_pool", "disagreement/specialist",
                   "risk_disagreement/candidate_pool", "risk_disagreement/exploration", "risk_disagreement/specialist"]
    lines = ["# 分歧选样失效诊断（事后分析，不训练）", "",
             "参考评估排除了构建原型的每类五个样本。候选池按原选样规则重建，未生成新查询。",
             "所有目标标签仅用于事后分析；下列各队列不是同一测试分布，不能将条件准确率视为全局准确率。", "",
             "| 队列 | 样本数 | 头准确率 | 原型准确率 | 崩溃类样本数 | 崩溃类比例 | 高置信错误数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"
    for name in key_cohorts:
        r = next(x for x in summary if x["cohort"] == name)
        lines.append(f"| {name} | {fmt(r['n_mean'], 1)} | {fmt(r['head_accuracy_mean'])} | {fmt(r['proto_accuracy_mean'])} | "
                     f"{fmt(r['collapse_samples_mean'], 1)} | {fmt(r['collapse_fraction_mean'])} | {fmt(r['high_conf_errors_mean'], 1)} |")
    lines += ["", "判读顺序：",
              "1. reference_heldout 已经很差：每类五例＋单一余弦均值原型可能不适合当前特征；不能直接归因于时间漂移。",
              "2. 参考表现好、target_all 明显差：与跨期几何失配一致，但该差值本身不是因果证明。",
              "3. target_disagreement 尚可、candidate_pool 更差：分数/截断可能富集了错误原型；candidate_pool 到 specialist 再下降则检查多样性抽样。",
              "4. 对照 exploration 和 specialist 的崩溃类比例，而非只看样本数；若专项不及共享随机探索，就没有定向发现收益。",
              "5. per_class.csv 对照 proto_predicted 与 true_support，检查是否存在被大量指向的原型类别；包括全部类别与零支持行。", "",
              "## 下一方向的起点：BADGE 前 20% 查询中的已确认错误", "",
              "| seed | 已用标签数 | 确认错分数 | 不同错分对数 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {r['seed']} | {r['labels_consumed']} | {r['confirmed_error_count']} | {r['distinct_error_pairs']} |" for r in scout)
    lines += ["", "这里仅统计侦察阶段是否发现足够错误；尚未搜索邻居，也未证明定向补采有效。",
              "下一候选可考虑在总预算内先用少量标签确认错分对，再从这些目标样本附近寻找同类错误；仍须保留探索来发现未被侦察覆盖的类别。",
              "区域主动学习和基于反馈的失效发现已有相关研究，二阶段流程本身不是创新证明。",
              "M12 结果属于开发性探索，不用于声称独立验证；新方案需冻结后另行验证。", ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def run(args):
    sources, choices, identity = verify_inputs(args)
    if args.mode == "preflight":
        kbs.device_for(args.device)
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with kbs.file_lock(out / ".lock"):
        manifest = out / "diagnosis_manifest.json"
        if manifest.exists() and pilot.read_json(manifest)["protocol"] != identity:
            raise ValueError("Diagnosis changed; use a new DIAG_OUTPUT_DIR")
        if not manifest.exists():
            kbs.atomic_json(manifest, {"protocol": identity, "environment": kbs.runtime_info()})
        pending = [s for s in identity["seeds"] if not pilot.completed(out / f"seed_{s}",
                    {"diagnosis": kbs.digest(identity), "seed": s}, SEED_FILES)]
        if pending:
            print("Loading existing reference/target cache for retrospective diagnosis...", flush=True)
            reference, target = [torch.load(Path(args.cache_dir) / f"{role}.pt", map_location="cpu", weights_only=True)
                                 for role in ("reference", "target")]
            classes = sources[0]["num_classes"]
            for role, payload in [("reference", reference), ("target", target)]:
                n = sources[0]["sample_counts"][role]
                y = payload["labels"].numpy()
                if len(payload["features"]) != n or payload["logits"].shape != (n, classes) or y.shape != (n,) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= classes)):
                    raise ValueError(f"Invalid {role} cache shapes/labels")
            summaries = (*pilot.logits_summary(reference["logits"], args.batch_size),
                         *pilot.logits_summary(target["logits"], args.batch_size))
            saved_rows = kbs.read_csv(Path(args.pilot_dir) / "evaluation/by_seed.csv")
            for seed in pending:
                rows, per_class, scout = diagnose_seed(args, sources, choices[seed], seed, reference, target, summaries)
                reconcile(rows, saved_rows, seed)
                directory = out / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                signature = {"diagnosis": kbs.digest(identity), "seed": seed}
                kbs.atomic_json(directory / "resolved_config.json", signature)
                kbs.write_csv(directory / "cohorts.csv", rows)
                kbs.write_csv(directory / "per_class.csv", per_class)
                kbs.atomic_json(directory / "scout_pairs.json", scout)
                kbs.finish(directory, signature, SEED_FILES)
                print(f"seed {seed}: reconciled with pilot evaluation; diagnosis saved.", flush=True)
        final_signature = {"diagnosis": kbs.digest(identity), "seed_completions": {
            str(s): kbs.file_sha(out / f"seed_{s}/complete.json") for s in identity["seeds"]}}
        final_files = ["resolved_config.json", "summary.csv", "scout_summary.csv", "report.md"]
        if not pilot.completed(out, final_signature, final_files):
            kbs.atomic_json(out / "resolved_config.json", final_signature)
            aggregate(out, identity)
            kbs.finish(out, final_signature, final_files)
        print(f"Results: {out / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "preflight"], nargs="?", default="run")
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--pilot-dir", default=str(kbs.CORE / "outputs/kbs_acquisition_pilot_v1"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_acquisition_diagnosis_v1"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", default=None, help="Default: all original pilot seeds")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--threads", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args)


if __name__ == "__main__":
    main()
