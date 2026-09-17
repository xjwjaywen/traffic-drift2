#!/usr/bin/env python3
"""Post-hoc diagnosis of learned versus oracle protection, without new training.

Reconciles saved results, expands both exclusions to one common mask, and measures
tail coverage, persistence outside BADGE training, and analytical CE gradients at
three shared heads. Full target truth is used only in this diagnostic process.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

import kbs_learned_protection as learned
import kbs_protection_budget as allocation
import kbs_repair_aware as pilot
import kbs_repair_diagnosis as diagnosis
import kbs_supplement as kbs

ORACLES = ("oracle_probe_damage_p05", "oracle_source_correct_p05")
METHODS = ("badge", *learned.METHODS, *ORACLES)
SETTINGS = {
    "schema": "care-protection-gap-diagnosis-v1",
    "evaluation": "union of both saved study exclusions, including undisplayed historical arms",
    "tail": "last 5% of each saved query; audit any difference in preceding 95%",
    "persistence": "probe damage also wrong after BADGE, excluding ALL BADGE training queries",
    "class_coverage": "common-test BADGE damage in true classes represented by tail probe-damage points",
    "gradients": "unweighted true-label CE, analytical weight+bias norms at shared source/probe/BADGE heads",
    "feature_diversity": "cosine within each selected tail/cohort only; zero vectors excluded",
    "new_training": 0, "new_queries": 0, "new_full_pool_inference": 0,
    "status": "post-hoc M12 diagnosis; oracle privilege retained; no causal or independent validation claim",
}
CSV_GROUPS = {
    "by_seed": ["method"], "paired_by_seed": ["comparison"],
    "selection_audit": ["method"], "cohorts": ["method", "cohort"],
    "gradients": ["method", "cohort", "head"],
}
SUMMARY_NAMES = {"by_seed": "summary", "paired_by_seed": "paired_summary",
                 "selection_audit": "selection_summary", "cohorts": "cohort_summary", "gradients": "gradient_summary"}
FILES = ["resolved_config.json", "excluded_ids.json", "per_class.csv", "report.md"] + [
    name + ".csv" for names in zip(CSV_GROUPS, SUMMARY_NAMES.values()) for name in names]


def verify_inputs(args):
    out = Path(args.output_dir).resolve()
    for name in ("learned_dir", "allocation_dir", "oracle_dir", "pilot_dir", "study_dir", "cache_dir"):
        source = Path(getattr(args, name)).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Use a separate sibling diagnostic output directory")
    if args.threads <= 0:
        raise ValueError("Threads must be positive")
    studies = {}
    # Do not forward diagnostic threads: historical verifiers insert original threads.
    base = {k: getattr(args, k) for k in ("pilot_dir", "oracle_dir", "study_dir", "cache_dir", "checkpoint")}
    for label, module, folder in [("learned", learned, args.learned_dir), ("allocation", allocation, args.allocation_dir)]:
        original = pilot.read_json(Path(folder) / "study_manifest.json")["identity"]
        source_args = argparse.Namespace(**base, output_dir=folder, seeds=",".join(map(str, original["seeds"])))
        verified = module.verify_sources(source_args)
        if verified[2] != original:
            raise ValueError(f"Saved {label} study identity changed")
        records = module.require_selections(source_args, verified)
        hashes = {}
        for seed in original["seeds"]:
            for method in module.METHODS:
                rd = pilot.run_dir(Path(folder), seed, method)
                if not pilot.completed(rd, pilot.train_signature(original, Path(folder), seed, method), pilot.TRAIN_FILES):
                    raise ValueError(f"Complete all source fits first: {label}/{seed}/{method}")
                hashes[f"{seed}/{method}"] = kbs.file_sha(rd / "complete.json")
        sig = {"identity_sha256": kbs.digest(original), "runs": hashes}
        if not pilot.completed(Path(folder) / "evaluation", sig, module.EVAL_FILES):
            raise ValueError(f"Complete source {label} evaluation first")
        studies[label] = {"verified": verified, "records": records, "identity": original,
            "evaluation_sha256": kbs.file_sha(Path(folder) / "evaluation/complete.json"), "runs": hashes}
    sources = studies["learned"]["verified"][0]
    other = studies["allocation"]["verified"][0][0]
    if sources[3] != other[3]:
        raise ValueError("Studies must share exactly the original pilot, cache and protocol")
    seeds = kbs.int_list(args.seeds)
    available = set(studies["learned"]["records"]) & set(studies["allocation"]["records"])
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= available:
        raise ValueError("Use unique seeds completed in both studies")
    identity = {"settings": SETTINGS, "seeds": seeds, "threads": args.threads,
        "sources": {name: {"identity_sha256": kbs.digest(v["identity"]), "runs": v["runs"],
                            "evaluation_sha256": v["evaluation_sha256"]} for name, v in studies.items()},
        "code_sha256": kbs.file_sha(Path(__file__)), "numpy_version": np.__version__, "torch_version": str(torch.__version__)}
    # Refuse changed settings even in preflight; do not create output until run.
    if (out / "study_manifest.json").exists():
        pilot.freeze_manifest(out, identity)
    return sources, studies, identity


def common_mask(n, *exclusions):
    excluded = sorted(set().union(*(set(x) for x in exclusions)))
    pilot.check_ids(excluded, n, len(excluded))
    mask = np.ones(n, dtype=bool); mask[excluded] = False
    if not mask.any():
        raise ValueError("No common evaluation samples remain")
    return mask, excluded


def feature_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid selected features")
    norms = np.linalg.norm(x, axis=1)
    z = x[norms > 0] / norms[norms > 0, None]
    pair, nearest = None, None
    if len(z) >= 2:
        similarity = np.clip(z @ z.T, -1, 1)
        pair = float(similarity[np.triu_indices(len(z), 1)].mean())
        np.fill_diagonal(similarity, -np.inf)
        nearest = float(similarity.max(1).mean())
    return {"feature_norm_mean": float(norms.mean()) if len(x) else None,
        "cosine_valid_samples": len(z), "zero_norm_samples": int((norms == 0).sum()),
        "pair_cosine_mean": pair, "nearest_cosine_mean": nearest}


def gradient_stats(x, y, state):
    """Exact per-example and mean CE gradients for a linear head, including bias.

    These are local gradient diagnostics, not actual AdamW/KD update magnitudes.
    No autograd, optimizer, model mutation, or full-pool model inference is used.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y)
    w, b = np.asarray(state["weight"], dtype=np.float64), np.asarray(state["bias"], dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or w.shape[1] != x.shape[1] or b.shape != (len(w),):
        raise ValueError("Incompatible feature/head shapes")
    diagnosis.vector(y, len(x), len(w), "selected truth")
    if not all(np.isfinite(v).all() for v in (x, w, b)):
        raise ValueError("Nonfinite feature/head values")
    keys = ("true_probability_mean", "ce_mean", "logit_gradient_norm_mean", "example_gradient_norm_mean",
            "example_gradient_norm_median", "mean_gradient_norm", "gradient_alignment_ratio")
    if not len(x):
        return {"n": 0, **dict.fromkeys(keys)}
    logits = x @ w.T + b
    shifted = logits - logits.max(1, keepdims=True)
    logprob = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    prob = np.exp(logprob)
    true_prob = prob[np.arange(len(y)), y].copy()
    residual = prob.copy(); residual[np.arange(len(y)), y] -= 1
    augmented = np.column_stack([x, np.ones(len(x))])
    logit_norm = np.linalg.norm(residual, axis=1)
    norms = logit_norm * np.linalg.norm(augmented, axis=1)
    average_norm = float(np.linalg.norm(residual.T @ augmented / len(x)))
    return {"n": len(x), "true_probability_mean": float(true_prob.mean()),
        "ce_mean": float(-logprob[np.arange(len(y)), y].mean()),
        "logit_gradient_norm_mean": float(logit_norm.mean()),
        "example_gradient_norm_mean": float(norms.mean()), "example_gradient_norm_median": float(np.median(norms)),
        "mean_gradient_norm": average_norm, "gradient_alignment_ratio": diagnosis.ratio(average_norm, float(norms.mean()))}


def tail_diagnostics(ids, badge, y, old, probe, badge_final, mask, p, features):
    prefix, _, slots = learned.parts(badge)
    tail = np.asarray(ids[len(prefix):], dtype=np.int64)
    if len(tail) != slots or len(features) != slots:
        raise ValueError("Tail must contain exactly 5% of the saved budget")
    pn = (old == y) & (probe != y)
    final_damage = (old == y) & (badge_final != y)
    outside = ~np.isin(tail, badge)
    outside_pn = outside & pn[tail]
    nc = ~np.isin(y, p["collapse_classes"])
    common_damage = mask & nc & final_damage
    protected_classes = np.unique(y[tail[pn[tail]]])
    covered_damage = int((common_damage & np.isin(y, protected_classes)).sum())
    audit = {"prefix_order_identical": int(list(ids[:len(prefix)]) == prefix.tolist()),
        "prefix_symmetric_difference": len(set(ids[:len(prefix)]) ^ set(prefix)),
        "tail_n": len(tail), "tail_overlap_badge_training": int((~outside).sum()),
        "probe_damage": int(pn[tail].sum()),
        "probe_correction": int(((old[tail] != y[tail]) & (probe[tail] == y[tail])).sum()),
        "outside_badge_training_n": int(outside.sum()),
        "outside_badge_final_damage_n": int((outside & final_damage[tail]).sum()),
        "outside_badge_final_damage_fraction": diagnosis.ratio(int((outside & final_damage[tail]).sum()), int(outside.sum())),
        "outside_badge_probe_damage_n": int(outside_pn.sum()),
        "outside_badge_persistent_damage_n": int((outside_pn & final_damage[tail]).sum()),
        "outside_badge_probe_damage_persistence": diagnosis.ratio(int((outside_pn & final_damage[tail]).sum()), int(outside_pn.sum())),
        "common_noncollapse_badge_damage_n": int(common_damage.sum()),
        "class_covered_noncollapse_badge_damage_n": covered_damage,
        "class_covered_noncollapse_badge_damage_fraction": diagnosis.ratio(covered_damage, int(common_damage.sum()))}
    cohorts = []
    for name, subset in [("all_tail", np.ones(len(tail), bool)), ("probe_damage_tail", pn[tail])]:
        labels = y[tail[subset]]
        counts = np.bincount(labels, minlength=p["num_classes"])
        cohorts.append({"cohort": name, "n": len(labels), "true_classes": int((counts > 0).sum()),
            "source_predicted_classes": len(np.unique(old[tail[subset]])),
            "collapse_queries": int(np.isin(labels, p["collapse_classes"]).sum()),
            "max_true_class_fraction": diagnosis.ratio(int(counts.max()), len(labels)),
            "effective_true_classes": diagnosis.ratio(len(labels)**2, int(counts @ counts)),
            **feature_stats(features[subset])})
    counts = np.bincount(y[tail], minlength=p["num_classes"])
    damage_counts = np.bincount(y[tail[pn[tail]]], minlength=p["num_classes"])
    common_counts = np.bincount(y[common_damage], minlength=p["num_classes"])
    class_rows = [{"class_id": c, "tail_queries": int(counts[c]), "tail_probe_damage": int(damage_counts[c]),
                   "common_noncollapse_badge_damage": int(common_counts[c])} for c in range(p["num_classes"])]
    return audit, cohorts, class_rows, pn[tail]


def evaluation_rows(folder, seeds, methods):
    rows = kbs.read_csv(Path(folder) / "evaluation/by_seed.csv")
    lookup = {(int(r["seed"]), r["method"]): r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {(s, m) for s in seeds for m in methods}:
        raise ValueError("Source evaluation row inventory changed")
    return lookup


def make_report(tables, identity):
    lines = ["# 保护选样与 oracle 差距诊断（事后分析，不训练）", "",
        f"种子 {identity['seeds']}；8 组预测在两个历史研究排除列表的并集之外重算。均值 ± 样本标准差。",
        "只读取已完成的选样、分类头和预测；没有新查询、修复训练、风险模型拟合或全池推断。",
        "oracle 两组保留完整目标真值选样特权，只作机制参照，不能当作同标注成本的可部署方法或严格上界。", ""]

    def table(title, rows, labels, fields, keys=("method",)):
        lines.extend([title, "", "| " + " | ".join([*keys, *labels]) + " |",
                      "|" + "|".join(["---"] * len(keys) + ["---:"] * len(fields)) + "|"])
        for r in rows:
            lines.append("| " + " | ".join([*[str(r[k]) for k in keys], *[diagnosis.stat(r, f) for f in fields]]) + " |")
        lines.append("")

    table("共同评估集结果", tables["summary"],
        ["整体 F1", "崩溃 F1", "稳定 F1", "非崩溃负翻转", "非崩溃正翻转", "新崩溃"],
        ["overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_after",
         "noncollapse_negative_flips", "noncollapse_positive_flips", "noncollapse_new_collapses"])
    table("配对差值（前者减后者；负翻转越小越好）", tables["paired_summary"],
        ["整体 F1 差", "崩溃 F1 差", "负翻转差", "正翻转差", "新崩溃差"],
        ["overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_negative_flips",
         "noncollapse_positive_flips", "noncollapse_new_collapses"], ("comparison",))
    table("尾部 5% 查询与最终 BADGE 损伤", tables["selection_summary"],
        ["预演误伤数", "与 BADGE 训练重合数", "非重合数", "非重合最终误伤率", "非重合预演误伤持续率", "类别覆盖损伤比例"],
        ["probe_damage", "tail_overlap_badge_training", "outside_badge_training_n",
         "outside_badge_final_damage_fraction", "outside_badge_probe_damage_persistence",
         "class_covered_noncollapse_badge_damage_fraction"])
    lines += ["率与比例均按 0–1 报告。BADGE 自身尾部全在 BADGE 训练中，因此非重合持续率为空，不能记为 0。",
        "非重合统计排除了全部 BADGE 训练查询，但仍是选样条件下的事后组成，不能与共同测试指标混用。",
        "类别覆盖损伤比例仅表示：尾部预演误伤样本涉及的真实类别，包含多少共同测试集非崩溃类 BADGE 损伤；不代表覆盖了相同特征区域。", ""]
    table("尾部类别与特征多样性", tables["cohort_summary"],
        ["样本数", "真实类别数", "有效类别数", "最大类别占比", "两两余弦均值", "最近邻余弦均值"],
        ["n", "true_classes", "effective_true_classes", "max_true_class_fraction", "pair_cosine_mean", "nearest_cosine_mean"],
        ("method", "cohort"))
    table("预演误伤子集的 CE 梯度（相同三个分类头）",
        [r for r in tables["gradient_summary"] if r["cohort"] == "probe_damage_tail"],
        ["样本数", "真类概率", "CE", "单样本梯度范数", "平均梯度范数", "方向一致性比"],
        ["n", "true_probability_mean", "ce_mean", "example_gradient_norm_mean", "mean_gradient_norm", "gradient_alignment_ratio"],
        ("method", "head"))
    differing = [r["method"] for r in tables["selection_summary"] if r["prefix_order_identical_mean"] != 1]
    lines += ["前 95% 查询审计：" + ("以下组存在与 BADGE 不同的前缀，不能把差异完全归于尾部：" + ", ".join(differing)
                                    if differing else "所有展示组的前 95% 查询顺序均与 BADGE 相同。"), "",
        "先核对共同表上的差距，再比较尾部覆盖、非重合持续率和同一分类头下的梯度。三者是诊断线索，不能单独证明失败原因。",
        "梯度包含线性头的权重和偏置：单样本范数为 ||p−onehot(y)|| × sqrt(||z||²+1)。",
        "平均梯度范数来自样本梯度先平均再求范数；方向一致性比为它除以单样本范数均值，不同子集大小会影响该比值。",
        "CE 梯度是未加权的局部诊断；不包含 KD、AdamW 状态、重复采样与后续训练轨迹，不能等同于实际保护贡献。",
        "只对已选尾部特征作 CPU float64 计算；误伤集合始终使用原运行保存的预测，避免重新推断改动近似并列的类别。",
        "零分母、空子集或不足两个非零特征的余弦统计为空；各指标有效种子数见 CSV。原始源结果在其原评估集上逐项核对后才重算。",
        "M12 已用于开发，种子共用一个源模型与月份；不搜索阈值、不选最优配置、不检验显著性，不产生独立验证或新方法结论。", ""]
    return "\n".join(lines)


def run(args, verified):
    sources, studies, identity = verified
    info, p, saved, _ = sources
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        pilot.freeze_manifest(out, identity, create=True)
        sig = {"identity_sha256": kbs.digest(identity)}
        if pilot.completed(out, sig, FILES):
            print(f"Verified report: {out / 'report.md'}", flush=True)
            return
        torch.set_num_threads(args.threads)
        target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True, mmap=True)
        n, classes = info["sample_counts"]["target"], p["num_classes"]
        y = diagnosis.vector(np.asarray(target["labels"]), n, classes, "truth")
        old = diagnosis.vector(target["logits"].argmax(1).numpy(), n, classes, "source")
        source_state = torch.load(Path(args.cache_dir) / "head.pt", map_location="cpu", weights_only=True)
        original_rows = {
            "learned": evaluation_rows(args.learned_dir, studies["learned"]["identity"]["seeds"], ("static", "badge", *learned.METHODS)),
            "allocation": evaluation_rows(args.allocation_dir, studies["allocation"]["identity"]["seeds"],
                ("static", *pilot.METHODS, *allocation.oracle.METHODS, *allocation.METHODS))}
        tables = {key: [] for key in [*CSV_GROUPS, "per_class"]}
        exclusions = {}
        for seed in identity["seeds"]:
            print(f"Diagnosing seed {seed}: reconciling source tables, then unified evaluation", flush=True)
            lr, ar = (studies[k]["records"][seed] for k in ("learned", "allocation"))
            mask, excluded = common_mask(n, lr["common_excluded_indices"], ar["common_excluded_indices"])
            exclusions[str(seed)] = excluded
            old_sd = pilot.selection_dir(Path(args.pilot_dir), seed)
            probe = diagnosis.vector(np.load(old_sd / "probe_predictions.npy", mmap_mode="r"), n, classes, "probe")
            saved_old = np.load(pilot.selection_dir(Path(args.learned_dir), seed) / "source_predictions.npy", mmap_mode="r")
            if not np.array_equal(saved_old, old):
                raise ValueError("Frozen source predictions differ from target cache")
            predictions, queries, directories, metrics = {"static": old}, {}, {}, {}
            for method in METHODS:
                folder = args.pilot_dir if method == "badge" else args.allocation_dir if method in ORACLES else args.learned_dir
                rd = pilot.run_dir(Path(folder), seed, method)
                directories[method] = rd
                predictions[method] = diagnosis.vector(np.load(rd / "predictions.npy", mmap_mode="r"), n, classes, method)
                ids = saved[seed] if method == "badge" else (ar if method in ORACLES else lr)["choices"][method]["row_indices"]
                ids = pilot.check_ids(ids, n, p["budget"])
                if not set(ids) <= set(excluded):
                    raise ValueError("A displayed query escaped common exclusion")
                query_rows = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query_rows] != list(ids) or [int(r["label"]) for r in query_rows] != y[ids].tolist():
                    raise ValueError("Training query IDs or labels disagree with the frozen selection/cache")
                queries[method] = ids
            for label, record, methods in [("learned", lr, ("static", "badge", *learned.METHODS)),
                                            ("allocation", ar, ("static", "badge", *ORACLES))]:
                original_mask, _ = common_mask(n, record["common_excluded_indices"])
                for method in methods:
                    m, _ = pilot.metrics(y, old, predictions[method], original_mask, p)
                    diagnosis.reconcile(m, original_rows[label][seed, method])
            heads = {"source": source_state,
                "probe": torch.load(old_sd / "probe_head.pt", map_location="cpu", weights_only=True),
                "badge_final": torch.load(directories["badge"] / "head.pt", map_location="cpu", weights_only=True)}
            for method, pred in predictions.items():
                m, cr = pilot.metrics(y, old, pred, mask, p)
                metrics[method] = m
                tables["by_seed"].append({"seed": seed, "method": method, **m})
                if method != "static":
                    ids = queries[method]
                    prefix, _, _ = learned.parts(saved[seed])
                    tail = np.asarray(ids[len(prefix):], dtype=np.int64)
                    x = target["features"][torch.as_tensor(tail.copy())].numpy().astype(np.float64)
                    audit, cohorts, selection_classes, pn = tail_diagnostics(ids, saved[seed], y, old, probe,
                        predictions["badge"], mask, p, x)
                    tables["selection_audit"].append({"seed": seed, "method": method, **audit})
                    tables["cohorts"].extend({"seed": seed, "method": method, **r} for r in cohorts)
                    for name, use in [("all_tail", np.ones(len(tail), bool)), ("probe_damage_tail", pn)]:
                        for head_name, state in heads.items():
                            tables["gradients"].append({"seed": seed, "method": method, "cohort": name,
                                "head": head_name, **gradient_stats(x[use], y[tail[use]], state)})
                    for row, extra in zip(cr, selection_classes):
                        if row["class_id"] != extra["class_id"]:
                            raise ValueError("Per-class ordering changed")
                        row.update(extra)
                tables["per_class"].extend({"seed": seed, "method": method, **r} for r in cr)
            comparisons = [(m, "badge") for m in METHODS[1:]] + [
                ("flip_risk50", "flip_uniform50"), ("flip_risk50", "flip_confidence50"),
                (ORACLES[0], "flip_confidence50"), (ORACLES[0], "flip_risk50"), (ORACLES[0], ORACLES[1])]
            for a, b in comparisons:
                tables["paired_by_seed"].append({"seed": seed, "comparison": f"{a} minus {b}", **{
                    k: metrics[a][k]-metrics[b][k] if metrics[a][k] is not None and metrics[b][k] is not None else None
                    for k in metrics[a]}})
        for name, keys in CSV_GROUPS.items():
            tables[SUMMARY_NAMES[name]] = diagnosis.aggregate(tables[name], keys)
        for name, rows in tables.items():
            kbs.write_csv(out / f"{name}.csv", rows)
        kbs.atomic_json(out / "resolved_config.json", sig)
        kbs.atomic_json(out / "excluded_ids.json", exclusions)
        (out / "report.md").write_text(make_report(tables, identity))
        kbs.finish(out, sig, FILES)
        print(f"Results: {out / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "run"])
    for flag, default in [("learned-dir", "outputs/kbs_learned_protection_v1"),
        ("allocation-dir", "outputs/kbs_protection_budget_v1"), ("oracle-dir", "outputs/kbs_oracle_protection_v1"),
        ("pilot-dir", "outputs/kbs_repair_aware_v1"), ("study-dir", "outputs/kbs_supplement_v1"),
        ("cache-dir", "outputs/kbs_supplement_v1/cache"), ("checkpoint", "outputs/tls22_cnn/best_model.pt"),
        ("output-dir", "outputs/kbs_protection_gap_diagnosis_v1")]:
        p.add_argument("--"+flag, default=str(kbs.CORE / default))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--threads", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        print(json.dumps({"settings": SETTINGS, "seeds": kbs.int_list(args.seeds), "device": "cpu"}, indent=2))
        return
    verified = verify_inputs(args)
    if args.command == "preflight":
        print(json.dumps({"status": "ready", "device": "cpu", "seeds": verified[2]["seeds"],
                          "new_training": 0, "new_queries": 0, "output_dir": args.output_dir}, indent=2))
    else:
        run(args, verified)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
