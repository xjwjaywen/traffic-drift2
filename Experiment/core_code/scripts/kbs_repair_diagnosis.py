#!/usr/bin/env python3
"""Read-only, CPU post-hoc diagnosis of the completed repair-aware pilot.

No model fitting, prediction, query creation, threshold tuning, or oracle policy.
BADGE final repair is the primary outcome; the other arms are sensitivity analyses.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

import kbs_repair_aware as pilot
import kbs_supplement as kbs

SETTINGS = {
    "schema": "care-repair-diagnosis-v1", "primary_outcome": "badge",
    "evaluation": "same per-seed four-arm query-union exclusion as the pilot",
    "flag": "source argmax != probe argmax (does not require labels)",
    "damage": "source correct and final repair wrong (post-hoc labels)",
    "probe_damage": "source correct and probe wrong (post-hoc labels; NOT a selectable flag)",
    "relation": "scout count[probe_prediction, source_prediction] > 0 inside flip pool",
    "enrichment": "damage precision in flagged region / damage prevalence in same evaluation scope",
    "absorber": "a class receiving at least one confirmed scout error; exposure-adjusted audit",
    "status": "M12 exploratory post-hoc analysis; no success/significance/novelty declaration",
}
FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "cohorts.csv", "cohort_summary.csv",
         "per_class.csv", "transitions.csv", "report.md"]


def verify_inputs(args):
    root, out = Path(args.pilot_dir).resolve(), Path(args.output_dir).resolve()
    for value in (root, Path(args.study_dir).resolve(), Path(args.cache_dir).resolve()):
        if out == value or value in out.parents or out in value.parents:
            raise ValueError("Diagnosis output must be a separate sibling of the pilot/study/cache")
    if args.threads <= 0:
        raise ValueError("Threads must be positive")
    original = pilot.read_json(root / "study_manifest.json")["identity"]
    # Validation retains the source runtime identity; no call requests its CUDA device.
    source_args = argparse.Namespace(study_dir=args.study_dir, cache_dir=args.cache_dir,
        checkpoint=args.checkpoint, output_dir=args.pilot_dir,
        seeds=",".join(map(str, original["seeds"])), device=original["device"],
        threads=original["threads"], batch_size=original["inference_batch_size"])
    sources = pilot.verify_sources(source_args)
    if sources[3] != original:
        raise ValueError("Original pilot inputs/code/environment changed; use its original environment")
    records = pilot.require_selections(root, sources)
    hashes = {}
    for seed in original["seeds"]:
        for method in pilot.METHODS:
            directory = pilot.run_dir(root, seed, method)
            if not pilot.completed(directory, pilot.train_signature(original, root, seed, method), pilot.TRAIN_FILES):
                raise ValueError(f"All original fits must be complete before diagnosis: {seed}/{method}")
            hashes[f"{seed}/{method}"] = kbs.file_sha(directory / "complete.json")
    signature = {"identity_sha256": kbs.digest(original), "runs": hashes}
    if not pilot.completed(root / "evaluation", signature, pilot.EVAL_FILES):
        raise ValueError("Finish the original pilot evaluation before diagnosis")
    identity = {"settings": SETTINGS, "source_identity_sha256": kbs.digest(original),
                "source_evaluation_sha256": kbs.file_sha(root / "evaluation/complete.json"),
                "source_run_sha256": hashes, "seeds": original["seeds"],
                "diagnosis_sha256": kbs.file_sha(Path(__file__)), "threads": args.threads,
                "numpy_version": np.__version__, "torch_version": str(torch.__version__)}
    return sources, records, identity


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def divide(a, b):
    return None if a is None or b is None or b == 0 else float(a / b)


def vector(value, n, classes, name):
    x = np.asarray(value)
    if x.shape != (n,) or x.dtype.kind not in "iu" or np.any((x < 0) | (x >= classes)):
        raise ValueError(f"Invalid {name}: expected {n} integer class IDs")
    return x


def diagnostic_metrics(y, old, probe, final, mask, counts, p):
    """All denominators are explicit; empty/zero-event rates are None, not zero."""
    n, classes = len(y), p["num_classes"]
    for name, x in [("truth", y), ("source", old), ("probe", probe), ("final", final)]:
        vector(x, n, classes, name)
    if mask.shape != (n,) or mask.dtype.kind != "b" or not mask.any():
        raise ValueError("Expected a nonempty common evaluation mask")
    if counts.shape != (classes, classes) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Invalid scout relation counts")
    correct = old == y
    flags = old != probe
    pn = correct & flags
    pp = (~correct) & (probe == y)
    negative = correct & (final != y)
    positive = (~correct) & (final == y)
    confirmed = flags & (counts[probe, old] > 0)
    absorber_classes = counts.sum(axis=0) > 0
    absorber = absorber_classes[y]
    outside_collapse = ~np.isin(y, p["collapse_classes"])
    main = []
    for scope, use in [("all", mask), ("noncollapse", mask & outside_collapse)]:
        total, source_correct = int(use.sum()), int((use & correct).sum())
        flagged, damage = int((use & flags).sum()), int((use & negative).sum())
        captured = int((use & flags & negative).sum())
        probe_damage = int((use & pn).sum())
        prior, precision = ratio(damage, total), ratio(captured, flagged)
        a_correct = int((use & correct & absorber).sum())
        a_damage = int((use & negative & absorber).sum())
        a_rate, other_rate = ratio(a_damage, a_correct), ratio(damage - a_damage, source_correct - a_correct)
        main.append({"scope": scope, "eval_samples": total, "source_correct": source_correct,
            "flagged_samples": flagged, "flagged_fraction": ratio(flagged, total),
            "final_negative_flips": damage, "damage_prevalence": prior,
            "damage_rate_on_source_correct": ratio(damage, source_correct),
            "captured_damage": captured, "missed_damage": damage - captured,
            "flag_precision": precision, "damage_coverage": ratio(captured, damage),
            "enrichment_vs_random": divide(precision, prior),
            "outside_flag_damage_rate": ratio(damage - captured, total - flagged),
            "probe_negative_flips": probe_damage,
            "probe_negative_persistence": ratio(captured, probe_damage),
            "probe_negative_resolved": probe_damage - captured,
            "probe_negative_share_of_flags": ratio(probe_damage, flagged),
            "probe_positive_share_of_flags": ratio(int((use & pp).sum()), flagged),
            "confirmed_absorber_classes": int(absorber_classes.sum()),
            "absorber_source_correct": a_correct, "absorber_final_damage": a_damage,
            "absorber_correct_exposure_share": ratio(a_correct, source_correct),
            "absorber_damage_share": ratio(a_damage, damage),
            "absorber_damage_rate": a_rate, "other_class_damage_rate": other_rate,
            "absorber_risk_ratio": divide(a_rate, other_rate)})
    cohorts = []
    for name, members in [("all", np.ones(n, dtype=bool)), ("probe_flip", flags),
                          ("probe_unchanged", ~flags), ("confirmed_relation_flip", confirmed),
                          ("unconfirmed_relation_flip", flags & ~confirmed), ("oracle_probe_negative", pn)]:
        use = mask & members
        size = int(use.sum())
        cohorts.append({"cohort": name, "n": size, "source_correct": int((use & correct).sum()),
            "probe_negative": int((use & pn).sum()), "probe_positive": int((use & pp).sum()),
            "probe_changed_both_wrong": int((use & flags & ~correct & (probe != y)).sum()),
            "probe_negative_fraction": ratio(int((use & pn).sum()), size),
            "probe_positive_fraction": ratio(int((use & pp).sum()), size),
            "final_negative": int((use & negative).sum()), "final_positive": int((use & positive).sum()),
            "final_negative_fraction": ratio(int((use & negative).sum()), size),
            "final_positive_fraction": ratio(int((use & positive).sum()), size)})

    metrics, class_rows, _ = kbs.compare_predictions(y, old, final, mask, classes, p["collapse_classes"],
        p["stable_classes"], p["collapse_recall_threshold"], p["f1_drop_threshold"])
    known_final_reverse = negative & (counts[final, old] > 0)
    class_counts = {name: np.bincount(y[mask & membership], minlength=classes) for name, membership in [
        ("source_correct", correct), ("probe_flags", flags), ("probe_negative", pn),
        ("negative_flips", negative), ("positive_flips", positive),
        ("captured_damage", negative & flags), ("confirmed_reverse_damage", known_final_reverse)]}
    for c, row in enumerate(class_rows):
        row.update({k: int(v[c]) for k, v in class_counts.items()})
        row.update({"scout_absorber_errors": int(counts[:, c].sum()), "scout_victim_errors": int(counts[c].sum()),
                    "is_confirmed_absorber": bool(absorber_classes[c]),
                    "damage_rate_on_source_correct": ratio(row["negative_flips"], row["source_correct"]),
                    "damage_coverage": ratio(row["captured_damage"], row["negative_flips"]),
                    "probe_negative_persistence": ratio(row["captured_damage"], row["probe_negative"]),
                    "new_collapse": bool(row["support"] > 0 and not row["is_collapse_group"] and
                        row["before_recall"] >= p["collapse_recall_threshold"] > row["after_recall"])})
    transitions = []
    use = mask & negative
    pair = old * classes + final
    total_edges = np.bincount(pair[use], minlength=classes**2).reshape(classes, classes)
    captured_edges = np.bincount(pair[use & flags], minlength=classes**2).reshape(classes, classes)
    same_edges = np.bincount(pair[use & (probe == final)], minlength=classes**2).reshape(classes, classes)
    for a, v in zip(*np.nonzero(total_edges)):
        transitions.append({"source_correct_class": int(a), "final_wrong_class": int(v),
            "negative_flips": int(total_edges[a, v]), "captured_by_probe": int(captured_edges[a, v]),
            "same_wrong_destination_in_probe": int(same_edges[a, v]),
            "confirmed_reverse_scout_errors": int(counts[v, a])})
    for scope, use in [("all", mask), ("noncollapse", mask & outside_collapse)]:
        metrics[f"{scope}_negative_flips"] = int((use & negative).sum())
        metrics[f"{scope}_positive_flips"] = int((use & positive).sum())
    if sum(r["negative_flips"] for r in transitions) != metrics["all_negative_flips"]:
        raise ValueError("Transition accounting failed")
    return main, cohorts, class_rows, transitions, metrics


def reconcile(calculated, saved):
    for key, expected in calculated.items():
        raw = saved.get(key)
        if expected is None:
            if raw not in (None, ""):
                raise ValueError(f"Metric mismatch: {key}")
        elif raw in (None, "") or not np.isclose(float(raw), expected, rtol=0, atol=1e-10):
            raise ValueError(f"Metric mismatch: {key} (recomputed {expected}, saved {raw})")


def aggregate(rows, keys):
    result = []
    for group in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
        selected = [r for r in rows if tuple(r[k] for k in keys) == group]
        out = {**dict(zip(keys, group)), "n_seeds": len(selected)}
        for field in selected[0]:
            if field in keys or field == "seed":
                continue
            values = [r[field] for r in selected if r[field] is not None]
            out[field + "_valid_n"] = len(values)
            out[field + "_mean"] = statistics.mean(values) if values else None
            out[field + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(out)
    return result


def stat(row, name, percent=False):
    mean, sd = row[name + "_mean"], row[name + "_sd"]
    if mean is None:
        return "—"
    scale, suffix = (100, "%") if percent else (1, "")
    return f"{mean*scale:.3f}{suffix} ± " + (f"{sd*scale:.3f}{suffix}" if sd is not None else "—")


def make_report(summary, cohorts, seeds):
    lines = ["# 修复预演的损伤预测诊断（事后分析，不训练）", "",
        f"种子 {seeds}；全部沿用原四组查询并集排除的共同评估集。BADGE 最终修复是主要参照。",
        "均值 ± 样本标准差；CSV 同时给每个指标的有效种子数，零分母记为空值，不算成 0。", "",
        "## 预演翻转区域能否覆盖最终损伤", "",
        "精确率 = 区域中最终负翻转数 / 区域样本数；覆盖率 = 被区域覆盖的最终负翻转数 / 全部最终负翻转数。",
        "随机基率 = 同一评估范围内最终负翻转数 / 全部样本数；富集倍数 = 区域精确率 / 随机基率。",
        "随机基率是均匀单次查询的精确期望，不是重新训练的随机策略结果。", "",
        "| 最终方法 | 范围 | 区域占比 | 随机基率 | 区域精确率 | 损伤覆盖率 | 富集倍数 | 预演误伤持续率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        cells = [stat(r, k, True) for k in ("flagged_fraction", "damage_prevalence", "flag_precision", "damage_coverage")]
        cells += [stat(r, "enrichment_vs_random"), stat(r, "probe_negative_persistence", True)]
        lines.append(f"| {r['method']} | {r['scope']} | " + " | ".join(cells) + " |")
    lines += ["", "预演误伤持续率 = 预演和最终都改错 / 预演改错；它使用事后真值，不能直接作为部署选样分数。",
        "最终负翻转必定源预测正确，因此翻转区域对最终损伤的覆盖，与事后预演负翻转集合的覆盖在数学上相同，不能算两份独立证据。",
        "noncollapse 是按原稿固定类组作事后分解，真实类别不能用于在线筛选。", "",
        "## BADGE 参照下的候选区域组成", "",
        "| 区域 | 样本数 | 预演误伤占比 | 预演纠错占比 | 最终误伤占比 |",
        "|---|---:|---:|---:|---:|"]
    for r in cohorts:
        if r["method"] == "badge":
            lines.append(f"| {r['cohort']} | {stat(r, 'n')} | {stat(r, 'probe_negative_fraction', True)} | "
                         f"{stat(r, 'probe_positive_fraction', True)} | {stat(r, 'final_negative_fraction', True)} |")
    lines += ["", "confirmed_relation_flip 与 unconfirmed_relation_flip 是同一翻转池的互斥分组。",
        "如果已确认关系区域主要富集预演纠错，而不是误伤，则与关系加权偏离保护目标的解释一致；不是因果证明。",
        "oracle_probe_negative 仅为事后诊断集合，没有生成利用未查询标签的查询策略。", "",
        "## BADGE 损伤是否集中在已确认吸收者类", "",
        "吸收者定义为共同侦察查询中至少接收过一次错分的类；不是使用完整目标真值重新识别的类。",
        "同时比较原本正确样本的暴露占比和负翻转占比，避免把高流量类别误判为高风险。", "",
        "| 范围 | 吸收者原正确样本占比 | 吸收者损伤占比 | 吸收者误伤率 | 其他类误伤率 | 风险比 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["method"] == "badge":
            cells = [stat(r, k, True) for k in ("absorber_correct_exposure_share", "absorber_damage_share",
                                                "absorber_damage_rate", "other_class_damage_rate")]
            lines.append(f"| {r['scope']} | " + " | ".join(cells) + f" | {stat(r, 'absorber_risk_ratio')} |")
    lines += ["", "风险比的分母为其他类误伤率；其他类无暴露或无损伤时不报告比值。",
        "逐类支持数、误伤率、新崩溃及 scout 计数见 per_class.csv；具体 a→v 误伤方向及反向关系支持见 transitions.csv。", "",
        "判读顺序：先看 BADGE 的覆盖率与富集程度，再看已确认关系区域是否比未确认区域更富集最终误伤。",
        "覆盖高但区域很大不代表选样精准；富集高但覆盖很低也不足以覆盖主要损伤。",
        "其他三组改变了最终修复模型，只作敏感性对照；本分析不能证明选择这些样本就能防止损伤。",
        "不新增种子、不搜索阈值、不自动判定通过或失败；没有 GPU 训练、模型推断或新增标注。",
        "M12 已用于开发；这些结果不是独立验证，不产生可直接替换论文主结果的新方法。", ""]
    return "\n".join(lines)


def analyze(args, verified):
    sources, records, identity = verified
    info, p, saved, original = sources
    root, out = Path(args.pilot_dir).resolve(), Path(args.output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        if pilot.completed(out, identity, FILES):
            print(f"Verified report: {out / 'report.md'}", flush=True)
            return
        torch.set_num_threads(args.threads)
        # mmap avoids eagerly loading the large feature tensor. No feature is accessed.
        payload = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True, mmap=True)
        y = np.asarray(payload["labels"])
        old = payload["logits"].argmax(1).numpy()
        del payload
        n, classes = info["sample_counts"]["target"], p["num_classes"]
        vector(y, n, classes, "truth"); vector(old, n, classes, "source")
        previous = kbs.read_csv(root / "evaluation/by_seed.csv")
        keyed = {(int(r["seed"]), r["method"]): r for r in previous}
        expected_keys = {(s, m) for s in original["seeds"] for m in ("static", *pilot.METHODS)}
        if len(keyed) != len(previous) or set(keyed) != expected_keys:
            raise ValueError("Original summary has missing/duplicate/unexpected seed-method rows")
        all_main, all_cohorts, all_classes, all_edges = [], [], [], []
        for seed, record in records.items():
            print(f"Diagnosing seed {seed}: saved predictions only...", flush=True)
            sd = pilot.selection_dir(root, seed)
            probe = vector(np.load(sd / "probe_predictions.npy", mmap_mode="r"), n, classes, "probe")
            scout = np.asarray(record["scout_indices"], dtype=np.int64)
            if not np.array_equal(np.load(sd / "scout_labels.npy"), y[scout]):
                raise ValueError("Scout labels changed")
            counts = np.load(sd / "relation_counts.npy")
            if not np.array_equal(counts, pilot.relation_counts(old[scout], y[scout], classes)):
                raise ValueError("Relation matrix differs from confirmed scout errors")
            if record != pilot.select_queries(saved[seed], old, probe, counts, seed):
                raise ValueError("Original selections differ from frozen policy")
            mask = np.ones(n, dtype=bool)
            mask[record["common_excluded_indices"]] = False
            static, _, _ = kbs.compare_predictions(y, old, old, mask, classes, p["collapse_classes"],
                p["stable_classes"], p["collapse_recall_threshold"], p["f1_drop_threshold"])
            reconcile(static, keyed[seed, "static"])
            for method in pilot.METHODS:
                rd = pilot.run_dir(root, seed, method)
                final = np.load(rd / "predictions.npy", mmap_mode="r")
                ids = np.asarray(record["choices"][method]["row_indices"], dtype=np.int64)
                query = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query] != ids.tolist() or [int(r["label"]) for r in query] != y[ids].tolist():
                    raise ValueError("Training queries do not match the frozen labels/rows")
                main, cohorts, class_rows, edges, recalculated = diagnostic_metrics(y, old, probe, final, mask, counts, p)
                reconcile(recalculated, keyed[seed, method])
                for dest, values in [(all_main, main), (all_cohorts, cohorts), (all_classes, class_rows), (all_edges, edges)]:
                    dest.extend({"seed": seed, "method": method, **r} for r in values)
        summary = aggregate(all_main, ["method", "scope"])
        cohort_summary = aggregate(all_cohorts, ["method", "cohort"])
        kbs.atomic_json(out / "resolved_config.json", identity)
        for name, rows in [("by_seed", all_main), ("summary", summary), ("cohorts", all_cohorts),
                           ("cohort_summary", cohort_summary), ("per_class", all_classes), ("transitions", all_edges)]:
            kbs.write_csv(out / f"{name}.csv", rows)
        (out / "report.md").write_text(make_report(summary, cohort_summary, identity["seeds"]))
        kbs.finish(out, identity, FILES)
        print(f"Results: {out / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "run"])
    p.add_argument("--pilot-dir", default=str(kbs.CORE / "outputs/kbs_repair_aware_v1"))
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_repair_diagnosis_v1"))
    p.add_argument("--threads", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        print(json.dumps({"settings": SETTINGS, "fits": 0, "model_inferences": 0,
                          "new_labels": 0, "seeds": "all seeds in completed pilot manifest"}, indent=2))
        return
    verified = verify_inputs(args)
    if args.command == "preflight":
        print(json.dumps({"status": "ready", "device": "cpu", "fits": 0,
                          "seeds": verified[2]["seeds"], "output_dir": args.output_dir}, indent=2))
    else:
        analyze(args, verified)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
