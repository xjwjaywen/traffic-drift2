#!/usr/bin/env python3
"""Read-only full-target audit of completed small repair heads, on CPU.

No queries, fitting, reference tensor loading or hyperparameter search. Target
truth is used only for offline diagnostics after all source stages are verified.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

import kbs_supplement as kbs
import small_repair_diagnostic as diag

SCHEMA = "small-repair-full-audit-v1"
SOURCE_ARTIFACTS = {
    "selection": {"resolved_config.json", "queries.json", "evaluation_rows.npy"},
    "run": {"resolved_config.json", "head.pt", "trace.json"},
    "evaluation": {"resolved_config.json", "by_seed.csv", "summary.csv", "per_class.csv",
                   "paired_by_seed.csv", "paired_summary.csv", "predictions.npz", "report.md"},
}


def completed(path, signature, kind):
    if not kbs.is_complete(path, signature):
        raise ValueError(f"Source {kind} incomplete: {path}")
    marker = json.loads((path / "complete.json").read_text())
    if set(marker["artifacts"]) != SOURCE_ARTIFACTS[kind]:
        raise ValueError(f"Source {kind} artifact list differs: {path}")
    if json.loads((path / "resolved_config.json").read_text()) != signature:
        raise ValueError(f"Source {kind} resolved settings differ: {path}")


def source_inputs(args):
    source, out = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    for path in (source, Path(args.cache_dir).resolve(), Path(args.checkpoint).resolve()):
        if path == out or path.is_relative_to(out) or out.is_relative_to(path):
            raise ValueError("Audit output must be separate from all source/cache/checkpoint paths.")
    manifest = json.loads((source / "diagnostic_manifest.json").read_text())
    protocol = manifest["protocol"]
    if protocol["schema"] != diag.SCHEMA or protocol["engines"] != list(diag.ENGINES):
        raise ValueError("Unsupported source diagnostic protocol.")
    if args.threads != protocol["threads"]:
        raise ValueError(f"Use the original CPU thread count: --threads {protocol['threads']}.")
    # Rebuild the original identity without changing its scripts or output.
    base = diag.parser().parse_args(["evaluate", "--cache-dir", args.cache_dir,
                                    "--checkpoint", args.checkpoint, "--output-dir", str(source)])
    for key in ("seeds", "selectors", "budget", "steps", "batch_size", "lr", "weight_decay",
                "threads", "replay_per_class", "hidden", "expected_classes", "expected_features",
                "eval_size", "eval_seed"):
        setattr(base, key, protocol[key])
    for key in ("collapse_classes", "stable_classes"):
        setattr(base, key, ",".join(map(str, protocol[key])))
    base.reference_period = protocol["periods"]["reference"]
    base.target_period = protocol["periods"]["target"]
    base.inference_batch_size = args.inference_batch_size
    info, reconstructed = diag.inputs(base)
    if reconstructed != protocol or manifest["source_manifest"] != info:
        raise ValueError("Source code/settings/environment/cache changed since fitting; restore the original inputs.")
    choices = diag.selections(source, protocol)
    completed(source / "selection", diag.selection_signature(protocol), "selection")
    n = info["sample_counts"]["target"]
    for choice in choices:
        if any(type(i) is not int or i < 0 or i >= n for i in choice["rows"]):
            raise ValueError("Frozen query row outside the target pool.")
    planned = [(c, e, diag.run_path(source, c, e)) for c in choices for e in diag.ENGINES]
    references, streams = {}, {}
    for c, e, path in planned:
        completed(path, diag.run_signature(protocol, c, e, source), "run")
        trace = json.loads((path / "trace.json").read_text())
        if (trace["query_rows"] != c["rows"] or len(trace["queried_labels"]) != protocol["budget"] or
                trace["optimizer_steps"] != protocol["steps"] or
                trace["target_presentations"] != protocol["steps"] * protocol["batch_size"]):
            raise ValueError(f"Source trace disagrees with frozen queries/steps: {path}")
        ref = (trace["reference_rows"], trace["reference_labels"])
        ref_y = np.asarray(ref[1])
        if (len(ref[0]) != protocol["expected_classes"] * protocol["replay_per_class"] or
                len(set(ref[0])) != len(ref[0]) or ref_y.dtype.kind not in "iu" or
                np.any(ref_y < 0) or np.any(ref_y >= protocol["expected_classes"]) or
                not np.all(np.bincount(ref_y, minlength=protocol["expected_classes"]) == protocol["replay_per_class"])):
            raise ValueError(f"Invalid balanced reference trace: {path}")
        stream = (trace["target_stream_sha256"], trace["reference_stream_sha256"])
        if c["seed"] in references and (references[c["seed"]] != ref or streams[c["seed"]] != stream):
            raise ValueError(f"Matched source references/streams differ: {path}")
        references[c["seed"]], streams[c["seed"]] = ref, stream
    evaluation_signature = {"protocol": protocol, "stage": "evaluate",
        "runs": {str(p.relative_to(source)): kbs.file_sha(p / "complete.json") for _, _, p in planned},
        "selection_sha256": kbs.file_sha(source / "selection/complete.json")}
    completed(source / "evaluation", evaluation_signature, "evaluation")
    signature = {"schema": SCHEMA, "source_protocol": protocol,
        "source_dir": str(source), "cache_dir": str(Path(args.cache_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_manifest_sha256": kbs.file_sha(source / "diagnostic_manifest.json"),
        "source_evaluation_sha256": kbs.file_sha(source / "evaluation/complete.json"),
        "source_selection_sha256": evaluation_signature["selection_sha256"],
        "source_runs": evaluation_signature["runs"],
        "code": {p.name: kbs.file_sha(p) for p in (Path(__file__), Path(__file__).with_name("run_small_repair_full_audit.sh"))},
        "device": "cpu", "threads": args.threads, "inference_batch_size": args.inference_batch_size,
        "evaluation": "all target rows excluding the frozen union of all selectors/seeds' queries"}
    return source, out, base, info, protocol, choices, planned, signature


def ratios(num, denom):
    return int(num) / int(denom) if denom else None


def scores(y, before, after, protocol, trace):
    metrics, classes, _ = kbs.compare_predictions(y, before, after, np.ones(len(y), dtype=bool),
        protocol["expected_classes"], protocol["collapse_classes"], protocol["stable_classes"],
        protocol["collapse_threshold"], protocol["f1_drop_threshold"])
    correct = before == y
    negative, positive = correct & (after != y), ~correct & (after == y)
    metrics.update(negative_flips=int(negative.sum()), positive_flips=int(positive.sum()),
                   negative_flip_rate=ratios(negative.sum(), correct.sum()),
                   positive_flip_rate=ratios(positive.sum(), (~correct).sum()))
    for key in ("query_train_accuracy", "query_train_ce", "trainable_parameters", "query_class_count", "train_wall_s"):
        metrics[key] = trace[key]
    return metrics, classes


def reconcile_csv(path, expected, keys):
    recorded = kbs.read_csv(path)
    lookup = {tuple(r[k] for k in keys): r for r in recorded}
    if len(lookup) != len(recorded) or len(recorded) != len(expected):
        raise ValueError(f"Source CSV row count/duplicates differ: {path}")
    for row in expected:
        saved = lookup.get(tuple(str(row[k]) for k in keys))
        if saved is None:
            raise ValueError(f"Source CSV row identity differs: {path}")
        for key, value in row.items():
            got = saved.get(key)
            if value is None:
                ok = got == ""
            elif isinstance(value, (bool, np.bool_)):
                ok = got == str(bool(value))
            elif isinstance(value, (int, float, np.number)):
                try:
                    ok = np.isclose(float(got), value, rtol=1e-10, atol=1e-12)
                except (TypeError, ValueError):
                    ok = False
            else:
                ok = got == str(value)
            if not ok:
                raise ValueError(f"Source CSV metric disagreement: {path}, {keys}, {key}")


@torch.no_grad()
def predict_rows(head, features, ids, batch):
    head.eval()
    result = np.empty(len(ids), dtype=np.int64)
    for start in range(0, len(ids), batch):
        part = ids[start:start + batch]
        result[start:start + len(part)] = head(features[part]).argmax(1).numpy()
    return result


def class_flips(y, before, after, classes, count):
    correct, repaired = before == y, after == y
    arrays = {"source_correct": np.bincount(y[correct], minlength=count),
              "source_wrong": np.bincount(y[~correct], minlength=count),
              "negative_flips": np.bincount(y[correct & ~repaired], minlength=count),
              "positive_flips": np.bincount(y[~correct & repaired], minlength=count),
              "wrong_to_other_wrong": np.bincount(y[~correct & ~repaired & (before != after)], minlength=count)}
    result = []
    for row in classes:
        c = row["class_id"]
        result.append({**row, "low_support_lt20": row["support"] < 20,
            **{k: int(v[c]) for k, v in arrays.items()},
            "negative_flip_rate": ratios(arrays["negative_flips"][c], arrays["source_correct"][c]),
            "positive_flip_rate": ratios(arrays["positive_flips"][c], arrays["source_wrong"][c])})
    return result


def transitions(y, before, after, count):
    changed = before != after
    codes, counts = np.unique((y[changed] * count + before[changed]) * count + after[changed], return_counts=True)
    result = []
    for code, n in zip(codes, counts):
        truth, old, new = int(code // (count * count)), int(code // count % count), int(code % count)
        kind = "negative" if old == truth else "positive" if new == truth else "wrong_to_other_wrong"
        result.append({"true_class": truth, "source_prediction": old, "repaired_prediction": new,
                       "kind": kind, "count": int(n)})
    return result


def summarize(rows, keys):
    """Unweighted seed means and sample SD; undefined rates stay undefined."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, members in groups.items():
        item = dict(zip(keys, group), n_seeds=len(members))
        for key in dict.fromkeys(k for r in members for k in r):
            if key in keys or key == "seed":
                continue
            values = [r[key] for r in members if r.get(key) is not None]
            if values and all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in values):
                item[key + "_mean"] = statistics.mean(map(float, values))
                item[key + "_sd"] = statistics.stdev(map(float, values)) if len(values) > 1 else None
                item[key + "_valid_seeds"] = len(values)
        result.append(item)
    return result


def transition_summary(rows, seeds):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in ("selector", "engine", "true_class", "source_prediction", "repaired_prediction", "kind"))
        groups.setdefault(key, {})[r["seed"]] = r["count"]
    result = []
    for key, values in sorted(groups.items()):
        counts = [values.get(seed, 0) for seed in seeds]  # absent edge means zero, not missing seed
        result.append(dict(zip(("selector", "engine", "true_class", "source_prediction", "repaired_prediction", "kind"), key),
            n_seeds=len(seeds), seeds_with_transition=sum(n > 0 for n in counts),
            count_mean=statistics.mean(counts), count_sd=statistics.stdev(counts) if len(counts) > 1 else None))
    return result


def query_coverage(target_y, target_before, choices, count):
    rows = []
    for choice in choices:
        idx = np.asarray(choice["rows"], dtype=np.int64)
        y, pred = target_y[idx], target_before[idx]
        support = np.bincount(y, minlength=count)
        anchors = np.bincount(y[y == pred], minlength=count)
        for c in range(count):
            rows.append({"seed": choice["seed"], "selector": choice["selector"], "class_id": c,
                "queried_count": int(support[c]), "source_correct_queried_count": int(anchors[c]),
                "source_wrong_queried_count": int(support[c] - anchors[c])})
    return rows


def report(protocol, summary, pairs, per_class, edges, coverage, baseline, n, union_size, mismatches):
    def fmt(row, metric):
        mean, sd = row.get(metric + "_mean"), row.get(metric + "_sd")
        return "NA" if mean is None else f"{mean:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    lines = ["# CARE 完整 M12 修复复评", "",
        f"复用 {len(protocol['selectors']) * len(protocol['seeds']) * len(diag.ENGINES)} 个已训练分类头；CPU；新增训练 0 次、新增查询 0 条。",
        f"共同评估 {n:,} 条；排除原全部种子/选样的查询并集 {union_size:,} 条。",
        f"原抽样评估的行号、标签、预测、逐种子指标和逐类指标复核通过；完整推理在原抽样行上与原预测不同 {mismatches} 次。",
        f"Static Macro-F1={baseline['overall_macro_f1_before']:.6f}；崩溃组 F1={baseline['collapse_macro_f1_before']:.6f}；"
        f"有支持类别 {baseline['overall_supported_classes']}/{protocol['expected_classes']}。", "",
        "| 选样 | 分类头 | Macro-F1 | 崩溃组 F1 | 非崩溃类 ΔF1 | 稳定类 ΔF1 | 负翻转率 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    metrics = ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta", "stable_macro_f1_delta", "negative_flip_rate")
    for r in summary:
        lines.append(f"| {r['selector']} | {r['engine']} | " + " | ".join(fmt(r, k) for k in metrics) + " |")
    lines += ["", "## 配对诊断（前者减后者）", "",
        "| 对照 | Macro-F1 Δ | 崩溃组 F1 Δ | 非崩溃类 ΔF1 差 | 负翻转率差 |",
        "|---|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['comparison']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## CARE 各组损伤最大的非崩溃类别", "",
        "按平均 ΔF1 排序，每组最多 5 类；包含低支持类别，其支持数同时显示。负翻转率只衡量源正确样本被改错，F1 还受假阳性影响。", "",
        "| 选样/分类头 | 类别 ID | 支持数 | ΔF1 | 负翻转率 | 已查询源正确样本数 | 最大负翻转去向（平均条数） |",
        "|---|---:|---:|---:|---:|---:|---|"]
    qlookup = {(r['selector'], r['class_id']): r for r in coverage}
    for selector in protocol["selectors"]:
        for engine in ("linear_care", "residual_mlp_care"):
            damaged = sorted((r for r in per_class if r['selector'] == selector and r['engine'] == engine and
                r['class_id'] not in protocol['collapse_classes'] and r['support_mean'] > 0 and r['delta_f1_mean'] < 0),
                key=lambda r: (r['delta_f1_mean'], r['class_id']))[:5]
            for r in damaged:
                dest = sorted((e for e in edges if e['selector'] == selector and e['engine'] == engine and
                    e['true_class'] == r['class_id'] and e['kind'] == 'negative'), key=lambda e: -e['count_mean'])[:3]
                destinations = ", ".join(f"{e['repaired_prediction']} ({e['count_mean']:.1f})" for e in dest) or "无"
                lines.append(f"| {selector}/{engine} | {r['class_id']} | {r['support_mean']:.0f} | {fmt(r, 'delta_f1')} | "
                    f"{fmt(r, 'negative_flip_rate')} | {fmt(qlookup[selector, r['class_id']], 'source_correct_queried_count')} | {destinations} |")
    lines += ["", "## 统计口径与用途", "",
        "- 均值和样本标准差来自原训练种子；同一月份/源模型不是独立环境，Margin 查询集合相同。",
        "- Macro-F1 保留全部类别（含零支持类）；逐类表含支持数、精度、召回、F1、正负翻转及错→另一个错。",
        "- 负翻转率的分母为源正确样本；正翻转率的分母为源错误样本。空分母留空/NA。",
        "- transitions.csv 记录所有发生改变的真实类/源预测/修复预测三元组；汇总把某种子未出现的边计为零。",
        "- 已查询源正确样本是可用于后续保护的候选锚点。覆盖量仅是事后描述，不能证明加约束一定有效。",
        "- 完整推理改变了分批边界，若原抽样行出现预测差异，已计入上面的差异数；原抽样单独推理必须严格复现。",
        "- 本轮不更新模型、查询或原结果。M12 继续作为方法开发数据，不能据此宣称独立确认或方法新颖性。", ""]
    return "\n".join(lines)


def audit(args):
    source, out, base, info, protocol, choices, planned, signature = source_inputs(args)
    if out.exists() and not (out / "resolved_config.json").exists() and any(p.name != ".lock" for p in out.iterdir()):
        raise ValueError("Unregistered audit files exist; use a new output directory.")
    with kbs.file_lock(out / ".lock"):
        if kbs.is_complete(out, signature):
            print(f"SMALL_REPAIR_FULL_AUDIT_COMPLETE: {out / 'report.md'} (verified; skipped)", flush=True)
            return
        if args.command == "preflight":
            print(json.dumps({"status": "ready", "device": "cpu", "trained_heads": len(planned),
                "target_samples": info["sample_counts"]["target"], "new_fits": 0, "output": str(out)}, indent=2))
            return
        kbs.atomic_json(out / "resolved_config.json", signature)
        target = diag.load_role(base, info, "target")  # all fitting verified before offline truth access
        all_y = diag.numpy(target["labels"])
        count = protocol["expected_classes"]
        if all_y.min() < 0 or all_y.max() >= count:
            raise ValueError("Target labels outside the source class universe.")
        all_before = target.pop("logits").argmax(1).numpy()
        ids = diag.evaluation_indices(len(all_y), choices, 0, protocol["eval_seed"])
        small_ids = np.load(source / "selection/evaluation_rows.npy", allow_pickle=False)
        expected = diag.evaluation_indices(len(all_y), choices, protocol["eval_size"], protocol["eval_seed"])
        if not np.array_equal(small_ids, expected):
            raise ValueError("Source evaluation rows differ from frozen query exclusion/sampling.")
        y, before = all_y[ids], all_before[ids]
        coverage = query_coverage(all_y, all_before, choices, count)
        small_rows, small_classes, rows, classes, edges = [], [], [], [], []
        predictions = {"row_indices": ids, "labels": y, "source": before}
        mismatches, validated = 0, []
        with np.load(source / "evaluation/predictions.npz", allow_pickle=False) as old:
            expected_keys = {"row_indices", "labels", "source"} | {f"{c['selector']}_seed{c['seed']}_{e}" for c, e, _ in planned}
            if set(old.files) != expected_keys or not np.array_equal(old["row_indices"], small_ids) or not np.array_equal(
                    old["labels"], all_y[small_ids]) or not np.array_equal(old["source"], all_before[small_ids]):
                raise ValueError("Source predictions disagree with frozen rows/target labels/static predictions.")
            for i, (c, e, path) in enumerate(planned, 1):
                print(f"Reconcile sample {i}/{len(planned)}: {c['selector']} seed={c['seed']} {e}", flush=True)
                trace = json.loads((path / "trace.json").read_text())
                if trace["queried_labels"] != all_y[c["rows"]].tolist():
                    raise ValueError(f"Source queried truth differs: {path}")
                head = diag.head_from_cache(base)
                if e == "residual_mlp_care":
                    head = diag.ResidualHead(head, protocol["hidden"], c["seed"])
                head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
                if any(not torch.isfinite(p).all() for p in head.parameters()):
                    raise ValueError(f"Non-finite trained head: {path}")
                key = f"{c['selector']}_seed{c['seed']}_{e}"
                small_after = predict_rows(head, target["features"], small_ids, args.inference_batch_size)
                if not np.array_equal(small_after, old[key]):
                    raise ValueError(f"Saved head does not reproduce source sampled predictions: {key}. Check original threads/batch size.")
                meta = {"seed": c["seed"], "selector": c["selector"], "engine": e}
                sm, sc = scores(all_y[small_ids], all_before[small_ids], small_after, protocol, trace)
                small_rows.append({**meta, **sm})
                small_classes.extend({**meta, "low_support_lt20": r["support"] < 20, **r} for r in sc)
                validated.append((head, trace, meta, key))
            reconcile_csv(source / "evaluation/by_seed.csv", small_rows, ("seed", "selector", "engine"))
            reconcile_csv(source / "evaluation/per_class.csv", small_classes, ("seed", "selector", "engine", "class_id"))
            print("Original sampled predictions and metrics reconciled. Starting full evaluation.", flush=True)
            for i, (head, trace, meta, key) in enumerate(validated, 1):
                print(f"Full audit {i}/{len(validated)}: {key}", flush=True)
                started = time.perf_counter()
                after = predict_rows(head, target["features"], ids, args.inference_batch_size)
                differences = int((after[np.searchsorted(ids, small_ids)] != old[key]).sum())
                mismatches += differences
                metrics, per_class = scores(y, before, after, protocol, trace)
                rows.append({**meta, **metrics, "full_vs_sample_prediction_mismatches": differences,
                             "evaluation_wall_s": time.perf_counter() - started})
                classes.extend({**meta, **r} for r in class_flips(y, before, after, per_class, count))
                edges.extend({**meta, **r} for r in transitions(y, before, after, count))
                predictions[key] = after
                print(f"  done: Macro-F1={metrics['overall_macro_f1_after']:.6f}, negative flip rate={metrics['negative_flip_rate']}", flush=True)
        summaries = summarize(rows, ("selector", "engine"))
        pairs = diag.paired(rows, base)
        pair_summary = summarize(pairs, ("question", "comparison"))
        class_summary = summarize(classes, ("selector", "engine", "class_id"))
        edge_summary = transition_summary(edges, protocol["seeds"])
        coverage_summary = summarize(coverage, ("selector", "class_id"))
        kbs.atomic_json(out / "provenance.json", {"signature": signature, "environment": kbs.runtime_info(),
            "new_fits": 0, "new_queries": 0, "query_union_size": len(all_y) - len(ids),
            "source_sample_reconciliation": "passed", "source_sample_size": len(small_ids),
            "full_evaluation_size": len(ids), "full_vs_sample_prediction_mismatches": mismatches})
        artifacts = ["resolved_config.json", "provenance.json", "predictions.npz", "report.md"]
        for name, values in (("by_seed.csv", rows), ("summary.csv", summaries), ("per_class.csv", classes),
            ("per_class_summary.csv", class_summary), ("transitions.csv", edges), ("transitions_summary.csv", edge_summary),
            ("query_coverage.csv", coverage), ("query_coverage_summary.csv", coverage_summary),
            ("paired_by_seed.csv", pairs), ("paired_summary.csv", pair_summary)):
            kbs.write_csv(out / name, values)
            artifacts.append(name)
        np.savez_compressed(out / "predictions.npz", **predictions)
        (out / "report.md").write_text(report(protocol, summaries, pair_summary, class_summary, edge_summary,
            coverage_summary, rows[0], len(ids), len(all_y) - len(ids), mismatches))
        kbs.finish(out, signature, artifacts)
        print(f"SMALL_REPAIR_FULL_AUDIT_COMPLETE: {out / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "run"), nargs="?", default="run")
    p.add_argument("--source-dir", default=str(kbs.CORE / "outputs/small_repair_diagnostic_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/small_repair_full_audit_v1"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    return p


def main():
    args = parser().parse_args()
    if args.threads <= 0 or args.inference_batch_size <= 0:
        raise ValueError("Threads and inference batch size must be positive.")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    audit(args)
    print(f"Full audit wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
