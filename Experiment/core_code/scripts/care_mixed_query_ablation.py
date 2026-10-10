#!/usr/bin/env python3
"""Frozen-budget query ablation with ordinary CARE, on CPU.

Reuse completed Margin and feature-coverage CARE heads. Freeze every new query
before fitting; evaluate unqueried truth in a separate process after all fits.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

import care_target_protection as protect
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import small_repair_full_audit as full

SCHEMA = "care-mixed-query-ablation-v1"
SELECTORS = ("margin", "mixed_random", "mixed_pool_random", "mixed_balanced", "mixed_coverage")
REUSED = {"margin": "margin", "mixed_coverage": "mixed"}
RUN_ARTIFACTS = {"resolved_config.json", "head.pt", "trace.json"}
SELECTION_ARTIFACTS = {"resolved_config.json", "queries.json", "evaluation_rows.npy", "candidate_pools.json"}
PARENT_EVAL_ARTIFACTS = {"resolved_config.json", "predictions.npz", "report.md", "by_seed.csv", "summary.csv",
    "per_class.csv", "paired_by_seed.csv", "paired_summary.csv", "query_coverage.csv", "query_coverage_summary.csv"}


def completed(directory, signature, artifacts):
    if not kbs.is_complete(directory, signature):
        raise ValueError(f"Required stage incomplete: {directory}")
    marker = json.loads((directory / "complete.json").read_text())
    if set(marker["artifacts"]) != artifacts or json.loads((directory / "resolved_config.json").read_text()) != signature:
        raise ValueError(f"Completed stage artifact/configuration mismatch: {directory}")


def inputs(args):
    parent = Path(args.protection_dir).resolve()
    out = Path(args.output_dir).resolve()
    for path in (parent, Path(args.source_dir).resolve(), Path(args.source_audit_dir).resolve(),
                 Path(args.cache_dir).resolve(), Path(args.checkpoint).resolve()):
        if path == out or path.is_relative_to(out) or out.is_relative_to(path):
            raise ValueError("Ablation output must be separate from all completed studies/cache/checkpoint.")
    saved = json.loads((parent / "manifest.json").read_text())["protocol"]
    pa = protect.parser().parse_args(["evaluate", "--source-dir", args.source_dir,
        "--source-audit-dir", args.source_audit_dir, "--cache-dir", args.cache_dir,
        "--checkpoint", args.checkpoint, "--output-dir", str(parent), "--threads", str(args.threads),
        "--inference-batch-size", str(args.inference_batch_size)])
    for key in ("protection_budget", "candidate_cap", "selection_seed", "margin_fraction", "margin_cap",
                "constraint_tolerance", "eval_size", "eval_seed"):
        setattr(pa, key, saved[key])
    source, _, base, info, protocol, original, parent_identity = protect.inputs(pa)
    protect.root_manifest(parent, parent_identity)
    parent_choices = protect.selections(parent, parent_identity)
    completed(parent / "selection", {"protocol": parent_identity, "stage": "select"},
              {"resolved_config.json", "queries.json", "evaluation_rows.npy"})
    planned = [(c, e, protect.run_path(parent, c, e)) for c in parent_choices for e in protect.ENGINES]
    for c, e, path in planned:
        completed(path, protect.run_signature(parent, c, e, parent_identity), RUN_ARTIFACTS)
    signature = {"protocol": parent_identity, "stage": "evaluate",
        "selection_sha256": kbs.file_sha(parent / "selection/complete.json"),
        "runs": {str(p.relative_to(parent)): kbs.file_sha(p / "complete.json") for _, _, p in planned}}
    completed(parent / "evaluation", signature, PARENT_EVAL_ARTIFACTS)
    base.log_every = args.log_every
    parents = {}
    for seed in protocol["seeds"]:
        for selector, pq in REUSED.items():
            c = next(c for c in parent_choices if c["seed"] == seed and c["selector"] == pq)
            path = protect.run_path(parent, c, "care")
            parents[f"{selector}/seed_{seed}"] = {"directory": str(path),
                "complete_sha256": kbs.file_sha(path / "complete.json"), "head_sha256": kbs.file_sha(path / "head.pt")}
    identity = {"schema": SCHEMA, "source_protocol": protocol, "parent_protocol": parent_identity,
        "parent_evaluation_sha256": kbs.file_sha(parent / "evaluation/complete.json"), "reused_parents": parents,
        "selectors": list(SELECTORS), "target_samples": info["sample_counts"]["target"],
        "supplement_budget": saved["protection_budget"], "candidate_cap": saved["candidate_cap"],
        "selection_seed": saved["selection_seed"], "eval_size": args.eval_size, "eval_seed": args.eval_seed,
        "code": {p.name: kbs.file_sha(p) for p in (Path(__file__), Path(__file__).with_name("run_care_mixed_query_ablation.sh"))}}
    return dict(source=source, out=out, base=base, info=info, protocol=protocol, original=original,
                parent=parent, parent_choices=parent_choices, identity=identity)


def candidate_pools(logits, repair, cap, seed):
    """Reproduce the parent's upper-half per-predicted-class reservoirs exactly."""
    available = np.ones(len(logits), dtype=bool)
    available[repair] = False
    predicted = logits.argmax(1).numpy()
    top = torch.topk(logits, 2, dim=1).values
    margins = (top[:, 0] - top[:, 1]).numpy()
    pools = {}
    for c in np.unique(predicted[available]):
        rows = np.flatnonzero(available & (predicted == c))
        rows = rows[margins[rows] >= np.median(margins[rows])]
        if len(rows) > cap:
            rows = np.sort(np.random.default_rng(seed + int(c) * 1009).choice(rows, cap, replace=False))
        pools[int(c)] = rows
    return pools, available, predicted


def balanced_quotas(pools, budget, seed):
    order = np.random.default_rng(seed).permutation(sorted(pools)).tolist()
    quotas = {c: 0 for c in pools}
    while sum(quotas.values()) < budget:
        progressed = False
        for c in order:
            if quotas[c] == len(pools[c]):
                continue
            quotas[c] += 1
            progressed = True
            if sum(quotas.values()) == budget:
                break
        if not progressed:
            raise ValueError("Candidate reservoirs cannot fill the fixed supplement budget.")
    return quotas


def select_supplements(features, logits, budget, supplement_budget, cap, seed):
    """No label argument: confidence pool, class quotas and feature geometry are separate."""
    coverage = protect.mixed_selection(features, logits, budget, supplement_budget, cap, seed)
    repair = coverage["repair_rows"]
    pools, available, predicted = candidate_pools(logits, repair, cap, seed)
    quotas = balanced_quotas(pools, supplement_budget, seed)
    actual = np.bincount(predicted[coverage["protection_rows"]], minlength=logits.shape[1])
    if any(actual[c] != quotas[c] for c in pools):
        raise ValueError("Feature coverage and balanced control have different class quotas.")
    candidates = np.sort(np.concatenate(list(pools.values())))
    supplements = {
        "mixed_random": np.random.default_rng(seed + 101).choice(np.flatnonzero(available), supplement_budget, replace=False).tolist(),
        "mixed_pool_random": np.random.default_rng(seed + 211).choice(candidates, supplement_budget, replace=False).tolist(),
        "mixed_balanced": [int(i) for c in sorted(pools) for i in
            np.random.default_rng(seed + 307 + c * 1009).choice(pools[c], quotas[c], replace=False)],
        "mixed_coverage": coverage["protection_rows"],
    }
    queries = {q: {"rows": sorted(repair + rows), "repair_rows": repair,
                   "supplement_rows": sorted(rows)} for q, rows in supplements.items()}
    registry = {"repair_rows": repair, "pools": {str(c): v.tolist() for c, v in pools.items()},
                "quotas": {str(c): v for c, v in quotas.items()}, "candidate_count": len(candidates)}
    return queries, registry


def root_manifest(ctx, create=False):
    protect.root_manifest(ctx["out"], ctx["identity"], create)


def selection_signature(ctx):
    return {"protocol": ctx["identity"], "stage": "select"}


def select(args, ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    with kbs.file_lock(out / ".lock"):
        root_manifest(ctx, True)
        directory, signature = out / "selection", selection_signature(ctx)
        if kbs.is_complete(directory, signature):
            completed(directory, signature, SELECTION_ARTIFACTS)
            print("Verified frozen ablation queries; skipping.", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target", keep_labels=False)
        mixed, registry = select_supplements(target["features"], target["logits"], p["budget"],
            ident["supplement_budget"], ident["candidate_cap"], ident["selection_seed"])
        choices = []
        for seed in p["seeds"]:
            old = next(c for c in ctx["parent_choices"] if c["seed"] == seed and c["selector"] == "margin")
            cover = next(c for c in ctx["parent_choices"] if c["seed"] == seed and c["selector"] == "mixed")
            if (old["rows"] != diag.select_indices("margin", target["logits"], p["budget"], seed).tolist() or
                mixed["mixed_coverage"]["rows"] != cover["rows"] or
                registry["repair_rows"] != cover["repair_rows"] or
                mixed["mixed_coverage"]["supplement_rows"] != cover["protection_rows"]):
                raise ValueError("Reconstructed queries disagree with the completed Margin/coverage parents.")
            choices.append({"seed": seed, "selector": "margin", "rows": old["rows"],
                            "repair_rows": old["rows"], "supplement_rows": []})
            choices.extend({"seed": seed, "selector": q, **mixed[q]} for q in SELECTORS[1:])
        ids = diag.evaluation_indices(ident["target_samples"], ctx["original"] + ctx["parent_choices"] + choices,
                                      args.eval_size, args.eval_seed)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        kbs.atomic_json(directory / "queries.json", choices)
        kbs.atomic_json(directory / "candidate_pools.json", registry)
        np.save(directory / "evaluation_rows.npy", ids)
        kbs.finish(directory, signature, sorted(SELECTION_ARTIFACTS))
        print(f"All queries frozen: {len(choices)} arms/seeds, {p['budget']} labels each; common evaluation {len(ids):,}.", flush=True)


def selections(ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    completed(out / "selection", selection_signature(ctx), SELECTION_ARTIFACTS)
    choices = json.loads((out / "selection/queries.json").read_text())
    expected = {(s, q) for s in p["seeds"] for q in SELECTORS}
    if len(choices) != len(expected) or {(c["seed"], c["selector"]) for c in choices} != expected:
        raise ValueError("Frozen ablation query grid mismatch.")
    registry = json.loads((out / "selection/candidate_pools.json").read_text())
    pool_rows = [i for rows in registry["pools"].values() for i in rows]
    def valid(rows):
        return rows == sorted(set(rows)) and all(type(i) is int and 0 <= i < ident["target_samples"] for i in rows)
    if (not valid(sorted(pool_rows)) or len(pool_rows) != registry["candidate_count"] or
        not valid(registry["repair_rows"]) or set(pool_rows) & set(registry["repair_rows"])):
        raise ValueError("Invalid frozen candidate registry.")
    parent_cover = next(c for c in ctx["parent_choices"] if c["selector"] == "mixed")
    pools = {int(cid): np.asarray(v) for cid, v in registry["pools"].items()}
    if (registry["repair_rows"] != parent_cover["repair_rows"] or
        any(not valid(v) or len(v) > ident["candidate_cap"] or not 0 <= int(cid) < p["expected_classes"]
            for cid, v in registry["pools"].items()) or
        registry["quotas"] != {str(c): n for c, n in balanced_quotas(pools, ident["supplement_budget"], ident["selection_seed"]).items()}):
        raise ValueError("Frozen candidate pools/quotas disagree with inherited query settings.")
    by_id = {(c["seed"], c["selector"]): c for c in choices}
    for c in choices:
        rows, repair, extra = c["rows"], c["repair_rows"], c["supplement_rows"]
        if (not all(valid(v) for v in (rows, repair, extra)) or len(rows) != p["budget"] or
            rows != sorted(repair + extra)):
            raise ValueError("Invalid frozen query partition.")
        pq = REUSED.get(c["selector"])
        if pq:
            parent = next(v for v in ctx["parent_choices"] if v["seed"] == c["seed"] and v["selector"] == pq)
            if rows != parent["rows"]:
                raise ValueError("Reused parent query mismatch.")
            if c["selector"] == "mixed_coverage" and extra != parent["protection_rows"]:
                raise ValueError("Reused coverage supplement partition mismatch.")
        if c["selector"] == "margin":
            if extra or repair != rows:
                raise ValueError("Invalid pure Margin query partition.")
        else:
            if len(extra) != ident["supplement_budget"] or repair != registry["repair_rows"]:
                raise ValueError("Mixed arms must have identical repair queries and supplement budgets.")
            if c["selector"] != "mixed_random" and not set(extra).issubset(pool_rows):
                raise ValueError("Supplement outside the common confidence candidate pool.")
        if c["rows"] != by_id[p["seeds"][0], c["selector"]]["rows"]:
            raise ValueError("Training seeds must share each frozen acquisition set.")
        if c["selector"] in ("mixed_balanced", "mixed_coverage"):
            selected = set(extra)
            if any(len(selected.intersection(registry["pools"][cid])) != quota for cid, quota in registry["quotas"].items()):
                raise ValueError("Balanced/coverage class quotas differ.")
    return choices


def run_path(ctx, choice):
    return ctx["out"] / "runs" / choice["selector"] / f"seed_{choice['seed']}" / "care"


def run_signature(ctx, choice):
    return {"protocol": ctx["identity"], "selector": choice["selector"], "seed": choice["seed"],
            "engine": "care", "selection_sha256": kbs.file_sha(ctx["out"] / "selection/complete.json")}


def annotate(head, trace, target, choice, classes):
    idx = np.asarray(choice["rows"], dtype=np.int64)
    y = diag.numpy(target["labels"][idx])
    if y.min() < 0 or y.max() >= classes:
        raise ValueError("Queried labels outside source class universe.")
    before = target["logits"][idx].argmax(1).numpy()
    extra = np.isin(idx, choice["supplement_rows"])
    correct = before == y
    with torch.no_grad():
        logits = head(target["features"][idx])
        trace.update(query_rows=idx.tolist(), queried_labels=y.tolist(),
            query_class_count=len(np.unique(y)), queried_class_histogram=np.bincount(y, minlength=classes).tolist(),
            query_train_accuracy=float((logits.argmax(1).numpy() == y).mean()),
            query_train_ce=float(F.cross_entropy(logits, torch.as_tensor(y).long())),
            trainable_parameters=sum(v.numel() for v in head.parameters()),
            source_correct_paid_count=int(correct.sum()), source_correct_paid_class_count=len(np.unique(y[correct])),
            source_correct_paid_histogram=np.bincount(y[correct], minlength=classes).tolist(),
            supplement_count=int(extra.sum()), supplement_true_class_count=len(np.unique(y[extra])),
            supplement_predicted_class_count=len(np.unique(before[extra])),
            supplement_source_correct_count=int((extra & correct).sum()),
            paid_negative_flips=int((correct & (logits.argmax(1).numpy() != y)).sum()))
    return trace


def train(args, ctx):
    root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        if all(kbs.is_complete(run_path(ctx, c), run_signature(ctx, c)) for c in choices):
            for c in choices:
                completed(run_path(ctx, c), run_signature(ctx, c), RUN_ARTIFACTS)
            print("All CARE fits/reused heads verified; skipping tensors/training.", flush=True)
            return
        base, p = ctx["base"], ctx["protocol"]
        head0 = diag.head_from_cache(base)
        target, reference = diag.load_role(base, ctx["info"], "target"), diag.load_role(base, ctx["info"], "reference")
        refs = {s: kbs.replay_indices(diag.numpy(reference["labels"]), p["expected_classes"], p["replay_per_class"], s) for s in p["seeds"]}
        for c in choices:
            directory, signature = run_path(ctx, c), run_signature(ctx, c)
            if kbs.is_complete(directory, signature):
                completed(directory, signature, RUN_ARTIFACTS)
                continue
            ridx = refs[c["seed"]]
            template = ctx["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old = json.loads((Path(template["directory"]) / "trace.json").read_text())
            if old["reference_rows"] != ridx.tolist() or old["reference_labels"] != diag.numpy(reference["labels"][ridx]).tolist():
                raise ValueError("Historical replay differs from the matched original seed.")
            print(f"Prepare {c['selector']} seed={c['seed']} ordinary CARE", flush=True)
            kbs.atomic_json(directory / "resolved_config.json", signature)
            if c["selector"] in REUSED:
                parent = ctx["identity"]["reused_parents"][f"{c['selector']}/seed_{c['seed']}"]
                path = Path(parent["directory"])
                trace = json.loads((path / "trace.json").read_text())
                if trace["query_rows"] != c["rows"] or trace["queried_labels"] != diag.numpy(target["labels"][c["rows"]]).tolist():
                    raise ValueError("Reused CARE paid queries/labels mismatch.")
                head = copy.deepcopy(head0)
                head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
                trace.update(new_optimizer_steps=0, ablation_reused_parent=parent)
            else:
                head, trace = diag.fit_one(base, p, head0, target, reference, c, ridx, "linear_care")
                trace.update(new_optimizer_steps=p["steps"], ablation_reused_parent=None)
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite completed CARE head.")
            trace = annotate(head, trace, target, c, p["expected_classes"])
            if c["selector"] in REUSED:
                shutil.copyfile(Path(parent["directory"]) / "head.pt", directory / "head.pt")
            else:
                torch.save(head.state_dict(), directory / "head.pt")
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, signature, sorted(RUN_ARTIFACTS))
        print("All fits complete; separate evaluation may now read held-out truth.", flush=True)


def paired(rows, p):
    lookup = {(r["seed"], r["selector"]): r for r in rows}
    comparisons = [("mixed_pool_random", "mixed_random", "confidence_candidate_pool"),
        ("mixed_balanced", "mixed_pool_random", "predicted_class_balance"),
        ("mixed_coverage", "mixed_balanced", "feature_geometry")]
    comparisons += [(q, "margin", "mixed_vs_margin") for q in SELECTORS[1:]]
    result = []
    for seed in p["seeds"]:
        for a, b, question in comparisons:
            item = {"seed": seed, "comparison": f"{a} - {b}", "question": question}
            for key in ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
                        "stable_macro_f1_delta", "negative_flip_rate", "positive_flip_rate"):
                x, y = lookup[seed, a][key], lookup[seed, b][key]
                item[key] = x - y if x is not None and y is not None else None
            result.append(item)
    return result


def report(ctx, summaries, pairs, size, static):
    def fmt(r, key):
        value, sd = r.get(key + "_mean"), r.get(key + "_sd")
        return "NA" if value is None else f"{value:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    p, ident = ctx["protocol"], ctx["identity"]
    lines = ["# CARE 混合查询消融", "",
        f"CPU；目标标签预算 {p['budget']}；四个 mixed 臂共用 {p['budget'] - ident['supplement_budget']} 个 Margin 修复查询＋{ident['supplement_budget']} 个补充查询。",
        f"训练种子 {p['seeds']}；每种子新增 3 次普通 CARE 拟合，复用 Margin/特征覆盖头；没有硬约束。",
        f"共同评价 {size:,} 条，排除全部新旧查询并集；Static Macro-F1={static:.6f}。", "",
        "| 查询 | Macro-F1 | 崩溃组 F1 | 稳定组 ΔF1 | 负翻转率 | 正翻转率 | 已付费源正确类别数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['selector']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_delta", "negative_flip_rate",
             "positive_flip_rate", "source_correct_paid_class_count")) + " |")
    lines += ["", "## 配对消融（前者减后者）", "",
        "| 对照 | Macro-F1 Δ | 崩溃组 F1 Δ | 稳定组 ΔF1 差 | 负翻转率差 |",
        "|---|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['comparison']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 补充查询覆盖（事后标签统计）", "",
        "| 查询 | 补充查询数量 | 预测类别数 | 真实类别数 | 补充中源正确数 | 全部已付费源正确数 | 已付费负翻转 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['selector']} | " + " | ".join(fmt(r, k) for k in
            ("supplement_count", "supplement_predicted_class_count", "supplement_true_class_count",
             "supplement_source_correct_count", "source_correct_paid_count", "paid_negative_flips")) + " |")
    lines += ["", "## 判读", "",
        "- mixed_random：排除共享修复查询后，在全部剩余目标行中均匀随机抽样。",
        "- mixed_pool_random：在特征覆盖使用的相同高置信候选池内均匀随机抽样。",
        "- mixed_balanced：在同一候选池内按源预测类别随机抽样，类别配额与特征覆盖完全相同。",
        "- mixed_coverage：复用上一轮查询和普通 CARE 头；类别首点接近特征中心，再按最远点补覆盖。",
        "- pool_random−random 检查候选池效应；balanced−pool_random 检查类别均衡；coverage−balanced 检查特征几何。",
        "- 候选池沿用各预测类间隔上半区及固定随机 reservoir；类别均衡针对预测类别，未使用真实类别。",
        "- 四个 mixed 臂总预算、修复查询、CE/KD 权重、初始化、步数、各种子的参考行和抽样索引流均匹配。",
        "- 标签只在全部查询冻结后用于各臂已付费查询的拟合；全部分类头完成后才独立评价未查询真值。",
        "- 本轮评价排除了更大的查询并集；表中的复用基线已重评，不能直接拿上一轮表中数值计算差异。",
        "- 三个训练种子共用每臂同一个查询集；± 是种子标准差，不是置信区间或选样重复。",
        "- 同一 M12 开发数据的消融不证明跨月份收益、全池安全或方法新颖性。", ""]
    return "\n".join(lines)


def evaluate(args, ctx):
    root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        for c in choices:
            completed(run_path(ctx, c), run_signature(ctx, c), RUN_ARTIFACTS)
        signature = {"protocol": ctx["identity"], "stage": "evaluate",
            "selection_sha256": kbs.file_sha(ctx["out"] / "selection/complete.json"),
            "runs": {str(run_path(ctx, c).relative_to(ctx["out"])): kbs.file_sha(run_path(ctx, c) / "complete.json") for c in choices}}
        directory = ctx["out"] / "evaluation"
        if kbs.is_complete(directory, signature):
            print(f"CARE_MIXED_QUERY_ABLATION_COMPLETE: {directory / 'report.md'} (verified; skipped)", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target")
        p = ctx["protocol"]
        ids = np.load(ctx["out"] / "selection/evaluation_rows.npy", allow_pickle=False)
        expected = diag.evaluation_indices(ctx["identity"]["target_samples"],
            ctx["original"] + ctx["parent_choices"] + choices, args.eval_size, args.eval_seed)
        if not np.array_equal(ids, expected):
            raise ValueError("Evaluation rows differ from the complete new/old query exclusion.")
        y = diag.numpy(target["labels"][ids])
        before = target["logits"][ids].argmax(1).numpy()
        if y.min() < 0 or y.max() >= p["expected_classes"]:
            raise ValueError("Evaluation labels outside source class universe.")
        predictions = {"row_indices": ids, "labels": y, "source": before}
        rows, classes, coverage = [], [], []
        for i, c in enumerate(choices, 1):
            path = run_path(ctx, c)
            print(f"Evaluate {i}/{len(choices)}: {c['selector']} seed={c['seed']}", flush=True)
            head = diag.head_from_cache(ctx["base"])
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite completed CARE head.")
            trace = json.loads((path / "trace.json").read_text())
            wanted = annotate(head, copy.deepcopy(trace), target, c, p["expected_classes"])
            checked = ("query_rows", "queried_labels", "query_class_count", "queried_class_histogram",
                "query_train_accuracy", "query_train_ce", "trainable_parameters", "source_correct_paid_count",
                "source_correct_paid_class_count", "source_correct_paid_histogram", "supplement_count",
                "supplement_true_class_count", "supplement_predicted_class_count", "supplement_source_correct_count", "paid_negative_flips")
            if any(trace[key] != wanted[key] for key in checked):
                raise ValueError("Paid query trace disagrees with saved head/queries.")
            template = ctx["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            original_trace = json.loads((Path(template["directory"]) / "trace.json").read_text())
            for key in ("reference_rows", "reference_labels", "target_stream_sha256", "reference_stream_sha256",
                        "optimizer_steps", "target_presentations", "reference_ce_presentations", "reference_kd_presentations"):
                if trace[key] != original_trace[key]:
                    raise ValueError(f"Matched CARE replay/streams/steps differ: {key}")
            parent = ctx["identity"]["reused_parents"].get(f"{c['selector']}/seed_{c['seed']}")
            if (trace["ablation_reused_parent"] != parent or
                trace["new_optimizer_steps"] != (0 if parent else p["steps"]) or
                (parent and kbs.file_sha(path / "head.pt") != parent["head_sha256"])):
                raise ValueError("Reused head identity or new optimizer step count differs.")
            after = full.predict_rows(head, target["features"], ids, args.inference_batch_size)
            metrics, per_class = full.scores(y, before, after, p, trace)
            meta = {"selector": c["selector"], "seed": c["seed"], "engine": "care"}
            rows.append({**meta, **metrics, **{k: trace[k] for k in checked if k not in
                ("query_rows", "queried_labels", "queried_class_histogram", "source_correct_paid_histogram")},
                "new_optimizer_steps": trace["new_optimizer_steps"]})
            classes.extend({**meta, **r} for r in full.class_flips(y, before, after, per_class, p["expected_classes"]))
            coverage.extend({**meta, "class_id": cid, "queried_count": q, "source_correct_paid_count": a}
                for cid, (q, a) in enumerate(zip(trace["queried_class_histogram"], trace["source_correct_paid_histogram"])))
            predictions[f"{c['selector']}_seed{c['seed']}"] = after
        summaries = full.summarize(rows, ("selector",))
        pairs = paired(rows, p)
        pair_summary = full.summarize(pairs, ("question", "comparison"))
        kbs.atomic_json(directory / "resolved_config.json", signature)
        artifacts = ["resolved_config.json", "predictions.npz", "report.md"]
        for name, values in (("by_seed.csv", rows), ("summary.csv", summaries), ("per_class.csv", classes),
                ("paired_by_seed.csv", pairs), ("paired_summary.csv", pair_summary), ("query_coverage.csv", coverage)):
            kbs.write_csv(directory / name, values)
            artifacts.append(name)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        (directory / "report.md").write_text(report(ctx, summaries, pair_summary, len(ids), rows[0]["overall_macro_f1_before"]))
        kbs.finish(directory, signature, artifacts)
        print(f"CARE_MIXED_QUERY_ABLATION_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "preflight", "select", "train", "evaluate"))
    for name, default in (("source-dir", "small_repair_diagnostic_v1"), ("source-audit-dir", "small_repair_full_audit_v1"),
                          ("protection-dir", "care_target_protection_v1"), ("output-dir", "care_mixed_query_ablation_v1")):
        p.add_argument("--" + name, default=str(kbs.CORE / "outputs" / default))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--eval-size", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1729)
    p.add_argument("--log-every", type=int, default=460)
    return p


def main():
    args = parser().parse_args()
    if min(args.threads, args.inference_batch_size) <= 0 or min(args.eval_size, args.eval_seed, args.log_every) < 0:
        raise ValueError("Invalid CPU/evaluation settings.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"device": "cpu", "selectors": SELECTORS, "new_fits_per_seed": 3,
                          "reused_fits_per_seed": 2, "eval_size": args.eval_size}, indent=2))
        return
    started = time.perf_counter()
    ctx = inputs(args)
    if args.command == "preflight":
        if (ctx["out"] / "manifest.json").exists():
            root_manifest(ctx)
        elif ctx["out"].exists() and any(p.name != ".lock" for p in ctx["out"].iterdir()):
            raise ValueError("Unregistered output files; use a new output directory.")
        print(json.dumps({"status": "ready", "device": "cpu", "seeds": ctx["protocol"]["seeds"],
            "budget": ctx["protocol"]["budget"], "supplement_budget": ctx["identity"]["supplement_budget"],
            "new_fits": 3 * len(ctx["protocol"]["seeds"])}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, ctx)
    print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
