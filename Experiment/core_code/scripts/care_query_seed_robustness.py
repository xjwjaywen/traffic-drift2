#!/usr/bin/env python3
"""Fresh acquisition-seed replication of balanced vs feature-coverage CARE.

Keep the original acquisition seed as a development control. Only fresh query
seeds enter the primary summary, after averaging paired training seeds within
each query seed. Freeze all queries before fitting and evaluate truth separately.
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

import care_mixed_query_ablation as ab
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import small_repair_full_audit as full

SCHEMA = "care-query-seed-robustness-v1"
SELECTORS = ("mixed_balanced", "mixed_coverage")
FRESH_SEEDS = [6174, 6175, 6176, 6177, 6178]
PARENT_EVAL_ARTIFACTS = {"resolved_config.json", "predictions.npz", "report.md", "by_seed.csv", "summary.csv",
    "per_class.csv", "paired_by_seed.csv", "paired_summary.csv", "query_coverage.csv"}
SELECTION_ARTIFACTS = {"resolved_config.json", "queries.json", "candidate_pools.json", "evaluation_rows.npy"}
TRACE_FIELDS = ("query_rows", "queried_labels", "query_class_count", "queried_class_histogram",
    "query_train_accuracy", "query_train_ce", "trainable_parameters", "source_correct_paid_count",
    "source_correct_paid_class_count", "source_correct_paid_histogram", "supplement_count",
    "supplement_true_class_count", "supplement_predicted_class_count", "supplement_source_correct_count", "paid_negative_flips")
METRICS = ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
           "stable_macro_f1_delta", "negative_flip_rate", "positive_flip_rate")


def seed_list(value):
    result = kbs.int_list(value)
    if not result or len(result) != len(set(result)) or any(i < 0 for i in result):
        raise ValueError("Fresh query seeds must be distinct nonnegative integers.")
    return result


def inventory(args):
    """Read small cache manifests only; presence is not tensor/hash validation."""
    root = Path(args.scan_root).resolve()
    started, visited, rows, errors = time.monotonic(), 0, [], []
    truncated = False
    if root.is_dir():
        for directory, dirs, files in os.walk(root, followlinks=False):
            path = Path(directory)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (path / d).is_symlink()
                             and d not in {"__pycache__", "node_modules"} and len(path.relative_to(root).parts) < 6)
            visited += len(dirs) + len(files)
            if visited > 100000 or time.monotonic() - started > 10:
                truncated = True
                break
            manifest = path / "manifest.json"
            if "manifest.json" not in files:
                continue
            try:
                if manifest.stat().st_size > 1024 ** 2:
                    continue
                data = json.loads(manifest.read_text())
                if (isinstance(data, dict) and isinstance(data.get("periods"), dict) and
                    isinstance(data.get("files"), dict) and set(data["files"]) == {"head.pt", "reference.pt", "target.pt"}):
                    rows.append({"directory": str(path), "periods": data["periods"], "num_classes": data.get("num_classes"),
                                 "sample_counts": data.get("sample_counts"),
                                 "all_tensor_files_present": all((path / n).is_file() for n in data["files"])})
            except (OSError, ValueError, TypeError) as exc:
                errors.append({"path": str(manifest), "error": str(exc)})
    result = {"scan_root": str(root), "root_exists": root.is_dir(), "truncated": truncated,
        "tensor_hashes_checked": False, "cache_metadata": rows, "errors": errors[:10],
        "prespecified_next_month": "M-2022-11",
        "m11_cache_metadata_found": any(r["periods"].get("reference") == "M-2022-4" and
            r["periods"].get("target") == "M-2022-11" and r["all_tensor_files_present"] for r in rows),
        "raw_year_h5_present": (kbs.CORE / "data/tls22/S/CESNET-TLS-Year22-S.h5").is_file(),
        "scope": "Metadata/presence only. This does not establish a usable month cache or extract new features."}
    print("MONTH_CACHE_INVENTORY: " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def inputs(args):
    out, parent = Path(args.output_dir).resolve(), Path(args.ablation_dir).resolve()
    for path in (parent, Path(args.source_dir).resolve(), Path(args.source_audit_dir).resolve(),
                 Path(args.protection_dir).resolve(), Path(args.cache_dir).resolve(), Path(args.checkpoint).resolve()):
        if path == out or path.is_relative_to(out) or out.is_relative_to(path):
            raise ValueError("Robustness output must be separate from all prior studies/cache/checkpoint.")
    saved = json.loads((parent / "manifest.json").read_text())["protocol"]
    aa = ab.parser().parse_args(["evaluate", "--source-dir", args.source_dir, "--source-audit-dir", args.source_audit_dir,
        "--protection-dir", args.protection_dir, "--cache-dir", args.cache_dir, "--checkpoint", args.checkpoint,
        "--output-dir", str(parent), "--threads", str(args.threads), "--inference-batch-size", str(args.inference_batch_size)])
    aa.eval_size, aa.eval_seed = saved["eval_size"], saved["eval_seed"]
    pc = ab.inputs(aa)
    ab.root_manifest(pc)
    previous = ab.selections(pc)
    for c in previous:
        ab.completed(ab.run_path(pc, c), ab.run_signature(pc, c), ab.RUN_ARTIFACTS)
    signature = {"protocol": pc["identity"], "stage": "evaluate",
        "selection_sha256": kbs.file_sha(parent / "selection/complete.json"),
        "runs": {str(ab.run_path(pc, c).relative_to(parent)): kbs.file_sha(ab.run_path(pc, c) / "complete.json") for c in previous}}
    ab.completed(parent / "evaluation", signature, PARENT_EVAL_ARTIFACTS)
    development = pc["identity"]["selection_seed"]
    fresh = seed_list(args.query_seeds)
    if development in fresh:
        raise ValueError("The development query seed must not enter the fresh replication set.")
    pc["base"].log_every = args.log_every
    parents = {}
    for c in previous:
        if c["selector"] in (*SELECTORS, "margin"):
            path = ab.run_path(pc, c)
            parents[f"{c['selector']}/seed_{c['seed']}"] = {"directory": str(path),
                "head_sha256": kbs.file_sha(path / "head.pt"), "complete_sha256": kbs.file_sha(path / "complete.json")}
    identity = {"schema": SCHEMA, "source_protocol": pc["protocol"], "parent_protocol": pc["identity"],
        "parent_evaluation_sha256": kbs.file_sha(parent / "evaluation/complete.json"), "reused_parents": parents,
        "development_query_seed": development, "fresh_query_seeds": fresh, "selectors": list(SELECTORS),
        "supplement_budget": pc["identity"]["supplement_budget"], "candidate_cap": pc["identity"]["candidate_cap"],
        "target_samples": pc["identity"]["target_samples"], "eval_size": args.eval_size, "eval_seed": args.eval_seed,
        "primary_summary": "paired training-seed means within each fresh query seed, then sample SD across fresh query seeds; development excluded",
        "code": {p.name: kbs.file_sha(p) for p in (Path(__file__), Path(__file__).with_name("run_care_query_seed_robustness.sh"))}}
    return dict(out=out, base=pc["base"], info=pc["info"], protocol=pc["protocol"], identity=identity,
                previous=previous, old_choices=pc["original"] + pc["parent_choices"] + previous)


def root_manifest(ctx, create=False):
    ab.root_manifest(ctx, create)


def selection_signature(ctx):
    return {"protocol": ctx["identity"], "stage": "select"}


def choice_key(c):
    return c["query_seed"], c["seed"], c["selector"]


def select(args, ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    with kbs.file_lock(out / ".lock"):
        root_manifest(ctx, True)
        directory, signature = out / "selection", selection_signature(ctx)
        if kbs.is_complete(directory, signature):
            ab.completed(directory, signature, SELECTION_ARTIFACTS)
            print("All query-seed replication selections verified; skipping.", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target", keep_labels=False)
        choices, registries = [], {}
        for qs in [ident["development_query_seed"], *ident["fresh_query_seeds"]]:
            mixed, registry = ab.select_supplements(target["features"], target["logits"], p["budget"],
                                                  ident["supplement_budget"], ident["candidate_cap"], qs)
            registries[str(qs)] = registry
            for seed in p["seeds"]:
                for q in SELECTORS:
                    c = {"query_seed": qs, "seed": seed, "selector": q, **mixed[q]}
                    if qs == ident["development_query_seed"]:
                        old = next(v for v in ctx["previous"] if v["seed"] == seed and v["selector"] == q)
                        if any(c[k] != old[k] for k in ("rows", "repair_rows", "supplement_rows")):
                            raise ValueError("Reconstructed development queries differ from completed ablation.")
                    choices.append(c)
        for seed in p["seeds"]:
            old = next(v for v in ctx["previous"] if v["seed"] == seed and v["selector"] == "margin")
            if old["rows"] != diag.select_indices("margin", target["logits"], p["budget"], seed).tolist():
                raise ValueError("Pure Margin source queries differ from current cache.")
            choices.append({"query_seed": -1, **old})
        ids = diag.evaluation_indices(ident["target_samples"], ctx["old_choices"] + choices, args.eval_size, args.eval_seed)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        kbs.atomic_json(directory / "queries.json", choices)
        kbs.atomic_json(directory / "candidate_pools.json", registries)
        np.save(directory / "evaluation_rows.npy", ids)
        kbs.finish(directory, signature, sorted(SELECTION_ARTIFACTS))
        print(f"Frozen every selection: {len(choices)} fits/reused heads; common evaluation {len(ids):,}.", flush=True)


def selections(ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    ab.completed(out / "selection", selection_signature(ctx), SELECTION_ARTIFACTS)
    choices = json.loads((out / "selection/queries.json").read_text())
    registries = json.loads((out / "selection/candidate_pools.json").read_text())
    query_seeds = [ident["development_query_seed"], *ident["fresh_query_seeds"]]
    expected = {(qs, s, q) for qs in query_seeds for s in p["seeds"] for q in SELECTORS}
    expected |= {(-1, s, "margin") for s in p["seeds"]}
    if len(choices) != len(expected) or {choice_key(c) for c in choices} != expected or set(registries) != set(map(str, query_seeds)):
        raise ValueError("Frozen query-seed replication grid mismatch.")
    lookup = {choice_key(c): c for c in choices}
    def valid(rows):
        return rows == sorted(set(rows)) and all(type(i) is int and 0 <= i < ident["target_samples"] for i in rows)
    repair = next(c["repair_rows"] for c in ctx["previous"] if c["selector"] == "mixed_coverage")
    for qs, reg in registries.items():
        pools = {int(cid): np.asarray(rows) for cid, rows in reg["pools"].items()}
        pool_rows = [i for v in reg["pools"].values() for i in v]
        if (reg["repair_rows"] != repair or not valid(sorted(pool_rows)) or set(pool_rows) & set(repair) or
            len(pool_rows) != reg["candidate_count"] or
            any(not valid(v) or len(v) > ident["candidate_cap"] or not 0 <= int(cid) < p["expected_classes"] for cid, v in reg["pools"].items()) or
            reg["quotas"] != {str(cid): n for cid, n in ab.balanced_quotas(pools, ident["supplement_budget"], int(qs)).items()}):
            raise ValueError("Invalid frozen candidate pools/quotas.")
    for c in choices:
        if (not all(valid(c[k]) for k in ("rows", "repair_rows", "supplement_rows")) or len(c["rows"]) != p["budget"] or
            c["rows"] != sorted(c["repair_rows"] + c["supplement_rows"])):
            raise ValueError("Invalid frozen query partition.")
        if c["selector"] == "margin" or c["query_seed"] == ident["development_query_seed"]:
            old = next(v for v in ctx["previous"] if v["seed"] == c["seed"] and v["selector"] == c["selector"])
            if any(c[k] != old[k] for k in ("rows", "repair_rows", "supplement_rows")):
                raise ValueError("Reused query/partition identity mismatch.")
        if c["selector"] == "margin":
            continue
        reg = registries[str(c["query_seed"])]
        extra = set(c["supplement_rows"])
        if c["repair_rows"] != repair or len(extra) != ident["supplement_budget"]:
            raise ValueError("Every mixed replication must share repair queries and label budget.")
        if any(len(extra.intersection(reg["pools"][cid])) != n for cid, n in reg["quotas"].items()):
            raise ValueError("Balanced and coverage class quotas differ.")
        if c["rows"] != lookup[c["query_seed"], p["seeds"][0], c["selector"]]["rows"]:
            raise ValueError("A query replication must be shared across its training seeds.")
    return choices


def run_path(ctx, c):
    qs = "baseline" if c["query_seed"] == -1 else str(c["query_seed"])
    return ctx["out"] / "runs" / f"query_seed_{qs}" / c["selector"] / f"seed_{c['seed']}" / "care"


def run_signature(ctx, c):
    return {"protocol": ctx["identity"], "query_seed": c["query_seed"], "selector": c["selector"], "seed": c["seed"],
            "engine": "care", "selection_sha256": kbs.file_sha(ctx["out"] / "selection/complete.json")}


def reused_parent(ctx, c):
    if c["selector"] == "margin" or c["query_seed"] == ctx["identity"]["development_query_seed"]:
        return ctx["identity"]["reused_parents"][f"{c['selector']}/seed_{c['seed']}"]
    return None


def train(args, ctx):
    root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        if all(kbs.is_complete(run_path(ctx, c), run_signature(ctx, c)) for c in choices):
            for c in choices:
                ab.completed(run_path(ctx, c), run_signature(ctx, c), ab.RUN_ARTIFACTS)
            print("All replicated fits/reused heads verified; skipping tensors/training.", flush=True)
            return
        base, p = ctx["base"], ctx["protocol"]
        head0 = diag.head_from_cache(base)
        target, reference = diag.load_role(base, ctx["info"], "target"), diag.load_role(base, ctx["info"], "reference")
        refs = {s: kbs.replay_indices(diag.numpy(reference["labels"]), p["expected_classes"], p["replay_per_class"], s) for s in p["seeds"]}
        for c in choices:
            directory, sig = run_path(ctx, c), run_signature(ctx, c)
            if kbs.is_complete(directory, sig):
                ab.completed(directory, sig, ab.RUN_ARTIFACTS)
                continue
            ridx = refs[c["seed"]]
            template = ctx["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old = json.loads((Path(template["directory"]) / "trace.json").read_text())
            if old["reference_rows"] != ridx.tolist() or old["reference_labels"] != diag.numpy(reference["labels"][ridx]).tolist():
                raise ValueError("Matched historical replay changed.")
            print(f"Prepare query_seed={c['query_seed']} {c['selector']} training_seed={c['seed']}", flush=True)
            kbs.atomic_json(directory / "resolved_config.json", sig)
            parent = reused_parent(ctx, c)
            if parent:
                path = Path(parent["directory"])
                trace = json.loads((path / "trace.json").read_text())
                if trace["query_rows"] != c["rows"] or trace["queried_labels"] != diag.numpy(target["labels"][c["rows"]]).tolist():
                    raise ValueError("Reused head paid queries/labels mismatch.")
                head = copy.deepcopy(head0)
                head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            else:
                head, trace = diag.fit_one(base, p, head0, target, reference, c, ridx, "linear_care")
            trace.pop("ablation_reused_parent", None)
            trace.update(query_seed=c["query_seed"], new_optimizer_steps=0 if parent else p["steps"], robustness_reused_parent=parent)
            trace = ab.annotate(head, trace, target, c, p["expected_classes"])
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite replicated CARE head.")
            if parent:
                shutil.copyfile(Path(parent["directory"]) / "head.pt", directory / "head.pt")
            else:
                torch.save(head.state_dict(), directory / "head.pt")
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, sig, sorted(ab.RUN_ARTIFACTS))
        print("All replicated fits complete; independent evaluation may read held-out truth.", flush=True)


def paired(rows, ctx):
    p, ident = ctx["protocol"], ctx["identity"]
    lookup = {choice_key(r): r for r in rows}
    comparisons = [("mixed_coverage", "mixed_balanced"), ("mixed_balanced", "margin"), ("mixed_coverage", "margin")]
    pairs = []
    for qs in [ident["development_query_seed"], *ident["fresh_query_seeds"]]:
        for seed in p["seeds"]:
            for a, b in comparisons:
                item = {"query_seed": qs, "seed": seed, "comparison": f"{a} - {b}"}
                for key in METRICS:
                    x, y = lookup[qs, seed, a][key], lookup[-1 if b == "margin" else qs, seed, b][key]
                    item[key] = x - y if x is not None and y is not None else None
                pairs.append(item)
    return pairs


def summarize_fresh(rows, pairs, ctx):
    """The experimental unit is a query draw, not a pooled optimizer replicate."""
    fresh = set(ctx["identity"]["fresh_query_seeds"])
    by_query = full.summarize(pairs, ("query_seed", "comparison"))
    query_arms = full.summarize(rows, ("query_seed", "selector"))
    units = [{"query_seed": r["query_seed"], "comparison": r["comparison"],
              **{k: r.get(k + "_mean") for k in METRICS}} for r in by_query if r["query_seed"] in fresh]
    primary = full.summarize([{k: v for k, v in r.items() if k != "query_seed"} for r in units], ("comparison",))
    for r in primary:
        members = [u for u in units if u["comparison"] == r["comparison"]]
        r["n_query_seeds"] = r.pop("n_seeds")
        r["macro_positive_query_seeds"] = sum(u["overall_macro_f1_after"] is not None and u["overall_macro_f1_after"] > 0 for u in members)
        r["collapse_positive_query_seeds"] = sum(u["collapse_macro_f1_after"] is not None and u["collapse_macro_f1_after"] > 0 for u in members)
        r["negative_flip_reduced_query_seeds"] = sum(u["negative_flip_rate"] is not None and u["negative_flip_rate"] < 0 for u in members)
    arm_units = [{"query_seed": r["query_seed"], "selector": r["selector"],
                  **{k: r.get(k + "_mean") for k in METRICS}} for r in query_arms if r["query_seed"] in fresh]
    fresh_arms = full.summarize([{k: v for k, v in r.items() if k != "query_seed"} for r in arm_units], ("selector",))
    for r in fresh_arms:
        r["n_query_seeds"] = r.pop("n_seeds")
    return query_arms, by_query, fresh_arms, primary


def report(ctx, query_arms, by_query, fresh_arms, primary, size, static):
    def fmt(r, key):
        mean, sd = r.get(key + "_mean"), r.get(key + "_sd")
        return "NA" if mean is None else f"{mean:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    p, ident = ctx["protocol"], ctx["identity"]
    baseline = next(r for r in query_arms if r["selector"] == "margin")
    lines = ["# CARE 类别均衡与特征覆盖：新选样种子复验", "",
        f"CPU；每臂标签 {p['budget']}＝共享修复 {p['budget'] - ident['supplement_budget']}＋补充 {ident['supplement_budget']}。",
        f"新选样种子 {ident['fresh_query_seeds']}，各配对训练种子 {p['seeds']}；新增 {2 * len(ident['fresh_query_seeds']) * len(p['seeds'])} 次拟合。",
        f"开发选样种子 {ident['development_query_seed']} 只作对照，不进入主汇总；纯 Margin 头也复用。",
        f"共同完整评价 {size:,} 行；Static Macro-F1={static:.6f}；复用 Margin Macro-F1={fmt(baseline, 'overall_macro_f1_after')}（± 为训练种子 SD）。", "",
        "## 新选样种子主结果（先平均训练种子，± 为选样种子 SD）", "",
        "| 查询 | Macro-F1 | 崩溃组 F1 | 稳定组 ΔF1 | 负翻转率 |",
        "|---|---:|---:|---:|---:|"]
    for r in fresh_arms:
        lines.append(f"| {r['selector']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 配对主结果：仅新选样种子", "",
        "| 对照 | Macro-F1 Δ | 崩溃组 F1 Δ | 负翻转率差 | Macro 增益种子数 | 崩溃增益种子数 | 误伤下降种子数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        n = r["n_query_seeds"]
        lines.append(f"| {r['comparison']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "negative_flip_rate")) +
            f" | {r['macro_positive_query_seeds']}/{n} | {r['collapse_positive_query_seeds']}/{n} | {r['negative_flip_reduced_query_seeds']}/{n} |")
    lines += ["", "## coverage−balanced：各选样种子的配对差异（± 为训练种子 SD）", "",
        "| 选样种子 | 用途 | Macro-F1 Δ | 崩溃组 F1 Δ | 稳定组 ΔF1 差 | 负翻转率差 |",
        "|---|---|---:|---:|---:|---:|"]
    for r in by_query:
        if r["comparison"] == "mixed_coverage - mixed_balanced":
            role = "开发对照" if r["query_seed"] == ident["development_query_seed"] else "新复验"
            lines.append(f"| {r['query_seed']} | {role} | " + " | ".join(fmt(r, k) for k in
                ("overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 判读", "",
        "- 主问题是特征覆盖在新查询抽样下是否仍优于相同候选池/相同预测类配额的类别均衡随机。",
        "- 每个选样种子独立重抽各类候选 reservoir、配额轮转顺序及层内随机样本；修复 800 行不变。",
        "- 每个选样种子内，两臂使用相同候选池/配额、训练初始化、参考行、抽样索引流和普通 CARE 损失。",
        "- 先计算每个选样种子内的配对训练种子均值，再以新选样种子作为主汇总单位；不把所有训练重复当作独立选样。",
        "- 所有查询先冻结；只拟合对应付费标签；全部新旧分类头完成后才评价未查询目标真值。",
        "- 所有头在同一新旧查询并集排除集重评，不直接比较旧表数字；开发选样种子从主汇总排除。",
        "- Macro/崩溃 F1 的改善与负翻转率下降分别判断；修复收益不等于保护有效。",
        "- 没有依据开发结果选择最优新种子或改超参数；主汇总包含全部预先指定的新选样种子。",
        "- 这仍是 M12 上的选样复验，不能证明跨月份或方法新颖性；缓存清单只是元数据/文件存在性检查。", ""]
    return "\n".join(lines)


def evaluate(args, ctx):
    root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        for c in choices:
            ab.completed(run_path(ctx, c), run_signature(ctx, c), ab.RUN_ARTIFACTS)
        signature = {"protocol": ctx["identity"], "stage": "evaluate",
            "selection_sha256": kbs.file_sha(ctx["out"] / "selection/complete.json"),
            "runs": {str(run_path(ctx, c).relative_to(ctx["out"])): kbs.file_sha(run_path(ctx, c) / "complete.json") for c in choices}}
        directory = ctx["out"] / "evaluation"
        if kbs.is_complete(directory, signature):
            print(f"CARE_QUERY_SEED_ROBUSTNESS_COMPLETE: {directory / 'report.md'} (verified; skipped)", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target")
        p = ctx["protocol"]
        ids = np.load(ctx["out"] / "selection/evaluation_rows.npy", allow_pickle=False)
        expected = diag.evaluation_indices(ctx["identity"]["target_samples"], ctx["old_choices"] + choices, args.eval_size, args.eval_seed)
        if not np.array_equal(ids, expected):
            raise ValueError("Evaluation rows disagree with the complete query union exclusion.")
        y, before = diag.numpy(target["labels"][ids]), target["logits"][ids].argmax(1).numpy()
        if y.min() < 0 or y.max() >= p["expected_classes"]:
            raise ValueError("Evaluation labels outside source class universe.")
        predictions = {"row_indices": ids, "labels": y, "source": before}
        rows, classes, coverage = [], [], []
        for i, c in enumerate(choices, 1):
            directory_run = run_path(ctx, c)
            print(f"Evaluate {i}/{len(choices)}: query_seed={c['query_seed']} {c['selector']} train_seed={c['seed']}", flush=True)
            head = diag.head_from_cache(ctx["base"])
            head.load_state_dict(torch.load(directory_run / "head.pt", map_location="cpu", weights_only=True))
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite replicated CARE head.")
            trace = json.loads((directory_run / "trace.json").read_text())
            wanted = ab.annotate(head, copy.deepcopy(trace), target, c, p["expected_classes"])
            if trace["query_seed"] != c["query_seed"] or any(trace[k] != wanted[k] for k in TRACE_FIELDS):
                raise ValueError("Replicated paid query trace disagrees with saved head/selection.")
            template = ctx["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old = json.loads((Path(template["directory"]) / "trace.json").read_text())
            for key in ("reference_rows", "reference_labels", "target_stream_sha256", "reference_stream_sha256",
                        "optimizer_steps", "target_presentations", "reference_ce_presentations", "reference_kd_presentations"):
                if trace[key] != old[key]:
                    raise ValueError(f"Matched replication training stream/replay differs: {key}")
            parent = reused_parent(ctx, c)
            if (trace["robustness_reused_parent"] != parent or trace["new_optimizer_steps"] != (0 if parent else p["steps"]) or
                (parent and kbs.file_sha(directory_run / "head.pt") != parent["head_sha256"])):
                raise ValueError("Reused replication head identity or step count differs.")
            after = full.predict_rows(head, target["features"], ids, args.inference_batch_size)
            metrics, per_class = full.scores(y, before, after, p, trace)
            meta = {"query_seed": c["query_seed"], "seed": c["seed"], "selector": c["selector"], "engine": "care"}
            rows.append({**meta, **metrics, "new_optimizer_steps": trace["new_optimizer_steps"],
                **{k: trace[k] for k in TRACE_FIELDS if k not in ("query_rows", "queried_labels", "queried_class_histogram", "source_correct_paid_histogram")}})
            classes.extend({**meta, **r} for r in full.class_flips(y, before, after, per_class, p["expected_classes"]))
            coverage.extend({**meta, "class_id": cid, "queried_count": q, "source_correct_paid_count": a}
                for cid, (q, a) in enumerate(zip(trace["queried_class_histogram"], trace["source_correct_paid_histogram"])))
            predictions[f"query{c['query_seed']}_{c['selector']}_seed{c['seed']}"] = after
        pairs = paired(rows, ctx)
        query_arms, by_query, fresh_arms, primary = summarize_fresh(rows, pairs, ctx)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        artifacts = ["resolved_config.json", "predictions.npz", "report.md"]
        for name, values in (("by_seed.csv", rows), ("per_query_summary.csv", query_arms),
            ("fresh_arm_summary.csv", fresh_arms), ("paired_by_seed.csv", pairs), ("paired_by_query.csv", by_query),
            ("fresh_paired_summary.csv", primary), ("per_class.csv", classes), ("query_coverage.csv", coverage)):
            kbs.write_csv(directory / name, values)
            artifacts.append(name)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        (directory / "report.md").write_text(report(ctx, query_arms, by_query, fresh_arms, primary, len(ids), rows[0]["overall_macro_f1_before"]))
        kbs.finish(directory, signature, artifacts)
        print(f"CARE_QUERY_SEED_ROBUSTNESS_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "inventory", "preflight", "select", "train", "evaluate"))
    for name, default in (("source-dir", "small_repair_diagnostic_v1"), ("source-audit-dir", "small_repair_full_audit_v1"),
        ("protection-dir", "care_target_protection_v1"), ("ablation-dir", "care_mixed_query_ablation_v1"),
        ("output-dir", "care_query_seed_robustness_v1")):
        p.add_argument("--" + name, default=str(kbs.CORE / "outputs" / default))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--scan-root", default=str(kbs.CORE / "outputs"))
    p.add_argument("--query-seeds", default=",".join(map(str, FRESH_SEEDS)))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--eval-size", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1729)
    p.add_argument("--log-every", type=int, default=460)
    return p


def main():
    args = parser().parse_args()
    fresh = seed_list(args.query_seeds)
    if min(args.threads, args.inference_batch_size) <= 0 or min(args.eval_size, args.eval_seed, args.log_every) < 0:
        raise ValueError("Invalid CPU/evaluation settings.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"device": "cpu", "fresh_query_seeds": fresh, "new_fits_per_training_seed": 2 * len(fresh),
            "reused_fits_per_training_seed": 3, "selectors": SELECTORS,
            "primary_unit": "fresh query-seed paired means; development query seed excluded", "eval_size": args.eval_size}, indent=2))
        return
    if args.command == "inventory":
        inventory(args)
        return
    started = time.perf_counter()
    ctx = inputs(args)
    if args.command == "preflight":
        if (ctx["out"] / "manifest.json").exists():
            root_manifest(ctx)
        elif ctx["out"].exists() and any(p.name != ".lock" for p in ctx["out"].iterdir()):
            raise ValueError("Unregistered output files; use a new output directory.")
        print(json.dumps({"status": "ready", "device": "cpu", "training_seeds": ctx["protocol"]["seeds"],
            "fresh_query_seeds": fresh, "development_query_seed": ctx["identity"]["development_query_seed"],
            "new_fits": 2 * len(fresh) * len(ctx["protocol"]["seeds"])}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, ctx)
    print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
