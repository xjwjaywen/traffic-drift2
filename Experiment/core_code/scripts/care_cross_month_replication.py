#!/usr/bin/env python3
"""Prespecified M11 replication of ordinary CARE query policies, on CPU.

Reuse the verified M4 cache; extract M11 once with the frozen source backbone.
Freeze every M11 query before fitting. Only a separate, all-fits-complete
evaluation stage reads held-out M11 truth. Stored labels remain in legacy .pt
files, so label separation is an API contract, not physical isolation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

import care_query_seed_robustness as robust
import care_mixed_query_ablation as ab
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import small_repair_full_audit as full

SCHEMA = "care-m11-replication-v1"
TARGET = "M-2022-11"
PARENT_ARTIFACTS = {"resolved_config.json", "predictions.npz", "report.md", "by_seed.csv",
    "per_query_summary.csv", "fresh_arm_summary.csv", "paired_by_seed.csv", "paired_by_query.csv",
    "fresh_paired_summary.csv", "per_class.csv", "query_coverage.csv"}
EVAL_ARTIFACTS = PARENT_ARTIFACTS | {"evaluation_groups.json"}
STREAM_FIELDS = ("reference_rows", "reference_labels", "target_stream_sha256", "reference_stream_sha256",
    "optimizer_steps", "target_presentations", "reference_ce_presentations", "reference_kd_presentations")
METRICS = robust.METRICS + ("m11_low_f1_macro_after", "m11_low_f1_macro_delta")


def code_identity():
    paths = (Path(__file__), Path(__file__).with_name("run_care_cross_month_replication.sh"))
    return {p.name: kbs.file_sha(p) for p in paths}


def separate(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError(f"New cache/output must be separate from every input: {a}, {b}")


def parent_context(args):
    parent = Path(args.replication_dir).resolve()
    saved = json.loads((parent / "manifest.json").read_text())["protocol"]
    ra = robust.parser().parse_args(["evaluate", "--source-dir", args.source_dir,
        "--source-audit-dir", args.source_audit_dir, "--protection-dir", args.protection_dir,
        "--ablation-dir", args.ablation_dir, "--cache-dir", args.source_cache_dir,
        "--checkpoint", args.checkpoint, "--output-dir", str(parent), "--threads", str(args.threads),
        "--query-seeds", ",".join(map(str, saved["fresh_query_seeds"])),
        "--eval-size", str(saved["eval_size"]), "--eval-seed", str(saved["eval_seed"])])
    pc = robust.inputs(ra)
    robust.root_manifest(pc)
    choices = robust.selections(pc)
    for c in choices:
        ab.completed(robust.run_path(pc, c), robust.run_signature(pc, c), ab.RUN_ARTIFACTS)
    signature = {"protocol": pc["identity"], "stage": "evaluate",
        "selection_sha256": kbs.file_sha(parent / "selection/complete.json"),
        "runs": {str(robust.run_path(pc, c).relative_to(parent)):
            kbs.file_sha(robust.run_path(pc, c) / "complete.json") for c in choices}}
    ab.completed(parent / "evaluation", signature, PARENT_ARTIFACTS)
    if saved["fresh_query_seeds"] != robust.FRESH_SEEDS or saved["eval_size"] != 0:
        raise ValueError("M11 protocol requires the five prespecified query seeds and completed full M12 evaluation.")
    old_paths = [args.source_dir, args.source_audit_dir, args.protection_dir, args.ablation_dir,
                 args.replication_dir, args.source_cache_dir, args.checkpoint]
    for new in (args.cache_dir, args.output_dir):
        for old in old_paths:
            separate(new, old)
    separate(args.cache_dir, args.output_dir)
    return pc


def cache_signature(args, pc):
    cfg = copy.deepcopy(pc["info"]["spec"]["data_config"])
    if args.data_dir:
        cfg["data_dir"] = str(Path(args.data_dir).expanduser().resolve())
    spec = {**pc["info"]["spec"], "data_config": cfg,
            "periods": {"reference": "M-2022-4", "target": TARGET}}
    if (cfg.get("dataset") != "tls22" or spec["data_seed"] != 0 or
        pc["info"]["periods"] != {"reference": "M-2022-4", "target": "M-2022-12"}):
        raise ValueError("Expected TLS22 M4/M12 source cache with data seed zero.")
    separate(args.cache_dir, cfg["data_dir"])
    separate(args.output_dir, cfg["data_dir"])
    return {"schema": SCHEMA, "stage": "prepare", "spec": spec, "code": code_identity(),
        "source_manifest_sha256": kbs.file_sha(Path(args.source_cache_dir) / "manifest.json"),
        "source_fingerprint": pc["info"]["fingerprint"],
        "reused_files": {n: pc["info"]["files"][n] for n in ("head.pt", "reference.pt")},
        "reference_input_sha256": pc["info"]["input_stream_sha256"]["reference"],
        "parent_evaluation_sha256": kbs.file_sha(Path(args.replication_dir) / "evaluation/complete.json")}


def validate_month_cache(args, pc, sig):
    root = Path(args.cache_dir).resolve()
    info = json.loads((root / "manifest.json").read_text())
    if (info.get("m11_preparation") != sig or info["spec"] != sig["spec"] or
        info["periods"] != sig["spec"]["periods"] or
        info["checkpoint_sha256"] != pc["info"]["checkpoint_sha256"] or
        info["num_classes"] != pc["info"]["num_classes"] or
        set(info["files"]) != {"head.pt", "reference.pt", "target.pt"} or
        any(info["files"][n] != sha for n, sha in sig["reused_files"].items()) or
        info["sample_counts"]["reference"] != pc["info"]["sample_counts"]["reference"] or
        info["input_stream_sha256"]["reference"] != sig["reference_input_sha256"] or
        info["fingerprint"] != kbs.digest({"spec": info["spec"], "inputs": info["input_stream_sha256"]})):
        raise ValueError("M11 cache provenance/source/reference mismatch; use a new cache directory.")
    kbs.validate_cache_files(root, info)
    if info["sample_counts"]["target"] <= pc["protocol"]["budget"]:
        raise ValueError("M11 target pool cannot support the fixed label budget.")
    return info


def collect_target(args, spec, pc):
    """Only source inference; never fit, score, select, or tune using cached labels."""
    import prototype_recalibration_tls22 as proto
    kbs.seed_all(0)
    model, _, classes = proto.load_source_model(args.checkpoint, torch.device("cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    expected = torch.load(Path(args.source_cache_dir) / "head.pt", map_location="cpu", weights_only=True)
    state = model.cls_head.state_dict()
    if set(state) != {"fc.weight", "fc.bias"} or classes != pc["info"]["num_classes"]:
        raise ValueError("Source backbone class count/head format differs from verified cache.")
    if any(not torch.equal(state["fc." + k].cpu(), v) for k, v in expected.items()):
        raise ValueError("Source backbone head differs from cached head.")
    cfg = {"data": {**spec["data_config"], "num_classes": classes}}
    kbs.seed_all(0)
    print(f"Building {TARGET} loader from {cfg['data']['data_dir']} (CPU; no download provisioning).", flush=True)
    loader, loader_classes = proto.make_test_loader(cfg, TARGET)
    if loader_classes != classes:
        raise ValueError("M11 loader/source class universes differ.")
    # Loader initialization can create its own local metadata. Bind the stable
    # inventory after initialization, then reject raw file changes while collecting.
    inventory = kbs.data_inventory(Path(cfg["data"]["data_dir"]))
    xs, logits_all, ys, raw = [], [], [], hashlib.sha256()
    count, started = 0, time.perf_counter()
    for i, batch in enumerate(loader):
        for key in ("ppi", "flow_stats", "label"):
            if key in batch:
                value = batch[key].detach().cpu().contiguous()
                raw.update(kbs.canonical([key, list(value.shape), str(value.dtype)]).encode())
                raw.update(value.numpy().tobytes())
        with torch.no_grad():
            logits, features = model(batch["ppi"], batch.get("flow_stats"), return_repr=True)
        xs.append(features.detach().cpu())
        logits_all.append(logits.detach().cpu())
        ys.append(batch["label"].detach().cpu().long())
        count += len(ys[-1])
        if i % 100 == 0:
            print(f"M11 features: {i + 1} batches, {count:,} rows, wall_s={time.perf_counter() - started:.1f}", flush=True)
    if not xs:
        raise ValueError("M11 loader returned no samples; annual H5 presence does not prove month availability.")
    payload = {"features": torch.cat(xs), "logits": torch.cat(logits_all), "labels": torch.cat(ys)}
    if (tuple(payload["features"].shape) != (count, pc["protocol"]["expected_features"]) or
        tuple(payload["logits"].shape) != (count, classes) or tuple(payload["labels"].shape) != (count,) or
        any(payload[k].dtype != torch.float32 or not torch.isfinite(payload[k]).all() for k in ("features", "logits")) or
        payload["labels"].min() < 0 or payload["labels"].max() >= classes):
        raise ValueError("M11 feature payload has invalid dimensions/dtypes/values/class IDs.")
    if inventory != kbs.data_inventory(Path(cfg["data"]["data_dir"])):
        raise ValueError("Raw data changed during M11 collection; cache left incomplete.")
    return payload, raw.hexdigest(), inventory


def prepare(args, pc):
    sig = cache_signature(args, pc)
    root = Path(args.cache_dir).resolve()
    with kbs.file_lock(root / ".lock"):
        if (root / "manifest.json").exists():
            validate_month_cache(args, pc, sig)
            print(f"M11_CACHE_COMPLETE: {root} (hashes verified; skipped)", flush=True)
            return
        pending = root / "prepare_config.json"
        if pending.exists():
            if json.loads(pending.read_text()) != sig:
                raise ValueError("Interrupted M11 cache has a different configuration; use a new cache directory.")
        elif any(p.name != ".lock" for p in root.iterdir()):
            raise ValueError("Unregistered files in M11 cache directory; use a new cache directory.")
        data = Path(sig["spec"]["data_config"]["data_dir"])
        if not data.is_dir() or not any(p.is_file() for p in data.rglob("*.h5")):
            raise ValueError(f"Existing TLS H5 data not found at {data}; set CARE_M11_DATA_DIR to its directory.")
        kbs.atomic_json(pending, sig)
        started = time.perf_counter()
        for n, sha in sig["reused_files"].items():
            if not (root / n).exists() or kbs.file_sha(root / n) != sha:
                shutil.copyfile(Path(args.source_cache_dir) / n, root / n)
        payload, raw, inventory = collect_target(args, sig["spec"], pc)
        torch.save(payload, root / "target.pt")
        hashes = {"reference": sig["reference_input_sha256"], "target": raw}
        info = {"spec": sig["spec"], "checkpoint_sha256": pc["info"]["checkpoint_sha256"],
            "periods": sig["spec"]["periods"], "num_classes": pc["info"]["num_classes"],
            "input_stream_sha256": hashes,
            "sample_counts": {"reference": pc["info"]["sample_counts"]["reference"], "target": len(payload["labels"])},
            "sample_id_definition": "fingerprint:role:row_index (snapshot rows, not original network flow IDs)",
            "fingerprint": kbs.digest({"spec": sig["spec"], "inputs": hashes}), "data_inventory": inventory,
            "files": {n: kbs.file_sha(root / n) for n in ("head.pt", "reference.pt", "target.pt")},
            "m11_preparation": sig, "prepare_wall_s": time.perf_counter() - started, "environment": kbs.runtime_info()}
        if any(info["files"][n] != sha for n, sha in sig["reused_files"].items()):
            raise ValueError("Copied reference/source head failed checksum.")
        # Final manifest is the cache completion marker, written after every tensor/hash.
        kbs.atomic_json(root / "manifest.json", info)
        validate_month_cache(args, pc, sig)
        print(f"M11_CACHE_COMPLETE: {root}; samples={info['sample_counts']}", flush=True)


def inputs(args, pc):
    info = validate_month_cache(args, pc, cache_signature(args, pc))
    base = copy.deepcopy(pc["base"])
    base.cache_dir, base.output_dir = args.cache_dir, args.output_dir
    base.target_period, base.eval_size, base.eval_seed = TARGET, 0, 1729
    base.log_every = args.log_every
    checked, protocol = diag.inputs(base)
    if checked != info:
        raise ValueError("M11 cache changed while validating.")
    identity = {"schema": SCHEMA, "source_protocol": protocol, "parent_protocol": pc["identity"],
        "parent_evaluation_sha256": kbs.file_sha(Path(args.replication_dir) / "evaluation/complete.json"),
        "fresh_query_seeds": list(robust.FRESH_SEEDS), "selectors": list(robust.SELECTORS),
        "supplement_budget": pc["identity"]["supplement_budget"], "candidate_cap": pc["identity"]["candidate_cap"],
        "target_samples": info["sample_counts"]["target"], "eval_size": 0, "eval_seed": 1729,
        "evaluation_groups": "fixed M12 stress/stable IDs are transfer groups; M11 supported source F1<0.1 is a separate offline-only descriptive group",
        "primary_summary": "training-seed paired means within query seed, then sample SD across all five prespecified query seeds",
        "code": code_identity()}
    return dict(out=Path(args.output_dir).resolve(), base=base, info=info, protocol=protocol, identity=identity,
                old_choices=[], parent=pc)


def select(args, ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    with kbs.file_lock(out / ".lock"):
        robust.root_manifest(ctx, True)
        directory, sig = out / "selection", robust.selection_signature(ctx)
        if kbs.is_complete(directory, sig):
            selections(ctx)
            print("All M11 queries verified; skipping tensors.", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target", keep_labels=False)
        choices, registries = [], {}
        for qs in ident["fresh_query_seeds"]:
            mixed, reg = ab.select_supplements(target["features"], target["logits"], p["budget"],
                ident["supplement_budget"], ident["candidate_cap"], qs)
            registries[str(qs)] = reg
            for seed in p["seeds"]:
                for selector in robust.SELECTORS:
                    choices.append({"query_seed": qs, "seed": seed, "selector": selector, **mixed[selector]})
        margin = diag.select_indices("margin", target["logits"], p["budget"], 0).tolist()
        for seed in p["seeds"]:
            choices.append({"query_seed": -1, "seed": seed, "selector": "margin", "rows": margin,
                            "repair_rows": margin, "supplement_rows": []})
        # M12 row IDs belong to another snapshot and must never enter this mask.
        ids = diag.evaluation_indices(ident["target_samples"], choices, 0, 1729)
        kbs.atomic_json(directory / "resolved_config.json", sig)
        kbs.atomic_json(directory / "queries.json", choices)
        kbs.atomic_json(directory / "candidate_pools.json", registries)
        np.save(directory / "evaluation_rows.npy", ids)
        kbs.finish(directory, sig, sorted(robust.SELECTION_ARTIFACTS))
        selections(ctx)
        print(f"M11 queries frozen: {len(choices)} new fits; common full evaluation {len(ids):,} rows.", flush=True)


def selections(ctx):
    out, p, ident = ctx["out"], ctx["protocol"], ctx["identity"]
    ab.completed(out / "selection", robust.selection_signature(ctx), robust.SELECTION_ARTIFACTS)
    choices = json.loads((out / "selection/queries.json").read_text())
    regs = json.loads((out / "selection/candidate_pools.json").read_text())
    expected = {(qs, s, q) for qs in ident["fresh_query_seeds"] for s in p["seeds"] for q in robust.SELECTORS}
    expected |= {(-1, s, "margin") for s in p["seeds"]}
    if len(choices) != len(expected) or {robust.choice_key(c) for c in choices} != expected or set(regs) != set(map(str, ident["fresh_query_seeds"])):
        raise ValueError("M11 frozen query grid mismatch.")
    lookup = {robust.choice_key(c): c for c in choices}
    def valid(rows):
        return rows == sorted(set(rows)) and all(type(i) is int and 0 <= i < ident["target_samples"] for i in rows)
    repair = regs[str(ident["fresh_query_seeds"][0])]["repair_rows"]
    for qs, reg in regs.items():
        pools = {int(cid): np.asarray(v) for cid, v in reg["pools"].items()}
        all_rows = [i for v in reg["pools"].values() for i in v]
        if (not valid(repair) or len(repair) != p["budget"] - ident["supplement_budget"] or reg["repair_rows"] != repair or
            len(set(all_rows)) != len(all_rows) or not valid(sorted(all_rows)) or set(all_rows) & set(repair) or
            len(all_rows) != reg["candidate_count"] or any(not valid(v) or not v or len(v) > ident["candidate_cap"] or
                not 0 <= int(cid) < p["expected_classes"] for cid, v in reg["pools"].items()) or
            reg["quotas"] != {str(c): n for c, n in ab.balanced_quotas(pools, ident["supplement_budget"], int(qs)).items()}):
            raise ValueError("M11 candidate pool/repair/quota mismatch.")
    baseline = lookup[-1, p["seeds"][0], "margin"]
    # Arrays are sorted by row, so the 800 repair rows need only be contained.
    if not set(repair).issubset(baseline["rows"]):
        raise ValueError("M11 shared repair core is not part of the pure Margin queries.")
    for c in choices:
        if (not all(valid(c[k]) for k in ("rows", "repair_rows", "supplement_rows")) or
            len(c["rows"]) != p["budget"] or c["rows"] != sorted(c["repair_rows"] + c["supplement_rows"]) or
            c["rows"] != lookup[c["query_seed"], p["seeds"][0], c["selector"]]["rows"]):
            raise ValueError("M11 query partition/training-seed sharing mismatch.")
        if c["selector"] == "margin":
            if c["supplement_rows"] or c["repair_rows"] != c["rows"]:
                raise ValueError("Pure Margin partition mismatch.")
            continue
        reg, extra = regs[str(c["query_seed"])], set(c["supplement_rows"])
        if (c["repair_rows"] != repair or len(extra) != ident["supplement_budget"] or
            any(len(extra.intersection(reg["pools"][cid])) != n for cid, n in reg["quotas"].items())):
            raise ValueError("M11 balanced/coverage paid budgets or class quotas differ.")
    return choices


def train(args, ctx):
    robust.root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        pending = []
        for c in choices:
            path, sig = robust.run_path(ctx, c), robust.run_signature(ctx, c)
            if kbs.is_complete(path, sig):
                ab.completed(path, sig, ab.RUN_ARTIFACTS)
            else:
                pending.append(c)
        if not pending:
            print("All M11 fits verified; skipping tensors/training.", flush=True)
            return
        base, p = ctx["base"], ctx["protocol"]
        source = diag.head_from_cache(base)
        target, ref = diag.load_role(base, ctx["info"], "target"), diag.load_role(base, ctx["info"], "reference")
        refs = {s: kbs.replay_indices(diag.numpy(ref["labels"]), p["expected_classes"], p["replay_per_class"], s) for s in p["seeds"]}
        for i, c in enumerate(pending, 1):
            path, sig = robust.run_path(ctx, c), robust.run_signature(ctx, c)
            old = ctx["parent"]["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old_trace = json.loads((Path(old["directory"]) / "trace.json").read_text())
            ridx = refs[c["seed"]]
            if old_trace["reference_rows"] != ridx.tolist() or old_trace["reference_labels"] != diag.numpy(ref["labels"][ridx]).tolist():
                raise ValueError("M11 historical replay differs from completed M12 reference.")
            print(f"M11 fit {i}/{len(pending)}: query_seed={c['query_seed']} {c['selector']} train_seed={c['seed']}", flush=True)
            kbs.atomic_json(path / "resolved_config.json", sig)
            head, trace = diag.fit_one(base, p, source, target, ref, c, ridx, "linear_care")
            trace = ab.annotate(head, trace, target, c, p["expected_classes"])
            trace.update(query_seed=c["query_seed"], new_optimizer_steps=p["steps"], month=TARGET)
            if any(trace[k] != old_trace[k] for k in STREAM_FIELDS):
                raise ValueError("M11 replay/optimizer sampling stream differs from fixed M12 protocol.")
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite M11 head.")
            torch.save(head.state_dict(), path / "head.pt")
            kbs.atomic_json(path / "trace.json", trace)
            kbs.finish(path, sig, sorted(ab.RUN_ARTIFACTS))
        print("All M11 fits complete; separate evaluation may now read held-out truth.", flush=True)


def paired(rows, ctx):
    lookup = {robust.choice_key(r): r for r in rows}
    result = []
    for qs in ctx["identity"]["fresh_query_seeds"]:
        for seed in ctx["protocol"]["seeds"]:
            for a, b in (("mixed_coverage", "mixed_balanced"), ("mixed_balanced", "margin"), ("mixed_coverage", "margin")):
                x, y = lookup[qs, seed, a], lookup[-1 if b == "margin" else qs, seed, b]
                result.append({"query_seed": qs, "seed": seed, "comparison": f"{a} - {b}",
                    **{k: x[k] - y[k] if x[k] is not None and y[k] is not None else None for k in METRICS}})
    return result


def summaries(rows, pairs, ctx):
    query_arms = full.summarize(rows, ("query_seed", "selector"))
    by_query = full.summarize(pairs, ("query_seed", "comparison"))
    fresh = set(ctx["identity"]["fresh_query_seeds"])
    arm_units = [{"selector": r["selector"], **{k: r.get(k + "_mean") for k in METRICS}}
                 for r in query_arms if r["query_seed"] in fresh]
    pair_units = [{"comparison": r["comparison"], **{k: r.get(k + "_mean") for k in METRICS}}
                  for r in by_query if r["query_seed"] in fresh]
    arms, primary = full.summarize(arm_units, ("selector",)), full.summarize(pair_units, ("comparison",))
    for r in arms + primary:
        r["n_query_seeds"] = r.pop("n_seeds")
    for r in primary:
        units = [u for u in pair_units if u["comparison"] == r["comparison"]]
        r["macro_positive_query_seeds"] = sum(u["overall_macro_f1_after"] is not None and u["overall_macro_f1_after"] > 0 for u in units)
        r["negative_flip_reduced_query_seeds"] = sum(u["negative_flip_rate"] is not None and u["negative_flip_rate"] < 0 for u in units)
    return query_arms, by_query, arms, primary


def report(ctx, query_arms, by_query, arms, primary, size, groups, static):
    def fmt(r, k):
        m, sd = r.get(k + "_mean"), r.get(k + "_sd")
        return "NA" if m is None else f"{m:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    margin = next(r for r in query_arms if r["selector"] == "margin")
    p = ctx["protocol"]
    lines = ["# CARE M11 跨月复验", "", f"目标 {TARGET}；CPU；每臂 {p['budget']} 标签；新增 {len(query_arms) * len(p['seeds'])} 次拟合。",
        f"查询种子 {ctx['identity']['fresh_query_seeds']}，训练种子 {p['seeds']}；沿用 M12 参数，M11 从源头初始化重新拟合。",
        f"完整共同评价 {size:,} 行；Static Macro-F1={static:.6f}；Margin Macro-F1={fmt(margin, 'overall_macro_f1_after')}（训练种子 SD）。",
        "M12 预定问题/稳定类别是迁移分组，不能直接称 M11 崩溃/稳定组。",
        f"M11 本月源模型低 F1 组：有评价支持且 Static F1<0.1，{len(groups['m11_low_f1_classes'])} 类；只在全部拟合完成后定义。", "",
        "## 主结果：先平均训练种子，± 为五个查询种子 SD", "",
        "| 查询 | Macro-F1 | M12 预定问题类 F1 | M11 低 F1 类 F1 | M12 预定稳定类 ΔF1 | 负翻转率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in arms:
        lines.append(f"| {r['selector']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "m11_low_f1_macro_after", "stable_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 配对主结果", "", "| 对照 | Macro-F1 Δ | M12 问题类 F1 Δ | M11 低 F1 类 Δ | 负翻转率差 | Macro 增益种子数 | 误伤下降种子数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        n = r["n_query_seeds"]
        lines.append(f"| {r['comparison']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "m11_low_f1_macro_after", "negative_flip_rate")) +
            f" | {r['macro_positive_query_seeds']}/{n} | {r['negative_flip_reduced_query_seeds']}/{n} |")
    lines += ["", "## coverage−balanced：逐查询种子（± 为训练种子 SD）", "",
              "| 查询种子 | Macro-F1 Δ | M12 问题类 F1 Δ | M11 低 F1 类 Δ | 负翻转率差 |", "|---|---:|---:|---:|---:|"]
    for r in by_query:
        if r["comparison"] == "mixed_coverage - mixed_balanced":
            lines.append(f"| {r['query_seed']} | " + " | ".join(fmt(r, k) for k in
                ("overall_macro_f1_after", "collapse_macro_f1_after", "m11_low_f1_macro_after", "negative_flip_rate")) + " |")
    lines += ["", "## 判读", "", "- 主对照是相同候选池/预测类别配额的 coverage−balanced；不挑月份、种子或超参数。",
        "- Margin 1000 与混合 800＋200 均为每次查询、每臂预算；全部重复的标签并集会超过 1000。",
        "- 所有 M11 查询在拟合前冻结，所有头使用 M11 查询并集排除集；M12 行号不参与排除。",
        "- M4 参考缓存和源分类头按字节校验复用；33 个 M11 头均从相同源模型重新拟合。",
        "- 配对参考行与抽样索引流逐项检查；只有付费目标标签参与拟合和付费覆盖记录。",
        "- 五个查询均值是主统计单位，不能将 15 个训练重复当作独立查询；± 是 SD，不是置信区间。",
        "- M11 低 F1 分组是预定规则下的事后描述，未用于查询、拟合或选择参数；不代表新方法检测到了这些类。",
        "- M12 和 M11 的绝对 F1 不直接作差；分别看 coverage−balanced 是否同方向。",
        "- 若 M11 优势消失或反号，收缩特征覆盖的跨月主张；若同方向，也仍需其他月份、基线和机制证据。",
        "- 收益、误伤和创新性分别判断；单个新月份不能证明普遍有效或方法新颖性。", ""]
    return "\n".join(lines)


def evaluate(args, ctx):
    robust.root_manifest(ctx)
    with kbs.file_lock(ctx["out"] / ".lock"):
        choices = selections(ctx)
        for c in choices:
            ab.completed(robust.run_path(ctx, c), robust.run_signature(ctx, c), ab.RUN_ARTIFACTS)
        sig = {"protocol": ctx["identity"], "stage": "evaluate",
            "selection_sha256": kbs.file_sha(ctx["out"] / "selection/complete.json"),
            "runs": {str(robust.run_path(ctx, c).relative_to(ctx["out"])):
                kbs.file_sha(robust.run_path(ctx, c) / "complete.json") for c in choices}}
        directory = ctx["out"] / "evaluation"
        if kbs.is_complete(directory, sig):
            ab.completed(directory, sig, EVAL_ARTIFACTS)
            print(f"CARE_M11_REPLICATION_COMPLETE: {directory / 'report.md'} (verified; skipped)", flush=True)
            return
        target = diag.load_role(ctx["base"], ctx["info"], "target")
        p = ctx["protocol"]
        ids = np.load(ctx["out"] / "selection/evaluation_rows.npy", allow_pickle=False)
        if not np.array_equal(ids, diag.evaluation_indices(ctx["identity"]["target_samples"], choices, 0, 1729)):
            raise ValueError("M11 evaluation mask differs from its full query union exclusion.")
        y, before = diag.numpy(target["labels"][ids]), target["logits"][ids].argmax(1).numpy()
        if y.min() < 0 or y.max() >= p["expected_classes"]:
            raise ValueError("M11 evaluation labels outside source class universe.")
        _, static_classes, _ = kbs.compare_predictions(y, before, before, np.ones(len(y), bool),
            p["expected_classes"], p["collapse_classes"], p["stable_classes"], .1, .05)
        low = [r["class_id"] for r in static_classes if r["support"] > 0 and r["before_f1"] < .1]
        groups = {"m12_prespecified_stress_classes": p["collapse_classes"], "m12_prespecified_stable_classes": p["stable_classes"],
            "m11_low_f1_rule": "supported on common M11 evaluation and Static F1<0.1; evaluation-only", "m11_low_f1_classes": low}
        predictions = {"row_indices": ids, "labels": y, "source": before}
        rows, classes, coverage = [], [], []
        for i, c in enumerate(choices, 1):
            path = robust.run_path(ctx, c)
            print(f"M11 evaluate {i}/{len(choices)}: query_seed={c['query_seed']} {c['selector']} train_seed={c['seed']}", flush=True)
            head = diag.head_from_cache(ctx["base"])
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite saved M11 head.")
            trace = json.loads((path / "trace.json").read_text())
            wanted = ab.annotate(head, copy.deepcopy(trace), target, c, p["expected_classes"])
            old = ctx["parent"]["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old_trace = json.loads((Path(old["directory"]) / "trace.json").read_text())
            if (any(trace[k] != wanted[k] for k in robust.TRACE_FIELDS) or
                any(trace[k] != old_trace[k] for k in STREAM_FIELDS) or trace["query_seed"] != c["query_seed"] or
                trace["month"] != TARGET or trace["new_optimizer_steps"] != p["steps"]):
                raise ValueError("M11 paid-query trace/replay/stream/step mismatch.")
            after = full.predict_rows(head, target["features"], ids, args.inference_batch_size)
            metrics, per_class = full.scores(y, before, after, p, trace)
            chosen = [r for r in per_class if r["class_id"] in low]
            metrics.update(m11_low_f1_macro_before=float(np.mean([r["before_f1"] for r in chosen])) if chosen else None,
                m11_low_f1_macro_after=float(np.mean([r["after_f1"] for r in chosen])) if chosen else None,
                m11_low_f1_macro_delta=float(np.mean([r["delta_f1"] for r in chosen])) if chosen else None)
            meta = {"query_seed": c["query_seed"], "seed": c["seed"], "selector": c["selector"], "engine": "care"}
            rows.append({**meta, **metrics, "new_optimizer_steps": trace["new_optimizer_steps"]})
            classes.extend({**meta, **r} for r in full.class_flips(y, before, after, per_class, p["expected_classes"]))
            coverage.extend({**meta, "class_id": cid, "queried_count": q, "source_correct_paid_count": a}
                for cid, (q, a) in enumerate(zip(trace["queried_class_histogram"], trace["source_correct_paid_histogram"])))
            predictions[f"query{c['query_seed']}_{c['selector']}_seed{c['seed']}"] = after
        pairs = paired(rows, ctx)
        query_arms, by_query, arms, primary = summaries(rows, pairs, ctx)
        kbs.atomic_json(directory / "resolved_config.json", sig)
        kbs.atomic_json(directory / "evaluation_groups.json", groups)
        for name, values in (("by_seed.csv", rows), ("per_query_summary.csv", query_arms), ("fresh_arm_summary.csv", arms),
            ("paired_by_seed.csv", pairs), ("paired_by_query.csv", by_query), ("fresh_paired_summary.csv", primary),
            ("per_class.csv", classes), ("query_coverage.csv", coverage)):
            kbs.write_csv(directory / name, values)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        (directory / "report.md").write_text(report(ctx, query_arms, by_query, arms, primary, len(ids), groups, rows[0]["overall_macro_f1_before"]))
        kbs.finish(directory, sig, sorted(EVAL_ARTIFACTS))
        print(f"CARE_M11_REPLICATION_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "preflight", "prepare", "select", "train", "evaluate"))
    for name, default in (("source-dir", "small_repair_diagnostic_v1"), ("source-audit-dir", "small_repair_full_audit_v1"),
        ("protection-dir", "care_target_protection_v1"), ("ablation-dir", "care_mixed_query_ablation_v1"),
        ("replication-dir", "care_query_seed_robustness_v1"), ("source-cache-dir", "kbs_supplement_v1/cache"),
        ("cache-dir", "care_m11_cache_v1"), ("output-dir", "care_m11_replication_v1")):
        p.add_argument("--" + name, default=str(kbs.CORE / "outputs" / default))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--data-dir", help="Existing raw TLS data directory; defaults to the source cache's recorded directory.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=460)
    return p


def main():
    args = parser().parse_args()
    if min(args.threads, args.inference_batch_size) <= 0 or args.log_every < 0:
        raise ValueError("Invalid CPU settings.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"target": TARGET, "device": "cpu", "query_seeds": robust.FRESH_SEEDS,
            "new_fits_default": 33, "prepare": "reuse verified M4/source head; extract M11 with frozen CPU backbone",
            "selectors": ["margin", *robust.SELECTORS], "eval_size": "all M11 rows excluding all M11 queries",
            "primary_unit": "paired training-seed means within each query seed"}, indent=2))
        return
    started = time.perf_counter()
    pc = parent_context(args)
    if args.command == "prepare":
        prepare(args, pc)
    elif args.command == "preflight":
        sig = cache_signature(args, pc)
        root = Path(args.cache_dir)
        if (root / "manifest.json").exists():
            info = validate_month_cache(args, pc, sig)
            status = f"M11 cache verified, {info['sample_counts']['target']} rows"
        else:
            status = "M11 cache not prepared; prepare stage will test loader availability and extract features"
        print(json.dumps({"status": status, "target": TARGET, "device": "cpu", "data_dir": sig["spec"]["data_config"]["data_dir"],
            "new_fits": (2 * len(robust.FRESH_SEEDS) + 1) * len(pc["protocol"]["seeds"])}, indent=2))
    else:
        ctx = inputs(args, pc)
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, ctx)
    print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
