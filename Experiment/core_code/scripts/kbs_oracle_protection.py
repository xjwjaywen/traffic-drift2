#!/usr/bin/env python3
"""Privileged-label mechanism control, NOT a deployable or equal-label-cost method.

Reuse 80% BADGE scout and its saved probe; two oracle tails, two fresh head fits.
Freeze every selection before fitting; evaluate only when every requested fit is done.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import kbs_repair_aware as pilot
import kbs_repair_diagnosis as diagnosis
import kbs_supplement as kbs

METHODS = ("oracle_probe_damage", "oracle_source_correct")
SETTINGS = {
    "schema": "care-oracle-protection-v1",
    "status": "M12 exploratory privileged-label mechanism control; not deployable, not a strict upper bound",
    "scout": "reuse exact first 80% BADGE indices and saved probe from completed repair-aware pilot",
    "oracle_probe_damage": "uniform without replacement: source correct AND probe wrong, excluding scout",
    "oracle_source_correct": "uniform without replacement: source correct, excluding scout; includes damage region",
    "label_access": "selection inspects FULL target truth; budget counts TRAINING rows only, not label cost",
    "selection_rng": "default_rng(seed + 52021) separately for each pool; sorted supplemental IDs",
    "insufficient_pool": "stop before training; no replacement or fallback",
    "repair": "unchanged target CE + reference KD; fresh source head, original reference IDs/streams/steps",
    "evaluation": "per-seed union exclusion of all four old and both new queries; old predictions reused",
    "primary_comparison": "oracle_probe_damage minus oracle_source_correct",
    "no_tuning": "no threshold search, best-seed selection, significance or automatic success declaration",
}
SELECT_FILES = ["resolved_config.json", "selection.json", "query_labels.json"]
TRAIN_FILES = pilot.TRAIN_FILES
EVAL_FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "paired_by_seed.csv",
              "paired_summary.csv", "per_class.csv", "selection_audit.csv", "report.md"]


def verify_sources(args):
    previous = pilot.read_json(Path(args.pilot_dir) / "study_manifest.json")["identity"]
    verifier = argparse.Namespace(**vars(args), threads=previous["threads"])
    # Original hashes, all original seeds/runs, and original evaluation must already pass.
    sources, records, audit = diagnosis.verify_inputs(verifier)
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(previous["seeds"]):
        raise ValueError("Choose unique seeds present in the completed original pilot")
    runtime = {k: previous[k] for k in ("device", "threads", "inference_batch_size")}
    identity = {"settings": SETTINGS, "seeds": seeds, "source_audit": audit, "runtime": runtime,
                "code_sha256": kbs.file_sha(Path(__file__))}
    return sources, records, identity


def selection_dir(out, seed):
    return Path(out) / "selections" / f"seed_{seed}"


def run_dir(out, seed, method):
    return Path(out) / "runs" / method / f"seed_{seed}"


def signature(identity, seed, stage, method=None):
    return {"identity_sha256": kbs.digest(identity), "seed": seed, "stage": stage, "method": method}


def train_signature(identity, out, seed, method):
    return {**signature(identity, seed, "train", method),
            "selection_complete_sha256": kbs.file_sha(selection_dir(out, seed) / "complete.json")}


def select_queries(old_record, y, old, probe, classes, seed):
    """Oracle API deliberately accepts full truth, but never final repair predictions."""
    n = len(y)
    for name, value in (("truth", y), ("source", old), ("probe", probe)):
        diagnosis.vector(value, n, classes, name)
    badge = np.asarray(old_record["choices"]["badge"]["row_indices"], dtype=np.int64)
    budget = len(badge)
    scout = pilot.check_ids(old_record["scout_indices"], n, budget * 4 // 5)
    if not np.array_equal(scout, badge[:len(scout)]):
        raise ValueError("Scout must be the original BADGE prefix")
    available = np.ones(n, dtype=bool)
    available[scout] = False
    correct = (old == y) & available
    pool_masks = {METHODS[0]: correct & (probe != y), METHODS[1]: correct}
    choices = {}
    for method, mask in pool_masks.items():
        pool = np.flatnonzero(mask)
        size = budget - len(scout)
        if len(pool) < size:
            raise ValueError(f"Insufficient oracle pool for {method}, seed {seed}: {len(pool)} < {size}; no fallback")
        tail = np.sort(np.random.default_rng(seed + 52021).choice(pool, size=size, replace=False))
        choices[method] = {"row_indices": np.concatenate([scout, tail]).tolist(), "pool_size": len(pool)}
    union = sorted(set(old_record["common_excluded_indices"]) |
                   {i for c in choices.values() for i in c["row_indices"]})
    if len(union) >= n:
        raise ValueError("Common unqueried evaluation set is empty")
    return {"scout_indices": scout.tolist(), "choices": choices, "training_rows_per_arm": budget,
            "oracle_truth_rows_inspected": n, "common_excluded_indices": union}


def validate_selection(record, old_record, p, n):
    budget = p["budget"]
    if record["scout_indices"] != old_record["scout_indices"] or set(record["choices"]) != set(METHODS):
        raise ValueError("Oracle selection changed its scout or arms")
    if record["training_rows_per_arm"] != budget or record["oracle_truth_rows_inspected"] != n:
        raise ValueError("Incorrect oracle/training-label accounting")
    for choice in record["choices"].values():
        ids = pilot.check_ids(choice["row_indices"], n, budget)
        size = len(record["scout_indices"])
        if ids[:size].tolist() != record["scout_indices"] or ids[size:].tolist() != sorted(ids[size:]):
            raise ValueError("Oracle query order/prefix changed")
        if not budget - size <= choice["pool_size"] <= n - size:
            raise ValueError("Invalid oracle pool size")
    union = sorted(set(old_record["common_excluded_indices"]) |
                   {i for c in record["choices"].values() for i in c["row_indices"]})
    if record["common_excluded_indices"] != union or len(union) >= n:
        raise ValueError("Common query union does not include all six arms")


def require_selections(args, verified):
    sources, previous, identity = verified
    info, p, _, _ = sources
    records = {}
    pilot.freeze_manifest(Path(args.output_dir), identity)
    for seed in identity["seeds"]:
        directory = selection_dir(args.output_dir, seed)
        if not pilot.completed(directory, signature(identity, seed, "select"), SELECT_FILES):
            raise ValueError(f"Every oracle selection must finish before training: missing seed {seed}")
        record = pilot.read_json(directory / "selection.json")
        validate_selection(record, previous[seed], p, info["sample_counts"]["target"])
        labels = pilot.read_json(directory / "query_labels.json")
        if set(labels) != set(METHODS):
            raise ValueError("Missing frozen query labels")
        for method in METHODS:
            diagnosis.vector(labels[method], p["budget"], p["num_classes"], "frozen query labels")
        records[seed] = record
    return records


def load_truth_and_source(args):
    target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True, mmap=True)
    return np.asarray(target["labels"]), target["logits"].argmax(1).numpy()


def select(args, verified):
    sources, previous, identity = verified
    info, p, _, _ = sources
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        pilot.freeze_manifest(out, identity, create=True)
        todo = [s for s in identity["seeds"] if not pilot.completed(
            selection_dir(out, s), signature(identity, s, "select"), SELECT_FILES)]
        if not todo:
            require_selections(args, verified)
            print("All oracle queries verified; no selection changes.", flush=True)
            return
        torch.set_num_threads(identity["runtime"]["threads"])
        y, old = load_truth_and_source(args)
        diagnosis.vector(y, info["sample_counts"]["target"], p["num_classes"], "truth")
        # Check all pools before writing any new selection; no fits in this process.
        pending = {}
        for seed in todo:
            sd = pilot.selection_dir(Path(args.pilot_dir), seed)
            probe = np.load(sd / "probe_predictions.npy", mmap_mode="r")
            if not np.array_equal(np.load(sd / "scout_labels.npy"), y[previous[seed]["scout_indices"]]):
                raise ValueError("Original scout truth changed")
            pending[seed] = select_queries(previous[seed], y, old, probe, p["num_classes"], seed)
        for seed, record in pending.items():
            directory = selection_dir(out, seed)
            sig = signature(identity, seed, "select")
            kbs.atomic_json(directory / "resolved_config.json", sig)
            kbs.atomic_json(directory / "selection.json", record)
            kbs.atomic_json(directory / "query_labels.json", {
                m: y[c["row_indices"]].tolist() for m, c in record["choices"].items()})
            kbs.finish(directory, sig, SELECT_FILES)
            print(f"Frozen oracle seed {seed}: training rows={p['budget']}/arm, "
                  f"full truth inspected={len(y)}, pools="
                  f"{ {m: c['pool_size'] for m, c in record['choices'].items()} }", flush=True)


def train(args, verified):
    sources, previous, identity = verified
    _, p, _, _ = sources
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)  # all seeds frozen before loading labels
        todo = [(s, m) for s in identity["seeds"] for m in METHODS if not pilot.completed(
            run_dir(out, s, m), train_signature(identity, out, s, m), TRAIN_FILES)]
        if not todo:
            print("All oracle repairs verified; zero additional fits.", flush=True)
            return
        runtime = identity["runtime"]
        device = kbs.device_for(runtime["device"])
        torch.set_num_threads(runtime["threads"])
        head, ref, target, labels = pilot.load_inputs(args)
        old = target["logits"].argmax(1).numpy()
        for seed, method in todo:
            record = records[seed]
            directory = run_dir(out, seed, method)
            sig = train_signature(identity, out, seed, method)
            sd = pilot.selection_dir(Path(args.pilot_dir), seed)
            ids = np.asarray(record["choices"][method]["row_indices"], dtype=np.int64)
            y = pilot.purchased_labels(labels, ids, p["num_classes"])
            frozen = pilot.read_json(selection_dir(out, seed) / "query_labels.json")[method]
            size = len(record["scout_indices"])
            if y.tolist() != frozen or not np.array_equal(y[:size], np.load(sd / "scout_labels.npy")):
                raise ValueError("Frozen query labels changed")
            probe = np.load(sd / "probe_predictions.npy", mmap_mode="r")
            if not np.all(old[ids[size:]] == y[size:]) or (
                method == METHODS[0] and not np.all(probe[ids[size:]] != y[size:])):
                raise ValueError("Selected rows do not satisfy the oracle condition")
            ridx = np.load(sd / "reference_ids.npy")
            if not np.array_equal(ridx, kbs.replay_indices(np.asarray(ref["labels"]), p["num_classes"], 5, seed)):
                raise ValueError("Original reference sampling changed")
            kbs.atomic_json(directory / "resolved_config.json", sig)
            print(f"Repair {method} seed {seed}: {len(ids)} TRAINING labels; oracle truth privilege", flush=True)
            kbs.seed_all(seed)
            repaired, trace = pilot.fit(head, ref, target, ids, y, ridx, p, seed, device)
            baseline_trace = pilot.read_json(pilot.run_dir(Path(args.pilot_dir), seed, "badge") / "trace.json")
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256"):
                if trace[key] != baseline_trace[key]:
                    raise ValueError(f"Training is not paired with original BADGE: {key}")
            started = time.perf_counter()
            pred = kbs.predict(repaired, target["features"], device, runtime["inference_batch_size"])
            trace.update({"full_pool_inference_wall_s": time.perf_counter() - started,
                "target_labels": len(ids), "oracle_truth_rows_inspected": len(old),
                "reference_labels": len(ridx), "replay_ce": False,
                "kd_weight": pilot.SETTINGS["kd_weight"], "temperature": pilot.SETTINGS["temperature"],
                "initialization": "source_head"})
            torch.save(repaired.state_dict(), directory / "head.pt")
            np.save(directory / "predictions.npy", pred)
            kbs.write_csv(directory / "query_ids.csv", [
                {"row_index": int(i), "label": int(v), "phase": "scout" if j < size else "oracle_supplement"}
                for j, (i, v) in enumerate(zip(ids, y))])
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, sig, TRAIN_FILES)


def report(summary, paired, audit, p, seeds):
    fields = ["overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_after",
              "noncollapse_new_collapses", "collapse_residual_count", "noncollapse_negative_flips",
              "noncollapse_positive_flips"]
    lines = ["# 理想保护样本机制对照（使用完整目标真值）", "",
        f"配对种子 {seeds}；每组训练 {p['budget']} 行，共同前 {p['budget']*4//5} 行沿用原 BADGE 查询。",
        "两组 oracle 在选样时均查看完整目标真值；训练行数相等不代表标注成本相等。",
        "这是开发性机制诊断，不是可部署方法、严格性能上界或同标注预算公平比较。",
        "四个旧方法只复用预测；下表全部重新按六组查询并集排除，在同一组种子上计算。",
        "均值 ± 样本标准差；CSV 同时报告每个指标的有效种子数。", "",
        "| 方法 | 整体 F1 | 崩溃类 F1 | 稳定类 F1 | 新崩溃 | 原崩溃残留 | 非崩溃类负翻转 | 非崩溃类正翻转 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| " + r["method"] + " | " + " | ".join(pilot.format_stat(r, k) for k in fields) + " |")
    lines += ["", "主要对照：oracle_probe_damage − oracle_source_correct；其余比较仅提供背景。",
        "差值为前者减后者；负翻转/新崩溃越少越好，但需同时看崩溃类恢复及正翻转。", "",
        "| 对照 | 整体 F1 差 | 崩溃类 F1 差 | 稳定类 F1 差 | 新崩溃差 | 非崩溃类负翻转差 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append("| " + r["comparison"] + " | " + " | ".join(pilot.format_stat(r, k)
                     for k in (fields[0], fields[1], fields[2], fields[3], fields[5])) + " |")
    lines += ["", "后 20% 查询组成（样本数；真值用于筛选，不能解释为无标签发现能力）：", "",
        "| 方法 | 原本正确 | 预演误伤 | 崩溃类 | 覆盖类别数 |",
        "|---|---:|---:|---:|---:|"]
    for r in diagnosis.aggregate(audit, ["method"]):
        lines.append("| " + r["method"] + " | " + " | ".join(pilot.format_stat(r, k) for k in
                     ("supplement_source_correct", "supplement_probe_damage", "supplement_collapse_rows", "supplement_classes")) + " |")
    lines += ["", "oracle_probe_damage 在源正确且预演错误的池内均匀采样；oracle_source_correct 在全部源正确样本中采样，允许覆盖前者区域。",
        "两组均排除共同侦察行；不足额则停止，无回填。没有按最终修复错误、测试 F1 或事后最佳种子选择样本。",
        "修复器、原参考行、学习率、损失权重、优化步数与训练抽样位置流均保持原协议；每次从同一源头部初始化。",
        "减少误伤但降低恢复仍是取舍，不自动判断成功；单个工作点也不能证明整条恢复—损伤曲线更优。",
        "定向保护优势若出现，只支持当前修复器下这些标签有用；仍需开发不查看未查询真值的选样方法。",
        "没有优势也不能证明所有保护策略无效：当前标签使用方式、采样覆盖及固定特征均可能限制收益。",
        "全类查询分布和逐类损伤见 per_class.csv，逐种子配对差见 paired_by_seed.csv。",
        "种子共用一个源模型和 M12；这是开发性探索，不是独立跨环境重复，不宣称显著性或新颖性。", ""]
    return "\n".join(lines)


def evaluate(args, verified):
    sources, previous, identity = verified
    info, p, saved_badge, _ = sources
    out, root = Path(args.output_dir), Path(args.pilot_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)
        hashes = {}
        for seed in identity["seeds"]:
            for method in METHODS:
                rd = run_dir(out, seed, method)
                if not pilot.completed(rd, train_signature(identity, out, seed, method), TRAIN_FILES):
                    raise ValueError(f"Every new fit must finish before evaluation: {seed}/{method}")
                hashes[f"{seed}/{method}"] = kbs.file_sha(rd / "complete.json")
        sig = {"identity_sha256": kbs.digest(identity), "runs": hashes}
        directory = out / "evaluation"
        if pilot.completed(directory, sig, EVAL_FILES):
            print(f"Verified report: {directory / 'report.md'}", flush=True)
            return
        torch.set_num_threads(identity["runtime"]["threads"])
        y, old = load_truth_and_source(args)
        n, classes = info["sample_counts"]["target"], p["num_classes"]
        diagnosis.vector(y, n, classes, "truth")
        prior_rows = kbs.read_csv(root / "evaluation/by_seed.csv")
        prior = {(int(r["seed"]), r["method"]): r for r in prior_rows}
        expected_keys = {(s, m) for s in sources[3]["seeds"] for m in ("static", *pilot.METHODS)}
        if len(prior) != len(prior_rows) or set(prior) != expected_keys:
            raise ValueError("Original evaluation has unexpected/missing rows")
        rows, class_rows, paired, audit = [], [], [], []
        for seed, record in records.items():
            print(f"Evaluating seed {seed}: recomputing all six arms on common exclusion...", flush=True)
            sd = pilot.selection_dir(root, seed)
            probe = diagnosis.vector(np.load(sd / "probe_predictions.npy"), n, classes, "probe")
            if record != select_queries(previous[seed], y, old, probe, classes, seed):
                raise ValueError("Frozen oracle selection violates the declared policy")
            scout = np.asarray(record["scout_indices"], dtype=np.int64)
            counts = pilot.relation_counts(old[scout], y[scout], classes)
            if not np.array_equal(np.load(sd / "scout_labels.npy"), y[scout]) or not np.array_equal(
                    np.load(sd / "relation_counts.npy"), counts) or previous[seed] != pilot.select_queries(
                        saved_badge[seed], old, probe, counts, seed):
                raise ValueError("Original scout or selection semantics changed")
            mask = np.ones(n, dtype=bool); mask[record["common_excluded_indices"]] = False
            old_mask = np.ones(n, dtype=bool); old_mask[previous[seed]["common_excluded_indices"]] = False
            original_static, _ = pilot.metrics(y, old, old, old_mask, p)
            diagnosis.reconcile(original_static, prior[seed, "static"])
            static, cr = pilot.metrics(y, old, old, mask, p)
            rows.append({"seed": seed, "method": "static", **static})
            class_rows.extend({"seed": seed, "method": "static", **r} for r in cr)
            metrics, traces = {}, []
            frozen_labels = pilot.read_json(selection_dir(out, seed) / "query_labels.json")
            for method in (*pilot.METHODS, *METHODS):
                is_new = method in METHODS
                rd = run_dir(out, seed, method) if is_new else pilot.run_dir(root, seed, method)
                ids = np.asarray((record if is_new else previous[seed])["choices"][method]["row_indices"], dtype=np.int64)
                query = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query] != ids.tolist() or [int(r["label"]) for r in query] != y[ids].tolist():
                    raise ValueError("Training queries differ from frozen rows/truth")
                if is_new and frozen_labels[method] != y[ids].tolist():
                    raise ValueError("Frozen oracle labels differ from truth")
                pred = diagnosis.vector(np.load(rd / "predictions.npy", mmap_mode="r"), n, classes, "final")
                if not is_new:
                    original_metrics, _ = pilot.metrics(y, old, pred, old_mask, p)
                    diagnosis.reconcile(original_metrics, prior[seed, method])
                m, cr = pilot.metrics(y, old, pred, mask, p)
                metrics[method] = m
                rows.append({"seed": seed, "method": method, **m})
                query_counts = np.bincount(y[ids], minlength=classes)
                tail = ids[len(scout):]
                tail_counts = np.bincount(y[tail], minlength=classes)
                class_rows.extend({"seed": seed, "method": method, **r,
                    "query_count": int(query_counts[r["class_id"]]),
                    "supplement_query_count": int(tail_counts[r["class_id"]])} for r in cr)
                audit.append({"seed": seed, "method": method, "training_rows": len(ids),
                    "oracle_truth_rows_inspected": n if is_new else 0,
                    "supplement_source_correct": int((old[tail] == y[tail]).sum()),
                    "supplement_probe_damage": int(((old[tail] == y[tail]) & (probe[tail] != y[tail])).sum()),
                    "supplement_collapse_rows": int(np.isin(y[tail], p["collapse_classes"]).sum()),
                    "supplement_classes": len(np.unique(y[tail]))})
                traces.append(pilot.read_json(rd / "trace.json"))
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256",
                        "target_labels", "reference_labels", "replay_ce", "kd_weight", "temperature", "initialization"):
                if len({t[key] for t in traces}) != 1:
                    raise ValueError(f"Training protocol differs across arms: {key}")
            for a, b in [(METHODS[0], METHODS[1]), (METHODS[0], "badge"), (METHODS[1], "badge"),
                         (METHODS[0], "flip_uniform"), (METHODS[1], "random")]:
                paired.append({"seed": seed, "comparison": f"{a} minus {b}", **{
                    k: metrics[a][k] - metrics[b][k] if metrics[a][k] is not None and metrics[b][k] is not None else None
                    for k in metrics[a]}})
        summary = diagnosis.aggregate(rows, ["method"])
        paired_summary = diagnosis.aggregate(paired, ["comparison"])
        kbs.atomic_json(directory / "resolved_config.json", sig)
        for name, values in [("by_seed", rows), ("summary", summary), ("paired_by_seed", paired),
                             ("paired_summary", paired_summary), ("per_class", class_rows), ("selection_audit", audit)]:
            kbs.write_csv(directory / f"{name}.csv", values)
        (directory / "report.md").write_text(report(summary, paired_summary, audit, p, identity["seeds"]))
        kbs.finish(directory, sig, EVAL_FILES)
        print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "select", "train", "evaluate"])
    p.add_argument("--pilot-dir", default=str(kbs.CORE / "outputs/kbs_repair_aware_v1"))
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_oracle_protection_v1"))
    p.add_argument("--seeds", default="0,1,2")
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        seeds = kbs.int_list(args.seeds)
        print(json.dumps({"settings": SETTINGS, "seeds": seeds, "new_fits": 2 * len(seeds),
                          "probe_fits": 0, "old_baseline_fits": 0}, indent=2))
        return
    verified = verify_sources(args)
    if args.command == "preflight":
        p, identity = verified[0][1], verified[2]
        kbs.device_for(identity["runtime"]["device"])
        print(json.dumps({"status": "ready", "seeds": identity["seeds"], "new_fits": 2 * len(identity["seeds"]),
            "training_rows_per_arm": p["budget"], "shared_scout_rows": p["budget"] * 4 // 5,
            "steps_per_fit": p["optimizer_steps"], "runtime": identity["runtime"],
            "warning": "FULL target truth used for oracle selection; not equal-label-cost comparison"}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, verified)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
