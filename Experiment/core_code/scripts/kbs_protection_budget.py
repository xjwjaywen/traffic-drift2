#!/usr/bin/env python3
"""Nested 5%/10% oracle protection allocations; reuse 0%/20% endpoints.

Uses saved privileged-label queries, not a deployable active-learning policy.
No new target truth is consulted during allocation; the old oracle privilege remains.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import kbs_oracle_protection as oracle
import kbs_repair_aware as pilot
import kbs_repair_diagnosis as diagnosis
import kbs_supplement as kbs

CONFIGS = tuple((f"{family}_p{pct:02d}", family, pct) for family in oracle.METHODS for pct in (5, 10))
METHODS = tuple(c[0] for c in CONFIGS)
SETTINGS = {
    "schema": "care-protection-budget-v1", "new_percentages": [5, 10], "reused_percentages": [0, 20],
    "selection": "fixed permutation(seed+62021) of saved 20% protection tail; nested first k entries",
    "query_order": "take first B-k original BADGE IDs excluding reserved protection; append sorted reserved IDs",
    "shared_scout": "original first 80% retained exactly; no probe refit",
    "evaluation": "EXACT original six-arm oracle query union; never shrink the mask",
    "repair": "unchanged original CE+reference KD engine, reference rows and paired training streams",
    "label_privilege": "inherits full-target-truth selection from oracle source; not equal-label-cost",
    "status": "M12 development grid after prior results; all four allocation points reported, no winner or significance claim",
}
SELECT_FILES = ["resolved_config.json", "selection.json", "query_labels.json"]
EVAL_FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "paired_by_seed.csv",
              "paired_summary.csv", "curve_by_seed.csv", "curve_summary.csv", "per_class.csv",
              "allocation_audit.csv", "report.md"]


def verify_sources(args):
    out = Path(args.output_dir).resolve()
    for name in ("oracle_dir", "pilot_dir", "study_dir", "cache_dir"):
        source = Path(getattr(args, name)).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Use a separate sibling output directory")
    original_identity = pilot.read_json(Path(args.oracle_dir) / "study_manifest.json")["identity"]
    source_args = argparse.Namespace(**{**vars(args), "output_dir": args.oracle_dir,
        "seeds": ",".join(map(str, original_identity["seeds"]))})
    verified = oracle.verify_sources(source_args)
    if verified[2] != original_identity:
        raise ValueError("Original oracle identity changed")
    records = oracle.require_selections(source_args, verified)
    hashes = {}
    for seed in original_identity["seeds"]:
        for method in oracle.METHODS:
            rd = oracle.run_dir(args.oracle_dir, seed, method)
            if not pilot.completed(rd, oracle.train_signature(original_identity, args.oracle_dir, seed, method), oracle.TRAIN_FILES):
                raise ValueError(f"Complete all source oracle fits first: {seed}/{method}")
            hashes[f"{seed}/{method}"] = kbs.file_sha(rd / "complete.json")
    sig = {"identity_sha256": kbs.digest(original_identity), "runs": hashes}
    evaluation = Path(args.oracle_dir) / "evaluation"
    if not pilot.completed(evaluation, sig, oracle.EVAL_FILES):
        raise ValueError("Complete source oracle evaluation first")
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(original_identity["seeds"]):
        raise ValueError("Use unique seeds from the completed oracle study")
    budget = verified[0][1]["budget"]
    if budget % 20:
        raise ValueError("Budget must be divisible by 20 for exact 5%/10% allocations")
    identity = {"settings": SETTINGS, "seeds": seeds, "runtime": original_identity["runtime"],
        "source_identity_sha256": kbs.digest(original_identity),
        "source_evaluation_sha256": kbs.file_sha(evaluation / "complete.json"),
        "source_runs": hashes, "code_sha256": kbs.file_sha(Path(__file__))}
    return verified, records, identity


def mixed_queries(badge, protected_tail, count, seed):
    """Pure ID-only allocation. Returns exact endpoint order when count is 0/full."""
    badge = np.asarray(badge, dtype=np.int64)
    tail = np.asarray(protected_tail, dtype=np.int64)
    if not 0 <= count <= len(tail) or len(tail) * 5 != len(badge):
        raise ValueError("Invalid allocation size")
    if len(np.unique(badge)) != len(badge) or len(np.unique(tail)) != len(tail):
        raise ValueError("Duplicate source query IDs")
    if set(tail) & set(badge[:len(badge) - len(tail)]):
        raise ValueError("Protection tail overlaps the shared scout")
    permutation = np.random.default_rng(seed + 62021).permutation(len(tail))
    protected = np.sort(tail[permutation[:count]]).tolist()
    reserved = set(protected)
    fill = [int(i) for i in badge if i not in reserved][:len(badge) - count]
    ids = fill + protected
    if len(ids) != len(badge) or len(set(ids)) != len(ids):
        raise ValueError("Allocation did not preserve the training budget")
    return {"row_indices": ids, "protected_indices": protected, "badge_fill_indices": fill,
            "protection_count": count, "badge_fill_count": len(fill),
            "protected_overlap_badge_count": len(reserved & set(badge)),
            "rows_outside_badge_count": len(set(ids) - set(badge))}


def select_queries(badge, original, seed):
    budget = len(badge)
    if budget % 20:
        raise ValueError("Budget must be divisible by 20")
    scout = original["scout_indices"]
    if scout != list(badge[:budget * 4 // 5]):
        raise ValueError("Shared scout differs from BADGE")
    choices = {}
    for method, family, pct in CONFIGS:
        source = original["choices"][family]["row_indices"]
        tail = source[len(scout):]
        if source[:len(scout)] != scout or mixed_queries(badge, tail, len(tail), seed)["row_indices"] != source:
            raise ValueError("20% endpoint order is not identical to the saved oracle run")
        if mixed_queries(badge, tail, 0, seed)["row_indices"] != list(badge):
            raise ValueError("0% endpoint differs from original BADGE")
        choices[method] = {"family": family, "protection_percent": pct,
                           **mixed_queries(badge, tail, budget * pct // 100, seed)}
    excluded = original["common_excluded_indices"]
    if not all(set(c["row_indices"]) <= set(excluded) for c in choices.values()):
        raise ValueError("New allocation escapes the original evaluation exclusion")
    return {"scout_indices": scout, "training_rows_per_arm": budget, "choices": choices,
            "common_excluded_indices": excluded,
            "inherited_oracle_truth_rows_inspected": original["oracle_truth_rows_inspected"],
            "new_truth_rows_inspected_for_allocation": 0}


def source_label_map(args, seed, old_record, original):
    """Only labels already saved with BADGE/oracle queries, not full target truth."""
    rows = kbs.read_csv(pilot.run_dir(Path(args.pilot_dir), seed, "badge") / "query_ids.csv")
    badge = old_record["choices"]["badge"]["row_indices"]
    if [int(r["row_index"]) for r in rows] != badge:
        raise ValueError("Saved BADGE query rows changed")
    labels = {int(r["row_index"]): int(r["label"]) for r in rows}
    frozen = pilot.read_json(oracle.selection_dir(args.oracle_dir, seed) / "query_labels.json")
    for method in oracle.METHODS:
        ids = original["choices"][method]["row_indices"]
        if len(frozen[method]) != len(ids):
            raise ValueError("Source query label length mismatch")
        for i, value in zip(ids, frozen[method]):
            if i in labels and labels[i] != value:
                raise ValueError("Conflicting source query labels")
            labels[i] = value
    return labels


def require_selections(args, verified):
    parent, originals, identity = verified
    sources, previous, _ = parent
    pilot.freeze_manifest(Path(args.output_dir), identity)
    records = {}
    for seed in identity["seeds"]:
        directory = oracle.selection_dir(args.output_dir, seed)
        if not pilot.completed(directory, oracle.signature(identity, seed, "select"), SELECT_FILES):
            raise ValueError(f"All allocations must be frozen before fitting: missing seed {seed}")
        record = pilot.read_json(directory / "selection.json")
        expected = select_queries(sources[2][seed], originals[seed], seed)
        if record != expected:
            raise ValueError("Allocation differs from the frozen nested-subset policy or evaluation mask")
        labels = source_label_map(args, seed, previous[seed], originals[seed])
        frozen = pilot.read_json(directory / "query_labels.json")
        if frozen != {m: [labels[i] for i in c["row_indices"]] for m, c in record["choices"].items()}:
            raise ValueError("Frozen allocation labels disagree with source queries")
        records[seed] = record
    return records


def select(args, verified):
    parent, originals, identity = verified
    sources, previous, _ = parent
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        pilot.freeze_manifest(out, identity, create=True)
        for seed in identity["seeds"]:
            directory = oracle.selection_dir(out, seed)
            sig = oracle.signature(identity, seed, "select")
            if pilot.completed(directory, sig, SELECT_FILES):
                continue
            record = select_queries(sources[2][seed], originals[seed], seed)
            labels = source_label_map(args, seed, previous[seed], originals[seed])
            kbs.atomic_json(directory / "resolved_config.json", sig)
            kbs.atomic_json(directory / "selection.json", record)
            kbs.atomic_json(directory / "query_labels.json", {
                m: [labels[i] for i in c["row_indices"]] for m, c in record["choices"].items()})
            kbs.finish(directory, sig, SELECT_FILES)
            print(f"Frozen seed {seed}: four allocations from saved query IDs; evaluation mask unchanged", flush=True)
        require_selections(args, verified)


def train(args, verified):
    parent, _, identity = verified
    p = parent[0][1]
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)
        todo = [(s, m) for s in identity["seeds"] for m in METHODS if not pilot.completed(
            oracle.run_dir(out, s, m), oracle.train_signature(identity, out, s, m), oracle.TRAIN_FILES)]
        if not todo:
            print("All allocation fits verified; zero new fits.", flush=True)
            return
        runtime = identity["runtime"]
        device = kbs.device_for(runtime["device"])
        torch.set_num_threads(runtime["threads"])
        head, ref, target, labels = pilot.load_inputs(args)
        old = target["logits"].argmax(1).numpy()
        for seed, method in todo:
            directory = oracle.run_dir(out, seed, method)
            sig = oracle.train_signature(identity, out, seed, method)
            choice = records[seed]["choices"][method]
            ids = np.asarray(choice["row_indices"], dtype=np.int64)
            y = pilot.purchased_labels(labels, ids, p["num_classes"])
            frozen = pilot.read_json(oracle.selection_dir(out, seed) / "query_labels.json")[method]
            if y.tolist() != frozen:
                raise ValueError("Training labels differ from frozen source-query labels")
            sd = pilot.selection_dir(Path(args.pilot_dir), seed)
            count = choice["protection_count"]
            probe = np.load(sd / "probe_predictions.npy", mmap_mode="r")
            if not np.all(old[ids[-count:]] == y[-count:]) or (choice["family"] == oracle.METHODS[0]
                    and not np.all(probe[ids[-count:]] != y[-count:])):
                raise ValueError("Protected rows violate their inherited oracle eligibility")
            ridx = np.load(sd / "reference_ids.npy")
            if not np.array_equal(ridx, kbs.replay_indices(np.asarray(ref["labels"]), p["num_classes"], 5, seed)):
                raise ValueError("Reference rows changed")
            kbs.atomic_json(directory / "resolved_config.json", sig)
            print(f"Repair {method} seed {seed}: {len(ids)-count} BADGE + {count} oracle protection", flush=True)
            kbs.seed_all(seed)
            repaired, trace = pilot.fit(head, ref, target, ids, y, ridx, p, seed, device)
            baseline = pilot.read_json(pilot.run_dir(Path(args.pilot_dir), seed, "badge") / "trace.json")
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256"):
                if trace[key] != baseline[key]:
                    raise ValueError(f"Training stream mismatch: {key}")
            started = time.perf_counter()
            pred = kbs.predict(repaired, target["features"], device, runtime["inference_batch_size"])
            trace.update({"full_pool_inference_wall_s": time.perf_counter() - started,
                "target_labels": len(ids), "reference_labels": len(ridx), "replay_ce": False,
                "kd_weight": pilot.SETTINGS["kd_weight"], "temperature": pilot.SETTINGS["temperature"],
                "initialization": "source_head", "protection_count": count,
                "inherited_oracle_truth_rows_inspected": records[seed]["inherited_oracle_truth_rows_inspected"]})
            torch.save(repaired.state_dict(), directory / "head.pt")
            np.save(directory / "predictions.npy", pred)
            kbs.write_csv(directory / "query_ids.csv", [{"row_index": int(i), "label": int(v),
                "phase": "badge_fill" if j < len(ids)-count else "oracle_protection"}
                for j, (i, v) in enumerate(zip(ids, y))])
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, sig, oracle.TRAIN_FILES)


def report(curves, paired, audit, seeds):
    fields = ["overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_after",
              "noncollapse_negative_flips", "noncollapse_positive_flips", "noncollapse_new_collapses"]
    lines = ["# 保护配额实验（继承 oracle 真值筛选）", "",
        f"配对种子 {seeds}；新训练为两组各 5%/10%，0% BADGE 和两组 20% 端点复用。",
        "全部工作点使用上一轮完全相同的六组查询并集排除评估集；均值 ± 样本标准差。",
        "只复用已有查询不消除原 oracle 的完整目标真值特权；本表不能解释为同标注成本的可部署方法。", "",
        "| 保护来源 | 保护数 | 整体 F1 | 崩溃类 F1 | 稳定类 F1 | 非崩溃类负翻转 | 非崩溃类正翻转 | 新崩溃 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in curves:
        lines.append(f"| {r['family']} | {r['protection_count']} | " +
                     " | ".join(pilot.format_stat(r, k) for k in fields) + " |")
    lines += ["", "两行 0 配额是同一个 BADGE 结果，为展示两组工作点而重复，不能算独立证据。",
        "重点先看相同配额的定向保护−普通正确样本，再对照 BADGE 的恢复与损伤变化。", "",
        "| 配对对照 | 崩溃类 F1 差 | 整体 F1 差 | 稳定类 F1 差 | 非崩溃类负翻转差 | 新崩溃差 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append("| " + r["comparison"] + " | " + " | ".join(pilot.format_stat(r, k)
                     for k in (fields[1], fields[0], fields[2], fields[3], fields[5])) + " |")
    lines += ["", "新配额组成（含普通 BADGE 部分意外选中的保护样本）：", "",
        "| 配置 | 指定保护数 | 与原 BADGE 重合的保护数 | 实际替换的 BADGE 行数 | 后 20% 中预演误伤数 | 后 20% 中崩溃类数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in diagnosis.aggregate(audit, ["method"]):
        lines.append("| " + r["method"] + " | " + " | ".join(pilot.format_stat(r, k) for k in
            ("protection_count", "protected_overlap_badge_count", "rows_outside_badge_count",
             "supplement_probe_damage", "supplement_collapse_rows")) + " |")
    lines += ["", "保护子集按固定随机顺序嵌套；BADGE 按原顺序补足并去重，保护行按 ID 排序放在末尾。",
        "前 80% 查询、参考行、源头部初始化、损失、优化步数及训练抽样位置流不变；不重训预演或旧端点。",
        "保护配额是分配给 oracle 子集的行数；BADGE 补充部分也可能自然包含源正确/预演误伤样本，不能把配额当作全部保护样本数量。",
        "报告全部四个工作点，不自动选赢家、不插值或外推恢复—损伤曲线、不自动声明通过或显著性。",
        "误伤减少若伴随恢复下降仍是取舍；逐种子差值、正翻转、新崩溃与逐类支持数必须一起看。",
        "本轮根据既有 M12 结果提出，属于开发性探索；三个种子不是独立环境重复，也不能证明新颖性。", ""]
    return "\n".join(lines)


def evaluate(args, verified):
    parent, originals, identity = verified
    sources, previous, source_identity = parent
    info, p, saved_badge, _ = sources
    out, root, source_root = Path(args.output_dir), Path(args.pilot_dir), Path(args.oracle_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)
        hashes = {}
        for seed in identity["seeds"]:
            for method in METHODS:
                rd = oracle.run_dir(out, seed, method)
                if not pilot.completed(rd, oracle.train_signature(identity, out, seed, method), oracle.TRAIN_FILES):
                    raise ValueError(f"All new fits must finish before evaluation: {seed}/{method}")
                hashes[f"{seed}/{method}"] = kbs.file_sha(rd / "complete.json")
        sig = {"identity_sha256": kbs.digest(identity), "runs": hashes}
        directory = out / "evaluation"
        if pilot.completed(directory, sig, EVAL_FILES):
            print(f"Verified report: {directory / 'report.md'}", flush=True)
            return
        torch.set_num_threads(identity["runtime"]["threads"])
        y, old = oracle.load_truth_and_source(args)
        n, classes = info["sample_counts"]["target"], p["num_classes"]
        diagnosis.vector(y, n, classes, "truth")
        old_rows = kbs.read_csv(source_root / "evaluation/by_seed.csv")
        old_metrics = {(int(r["seed"]), r["method"]): r for r in old_rows}
        expected_keys = {(s, m) for s in source_identity["seeds"] for m in ("static", *pilot.METHODS, *oracle.METHODS)}
        if len(old_metrics) != len(old_rows) or set(old_metrics) != expected_keys:
            raise ValueError("Source oracle evaluation rows are missing/duplicated")
        rows, paired, curves, per_class, audit = [], [], [], [], []
        for seed, record in records.items():
            print(f"Evaluating seed {seed}: unchanged oracle common mask", flush=True)
            sd = pilot.selection_dir(root, seed)
            probe = diagnosis.vector(np.load(sd / "probe_predictions.npy"), n, classes, "probe")
            original = originals[seed]
            if original != oracle.select_queries(previous[seed], y, old, probe, classes, seed):
                raise ValueError("Original oracle queries violate their declared policy")
            scout = np.asarray(record["scout_indices"], dtype=np.int64)
            counts = pilot.relation_counts(old[scout], y[scout], classes)
            if not np.array_equal(np.load(sd / "scout_labels.npy"), y[scout]) or not np.array_equal(
                    np.load(sd / "relation_counts.npy"), counts) or previous[seed] != pilot.select_queries(
                        saved_badge[seed], old, probe, counts, seed):
                raise ValueError("Original scout selection semantics changed")
            mask = np.ones(n, dtype=bool); mask[record["common_excluded_indices"]] = False
            static, cr = pilot.metrics(y, old, old, mask, p)
            diagnosis.reconcile(static, old_metrics[seed, "static"])
            rows.append({"seed": seed, "method": "static", **static})
            per_class.extend({"seed": seed, "method": "static", **r} for r in cr)
            metrics, traces = {}, []
            for method in (*pilot.METHODS, *oracle.METHODS, *METHODS):
                if method in METHODS:
                    rd, choice = oracle.run_dir(out, seed, method), record["choices"][method]
                elif method in oracle.METHODS:
                    rd, choice = oracle.run_dir(source_root, seed, method), original["choices"][method]
                else:
                    rd, choice = pilot.run_dir(root, seed, method), previous[seed]["choices"][method]
                ids = np.asarray(choice["row_indices"], dtype=np.int64)
                query = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query] != ids.tolist() or [int(r["label"]) for r in query] != y[ids].tolist():
                    raise ValueError("Saved query rows/labels differ from declared queries")
                pred = diagnosis.vector(np.load(rd / "predictions.npy", mmap_mode="r"), n, classes, "final")
                m, cr = pilot.metrics(y, old, pred, mask, p)
                if method not in METHODS:
                    diagnosis.reconcile(m, old_metrics[seed, method])
                metrics[method] = m
                rows.append({"seed": seed, "method": method, **m})
                query_counts = np.bincount(y[ids], minlength=classes)
                per_class.extend({"seed": seed, "method": method, **r,
                    "query_count": int(query_counts[r["class_id"]])} for r in cr)
                traces.append(pilot.read_json(rd / "trace.json"))
                if method in METHODS:
                    tail = ids[len(scout):]
                    audit.append({"seed": seed, "method": method, **{k: choice[k] for k in
                        ("protection_count", "protected_overlap_badge_count", "rows_outside_badge_count")},
                        "supplement_probe_damage": int(((old[tail] == y[tail]) & (probe[tail] != y[tail])).sum()),
                        "supplement_source_correct": int((old[tail] == y[tail]).sum()),
                        "supplement_collapse_rows": int(np.isin(y[tail], p["collapse_classes"]).sum()),
                        "supplement_classes": len(np.unique(y[tail]))})
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256",
                        "target_labels", "reference_labels", "replay_ce", "kd_weight", "temperature", "initialization"):
                if len({t[key] for t in traces}) != 1:
                    raise ValueError(f"Training protocol differs across arms: {key}")
            for family in oracle.METHODS:
                for pct in (0, 5, 10, 20):
                    method = "badge" if pct == 0 else family if pct == 20 else f"{family}_p{pct:02d}"
                    curves.append({"seed": seed, "family": family, "protection_percent": pct,
                                   "protection_count": p["budget"] * pct // 100, **metrics[method]})
            comparisons = [(f"{oracle.METHODS[0]}_p{pct:02d}", f"{oracle.METHODS[1]}_p{pct:02d}") for pct in (5, 10)]
            comparisons += [(oracle.METHODS[0], oracle.METHODS[1])]
            comparisons += [(m, "badge") for m in (*METHODS, *oracle.METHODS)]
            for a, b in comparisons:
                paired.append({"seed": seed, "comparison": f"{a} minus {b}", **{
                    k: metrics[a][k] - metrics[b][k] if metrics[a][k] is not None and metrics[b][k] is not None else None
                    for k in metrics[a]}})
        summary = diagnosis.aggregate(rows, ["method"])
        paired_summary = diagnosis.aggregate(paired, ["comparison"])
        curve_summary = diagnosis.aggregate(curves, ["family", "protection_percent", "protection_count"])
        kbs.atomic_json(directory / "resolved_config.json", sig)
        for name, values in [("by_seed", rows), ("summary", summary), ("paired_by_seed", paired),
            ("paired_summary", paired_summary), ("curve_by_seed", curves), ("curve_summary", curve_summary),
            ("per_class", per_class), ("allocation_audit", audit)]:
            kbs.write_csv(directory / f"{name}.csv", values)
        (directory / "report.md").write_text(report(curve_summary, paired_summary, audit, identity["seeds"]))
        kbs.finish(directory, sig, EVAL_FILES)
        print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "select", "train", "evaluate"])
    for flag, default in [("oracle-dir", "outputs/kbs_oracle_protection_v1"),
        ("pilot-dir", "outputs/kbs_repair_aware_v1"), ("study-dir", "outputs/kbs_supplement_v1"),
        ("cache-dir", "outputs/kbs_supplement_v1/cache"), ("checkpoint", "outputs/tls22_cnn/best_model.pt"),
        ("output-dir", "outputs/kbs_protection_budget_v1")]:
        p.add_argument("--" + flag, default=str(kbs.CORE / default))
    p.add_argument("--seeds", default="0,1,2")
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        seeds = kbs.int_list(args.seeds)
        print(json.dumps({"settings": SETTINGS, "seeds": seeds, "new_fits": len(seeds) * 4,
                          "probe_fits": 0, "endpoint_fits": 0}, indent=2))
        return
    verified = verify_sources(args)
    if args.command == "preflight":
        p, identity = verified[0][0][1], verified[2]
        kbs.device_for(identity["runtime"]["device"])
        print(json.dumps({"status": "ready", "seeds": identity["seeds"], "new_fits": len(identity["seeds"]) * 4,
            "training_rows": p["budget"], "new_protection_counts": [p["budget"] // 20, p["budget"] // 10],
            "steps_per_fit": p["optimizer_steps"], "runtime": identity["runtime"]}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, verified)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
