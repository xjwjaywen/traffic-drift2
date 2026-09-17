#!/usr/bin/env python3
"""Exploratory, fixed-budget repair-aware acquisition on the existing CARE cache.

Stages: select (only shared scout labels), train (only frozen query labels),
evaluate (all labels, only after every requested run is complete). No oracle arm.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

import kbs_supplement as kbs
from kbs_acquisition_pilot import check_ids, completed, read_json

METHODS = ("badge", "random", "flip_uniform", "flip_relation")
SETTINGS = {
    "schema": "care-repair-aware-v1",
    "scout_fraction": [4, 5],
    "scout": "first 80% of saved ordered BADGE query; A uses its remaining suffix",
    "probe": "same CE+reference KD engine and step count as final fits; scout labels only",
    "flip_pool": "source argmax differs from probe argmax; scout rows excluded",
    "relation": "count[true victim, source absorber] from scout errors only",
    "relation_weight": "1 + count[probe_prediction, source_prediction] inside flip pool",
    "fallback": "if flip pool is too small, take all and fill uniformly from remaining rows",
    "flip_query_order": "sort supplemental row IDs; identical C/D sets have identical training order",
    "repair": "fresh source head; target CE + reference KD; no reference CE",
    "kd_weight": 0.5, "temperature": 2.0, "reference_per_class": 5,
    "evaluation": "common per-seed exclusion of all four methods' queries",
    "status": "M12 development pilot; no independent validation or automatic success claim",
}
SELECT_FILES = ["resolved_config.json", "selection.json", "scout_labels.npy", "reference_ids.npy",
                "relation_counts.npy", "probe_head.pt", "probe_predictions.npy", "probe_trace.json"]
TRAIN_FILES = ["resolved_config.json", "head.pt", "predictions.npy", "query_ids.csv", "trace.json"]
EVAL_FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "paired_by_seed.csv",
              "paired_summary.csv", "per_class.csv", "acquisition.csv", "report.md"]


def verify_sources(args):
    study, cache, out = (Path(getattr(args, n)).resolve() for n in ("study_dir", "cache_dir", "output_dir"))
    for source in (study, cache):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError("Use a sibling output directory, separate from the old study/cache")
    info = read_json(cache / "manifest.json")
    p = read_json(study / "study_manifest.json")["protocol"]
    if info["spec"]["schema"] != kbs.SCHEMA or info["spec"]["loader_sha256"] != kbs.loader_sha():
        raise ValueError("Cache schema or loader changed")
    if set(info["files"]) != {"head.pt", "reference.pt", "target.pt"}:
        raise ValueError("Incomplete cache inventory")
    if info["fingerprint"] != kbs.digest({"spec": info["spec"], "inputs": info["input_stream_sha256"]}):
        raise ValueError("Invalid cache fingerprint")
    if p["cache"] != info["fingerprint"] or p["checkpoint_sha256"] != info["checkpoint_sha256"]:
        raise ValueError("Study/cache identity mismatch")
    if p["implementation_sha256"] != kbs.implementation_sha():
        raise ValueError("Original repair engine changed; do not mix implementations")
    if kbs.file_sha(args.checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the cache")
    if p["num_classes"] != info["num_classes"]:
        raise ValueError("Class universe mismatch")
    if info["periods"] != {"reference": "M-2022-4", "target": "M-2022-12"}:
        raise ValueError("v1 is a fixed M4-reference / M12-development pilot")
    n, budget = info["sample_counts"]["target"], p["budget"]
    if budget % 5 or not 5 <= budget < n:
        raise ValueError("Existing budget must be divisible by five and smaller than the pool")
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32 - 50000 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers within the supported range")
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("Threads and inference batch size must be positive")
    saved, hashes = {}, {}
    for seed in seeds:
        directory = study / "selections/badge" / f"seed_{seed}"
        sig = {"cache": info["fingerprint"], "implementation": kbs.implementation_sha(),
               "selector": "badge", "budget": budget, "seed": seed}
        if not completed(directory, sig, ["resolved_config.json", "selection.json"]):
            raise ValueError(f"Missing verified original BADGE selection: {directory}")
        saved[seed] = check_ids(read_json(directory / "selection.json")["row_indices"], n, budget)
        hashes[str(seed)] = kbs.file_sha(directory / "selection.json")
    print("Verifying existing feature cache (read-only; no extraction)...", flush=True)
    kbs.validate_cache_files(cache, info)
    identity = {"settings": SETTINGS, "seeds": seeds, "source_protocol": p,
                "cache_manifest_sha256": kbs.file_sha(cache / "manifest.json"),
                "study_manifest_sha256": kbs.file_sha(study / "study_manifest.json"),
                "baseline_selection_sha256": hashes,
                "code_sha256": {name: kbs.file_sha(Path(__file__).with_name(name)) for name in
                                ("kbs_repair_aware.py", "kbs_acquisition_pilot.py")},
                "device": args.device, "threads": args.threads, "inference_batch_size": args.batch_size,
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    return info, p, saved, identity


def freeze_manifest(out, identity, create=False):
    path = out / "study_manifest.json"
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError("Pilot inputs/settings changed; preserve results and use a new output directory")
    elif create:
        kbs.atomic_json(path, {"identity": identity, "environment": kbs.runtime_info()})
    else:
        raise ValueError("Run select before train/evaluate")


def signature(identity, seed, stage, method=None):
    return {"identity_sha256": kbs.digest(identity), "seed": seed, "stage": stage, "method": method}


def selection_dir(out, seed):
    return out / "selections" / f"seed_{seed}"


def run_dir(out, seed, method):
    return out / "runs" / method / f"seed_{seed}"


def relation_counts(old_scout, y_scout, classes):
    """An observed v->a error may make the reverse a->v update risky for true a."""
    old, y = np.asarray(old_scout), np.asarray(y_scout)
    if old.shape != y.shape or old.ndim != 1:
        raise ValueError("Invalid scout labels/predictions")
    if np.any((y < 0) | (y >= classes)) or np.any((old < 0) | (old >= classes)):
        raise ValueError("Scout label out of range")
    errors = old != y
    return np.bincount(y[errors] * classes + old[errors], minlength=classes**2).reshape(classes, classes)


def select_queries(saved_badge, old, probe, counts, seed):
    """Pure selection API: NO target labels, evaluation groups, or post-repair metrics."""
    n, budget = len(old), len(saved_badge)
    badge = check_ids(saved_badge, n, budget)
    if budget % 5 or budget >= n or budget < 5:
        raise ValueError("Invalid budget")
    old, probe = np.asarray(old), np.asarray(probe)
    if old.shape != probe.shape or counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("Invalid predictions/relation matrix")
    classes = len(counts)
    if np.any((old < 0) | (old >= classes)) or np.any((probe < 0) | (probe >= classes)):
        raise ValueError("Prediction outside the class universe")
    scout_n = budget * 4 // 5
    scout, slots = badge[:scout_n], budget - scout_n
    eligible = np.ones(n, dtype=bool)
    eligible[scout] = False
    remaining = np.flatnonzero(eligible)
    pool = np.flatnonzero(eligible & (old != probe))
    # Shared seed for C/D: identical weights give identical choices, including fallback.
    rng = np.random.default_rng(seed + 41011)
    choices = {
        "badge": {"row_indices": badge.tolist(), "fallback_count": 0},
        "random": {"row_indices": np.r_[scout, rng.choice(remaining, slots, replace=False)].tolist(),
                   "fallback_count": 0},
    }
    for method in ("flip_uniform", "flip_relation"):
        rng = np.random.default_rng(seed + 42011)
        weights = np.ones(len(pool), dtype=np.float64)
        if method == "flip_relation":
            weights += counts[probe[pool], old[pool]]
        take = min(slots, len(pool))
        # If the whole pool fits, weighting cannot change its membership. Do not
        # let weighted draw order alter either the fallback RNG or training order.
        picked = (pool.copy() if len(pool) <= slots else
                  rng.choice(pool, take, replace=False, p=weights / weights.sum()))
        fallback = slots - take
        if fallback:
            allowed = eligible.copy()
            allowed[picked] = False
            picked = np.r_[picked, rng.choice(np.flatnonzero(allowed), fallback, replace=False)]
        ids = check_ids(np.r_[scout, np.sort(picked)], n, budget)
        choices[method] = {"row_indices": ids.tolist(), "fallback_count": fallback}
    for choice in choices.values():
        check_ids(choice["row_indices"], n, budget)
    return {"scout_indices": scout.tolist(), "flip_pool_count": len(pool),
            "relation_supported_flip_count": int((counts[probe[pool], old[pool]] > 0).sum()),
            "choices": choices,
            "common_excluded_indices": sorted({i for c in choices.values() for i in c["row_indices"]})}


def load_inputs(args):
    cache = Path(args.cache_dir)
    state = torch.load(cache / "head.pt", map_location="cpu", weights_only=True)
    head = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
    head.load_state_dict(state)
    reference = torch.load(cache / "reference.pt", map_location="cpu", weights_only=True)
    target = torch.load(cache / "target.pt", map_location="cpu", weights_only=True)
    # Legacy .pt bundles truth with features. Keep truth out of all selector APIs.
    labels = target.pop("labels")
    return head, reference, target, labels


def purchased_labels(labels, ids, classes):
    """The only label access during selection/training is indexing declared queries."""
    values = np.asarray(labels[ids], dtype=np.int64)
    if values.shape != (len(ids),) or np.any((values < 0) | (values >= classes)):
        raise ValueError("Invalid purchased labels")
    return values


def fit(head, ref, target, ids, y, ridx, p, seed, device):
    return kbs.fit_controlled(
        head, target["features"][ids], y, ref["features"][ridx],
        np.asarray(ref["labels"])[ridx], ref["logits"][ridx],
        replay_ce=False, kd_weight=SETTINGS["kd_weight"], temperature=SETTINGS["temperature"],
        target_weight=p["target_weight"], replay_weight=p["replay_weight"],
        steps=p["optimizer_steps"], batch_size=p["batch_size"], lr=p["lr"],
        weight_decay=p["weight_decay"], seed=seed, device=device)


def validate_selection(record, badge, n, budget):
    scout = check_ids(record["scout_indices"], n, budget * 4 // 5)
    if not np.array_equal(scout, badge[:len(scout)]) or set(record["choices"]) != set(METHODS):
        raise ValueError("Scout prefix/methods differ from the fixed protocol")
    for method, choice in record["choices"].items():
        ids = check_ids(choice["row_indices"], n, budget)
        if not np.array_equal(ids[:len(scout)], scout):
            raise ValueError("Methods do not share the exact scout prefix")
        if method == "badge" and not np.array_equal(ids, badge):
            raise ValueError("BADGE baseline differs from the saved order")
    union = sorted({i for c in record["choices"].values() for i in c["row_indices"]})
    if union != record["common_excluded_indices"] or len(union) >= n:
        raise ValueError("Invalid common query union / empty evaluation set")


def require_selections(out, sources):
    info, p, saved, identity = sources
    records = {}
    for seed in identity["seeds"]:
        directory = selection_dir(out, seed)
        if not completed(directory, signature(identity, seed, "select"), SELECT_FILES):
            raise ValueError(f"Selection must finish for every requested seed: missing {seed}")
        record = read_json(directory / "selection.json")
        validate_selection(record, saved[seed], info["sample_counts"]["target"], p["budget"])
        records[seed] = record
    return records


def select(args, sources):
    info, p, saved, identity = sources
    out = Path(args.output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        freeze_manifest(out, identity, create=True)
        todo = [s for s in identity["seeds"] if not completed(
            selection_dir(out, s), signature(identity, s, "select"), SELECT_FILES)]
        if not todo:
            require_selections(out, sources)
            print("All selections verified; no probe retraining.", flush=True)
            return
        device = kbs.device_for(args.device)
        torch.set_num_threads(args.threads)
        head, ref, target, labels = load_inputs(args)
        old = target["logits"].argmax(1).numpy()
        for seed in todo:
            directory = selection_dir(out, seed)
            directory.mkdir(parents=True, exist_ok=True)
            sig = signature(identity, seed, "select")
            kbs.atomic_json(directory / "resolved_config.json", sig)
            scout = saved[seed][:p["budget"] * 4 // 5]
            y = purchased_labels(labels, scout, p["num_classes"])
            ridx = kbs.replay_indices(np.asarray(ref["labels"]), p["num_classes"], 5, seed)
            print(f"Probe seed {seed}: {len(scout)} target labels, {len(ridx)} reference rows", flush=True)
            kbs.seed_all(seed)
            started = time.perf_counter()
            probe_head, trace = fit(head, ref, target, scout, y, ridx, p, seed, device)
            selection_started = time.perf_counter()
            probe = kbs.predict(probe_head, target["features"], device, args.batch_size)
            counts = relation_counts(old[scout], y, p["num_classes"])
            record = select_queries(saved[seed], old, probe, counts, seed)
            validate_selection(record, saved[seed], len(old), p["budget"])
            trace["probe_inference_and_selection_wall_s"] = time.perf_counter() - selection_started
            trace["probe_total_wall_s"] = time.perf_counter() - started
            trace["labels_purchased_before_selection"] = len(scout)
            trace["source_cache_fingerprint"] = info["fingerprint"]
            np.save(directory / "scout_labels.npy", y)
            np.save(directory / "reference_ids.npy", ridx)
            np.save(directory / "relation_counts.npy", counts)
            np.save(directory / "probe_predictions.npy", probe)
            torch.save(probe_head.state_dict(), directory / "probe_head.pt")
            kbs.atomic_json(directory / "probe_trace.json", trace)
            kbs.atomic_json(directory / "selection.json", record)
            kbs.finish(directory, sig, SELECT_FILES)
            print(f"Frozen seed {seed}: flip pool={record['flip_pool_count']}, "
                  f"relation-supported={record['relation_supported_flip_count']}", flush=True)


def train_signature(identity, out, seed, method):
    return {**signature(identity, seed, "train", method),
            "selection_complete_sha256": kbs.file_sha(selection_dir(out, seed) / "complete.json")}


def train(args, sources):
    _, p, _, identity = sources
    out = Path(args.output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        freeze_manifest(out, identity)
        records = require_selections(out, sources)  # before loading any target labels
        todo = [(s, m) for s in identity["seeds"] for m in METHODS if not completed(
            run_dir(out, s, m), train_signature(identity, out, s, m), TRAIN_FILES)]
        if not todo:
            print("All final repairs verified; no retraining.", flush=True)
            return
        device = kbs.device_for(args.device)
        torch.set_num_threads(args.threads)
        head, ref, target, labels = load_inputs(args)
        for seed, method in todo:
            directory = run_dir(out, seed, method)
            directory.mkdir(parents=True, exist_ok=True)
            sig = train_signature(identity, out, seed, method)
            kbs.atomic_json(directory / "resolved_config.json", sig)
            ids = np.asarray(records[seed]["choices"][method]["row_indices"], dtype=np.int64)
            y = purchased_labels(labels, ids, p["num_classes"])
            sdir = selection_dir(out, seed)
            if not np.array_equal(y[:len(records[seed]["scout_indices"])], np.load(sdir / "scout_labels.npy")):
                raise ValueError("Scout truth changed between stages")
            ridx = np.load(sdir / "reference_ids.npy")
            if not np.array_equal(ridx, kbs.replay_indices(np.asarray(ref["labels"]), p["num_classes"], 5, seed)):
                raise ValueError("Reference sampling changed")
            print(f"Repair {method} seed {seed}: {len(ids)} labels, fresh source head", flush=True)
            kbs.seed_all(seed)
            repaired, trace = fit(head, ref, target, ids, y, ridx, p, seed, device)
            started = time.perf_counter()
            pred = kbs.predict(repaired, target["features"], device, args.batch_size)
            trace["full_pool_inference_wall_s"] = time.perf_counter() - started
            trace.update({"target_labels": len(ids), "reference_labels": len(ridx),
                          "replay_ce": False, "kd_weight": SETTINGS["kd_weight"],
                          "temperature": SETTINGS["temperature"], "initialization": "source_head"})
            torch.save(repaired.state_dict(), directory / "head.pt")
            np.save(directory / "predictions.npy", pred)
            kbs.write_csv(directory / "query_ids.csv", [
                {"row_index": int(i), "label": int(v), "phase": "scout" if j < len(records[seed]["scout_indices"]) else "supplement"}
                for j, (i, v) in enumerate(zip(ids, y))])
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, sig, TRAIN_FILES)


def metrics(y, old, pred, mask, p):
    summary, rows, _ = kbs.compare_predictions(
        y, old, pred, mask, p["num_classes"], p["collapse_classes"], p["stable_classes"],
        p["collapse_recall_threshold"], p["f1_drop_threshold"])
    positive, negative = (old != y) & (pred == y), (old == y) & (pred != y)
    for c, row in enumerate(rows):
        subset = mask & (y == c)
        row["positive_flips"] = int((subset & positive).sum())
        row["negative_flips"] = int((subset & negative).sum())
    for group, group_mask in [("all", np.ones(len(y), dtype=bool)),
                              ("noncollapse", ~np.isin(y, p["collapse_classes"]))]:
        use = group_mask & mask
        summary[f"{group}_positive_flips"] = int((use & positive).sum())
        summary[f"{group}_negative_flips"] = int((use & negative).sum())
        denom = int((use & (old == y)).sum())
        summary[f"{group}_negative_flip_rate_on_old_correct"] = float((use & negative).sum() / denom) if denom else None
    summary["accuracy_before"] = float((old[mask] == y[mask]).mean())
    summary["accuracy_after"] = float((pred[mask] == y[mask]).mean())
    return summary, rows


def aggregate(rows, key):
    result = []
    for name in dict.fromkeys(row[key] for row in rows):
        group = [r for r in rows if r[key] == name]
        entry = {key: name, "n_seeds": len(group)}
        for metric in group[0]:
            if metric in (key, "seed"):
                continue
            values = [r[metric] for r in group if r[metric] is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                entry[metric + "_mean"] = statistics.mean(values)
                entry[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(entry)
    return result


def format_stat(row, key):
    mean, sd = row.get(key + "_mean"), row.get(key + "_sd")
    return "—" if mean is None else f"{mean:.4f} ± {sd:.4f}" if sd is not None else f"{mean:.4f} ± —"


def report(summary, paired, acquisitions, p, seeds):
    fields = ["overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_after",
              "noncollapse_new_collapses", "collapse_residual_count", "noncollapse_negative_flips"]
    lines = ["# 修复预演选样：开发性预实验", "",
             f"完成 {len(seeds)} 个配对种子 {seeds}；每组总预算 {p['budget']}，共享前 80% BADGE 查询。",
             "修复器固定为目标 CE＋参考 KD；最终均从源分类头重新训练。表中只使用同一种子的四组查询并集排除评估。",
             "均值 ± 样本标准差；这些种子共用同一源模型和 M12，不是独立环境验证。", "",
             "| 方法 | 整体 F1 | 崩溃类 F1 | 稳定类 F1 | 新崩溃数 | 原崩溃残留 | 非崩溃类负翻转 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| " + row["method"] + " | " + " | ".join(format_stat(row, k) for k in fields) + " |")
    lines += ["", "配对增量为前者减后者；F1 越高越好，新增崩溃和负翻转越少越好。", "",
              "| 对照 | 整体 F1 差 | 崩溃类 F1 差 | 新崩溃差 | 非崩溃类负翻转差 |",
              "|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append("| " + row["comparison"] + " | " + " | ".join(format_stat(row, k) for k in
                     (fields[0], fields[1], fields[3], fields[5])) + " |")
    lines += ["", "后 20% 查询的事后诊断（不能替代最终修复结果）：", "",
              "| 方法 | 原本正确→预演错误 | 原本错误→预演正确 | 崩溃类查询数 | 随机回填数 |",
              "|---|---:|---:|---:|---:|"]
    for row in aggregate(acquisitions, "method"):
        lines.append("| " + row["method"] + " | " + " | ".join(format_stat(row, k) for k in
                     ("supplement_probe_negative", "supplement_probe_positive", "supplement_collapse_queries", "fallback_count")) + " |")
    lines += ["", "判读：先比较 flip_uniform 与 badge/random，再比较 flip_relation 与 flip_uniform。",
              "若减少损伤同时明显降低崩溃类恢复，只能说明取舍变化；不自动宣布成功、新颖性或显著性。",
              "flip_relation 在相同翻转池内按 1＋已确认反向错分次数加权；未知关系仍可被选中，不是硬门控。",
              "BADGE 后 20% 沿用源模型保存的原查询序列，并未按预演模型重新计算。",
              "共同预演也为基线生成，仅作配对分析；基线部署本身不需要预演。成本应分别报告预演、推断和最终训练。",
              "训练耗时不含旧特征提取和旧 BADGE 选样；不据此报告端到端效率。",
              "全部新目标标签（包括查询后不符合保护设想的样本）均计入预算。历史参考集另计且各组相同。",
              "旧缓存将标签与特征捆绑加载；选择阶段代码只索引共同侦察查询，训练阶段只索引已冻结查询。",
              "完整目标标签仅在四组、全部请求种子训练完成后的独立 evaluate 进程中用于离线评价；不计为方法可用标签。",
              "没有用评估类组或未查询真值选样；不声称 M10/M11 为从未使用的月份。", ""]
    return "\n".join(lines)


def evaluate(args, sources):
    info, p, _, identity = sources
    out = Path(args.output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        freeze_manifest(out, identity)
        records = require_selections(out, sources)
        hashes = {}
        for seed in identity["seeds"]:
            for method in METHODS:
                directory = run_dir(out, seed, method)
                if not completed(directory, train_signature(identity, out, seed, method), TRAIN_FILES):
                    raise ValueError(f"Every final fit must finish before evaluation: {method}, seed {seed}")
                hashes[f"{seed}/{method}"] = kbs.file_sha(directory / "complete.json")
        directory = out / "evaluation"
        sig = {"identity_sha256": kbs.digest(identity), "runs": hashes}
        if completed(directory, sig, EVAL_FILES):
            print(f"Verified report: {directory / 'report.md'}", flush=True)
            return
        # This is the first stage allowed to interpret truth outside the purchased queries.
        target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True)
        y = np.asarray(target.pop("labels"), dtype=np.int64)
        old = target["logits"].argmax(1).numpy()
        if len(y) != info["sample_counts"]["target"]:
            raise ValueError("Target size mismatch")
        del target
        rows, classes, acquisitions, paired_rows = [], [], [], []
        for seed, record in records.items():
            mask = np.ones(len(y), dtype=bool)
            mask[record["common_excluded_indices"]] = False
            probe = np.load(selection_dir(out, seed) / "probe_predictions.npy")
            scout = np.asarray(record["scout_indices"], dtype=np.int64)
            observed = np.load(selection_dir(out, seed) / "scout_labels.npy")
            if not np.array_equal(observed, y[scout]):
                raise ValueError("Saved scout labels do not match evaluation truth")
            if not np.array_equal(np.load(selection_dir(out, seed) / "relation_counts.npy"),
                                  relation_counts(old[scout], observed, p["num_classes"])):
                raise ValueError("Relation matrix does not match purchased scout errors")
            expected = select_queries(np.asarray(record["choices"]["badge"]["row_indices"]), old, probe,
                                      relation_counts(old[scout], observed, p["num_classes"]), seed)
            if record != expected:
                raise ValueError("Saved selection does not follow the frozen policy")
            static, per_class = metrics(y, old, old, mask, p)
            rows.append({"seed": seed, "method": "static", **static})
            classes.extend({"seed": seed, "method": "static", **r} for r in per_class)
            seed_metrics, traces = {}, []
            for method in METHODS:
                rd = run_dir(out, seed, method)
                pred = np.load(rd / "predictions.npy")
                if pred.shape != old.shape or pred.dtype.kind not in "iu" or np.any((pred < 0) | (pred >= p["num_classes"])):
                    raise ValueError("Invalid saved prediction rows")
                ids = np.asarray(record["choices"][method]["row_indices"], dtype=np.int64)
                query = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query] != ids.tolist() or [int(r["label"]) for r in query] != y[ids].tolist():
                    raise ValueError("Training query rows/truth differ from frozen selection")
                traces.append(read_json(rd / "trace.json"))
                m, cr = metrics(y, old, pred, mask, p)
                seed_metrics[method] = m
                rows.append({"seed": seed, "method": method, **m})
                classes.extend({"seed": seed, "method": method, **r} for r in cr)
                tail = ids[len(scout):]
                acquisitions.append({"seed": seed, "method": method, "total_labels": len(ids),
                    "total_old_correct": int((old[ids] == y[ids]).sum()),
                    "supplement_probe_negative": int(((old[tail] == y[tail]) & (probe[tail] != y[tail])).sum()),
                    "supplement_probe_positive": int(((old[tail] != y[tail]) & (probe[tail] == y[tail])).sum()),
                    "supplement_collapse_queries": int(np.isin(y[tail], p["collapse_classes"]).sum()),
                    "flip_pool_count": record["flip_pool_count"],
                    "relation_supported_flip_count": record["relation_supported_flip_count"],
                    "fallback_count": record["choices"][method]["fallback_count"]})
            for key in ("target_stream_sha256", "reference_stream_sha256", "optimizer_steps"):
                if len({t[key] for t in traces}) != 1:
                    raise ValueError("Final training streams/steps were not paired")
            for a, b in [("random", "badge"), ("flip_uniform", "badge"), ("flip_uniform", "random"),
                         ("flip_relation", "flip_uniform"), ("flip_relation", "badge")]:
                paired_rows.append({"seed": seed, "comparison": f"{a} minus {b}", **{
                    k: seed_metrics[a][k] - seed_metrics[b][k] if seed_metrics[a][k] is not None and seed_metrics[b][k] is not None else None
                    for k in seed_metrics[a]}})
        directory.mkdir(parents=True, exist_ok=True)
        kbs.atomic_json(directory / "resolved_config.json", sig)
        summaries, pair_summary = aggregate(rows, "method"), aggregate(paired_rows, "comparison")
        for name, content in [("by_seed", rows), ("summary", summaries), ("paired_by_seed", paired_rows),
                              ("paired_summary", pair_summary), ("per_class", classes), ("acquisition", acquisitions)]:
            kbs.write_csv(directory / f"{name}.csv", content)
        (directory / "report.md").write_text(report(summaries, pair_summary, acquisitions, p, identity["seeds"]))
        kbs.finish(directory, sig, EVAL_FILES)
        print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "select", "train", "evaluate"])
    p.add_argument("--study-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/kbs_repair_aware_v1"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096, help="Inference only; train batch inherited from original study")
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        seeds = kbs.int_list(args.seeds)
        print(json.dumps({"settings": SETTINGS, "methods": METHODS, "seeds": seeds,
                          "probe_fits": len(seeds), "final_fits": 4 * len(seeds),
                          "training_steps": "inherited from source study for probe and final fits"}, indent=2))
        return
    sources = verify_sources(args)
    if args.command == "preflight":
        kbs.device_for(args.device)
        print(json.dumps({"status": "ready", "budget": sources[1]["budget"],
                          "scout_labels": sources[1]["budget"] * 4 // 5,
                          "reference_labels": sources[1]["num_classes"] * 5,
                          "steps_per_fit": sources[1]["optimizer_steps"],
                          "seeds": sources[3]["seeds"], "environment": kbs.runtime_info()}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, sources)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
