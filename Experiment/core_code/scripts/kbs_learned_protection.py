#!/usr/bin/env python3
"""Fixed 950+50 acquisition pilot. Only 150 already purchased labels fit a risk ranker.

The saved probe used the first 800 queries; its next 150 BADGE queries are out of
probe training. No oracle selections, unqueried truth or final predictions enter
acquisition. This is M12 development, not an independent confirmation experiment.
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

METHODS = ("random50", "flip_uniform50", "flip_confidence50", "flip_risk50")
FEATURES = ("source_confidence", "source_margin", "probe_probability_of_source_class", "probe_margin")
SETTINGS = {
    "schema": "care-learned-protection-v1", "shared_badge_fraction": [19, 20],
    "probe": "reuse first-80%-BADGE probe; never refit",
    "ranker_labels": "only BADGE positions [80%,95%); never used to train the saved probe",
    "ranker_target": "source correct, conditional on source/probe argmax disagreement",
    "ranker_features": FEATURES, "ranker_model": "standardized four-feature logistic regression",
    "ranker_objective": "sum binary CE + 0.5*L2(weights)^2; unpenalized intercept; no class weighting",
    "ranker_l2": 1.0, "ranker_min_each_class": 5, "ranker_max_iterations": 50,
    "ranker_gradient_tolerance": 1e-8,
    "ranker_fallback": "source confidence when either binary class has fewer than 5 examples or solver fails",
    "selection": "top 5%-budget ranked flipped points, excluding first 95% BADGE; fixed random ties",
    "small_pool": "take all flips then uniform fill from unqueried pool; count every queried label",
    "rng": "default_rng(seed+72021); same random priority for ties and uniform policies",
    "repair": "original CE+reference KD, fresh source head, same rows/steps/streams; keep all query labels",
    "evaluation": "union of original four policies plus new four; recompute BADGE on expanded mask",
    "status": "M12 development after oracle exploration; no winner/significance/novelty claim",
}
SELECT_FILES = ["resolved_config.json", "selection.json", "ranker.json", "fit_labels.npy",
                "source_predictions.npy", "signals.npy", "selection_trace.json"]
TRAIN_FILES = pilot.TRAIN_FILES
EVAL_FILES = ["resolved_config.json", "by_seed.csv", "summary.csv", "paired_by_seed.csv",
              "paired_summary.csv", "per_class.csv", "acquisition.csv", "candidate_audit.csv", "report.md"]


def verify_sources(args):
    previous = pilot.read_json(Path(args.pilot_dir) / "study_manifest.json")["identity"]
    verifier = argparse.Namespace(**vars(args), threads=previous["threads"])
    sources, records, audit = diagnosis.verify_inputs(verifier)
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(previous["seeds"]):
        raise ValueError("Use unique seeds from the completed original pilot")
    if sources[1]["budget"] % 20:
        raise ValueError("Budget must be divisible by 20")
    identity = {"settings": SETTINGS, "seeds": seeds, "source_audit": audit,
        "runtime": {k: previous[k] for k in ("device", "threads", "inference_batch_size")},
        "code_sha256": kbs.file_sha(Path(__file__))}
    # Normalize tuples to the exact JSON representation used by freeze_manifest.
    return sources, records, json.loads(json.dumps(identity))


def parts(badge):
    badge = np.asarray(badge, dtype=np.int64)
    b = len(badge)
    if not b or b % 20 or len(np.unique(badge)) != b:
        raise ValueError("Invalid ordered BADGE query")
    return badge[:b*19//20], badge[b*4//5:b*19//20], b//20


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=np.float64)))


def fit_ranker(x, binary):
    """Small deterministic L2 logistic model; no candidate rows or test labels."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(binary, dtype=np.int64)
    if x.shape != (len(y), len(FEATURES)) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid ranker training data")
    positive = int(y.sum())
    meta = {"training_flips": len(y), "positive": positive, "negative": len(y)-positive,
            "status": "source_confidence_fallback", "iterations": 0}
    if min(positive, len(y)-positive) < SETTINGS["ranker_min_each_class"]:
        return {**meta, "reason": "insufficient_binary_support"}
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    design = np.column_stack([np.ones(len(y)), (x-mean)/scale])
    w = np.zeros(design.shape[1]); w[0] = np.log(positive/(len(y)-positive))
    penalty = np.diag([0.0] + [SETTINGS["ranker_l2"]]*len(FEATURES))

    def objective(v):
        z = design @ v
        return float(np.sum(np.logaddexp(0, z)-y*z) + .5*v@penalty@v)

    for iteration in range(SETTINGS["ranker_max_iterations"]):
        prob = sigmoid(design @ w)
        grad = design.T @ (prob-y) + penalty @ w
        if np.max(np.abs(grad)) <= SETTINGS["ranker_gradient_tolerance"]:
            return {**meta, "status": "fitted", "reason": None, "iterations": iteration,
                    "mean": mean.tolist(), "scale": scale.tolist(), "weights": w.tolist()}
        hessian = design.T @ (design * (prob*(1-prob))[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        rate, loss = 1.0, objective(w)
        for _ in range(30):
            trial = w-rate*step
            if np.isfinite(trial).all() and objective(trial) <= loss - 1e-4*rate*float(grad@step):
                w = trial
                break
            rate *= .5
        else:
            break
        meta["iterations"] = iteration+1
    return {**meta, "reason": "solver_did_not_converge"}


def rank_scores(signals, model):
    if model["status"] != "fitted":
        return np.asarray(signals[:, 0], dtype=np.float64)
    w = np.asarray(model["weights"])
    x = (np.asarray(signals, dtype=np.float64)-model["mean"])/model["scale"]
    return sigmoid(w[0]+x@w[1:])


def select_queries(badge, previous, old, probe, signals, fit_labels, classes, seed):
    """Acquisition API accepts ONLY the 15%-budget labels reserved for ranker fit."""
    n = len(old)
    prefix, fit_ids, slots = parts(badge)
    pilot.check_ids(badge, n, len(badge))
    for name, value in (("source", old), ("probe", probe)):
        diagnosis.vector(value, n, classes, name)
    diagnosis.vector(fit_labels, len(fit_ids), classes, "ranker labels")
    if signals.shape != (n, len(FEATURES)) or not np.isfinite(signals).all() or np.any((signals < 0) | (signals > 1)):
        raise ValueError("Invalid unlabeled ranker signals")
    if previous["scout_indices"] != list(badge[:len(badge)*4//5]):
        raise ValueError("Saved probe did not use the expected BADGE prefix")
    flipped_fit = old[fit_ids] != probe[fit_ids]
    model = fit_ranker(signals[fit_ids][flipped_fit], (old[fit_ids][flipped_fit] == fit_labels[flipped_fit]).astype(int))
    eligible = np.ones(n, dtype=bool); eligible[prefix] = False
    # One random priority supplies unbiased uniform choices and deterministic ties.
    priority = np.random.default_rng(seed+72021).permutation(np.flatnonzero(eligible))
    pool = priority[old[priority] != probe[priority]]
    choices = {}
    for method in METHODS:
        if method == "random50":
            picked, fallback = priority[:slots], 0
        else:
            ordered = pool
            if method in ("flip_confidence50", "flip_risk50"):
                scores = signals[pool, 0] if method == "flip_confidence50" else rank_scores(signals[pool], model)
                ordered = pool[np.argsort(-scores, kind="stable")]
            picked = ordered[:slots]
            fallback = slots-len(picked)
            if fallback:
                picked = np.r_[picked, priority[~np.isin(priority, picked)][:fallback]]
        ids = np.r_[prefix, np.sort(picked)]
        pilot.check_ids(ids, n, len(badge))
        choices[method] = {"row_indices": ids.tolist(), "fallback_count": fallback}
    union = sorted(set(previous["common_excluded_indices"]) | {i for c in choices.values() for i in c["row_indices"]})
    if len(union) >= n:
        raise ValueError("No common unqueried evaluation samples remain")
    return {"prefix_indices": prefix.tolist(), "fit_indices": fit_ids.tolist(), "supplement_count": slots,
            "flip_pool_count": len(pool), "labels_available_before_selection": len(prefix),
            "choices": choices, "common_excluded_indices": union}, model


@torch.no_grad()
def infer_signals(target, state, expected_probe, device, batch_size):
    """Four bounded features, batched head inference only; API has no label input."""
    n = len(target["features"])
    signals = np.empty((n, len(FEATURES)), dtype=np.float32)
    old = np.empty(n, dtype=np.int64)
    weight, bias = state["weight"].to(device), state["bias"].to(device)
    for start in range(0, n, batch_size):
        end = min(n, start+batch_size)
        a = target["logits"][start:end].to(device)
        b = torch.nn.functional.linear(target["features"][start:end].to(device), weight, bias)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite source/probe logits")
        source_ids = a.argmax(1)
        if not np.array_equal(b.argmax(1).cpu().numpy(), expected_probe[start:end]):
            raise ValueError("Saved probe predictions differ from reconstructed head; preserve original runtime")
        pa, pb = a.softmax(1), b.softmax(1)
        atop, btop = pa.topk(2, dim=1).values, pb.topk(2, dim=1).values
        signals[start:end] = torch.stack((atop[:, 0], atop[:, 0]-atop[:, 1],
            pb.gather(1, source_ids[:, None]).squeeze(1), btop[:, 0]-btop[:, 1]), dim=1).cpu().numpy()
        old[start:end] = source_ids.cpu().numpy()
    return signals, old


def require_selections(args, verified):
    sources, previous, identity = verified
    info, p, saved, _ = sources
    out = Path(args.output_dir)
    pilot.freeze_manifest(out, identity)
    records = {}
    for seed in identity["seeds"]:
        sd = pilot.selection_dir(out, seed)
        if not pilot.completed(sd, pilot.signature(identity, seed, "select"), SELECT_FILES):
            raise ValueError(f"All selections must finish before label access: missing seed {seed}")
        r = pilot.read_json(sd / "selection.json")
        prefix, fit_ids, slots = parts(saved[seed])
        if r["prefix_indices"] != prefix.tolist() or r["fit_indices"] != fit_ids.tolist() or r["supplement_count"] != slots:
            raise ValueError("Shared prefix/ranker fit partition changed")
        if set(r["choices"]) != set(METHODS) or r["labels_available_before_selection"] != len(prefix):
            raise ValueError("Acquisition label budget or methods changed")
        for c in r["choices"].values():
            pilot.check_ids(c["row_indices"], info["sample_counts"]["target"], p["budget"])
            if c["row_indices"][:len(prefix)] != prefix.tolist() or not 0 <= c["fallback_count"] <= slots:
                raise ValueError("Query prefix/fallback changed")
        union = sorted(set(previous[seed]["common_excluded_indices"]) | {i for c in r["choices"].values() for i in c["row_indices"]})
        if union != r["common_excluded_indices"] or len(union) >= info["sample_counts"]["target"]:
            raise ValueError("Common query exclusion changed")
        records[seed] = r
    return records


def verify_selection_semantics(args, verified, seed, record):
    sources, previous, _ = verified
    sd = pilot.selection_dir(Path(args.output_dir), seed)
    expected, model = select_queries(sources[2][seed], previous[seed],
        np.load(sd / "source_predictions.npy", mmap_mode="r"),
        np.load(pilot.selection_dir(Path(args.pilot_dir), seed) / "probe_predictions.npy", mmap_mode="r"),
        np.load(sd / "signals.npy", mmap_mode="r"), np.load(sd / "fit_labels.npy"), sources[1]["num_classes"], seed)
    if expected != record or model != pilot.read_json(sd / "ranker.json"):
        raise ValueError("Saved acquisition or ranker differs from the declared label-limited policy")


def select(args, verified):
    sources, previous, identity = verified
    _, p, saved, _ = sources
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        pilot.freeze_manifest(out, identity, create=True)
        todo = [s for s in identity["seeds"] if not pilot.completed(
            pilot.selection_dir(out, s), pilot.signature(identity, s, "select"), SELECT_FILES)]
        if not todo:
            require_selections(args, verified)
            print("All acquisition choices verified; no ranker or probe refit.", flush=True)
            return
        runtime = identity["runtime"]
        torch.set_num_threads(runtime["threads"])
        device = kbs.device_for(runtime["device"])
        target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True, mmap=True)
        labels = target.pop("labels")  # Legacy bundled labels; only fit_ids may be indexed below.
        for seed in todo:
            sd, old_sd = pilot.selection_dir(out, seed), pilot.selection_dir(Path(args.pilot_dir), seed)
            _, fit_ids, _ = parts(saved[seed])
            y = pilot.purchased_labels(labels, fit_ids, p["num_classes"])
            started = time.perf_counter()
            state = torch.load(old_sd / "probe_head.pt", map_location="cpu", weights_only=True)
            probe = np.load(old_sd / "probe_predictions.npy", mmap_mode="r")
            signals, old = infer_signals(target, state, probe, device, runtime["inference_batch_size"])
            inference_s = time.perf_counter()-started
            started = time.perf_counter()
            record, model = select_queries(saved[seed], previous[seed], old, probe, signals, y, p["num_classes"], seed)
            selection_s = time.perf_counter()-started
            sig = pilot.signature(identity, seed, "select")
            kbs.atomic_json(sd / "resolved_config.json", sig)
            kbs.atomic_json(sd / "selection.json", record)
            kbs.atomic_json(sd / "ranker.json", model)
            np.save(sd / "fit_labels.npy", y)
            np.save(sd / "source_predictions.npy", old)
            np.save(sd / "signals.npy", signals)
            kbs.atomic_json(sd / "selection_trace.json", {"head_signal_inference_s": inference_s,
                "ranker_and_four_selections_s": selection_s, "ranker_label_count": len(y),
                "labels_available_before_selection": len(record["prefix_indices"]),
                "probe_training_reused": True, "oracle_inputs_used": False})
            kbs.finish(sd, sig, SELECT_FILES)
            print(f"Frozen seed {seed}: ranker={model['status']}, positive={model['positive']}, "
                  f"negative={model['negative']}; {len(y)} paid labels outside probe training", flush=True)
        require_selections(args, verified)


def train(args, verified):
    sources, _, identity = verified
    p = sources[1]
    out = Path(args.output_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)
        todo = [(s, m) for s in identity["seeds"] for m in METHODS if not pilot.completed(
            pilot.run_dir(out, s, m), pilot.train_signature(identity, out, s, m), TRAIN_FILES)]
        if not todo:
            print("All final repairs verified; zero new fits.", flush=True)
            return
        for seed in sorted({s for s, _ in todo}):
            verify_selection_semantics(args, verified, seed, records[seed])
        runtime = identity["runtime"]
        torch.set_num_threads(runtime["threads"])
        device = kbs.device_for(runtime["device"])
        head, ref, target, labels = pilot.load_inputs(args)
        for seed, method in todo:
            record = records[seed]
            rd, sd = pilot.run_dir(out, seed, method), pilot.selection_dir(out, seed)
            old_sd = pilot.selection_dir(Path(args.pilot_dir), seed)
            ids = np.asarray(record["choices"][method]["row_indices"], dtype=np.int64)
            y = pilot.purchased_labels(labels, ids, p["num_classes"])
            scout_n, prefix_n = p["budget"]*4//5, len(record["prefix_indices"])
            if not np.array_equal(y[:scout_n], np.load(old_sd / "scout_labels.npy")) or not np.array_equal(
                    y[scout_n:prefix_n], np.load(sd / "fit_labels.npy")):
                raise ValueError("Already-purchased labels changed between selection and training")
            ridx = np.load(old_sd / "reference_ids.npy")
            if not np.array_equal(ridx, kbs.replay_indices(np.asarray(ref["labels"]), p["num_classes"], 5, seed)):
                raise ValueError("Reference sampling changed")
            sig = pilot.train_signature(identity, out, seed, method)
            kbs.atomic_json(rd / "resolved_config.json", sig)
            print(f"Repair {method} seed {seed}: {prefix_n}+{record['supplement_count']} paid labels; fresh source head", flush=True)
            kbs.seed_all(seed)
            repaired, trace = pilot.fit(head, ref, target, ids, y, ridx, p, seed, device)
            baseline = pilot.read_json(pilot.run_dir(Path(args.pilot_dir), seed, "badge") / "trace.json")
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256"):
                if trace[key] != baseline[key]:
                    raise ValueError(f"Original BADGE training stream differs: {key}")
            started = time.perf_counter()
            pred = kbs.predict(repaired, target["features"], device, runtime["inference_batch_size"])
            trace.update({"full_pool_inference_wall_s": time.perf_counter()-started,
                "target_labels": len(ids), "reference_labels": len(ridx), "replay_ce": False,
                "kd_weight": pilot.SETTINGS["kd_weight"], "temperature": pilot.SETTINGS["temperature"],
                "initialization": "source_head"})
            torch.save(repaired.state_dict(), rd / "head.pt")
            np.save(rd / "predictions.npy", pred)
            kbs.write_csv(rd / "query_ids.csv", [{"row_index": int(i), "label": int(v),
                "phase": "probe_scout" if j < scout_n else "ranker_fit" if j < prefix_n else "supplement"}
                for j, (i, v) in enumerate(zip(ids, y))])
            kbs.atomic_json(rd / "trace.json", trace)
            kbs.finish(rd, sig, TRAIN_FILES)


def report(summary, paired, acquisitions, candidates, seeds, budget):
    fields = ["overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_after",
              "noncollapse_negative_flips", "noncollapse_positive_flips",
              "noncollapse_new_collapses", "collapse_residual_count"]
    lines = ["# 已查询标签驱动的保护选样（M12 开发实验）", "",
        f"配对种子 {seeds}；每组目标标签总预算 {budget}，共同前 95% BADGE＋后 5% 补采。",
        "四个新方法从源分类头修复；复用旧预演和 BADGE。未使用 oracle 查询、未查询标签或最终修复预测选样。",
        "原四组查询与新四组查询的并集共同排除；BADGE/static 已重算，不能拼接旧 oracle 表。均值 ± 样本标准差。", "",
        "| 方法 | 整体 F1 | 崩溃类 F1 | 稳定类 F1 | 非崩溃类负翻转 | 非崩溃类正翻转 | 新崩溃 | 原崩溃残留 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| "+r["method"]+" | "+" | ".join(pilot.format_stat(r, f) for f in fields)+" |")
    lines += ["", "配对差值为前者减后者；负翻转与新崩溃越少越好。", "",
        "| 对照 | 整体 F1 差 | 崩溃类 F1 差 | 负翻转差 | 正翻转差 | 新崩溃差 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append("| "+r["comparison"]+" | "+" | ".join(pilot.format_stat(r, f)
            for f in (fields[0], fields[1], fields[3], fields[4], fields[5]))+" |")
    lines += ["", "后 5% 查询的事后组成：", "",
        "| 方法 | 预演误伤样本数 | 预演纠错样本数 | 崩溃类数 | 真实类别覆盖 | 随机回填数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in diagnosis.aggregate(acquisitions, ["method"]):
        lines.append("| "+r["method"]+" | "+" | ".join(pilot.format_stat(r, f) for f in
            ("probe_damage", "probe_correction", "collapse_queries", "classes_covered", "fallback_count"))+" |")
    lines += ["", "风险模型仅使用 BADGE 的 80%–95% 段：它们不参与旧预演训练。模型只在其中发生翻转的样本上拟合。", "",
        "| seed | 已付费拟合标签 | 其中翻转数 | 源正确正例 | 源错误反例 | 风险模型状态 | 候选翻转池预演误伤率（事后） |",
        "|---|---:|---:|---:|---:|---|---:|"]
    for r in candidates:
        fraction = r["pool_probe_damage_fraction"]
        lines.append(f"| {r['seed']} | {r['fit_labels']} | {r['training_flips']} | {r['positive']} | {r['negative']} | "
            f"{r['ranker_status']} ({r['fallback_reason'] or 'no fallback'}) | "+("—" if fraction is None else f"{fraction:.4f}")+" |")
    lines += ["", "优先看 flip_risk50 相对 flip_uniform50 / flip_confidence50 的增量，再看相对 BADGE 的恢复—损伤取舍。",
        "若风险模型回退，flip_risk50 与 flip_confidence50 应完全相同，不能将其视为两个独立有效方法。",
        "打分是 BADGE 选择分布上的小样本逻辑回归排序，不是经独立验证的概率校准；推向全池会存在分布偏差。",
        "翻转池中源正确就意味着预演错误，但预演误伤不等于最终误伤。命中更多保护样本不保证修复收益。",
        "所有补采标签均计入预算并用于最终修复，查询后不筛掉错误样本、不为凑够保护样本追加查询。",
        "源置信度高并不保证正确，置信度组仅为简单对照。随机回填和排序并列值规则均提前固定。",
        "共同评估集排除了全部方法查询；查询组成是训练样本的事后诊断，不能算独立测试结果。",
        "旧 .pt 缓存捆绑加载了标签，但选择进程只索引 15% 拟合标签，训练仅索引冻结查询；完整真值只用于最后独立评价。",
        "新增成本含每种子的旧预演头全池推断和小风险模型拟合；selection_trace.json 记录增量耗时。",
        "旧特征提取、BADGE 选样和预演训练虽复用，部署仍有这些成本；此处不作端到端效率结论。",
        "本轮方案在观察 M12 oracle 结果后提出。三个种子共用源模型与月份，不是独立确认；不自动宣布显著性、创新性或选赢家。", ""]
    return "\n".join(lines)


def evaluate(args, verified):
    sources, previous, identity = verified
    info, p, _, source_identity = sources
    out, root = Path(args.output_dir), Path(args.pilot_dir)
    with kbs.file_lock(out / ".lock"):
        records = require_selections(args, verified)
        hashes = {}
        for seed in identity["seeds"]:
            for method in METHODS:
                rd = pilot.run_dir(out, seed, method)
                if not pilot.completed(rd, pilot.train_signature(identity, out, seed, method), TRAIN_FILES):
                    raise ValueError(f"All new fits must finish before evaluation truth: {seed}/{method}")
                hashes[f"{seed}/{method}"] = kbs.file_sha(rd / "complete.json")
        sig = {"identity_sha256": kbs.digest(identity), "runs": hashes}
        directory = out / "evaluation"
        if pilot.completed(directory, sig, EVAL_FILES):
            print(f"Verified report: {directory / 'report.md'}", flush=True)
            return
        for seed in records:
            verify_selection_semantics(args, verified, seed, records[seed])
        torch.set_num_threads(identity["runtime"]["threads"])
        target = torch.load(Path(args.cache_dir) / "target.pt", map_location="cpu", weights_only=True, mmap=True)
        y, old = np.asarray(target["labels"]), target["logits"].argmax(1).numpy()
        del target
        n, classes = info["sample_counts"]["target"], p["num_classes"]
        diagnosis.vector(y, n, classes, "evaluation truth")
        old_rows = kbs.read_csv(root / "evaluation/by_seed.csv")
        old_metrics = {(int(r["seed"]), r["method"]): r for r in old_rows}
        expected = {(s, m) for s in source_identity["seeds"] for m in ("static", *pilot.METHODS)}
        if len(old_metrics) != len(old_rows) or set(old_metrics) != expected:
            raise ValueError("Original evaluation rows changed")
        rows, pairs, class_rows, acquisitions, candidates = [], [], [], [], []
        for seed, record in records.items():
            print(f"Evaluating seed {seed}: new common query exclusion; recomputing BADGE", flush=True)
            sd, old_sd = pilot.selection_dir(out, seed), pilot.selection_dir(root, seed)
            probe = diagnosis.vector(np.load(old_sd / "probe_predictions.npy"), n, classes, "probe")
            if not np.array_equal(np.load(sd / "source_predictions.npy"), old) or not np.array_equal(
                    np.load(sd / "fit_labels.npy"), y[record["fit_indices"]]):
                raise ValueError("Frozen source predictions/ranker labels changed")
            scout = previous[seed]["scout_indices"]
            counts = pilot.relation_counts(old[scout], y[scout], classes)
            if not np.array_equal(np.load(old_sd / "scout_labels.npy"), y[scout]) or not np.array_equal(
                    np.load(old_sd / "relation_counts.npy"), counts) or previous[seed] != pilot.select_queries(
                        sources[2][seed], old, probe, counts, seed):
                raise ValueError("Original scout/query semantics changed")
            old_mask = np.ones(n, bool); old_mask[previous[seed]["common_excluded_indices"]] = False
            m, _ = pilot.metrics(y, old, old, old_mask, p)
            diagnosis.reconcile(m, old_metrics[seed, "static"])
            for method in pilot.METHODS:
                pred = diagnosis.vector(np.load(pilot.run_dir(root, seed, method) / "predictions.npy", mmap_mode="r"), n, classes, method)
                m, _ = pilot.metrics(y, old, pred, old_mask, p)
                diagnosis.reconcile(m, old_metrics[seed, method])
            mask = np.ones(n, bool); mask[record["common_excluded_indices"]] = False
            static, cr = pilot.metrics(y, old, old, mask, p)
            rows.append({"seed": seed, "method": "static", **static})
            class_rows.extend({"seed": seed, "method": "static", **r} for r in cr)
            metrics, traces = {}, []
            for method in ("badge", *METHODS):
                rd = pilot.run_dir(root if method == "badge" else out, seed, method)
                choice = previous[seed]["choices"][method] if method == "badge" else record["choices"][method]
                ids = np.asarray(choice["row_indices"], dtype=np.int64)
                query = kbs.read_csv(rd / "query_ids.csv")
                if [int(r["row_index"]) for r in query] != ids.tolist() or [int(r["label"]) for r in query] != y[ids].tolist():
                    raise ValueError("Saved training queries/labels changed")
                pred = diagnosis.vector(np.load(rd / "predictions.npy", mmap_mode="r"), n, classes, method)
                m, cr = pilot.metrics(y, old, pred, mask, p)
                metrics[method] = m
                rows.append({"seed": seed, "method": method, **m})
                qcounts = np.bincount(y[ids], minlength=classes)
                class_rows.extend({"seed": seed, "method": method, **r, "query_count": int(qcounts[r["class_id"]])} for r in cr)
                traces.append(pilot.read_json(rd / "trace.json"))
                tail = ids[len(record["prefix_indices"]):]
                acquisitions.append({"seed": seed, "method": method, "total_paid_labels": len(ids),
                    "supplement_labels": len(tail), "probe_damage": int(((old[tail] == y[tail]) & (probe[tail] != y[tail])).sum()),
                    "probe_correction": int(((old[tail] != y[tail]) & (probe[tail] == y[tail])).sum()),
                    "collapse_queries": int(np.isin(y[tail], p["collapse_classes"]).sum()),
                    "classes_covered": len(np.unique(y[tail])), "source_predicted_classes": len(np.unique(old[tail])),
                    "fallback_count": choice["fallback_count"]})
            for key in ("optimizer_steps", "target_stream_sha256", "reference_stream_sha256", "target_labels",
                        "reference_labels", "replay_ce", "kd_weight", "temperature", "initialization"):
                if len({t[key] for t in traces}) != 1:
                    raise ValueError(f"Final training is not paired: {key}")
            comparisons = [(m, "badge") for m in METHODS] + [("flip_risk50", m) for m in METHODS[:-1]]
            comparisons += [("flip_confidence50", "flip_uniform50")]
            for a, b in comparisons:
                pairs.append({"seed": seed, "comparison": f"{a} minus {b}", **{
                    k: metrics[a][k]-metrics[b][k] if metrics[a][k] is not None and metrics[b][k] is not None else None for k in metrics[a]}})
            pool = old != probe; pool[record["prefix_indices"]] = False
            model = pilot.read_json(sd / "ranker.json")
            candidates.append({"seed": seed, "fit_labels": len(record["fit_indices"]),
                **{k: model[k] for k in ("training_flips", "positive", "negative")},
                "ranker_status": model["status"], "fallback_reason": model["reason"],
                "flip_pool_size": int(pool.sum()), "pool_probe_damage_fraction": diagnosis.ratio(int((pool & (old == y)).sum()), int(pool.sum()))})
        summary, pair_summary = diagnosis.aggregate(rows, ["method"]), diagnosis.aggregate(pairs, ["comparison"])
        kbs.atomic_json(directory / "resolved_config.json", sig)
        for name, values in [("by_seed", rows), ("summary", summary), ("paired_by_seed", pairs), ("paired_summary", pair_summary),
                             ("per_class", class_rows), ("acquisition", acquisitions), ("candidate_audit", candidates)]:
            kbs.write_csv(directory / f"{name}.csv", values)
        (directory / "report.md").write_text(report(summary, pair_summary, acquisitions, candidates, identity["seeds"], p["budget"]))
        kbs.finish(directory, sig, EVAL_FILES)
        print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "preflight", "select", "train", "evaluate"])
    for flag, default in [("pilot-dir", "outputs/kbs_repair_aware_v1"), ("study-dir", "outputs/kbs_supplement_v1"),
        ("cache-dir", "outputs/kbs_supplement_v1/cache"), ("checkpoint", "outputs/tls22_cnn/best_model.pt"),
        ("output-dir", "outputs/kbs_learned_protection_v1")]:
        p.add_argument("--"+flag, default=str(kbs.CORE / default))
    p.add_argument("--seeds", default="0,1,2")
    return p


def main():
    args = parser().parse_args()
    if args.command == "plan":
        seeds = kbs.int_list(args.seeds)
        print(json.dumps({"settings": SETTINGS, "seeds": seeds, "new_repair_fits": len(seeds)*4,
                          "small_ranker_attempts": len(seeds), "probe_fits": 0}, indent=2))
        return
    verified = verify_sources(args)
    if args.command == "preflight":
        p, identity = verified[0][1], verified[2]
        kbs.device_for(identity["runtime"]["device"])
        print(json.dumps({"status": "ready", "seeds": identity["seeds"], "new_repair_fits": len(identity["seeds"])*4,
            "paid_labels_per_arm": p["budget"], "prefix": p["budget"]*19//20, "supplement": p["budget"]//20,
            "ranker_labels_outside_probe_training": p["budget"]*3//20, "runtime": identity["runtime"]}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, verified)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
