#!/usr/bin/env python3
"""CPU mechanism pilot: budgeted target coverage and feasible linear CARE updates.

Selection is label-free; fitting reads only frozen paid target labels. Independent
evaluation reads held-out truth only after every fit/derived control is complete.
This is a mechanism test, not a claim of novelty or target-wide safety.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

import kbs_supplement as kbs
import small_repair_diagnostic as diag
import small_repair_full_audit as full

SCHEMA = "care-target-protection-v1"
SELECTORS = ("margin", "mixed")
ENGINES = ("care", "constrained", "interpolation")


def mixed_selection(features, logits, budget, protection_budget, candidate_cap, selection_seed):
    """Source-prediction strata, upper-half raw margins, then feature coverage.

The candidate cap is applied with a fixed label-free random reservoir per stratum.
One central point per stratum precedes farthest-point additions, in a shuffled
round-robin class order. No target labels or evaluation groups enter this API.
"""
    repair = diag.select_indices("margin", logits, budget - protection_budget, 0)
    available = np.ones(len(logits), dtype=bool)
    available[repair] = False
    predicted = logits.argmax(1).numpy()
    top = torch.topk(logits, 2, dim=1).values
    margin = (top[:, 0] - top[:, 1]).numpy()
    pools, vectors, minimum, selected_positions = {}, {}, {}, {}
    for c in np.unique(predicted[available]):
        rows = np.flatnonzero(available & (predicted == c))
        rows = rows[margin[rows] >= np.median(margin[rows])]
        if len(rows) > candidate_cap:
            rows = np.sort(np.random.default_rng(selection_seed + int(c) * 1009).choice(rows, candidate_cap, replace=False))
        pools[int(c)] = rows
        # Squared distances on unit-normalized cached features.
        x = F.normalize(features[rows].double(), dim=1).numpy()
        vectors[int(c)] = x
        minimum[int(c)] = np.full(len(rows), np.inf)
        selected_positions[int(c)] = []
    order = np.random.default_rng(selection_seed).permutation(sorted(pools)).tolist()
    protection = []
    while len(protection) < protection_budget:
        progressed = False
        for c in order:
            positions, x = selected_positions[c], vectors[c]
            if len(positions) == len(x):
                continue
            if not positions:
                pos = int(np.argmin(((x - x.mean(0)) ** 2).sum(1)))
            else:
                pos = int(np.argmax(minimum[c]))
            positions.append(pos)
            protection.append(int(pools[c][pos]))
            minimum[c] = np.minimum(minimum[c], ((x - x[pos]) ** 2).sum(1))
            minimum[c][positions] = -np.inf
            progressed = True
            if len(protection) == protection_budget:
                break
        if not progressed:
            raise ValueError("High-margin candidate reservoirs cannot fill the protection budget.")
    return {"rows": sorted(repair.tolist() + protection), "repair_rows": repair.tolist(),
            "protection_rows": sorted(protection), "candidate_counts": {str(c): len(v) for c, v in pools.items()},
            "source_predicted_protection_classes": len(np.unique(predicted[protection]))}


class AnchorConstraints:
    """All-rival linear margins on query-confirmed source-correct targets only."""
    def __init__(self, source, x, y, fraction, cap, tolerance):
        x, y = x.double(), torch.as_tensor(y, dtype=torch.long)
        with torch.no_grad():
            logits = F.linear(x, source.weight.double(), source.bias.double())
            gaps_all = logits.gather(1, y[:, None]) - logits
            gaps_all.scatter_(1, y[:, None], float("inf"))
            source_correct = logits.argmax(1) == y
            keep = source_correct & (gaps_all.min(1).values > 4 * tolerance / fraction)
            keep &= source(x.float()).argmax(1) == y
            self.excluded_weak_correct = int((source_correct & ~keep).sum())
            self.positions = torch.nonzero(keep).flatten()
            self.x, self.y = x[keep], y[keep]
            gaps = logits[keep].gather(1, self.y[:, None]) - logits[keep]
            self.required = gaps.clamp(min=0, max=cap) * fraction
            self.mask = torch.ones_like(gaps, dtype=torch.bool)
            if len(self.y):
                self.mask.scatter_(1, self.y[:, None], False)
        self.tolerance = tolerance

    @torch.no_grad()
    def gaps(self, head, state=None):
        weight, bias = (head.weight, head.bias) if state is None else (state["weight"], state["bias"])
        logits = F.linear(self.x, weight.double(), bias.double())
        return logits.gather(1, self.y[:, None]) - logits

    @torch.no_grad()
    def violation(self, head):
        if not len(self.y):
            return 0.
        return float((self.required - self.gaps(head))[self.mask].clamp_min(0).max())

    @torch.no_grad()
    def negative_flips(self, head):
        return int((head(self.x.float()).argmax(1) != self.y).sum()) if len(self.y) else 0

    @torch.no_grad()
    def accept(self, head, previous):
        """Largest feasible step along AdamW's proposed direction, then rounding check.

        This is a segment line search, not an Euclidean projection or an optimal
        constrained solver. AdamW moments retain the unscaled gradient history.
        """
        if not len(self.y):
            return 1., 0, 0.
        old, proposed = self.gaps(head, previous), self.gaps(head)
        goal = self.required - self.tolerance / 2
        crossed = self.mask & (proposed < goal)
        scale = 1.
        if crossed.any():
            scale = float(((old[crossed] - goal[crossed]) / (old[crossed] - proposed[crossed])).clamp(0, 1).min())
            scale *= 1 - 1e-7
        candidate = {k: v.detach().clone() for k, v in head.state_dict().items()}
        backtracks = 0
        while True:
            if scale < 1:
                for key, parameter in head.named_parameters():
                    parameter.copy_(previous[key] + scale * (candidate[key] - previous[key]))
            violation = self.violation(head)
            if violation <= self.tolerance and self.negative_flips(head) == 0:
                return scale, backtracks, violation
            if backtracks >= 20:
                head.load_state_dict(previous)
                if self.violation(head) > self.tolerance or self.negative_flips(head):
                    raise ValueError("Previous accepted head is infeasible; refusing a false safety claim.")
                return 0., backtracks, self.violation(head)
            scale *= .5
            backtracks += 1


def fit_constrained(source, x, y, rx, ry, teacher, protocol, seed, anchors, log_every):
    # Match the legacy CARE sampling streams and objective exactly.
    head = copy.deepcopy(source).train()
    opt = torch.optim.AdamW(head.parameters(), lr=protocol["lr"], weight_decay=protocol["weight_decay"])
    steps, batch = protocol["steps"], protocol["batch_size"]
    target_idx = torch.randint(len(x), (steps, batch), generator=torch.Generator().manual_seed(seed + 20011))
    ref_idx = torch.randint(len(rx), (steps, batch), generator=torch.Generator().manual_seed(seed + 30011))
    y, ry = torch.as_tensor(y).long(), torch.as_tensor(ry).long()
    trace = {"optimizer_steps": steps, "target_presentations": steps * batch,
        "reference_ce_presentations": steps * batch, "reference_kd_presentations": steps * batch,
        "target_stream_sha256": hashlib.sha256(target_idx.numpy().tobytes()).hexdigest(),
        "reference_stream_sha256": hashlib.sha256(ref_idx.numpy().tobytes()).hexdigest()}
    scales, violations, backtracks = [], [], []
    sums = {"target_ce": 0., "reference_ce": 0., "reference_kd": 0., "loss": 0.}
    started = time.perf_counter()
    for step in range(steps):
        tx, r = target_idx[step], ref_idx[step]
        ce = F.cross_entropy(head(x[tx]), y[tx])
        student = head(rx[r])
        rce = F.cross_entropy(student, ry[r])
        t = protocol["temperature"]
        kd = F.kl_div(F.log_softmax(student / t, dim=1), F.softmax(teacher[r] / t, dim=1), reduction="batchmean") * t ** 2
        loss = protocol["target_weight"] * ce + protocol["replay_weight"] * rce + protocol["kd_weight"] * kd
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite CARE objective at step {step + 1}")
        previous = {k: v.detach().clone() for k, v in head.state_dict().items()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError(f"Non-finite proposed parameters at step {step + 1}")
        scale, backtrack, violation = anchors.accept(head, previous)
        scales.append(scale)
        violations.append(violation)
        backtracks.append(backtrack)
        for key, value in (("target_ce", ce), ("reference_ce", rce), ("reference_kd", kd), ("loss", loss)):
            sums[key] += float(value.detach())
        if log_every and ((step + 1) % log_every == 0 or step + 1 == steps):
            print(f"  step {step + 1}/{steps} loss={float(loss.detach()):.6f} accepted_scale={scale:.6f}", flush=True)
    trace.update(train_wall_s=time.perf_counter() - started, mean_losses={k: v / steps for k, v in sums.items()},
        accepted_step_scale_mean=float(np.mean(scales)), scaled_steps=sum(s < 1 for s in scales),
        zero_scale_steps=sum(s <= 1e-12 for s in scales), backtracks=sum(backtracks),
        max_accepted_constraint_violation=max(violations))
    return head.eval(), trace


def inputs(args):
    source, out, base, info, protocol, original, _, source_signature = full.source_inputs(args)
    prior = Path(args.source_audit_dir).resolve()
    if out == prior or out.is_relative_to(prior) or prior.is_relative_to(out):
        raise ValueError("New output must be separate from the completed full audit.")
    if not kbs.is_complete(prior, source_signature):
        raise ValueError("Complete the full-target source audit first.")
    if not 0 < args.protection_budget < protocol["budget"]:
        raise ValueError("Protection budget must be positive and smaller than the fixed total budget.")
    if "margin" not in protocol["selectors"]:
        raise ValueError("Completed Margin CARE baselines are required.")
    identity = {"schema": SCHEMA, "source_signature": source_signature,
        "source_audit_dir": str(prior), "source_audit_complete_sha256": kbs.file_sha(prior / "complete.json"),
        "source_protocol": protocol, "selectors": list(SELECTORS), "engines": list(ENGINES),
        "protection_budget": args.protection_budget, "candidate_cap": args.candidate_cap,
        "selection_seed": args.selection_seed, "margin_fraction": args.margin_fraction,
        "margin_cap": args.margin_cap, "constraint_tolerance": args.constraint_tolerance,
        "eval_size": args.eval_size, "eval_seed": args.eval_seed, "target_samples": info["sample_counts"]["target"],
        "code": {p.name: kbs.file_sha(p) for p in (Path(__file__), Path(__file__).with_name("run_care_target_protection.sh"))}}
    return source, out, base, info, protocol, original, identity


def root_manifest(out, identity, create=False):
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["protocol"] != identity:
            raise ValueError("Experiment code/settings/environment changed; use a new output directory.")
    elif create:
        if out.exists() and any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Unregistered output files; use a new output directory.")
        kbs.atomic_json(path, {"protocol": identity, "environment": kbs.runtime_info()})
    else:
        raise ValueError("Freeze all queries with select before fitting/evaluation.")


def select(args, context):
    _, out, base, info, protocol, original, identity = context
    with kbs.file_lock(out / ".lock"):
        root_manifest(out, identity, True)
        signature = {"protocol": identity, "stage": "select"}
        directory = out / "selection"
        if kbs.is_complete(directory, signature):
            print("Verified frozen queries; skipping.", flush=True)
            return
        target = diag.load_role(base, info, "target", keep_labels=False)
        mix = mixed_selection(target["features"], target["logits"], protocol["budget"], args.protection_budget,
                              args.candidate_cap, args.selection_seed)
        choices = []
        for seed in protocol["seeds"]:
            margin = next(c for c in original if c["seed"] == seed and c["selector"] == "margin")
            expected = diag.select_indices("margin", target["logits"], protocol["budget"], seed).tolist()
            if margin["rows"] != expected or not set(mix["repair_rows"]).issubset(expected):
                raise ValueError("Original Margin queries do not match source logits/ranking.")
            choices.extend([{"seed": seed, "selector": "margin", "rows": margin["rows"],
                             "repair_rows": margin["rows"], "protection_rows": []},
                            {"seed": seed, "selector": "mixed", **mix}])
        ids = diag.evaluation_indices(len(target["features"]), original + choices, args.eval_size, args.eval_seed)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        kbs.atomic_json(directory / "queries.json", choices)
        np.save(directory / "evaluation_rows.npy", ids)
        kbs.finish(directory, signature, ["resolved_config.json", "queries.json", "evaluation_rows.npy"])
        print(f"Queries frozen: {len(choices)} sets, {protocol['budget']} labels each; common evaluation {len(ids):,}.", flush=True)


def selections(out, identity):
    signature = {"protocol": identity, "stage": "select"}
    if not kbs.is_complete(out / "selection", signature):
        raise ValueError("All target queries must be frozen before training.")
    choices = json.loads((out / "selection/queries.json").read_text())
    p = identity["source_protocol"]
    expected = {(seed, q) for seed in p["seeds"] for q in SELECTORS}
    if len(choices) != len(expected) or {(c["seed"], c["selector"]) for c in choices} != expected:
        raise ValueError("Frozen query grid mismatch.")
    for c in choices:
        rows = c["rows"]
        if (len(rows) != p["budget"] or rows != sorted(set(rows)) or rows != sorted(c["repair_rows"] + c["protection_rows"]) or
                any(type(i) is not int or i < 0 or i >= identity["target_samples"] for i in rows)):
            raise ValueError("Invalid frozen repair/protection query partition.")
        if c["selector"] == "mixed" and len(c["protection_rows"]) != identity["protection_budget"]:
            raise ValueError("Fixed protection budget mismatch.")
    return choices


def run_path(out, choice, engine):
    return out / "runs" / choice["selector"] / f"seed_{choice['seed']}" / engine


def run_signature(out, choice, engine, identity):
    return {"protocol": identity, "seed": choice["seed"], "selector": choice["selector"], "engine": engine,
            "selection_sha256": kbs.file_sha(out / "selection/complete.json")}


def annotate(head, trace, source_head, target, reference, choice, ridx, args, protocol):
    idx = np.asarray(choice["rows"], dtype=np.int64)
    x, y = target["features"][idx], diag.numpy(target["labels"][idx])
    anchors = AnchorConstraints(source_head, x, y, args.margin_fraction, args.margin_cap, args.constraint_tolerance)
    protection_positions = np.flatnonzero(np.isin(idx, choice["protection_rows"]))
    with torch.no_grad():
        logits = head(x)
        trace.update(query_train_accuracy=float((logits.argmax(1).numpy() == y).mean()),
            query_train_ce=float(F.cross_entropy(logits, torch.as_tensor(y).long())),
            trainable_parameters=sum(p.numel() for p in head.parameters()), query_class_count=len(np.unique(y)),
            queried_class_histogram=np.bincount(y, minlength=protocol["expected_classes"]).tolist(),
            query_rows=idx.tolist(), queried_labels=y.tolist(), reference_rows=ridx.tolist(),
            reference_labels=diag.numpy(reference["labels"][ridx]).tolist(),
            anchor_rows=idx[anchors.positions.numpy()].tolist(),
            anchor_count=len(anchors.y), anchor_class_count=len(torch.unique(anchors.y)),
            excluded_weak_correct=anchors.excluded_weak_correct,
            anchor_class_histogram=np.bincount(anchors.y.numpy(), minlength=protocol["expected_classes"]).tolist(),
            source_correct_protection_count=int((source_head(x[protection_positions]).argmax(1).numpy() == y[protection_positions]).sum())
                if len(protection_positions) else 0,
            final_anchor_constraint_violation=anchors.violation(head), anchor_negative_flips=anchors.negative_flips(head))
    return trace


def train(args, context):
    source, out, base, info, protocol, original, identity = context
    root_manifest(out, identity)
    with kbs.file_lock(out / ".lock"):
        choices = selections(out, identity)
        planned = [(c, e) for c in choices for e in ENGINES]
        if all(kbs.is_complete(run_path(out, c, e), run_signature(out, c, e, identity)) for c, e in planned):
            print("All fits and interpolation controls verified; skipping tensors/training.", flush=True)
            return
        head0 = diag.head_from_cache(base)
        target, reference = diag.load_role(base, info, "target"), diag.load_role(base, info, "reference")
        refs = {s: kbs.replay_indices(diag.numpy(reference["labels"]), protocol["expected_classes"], protocol["replay_per_class"], s)
                for s in protocol["seeds"]}
        for choice in choices:
            idx, ridx = np.asarray(choice["rows"], dtype=np.int64), refs[choice["seed"]]
            original_choice = next(c for c in original if c["seed"] == choice["seed"] and c["selector"] == "margin")
            baseline_path = diag.run_path(source, original_choice, "linear_care")
            old_trace = json.loads((baseline_path / "trace.json").read_text())
            if old_trace["reference_rows"] != ridx.tolist():
                raise ValueError("New replay differs from original matched-seed replay.")
            y = diag.numpy(target["labels"][idx])  # only frozen paid labels are indexed
            if y.min() < 0 or y.max() >= protocol["expected_classes"]:
                raise ValueError("Queried truth outside source classes.")
            x, rx = target["features"][idx], reference["features"][ridx]
            ry, teacher = diag.numpy(reference["labels"][ridx]), reference["logits"][ridx]
            for engine in ENGINES:
                directory, signature = run_path(out, choice, engine), run_signature(out, choice, engine, identity)
                if kbs.is_complete(directory, signature):
                    continue
                print(f"Prepare {choice['selector']} seed={choice['seed']} {engine}", flush=True)
                kbs.atomic_json(directory / "resolved_config.json", signature)
                anchors = AnchorConstraints(head0, x, y, args.margin_fraction, args.margin_cap, args.constraint_tolerance)
                if engine == "care" and choice["selector"] == "margin":
                    head = copy.deepcopy(head0)
                    head.load_state_dict(torch.load(baseline_path / "head.pt", map_location="cpu", weights_only=True))
                    trace = {**old_trace, "reused_original_fit": True, "new_optimizer_steps": 0}
                    if trace["query_rows"] != choice["rows"] or trace["queried_labels"] != y.tolist():
                        raise ValueError("Reused Margin baseline query/label mismatch.")
                elif engine == "care":
                    head, trace = diag.fit_one(base, protocol, head0, target, reference, choice, ridx, "linear_care")
                    trace.update(reused_original_fit=False, new_optimizer_steps=protocol["steps"])
                elif engine == "constrained":
                    head, trace = fit_constrained(head0, x, y, rx, ry, teacher, protocol, choice["seed"], anchors, args.log_every)
                    trace.update(reused_original_fit=False, new_optimizer_steps=protocol["steps"])
                else:
                    derive_started = time.perf_counter()
                    parent = run_path(out, choice, "care")
                    if not kbs.is_complete(parent, run_signature(out, choice, "care", identity)):
                        raise ValueError("Fit the corresponding CARE parent before deriving interpolation.")
                    head = copy.deepcopy(head0)
                    head.load_state_dict(torch.load(parent / "head.pt", map_location="cpu", weights_only=True))
                    scale, backtracks, violation = anchors.accept(head, head0.state_dict())
                    trace = {**json.loads((parent / "trace.json").read_text()), "new_optimizer_steps": 0,
                        "derived_control": True, "parent_complete_sha256": kbs.file_sha(parent / "complete.json"),
                        "interpolation_scale": scale, "interpolation_backtracks": backtracks,
                        "derivation_wall_s": time.perf_counter() - derive_started,
                        "final_anchor_constraint_violation": violation}
                    # Parent step statistics do not describe this one-shot control.
                    for key in ("accepted_step_scale_mean", "scaled_steps", "zero_scale_steps", "max_accepted_constraint_violation"):
                        trace.pop(key, None)
                trace = annotate(head, trace, head0, target, reference, choice, ridx, args, protocol)
                if engine != "care" and (trace["final_anchor_constraint_violation"] > args.constraint_tolerance or trace["anchor_negative_flips"]):
                    raise ValueError("A protected result is infeasible on confirmed target anchors.")
                if any(not torch.isfinite(p).all() for p in head.parameters()):
                    raise ValueError("Non-finite fitted head.")
                torch.save(head.state_dict(), directory / "head.pt")
                kbs.atomic_json(directory / "trace.json", trace)
                kbs.finish(directory, signature, ["resolved_config.json", "head.pt", "trace.json"])
        print("All fits complete; independent offline evaluation may now read held-out target truth.", flush=True)


def paired(rows, protocol):
    lookup = {(r["seed"], r["selector"], r["engine"]): r for r in rows}
    comparisons = [("mixed", e, "margin", e, "selection") for e in ("care", "constrained")]
    comparisons += [(q, "constrained", q, "care", "constraint") for q in SELECTORS]
    comparisons += [(q, "constrained", q, "interpolation", "step_vs_final_interpolation") for q in SELECTORS]
    result = []
    for seed in protocol["seeds"]:
        for aq, ae, bq, be, question in comparisons:
            a, b = lookup[seed, aq, ae], lookup[seed, bq, be]
            item = {"seed": seed, "question": question, "comparison": f"{aq}/{ae} - {bq}/{be}"}
            for key in ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
                        "stable_macro_f1_delta", "negative_flip_rate", "positive_flip_rate"):
                item[key] = a[key] - b[key] if a[key] is not None and b[key] is not None else None
            result.append(item)
    return result


def report(protocol, args, summaries, pairs, coverage, size, static):
    def fmt(r, key):
        mean, sd = r.get(key + "_mean"), r.get(key + "_sd")
        return "NA" if mean is None else f"{mean:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    lines = ["# CARE 目标保护机制试验", "",
        f"CPU；目标预算固定 {protocol['budget']}；mixed 分配修复 {protocol['budget'] - args.protection_budget}＋保护候选 {args.protection_budget}。",
        f"原种子 {protocol['seeds']}；四格主对照＋两个插值对照；复用原 Margin CARE，新增 {3 * len(protocol['seeds'])} 次拟合。",
        f"共同评估 {size:,} 条，排除全部旧查询与新查询并集；Static Macro-F1={static:.6f}。", "",
        "| 查询 | 修复 | Macro-F1 | 崩溃组 F1 | 稳定组 ΔF1 | 负翻转率 | 锚点数量 | 覆盖类别 | 锚点负翻转 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['selector']} | {r['engine']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "stable_macro_f1_delta", "negative_flip_rate",
             "anchor_count", "anchor_class_count", "anchor_negative_flips")) + " |")
    lines += ["", "## 配对比较（前者减后者）", "",
        "| 对照 | Macro-F1 Δ | 崩溃组 F1 Δ | 非崩溃类 ΔF1 差 | 负翻转率差 |",
        "|---|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['comparison']} | " + " | ".join(fmt(r, k) for k in
            ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 约束执行诊断", "",
        "| 查询/修复 | 平均接受步长比例 | 近零步数 | 最终最大间隔违反 | 最终插值比例 |",
        "|---|---:|---:|---:|---:|"]
    for r in summaries:
        if r["engine"] != "care":
            lines.append(f"| {r['selector']}/{r['engine']} | " + " | ".join(fmt(r, k) for k in
                ("accepted_step_scale_mean", "zero_scale_steps", "final_anchor_constraint_violation", "interpolation_scale")) + " |")
    lines += ["", "## 原先主要受损类别的锚点覆盖（仅事后展示）", "",
        "| 查询 | 类别 ID | 已查询数量 | 确认源正确且可约束数量 |",
        "|---|---:|---:|---:|"]
    # IDs do not enter selection or fitting; this table is a development diagnostic.
    for r in coverage:
        if r["class_id"] in (120, 94, 15, 53, 110, 32, 73):
            lines.append(f"| {r['selector']} | {r['class_id']} | {fmt(r, 'queried_count')} | {fmt(r, 'anchor_count')} |")
    lines += ["", "## 判读", "",
        "- care 保留原目标 CE＋历史参考 CE＋KD；constrained 只改变每次 AdamW 提议后的接受步长。",
        "- 对已查询且源正确、间隔足够的锚点，所有 rival 的间隔须不低于源间隔截断值的 "
        f"{args.margin_fraction} 倍（截断上限 {args.margin_cap}，数值容差 {args.constraint_tolerance}）。",
        "- 几何检查使用 float64，实际参数仍为 float32；每次接受同时检查锚点 float32 预测，失败回退。",
        "- 锚点约束不保证其他目标样本安全。看完整排除集的恢复/正负翻转、逐类精度和召回。",
        "- 插值对照只用同一付费锚点确定最大可行旧→CARE 步长，没有使用评估真值挑比例。",
        "- 若约束仅减少更新且恢复也下降，或不优于简单插值，不能当作机制成功。",
        "- 这是可行线段搜索，未声称解出最优约束问题；AdamW 动量不按接受比例缩放。",
        "- 更新步数和基础损失相同，额外约束检查增加计算量。日志/trace 记录实际时间和接受比例。",
        "- 所有种子共享月份/源模型，查询集合在种子间相同；M12 是开发数据，不证明跨月份效果或新颖性。", ""]
    return "\n".join(lines)


def evaluate(args, context):
    _, out, base, info, protocol, original, identity = context
    root_manifest(out, identity)
    with kbs.file_lock(out / ".lock"):
        choices = selections(out, identity)
        planned = [(c, e, run_path(out, c, e)) for c in choices for e in ENGINES]
        for c, e, path in planned:
            if not kbs.is_complete(path, run_signature(out, c, e, identity)):
                raise ValueError(f"Fit/control incomplete: {path}; finish every fit before evaluation.")
        signature = {"protocol": identity, "stage": "evaluate", "selection_sha256": kbs.file_sha(out / "selection/complete.json"),
            "runs": {str(p.relative_to(out)): kbs.file_sha(p / "complete.json") for _, _, p in planned}}
        directory = out / "evaluation"
        if kbs.is_complete(directory, signature):
            print(f"CARE_TARGET_PROTECTION_COMPLETE: {directory / 'report.md'} (verified; skipped)", flush=True)
            return
        target = diag.load_role(base, info, "target")
        ids = np.load(out / "selection/evaluation_rows.npy", allow_pickle=False)
        expected = diag.evaluation_indices(info["sample_counts"]["target"], original + choices, args.eval_size, args.eval_seed)
        if not np.array_equal(ids, expected):
            raise ValueError("Evaluation rows differ from all old/new query exclusion.")
        y, before = diag.numpy(target["labels"][ids]), target["logits"][ids].argmax(1).numpy()
        if y.min() < 0 or y.max() >= protocol["expected_classes"]:
            raise ValueError("Evaluation truth outside source classes.")
        predictions = {"row_indices": ids, "labels": y, "source": before}
        rows, classes, coverage, traces = [], [], [], {}
        for i, (c, e, path) in enumerate(planned, 1):
            print(f"Evaluate {i}/{len(planned)}: {c['selector']} seed={c['seed']} {e}", flush=True)
            head = diag.head_from_cache(base)
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise ValueError("Non-finite completed head.")
            after = full.predict_rows(head, target["features"], ids, args.inference_batch_size)
            trace = json.loads((path / "trace.json").read_text())
            if trace["query_rows"] != c["rows"] or trace["queried_labels"] != diag.numpy(target["labels"][c["rows"]]).tolist():
                raise ValueError("Run query rows/truth disagree with frozen selection.")
            anchors = AnchorConstraints(diag.head_from_cache(base), target["features"][c["rows"]],
                diag.numpy(target["labels"][c["rows"]]), args.margin_fraction, args.margin_cap, args.constraint_tolerance)
            expected_anchor_rows = np.asarray(c["rows"])[anchors.positions.numpy()].tolist()
            if (trace["anchor_rows"] != expected_anchor_rows or trace["anchor_count"] != len(anchors.y) or
                    trace["anchor_class_count"] != len(torch.unique(anchors.y)) or
                    trace["anchor_class_histogram"] != np.bincount(anchors.y.numpy(), minlength=protocol["expected_classes"]).tolist() or
                    trace["anchor_negative_flips"] != anchors.negative_flips(head) or
                    not np.isclose(trace["final_anchor_constraint_violation"], anchors.violation(head), atol=1e-12, rtol=1e-10)):
                raise ValueError("Anchor trace disagrees with saved head/paid queries.")
            if e == "interpolation" and trace["parent_complete_sha256"] != kbs.file_sha(run_path(out, c, "care") / "complete.json"):
                raise ValueError("Interpolation parent completion changed.")
            if e != "care" and (trace["final_anchor_constraint_violation"] > args.constraint_tolerance or trace["anchor_negative_flips"]):
                raise ValueError("Protected run trace reports infeasibility.")
            meta = {"seed": c["seed"], "selector": c["selector"], "engine": e}
            metrics, per_class = full.scores(y, before, after, protocol, trace)
            extra = {k: trace.get(k) for k in ("anchor_count", "anchor_class_count", "anchor_negative_flips",
                "accepted_step_scale_mean", "zero_scale_steps", "scaled_steps", "final_anchor_constraint_violation", "interpolation_scale")}
            rows.append({**meta, **metrics, **extra})
            classes.extend({**meta, **r} for r in full.class_flips(y, before, after, per_class, protocol["expected_classes"]))
            predictions[f"{c['selector']}_seed{c['seed']}_{e}"] = after
            traces[c["seed"], c["selector"], e] = trace
            if e == "care":
                for cid, (qcount, acount) in enumerate(zip(trace["queried_class_histogram"], trace["anchor_class_histogram"])):
                    coverage.append({"seed": c["seed"], "selector": c["selector"], "class_id": cid,
                                     "queried_count": qcount, "anchor_count": acount})
        for c in choices:
            members = [traces[c["seed"], c["selector"], e] for e in ENGINES]
            for key in ("target_stream_sha256", "reference_stream_sha256", "reference_rows", "reference_labels", "anchor_rows"):
                if any(m[key] != members[0][key] for m in members[1:]):
                    raise ValueError(f"Matched run streams/reference/anchors differ: {key}")
        summaries = full.summarize(rows, ("selector", "engine"))
        pairs = paired(rows, protocol)
        pair_summary = full.summarize(pairs, ("question", "comparison"))
        coverage_summary = full.summarize(coverage, ("selector", "class_id"))
        kbs.atomic_json(directory / "resolved_config.json", signature)
        artifacts = ["resolved_config.json", "predictions.npz", "report.md"]
        for name, values in (("by_seed.csv", rows), ("summary.csv", summaries), ("per_class.csv", classes),
                ("paired_by_seed.csv", pairs), ("paired_summary.csv", pair_summary),
                ("query_coverage.csv", coverage), ("query_coverage_summary.csv", coverage_summary)):
            kbs.write_csv(directory / name, values)
            artifacts.append(name)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        (directory / "report.md").write_text(report(protocol, args, summaries, pair_summary, coverage_summary,
            len(ids), rows[0]["overall_macro_f1_before"]))
        kbs.finish(directory, signature, artifacts)
        print(f"CARE_TARGET_PROTECTION_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "preflight", "select", "train", "evaluate"))
    p.add_argument("--source-dir", default=str(kbs.CORE / "outputs/small_repair_diagnostic_v1"))
    p.add_argument("--source-audit-dir", default=str(kbs.CORE / "outputs/small_repair_full_audit_v1"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/care_target_protection_v1"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--protection-budget", type=int, default=200)
    p.add_argument("--candidate-cap", type=int, default=256)
    p.add_argument("--selection-seed", type=int, default=6173)
    p.add_argument("--margin-fraction", type=float, default=.25)
    p.add_argument("--margin-cap", type=float, default=1.)
    p.add_argument("--constraint-tolerance", type=float, default=1e-6)
    p.add_argument("--eval-size", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=1729)
    p.add_argument("--log-every", type=int, default=460)
    return p


def main():
    args = parser().parse_args()
    for key in ("threads", "inference_batch_size", "candidate_cap", "protection_budget"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    if (any(not math.isfinite(v) for v in (args.margin_fraction, args.margin_cap, args.constraint_tolerance)) or
            not 0 < args.margin_fraction <= 1 or args.margin_cap <= 0 or args.constraint_tolerance <= 0 or
            args.margin_fraction * args.margin_cap <= 4 * args.constraint_tolerance or
            min(args.eval_size, args.eval_seed, args.selection_seed, args.log_every) < 0):
        raise ValueError("Invalid constraint/evaluation settings.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"device": "cpu", "selectors": SELECTORS, "engines": ENGINES,
            "protection_budget": args.protection_budget, "new_fits_per_seed": 3,
            "margin_fraction": args.margin_fraction, "margin_cap": args.margin_cap, "eval_size": args.eval_size}, indent=2))
        return
    started = time.perf_counter()
    context = inputs(args)
    if args.command == "preflight":
        out = context[1]
        if (out / "manifest.json").exists():
            root_manifest(out, context[-1])
        elif out.exists() and any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Unregistered output files; use a new output directory.")
        print(json.dumps({"status": "ready", "device": "cpu", "seeds": context[4]["seeds"],
            "total_target_budget": context[4]["budget"], "new_fits": 3 * len(context[4]["seeds"])}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, context)
    print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
