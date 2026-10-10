#!/usr/bin/env python3
"""CPU repair feasibility pilot on existing CARE caches; not a paper replication.

select, train and evaluate are separate processes. Only evaluate reads unqueried
target truth. The legacy cache bundles labels with features, so this is an API
access contract, not physical/process-level isolation of stored labels.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

import kbs_supplement as kbs

SCHEMA = "small-repair-diagnostic-v1"
# Preserve the tested legacy numerical engine; do not silently use server-local edits.
ENGINE_SHA256 = "8f391a4a6a41a6b10b883649202ec3002d84d7fea6a2a96aa2495250d9d1189c"
ENGINES = ("linear_ce", "linear_care", "residual_mlp_care")
CPU = torch.device("cpu")


class ResidualHead(torch.nn.Module):
    """Starts exactly at the source linear head; both branches can then adapt."""
    def __init__(self, source, hidden, seed):
        super().__init__()
        self.base = copy.deepcopy(source)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 40009)
            self.delta = torch.nn.Sequential(
                torch.nn.Linear(source.in_features, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, source.out_features))
            torch.nn.init.zeros_(self.delta[-1].weight)
            torch.nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x):
        return self.base(x) + self.delta(x)


def numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def select_indices(selector, logits, budget, seed):
    """No labels or evaluation class groups enter this API."""
    n = len(logits)
    if not 0 < budget < n:
        raise ValueError("Query budget must be positive and smaller than the target pool.")
    if selector == "random":
        rows = np.random.default_rng(seed + 50021).choice(n, budget, replace=False)
    elif selector == "margin":
        top = torch.topk(logits, 2, dim=1).values
        rows = torch.argsort(top[:, 0] - top[:, 1], stable=True)[:budget].numpy()
    else:
        raise ValueError(f"Unknown selector: {selector}")
    # Fixed row order avoids a ranking/order confound between selectors.
    return np.sort(rows).astype(np.int64)


def evaluation_indices(n, choices, size, seed):
    mask = np.ones(n, dtype=bool)
    for item in choices:
        mask[np.asarray(item["rows"], dtype=np.int64)] = False
    available = np.flatnonzero(mask)
    if not len(available):
        raise ValueError("The query union leaves no evaluation samples.")
    if size and len(available) > size:
        available = np.random.default_rng(seed).choice(available, size, replace=False)
    return np.sort(available)


def inputs(args):
    if kbs.file_sha(Path(kbs.__file__)) != ENGINE_SHA256:
        raise ValueError("Legacy kbs_supplement.py differs from the tested engine. Compare server edits with repository main.")
    root = Path(args.cache_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.output_dir).resolve()
    if root == out or root.is_relative_to(out) or out.is_relative_to(root) or checkpoint.is_relative_to(out):
        raise ValueError("Output must be separate from the input cache and checkpoint.")
    info = json.loads((root / "manifest.json").read_text())
    if info["spec"]["schema"] != kbs.SCHEMA or info["spec"]["loader_sha256"] != kbs.loader_sha():
        raise ValueError("Cache schema/loader differs from the original CARE engine.")
    if info["periods"] != {"reference": args.reference_period, "target": args.target_period}:
        raise ValueError("Cache periods do not match the requested reference/target months.")
    if info["num_classes"] != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} classes, found {info['num_classes']}.")
    if set(info["files"]) != {"head.pt", "reference.pt", "target.pt"}:
        raise ValueError("Manifest must bind all three input tensors.")
    print("Verifying cache/checkpoint hashes (disk I/O, no GPU)...", flush=True)
    kbs.validate_cache_files(root, info)
    if kbs.file_sha(checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("Source checkpoint does not match the feature cache.")
    classes = info["num_classes"]
    collapse, stable = kbs.int_list(args.collapse_classes), kbs.int_list(args.stable_classes)
    for group in (collapse, stable):
        if not group or len(group) != len(set(group)) or any(c < 0 or c >= classes for c in group):
            raise ValueError("Invalid evaluation class group.")
    if set(collapse) & set(stable):
        raise ValueError("Collapse and stable evaluation groups overlap.")
    identity = {
        "schema": SCHEMA, "cache_fingerprint": info["fingerprint"],
        "cache_manifest_sha256": kbs.file_sha(root / "manifest.json"),
        "checkpoint_sha256": info["checkpoint_sha256"], "periods": info["periods"],
        "code": {p.name: kbs.file_sha(p) for p in
                 (Path(__file__), Path(kbs.__file__), Path(__file__).with_name("run_small_repair_diagnostic.sh"))},
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "seeds": args.seeds, "selectors": args.selectors, "engines": list(ENGINES),
        "budget": args.budget, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "threads": args.threads,
        "replay_per_class": args.replay_per_class, "kd_weight": .5, "temperature": 2.,
        "target_weight": (2 * args.budget) / (2 * args.budget + classes * args.replay_per_class),
        "replay_weight": (classes * args.replay_per_class) / (2 * args.budget + classes * args.replay_per_class),
        "hidden": args.hidden, "expected_classes": classes, "expected_features": args.expected_features,
        "eval_size": args.eval_size, "eval_seed": args.eval_seed,
        "collapse_classes": collapse, "stable_classes": stable,
        "collapse_threshold": .1, "f1_drop_threshold": .05,
        "evaluation": "one label-free uniform sample excluding all selectors/seeds' query union",
    }
    return info, identity


def head_from_cache(args):
    state = torch.load(Path(args.cache_dir) / "head.pt", map_location="cpu", weights_only=True)
    if tuple(state["weight"].shape) != (args.expected_classes, args.expected_features):
        raise ValueError("Source head dimensions do not match expected classes/features.")
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Source head must contain finite float32 parameters.")
    head = torch.nn.Linear(args.expected_features, args.expected_classes)
    head.load_state_dict(state)
    return head.eval()


def load_role(args, info, role, keep_labels=True):
    p = torch.load(Path(args.cache_dir) / f"{role}.pt", map_location="cpu", weights_only=True)
    n = info["sample_counts"][role]
    if (tuple(p["features"].shape) != (n, args.expected_features) or
            tuple(p["logits"].shape) != (n, args.expected_classes) or tuple(p["labels"].shape) != (n,)):
        raise ValueError(f"{role}: tensor dimensions disagree with the manifest.")
    if p["labels"].dtype != torch.int64:
        raise ValueError(f"{role}: expected original int64 labels.")
    if not keep_labels:
        del p["labels"]
    for key in ("features", "logits"):
        if p[key].dtype != torch.float32:
            raise ValueError(f"{role}/{key}: expected original float32 cache.")
        for chunk in p[key].split(65536):
            if not torch.isfinite(chunk).all():
                raise ValueError(f"{role}/{key}: non-finite cache values.")
    return p


def root_manifest(args, info, identity, create=False):
    out = Path(args.output_dir).resolve()
    path = out / "diagnostic_manifest.json"
    if path.exists():
        if json.loads(path.read_text())["protocol"] != identity:
            raise ValueError("Diagnostic settings/code/environment changed; use a new output directory.")
    elif create:
        if out.exists() and any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Unregistered files exist in output; use a new output directory.")
        kbs.atomic_json(path, {"protocol": identity, "source_manifest": info, "environment": kbs.runtime_info()})
    else:
        raise ValueError("Run the select stage before training/evaluation.")
    return out


def selection_signature(identity):
    return {"protocol": identity, "stage": "select"}


def selections(out, identity):
    directory = out / "selection"
    if not kbs.is_complete(directory, selection_signature(identity)):
        raise ValueError("All query selections must be frozen before training/evaluation.")
    choices = json.loads((directory / "queries.json").read_text())
    expected = {(s, q) for s in identity["seeds"] for q in identity["selectors"]}
    if len(choices) != len(expected) or {(c["seed"], c["selector"]) for c in choices} != expected:
        raise ValueError("Frozen query grid differs from the planned grid.")
    for c in choices:
        rows = c["rows"]
        if len(rows) != identity["budget"] or len(set(rows)) != len(rows) or rows != sorted(rows):
            raise ValueError("Invalid frozen query rows.")
    return choices


def select(args, info, identity):
    out = Path(args.output_dir).resolve()
    with kbs.file_lock(out / ".lock"):
        root_manifest(args, info, identity, create=True)
        directory = out / "selection"
        signature = selection_signature(identity)
        if kbs.is_complete(directory, signature):
            print("Verified frozen selections; skipping.", flush=True)
            return
        target = load_role(args, info, "target", keep_labels=False)
        choices = [{"seed": seed, "selector": selector,
                    "rows": select_indices(selector, target["logits"], args.budget, seed).tolist()}
                   for seed in args.seeds for selector in args.selectors]
        rows = evaluation_indices(len(target["logits"]), choices, args.eval_size, args.eval_seed)
        kbs.atomic_json(directory / "resolved_config.json", signature)
        kbs.atomic_json(directory / "queries.json", choices)
        np.save(directory / "evaluation_rows.npy", rows)
        kbs.finish(directory, signature, ["resolved_config.json", "queries.json", "evaluation_rows.npy"])
        print(f"Queries frozen: {len(choices)} sets; common evaluation: {len(rows)} rows.", flush=True)


def run_path(out, choice, engine):
    return out / "runs" / choice["selector"] / f"seed_{choice['seed']}" / engine


def run_signature(identity, choice, engine, out):
    return {"protocol": identity, "seed": choice["seed"], "selector": choice["selector"],
            "engine": engine, "selection_sha256": kbs.file_sha(out / "selection/complete.json")}


def fit_one(args, identity, source, target, reference, choice, ridx, engine):
    idx = np.asarray(choice["rows"], dtype=np.int64)
    # No unqueried target labels are inspected; selected labels are the only y passed to fitting.
    y = numpy(target["labels"][idx])
    ry = numpy(reference["labels"][ridx])
    if any(a.min() < 0 or a.max() >= args.expected_classes for a in (y, ry)):
        raise ValueError("Queried/reference labels lie outside the source class universe.")
    source = ResidualHead(source, args.hidden, choice["seed"]) if engine == "residual_mlp_care" else source
    protected = engine != "linear_ce"
    head, trace = kbs.fit_controlled(
        source, target["features"][idx], y, reference["features"][ridx], ry, reference["logits"][ridx],
        replay_ce=protected, kd_weight=.5 if protected else 0., temperature=2.,
        target_weight=identity["target_weight"], replay_weight=identity["replay_weight"],
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=choice["seed"], device=CPU, log_every=args.log_every)
    with torch.no_grad():
        logits = head(target["features"][idx])
        trace.update({"query_train_accuracy": float((logits.argmax(1).numpy() == y).mean()),
                      "query_train_ce": float(F.cross_entropy(logits, torch.as_tensor(y).long())),
                      "trainable_parameters": sum(p.numel() for p in head.parameters()),
                      "query_class_count": len(np.unique(y)),
                      "queried_class_histogram": np.bincount(y, minlength=args.expected_classes).tolist(),
                      "query_rows": idx.tolist(), "queried_labels": y.tolist(),
                      "reference_rows": ridx.tolist(), "reference_labels": ry.tolist()})
    return head, trace


def train(args, info, identity):
    out = root_manifest(args, info, identity)
    with kbs.file_lock(out / ".lock"):
        choices = selections(out, identity)
        planned = [(c, e) for c in choices for e in ENGINES]
        pending = [(c, e) for c, e in planned if not kbs.is_complete(
            run_path(out, c, e), run_signature(identity, c, e, out))]
        if not pending:
            print("All head fits verified; skipping tensor loading/training.", flush=True)
            return
        source = head_from_cache(args)
        target, reference = load_role(args, info, "target"), load_role(args, info, "reference")
        # Reference labels are historical labels, used for the same balanced replay in each arm.
        ref_labels = numpy(reference["labels"])
        refs = {s: kbs.replay_indices(ref_labels, args.expected_classes, args.replay_per_class, s)
                for s in args.seeds}
        for i, (choice, engine) in enumerate(pending, 1):
            directory = run_path(out, choice, engine)
            signature = run_signature(identity, choice, engine, out)
            kbs.atomic_json(directory / "resolved_config.json", signature)
            print(f"Fit {i}/{len(pending)}: {choice['selector']} seed={choice['seed']} {engine}", flush=True)
            head, trace = fit_one(args, identity, source, target, reference, choice, refs[choice["seed"]], engine)
            torch.save(head.state_dict(), directory / "head.pt")
            kbs.atomic_json(directory / "trace.json", trace)
            kbs.finish(directory, signature, ["resolved_config.json", "head.pt", "trace.json"])
        print("All head fits complete. Offline evaluation can now read target truth.", flush=True)


def summarize(rows, keys):
    groups = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        groups.setdefault(group, []).append(row)
    result = []
    for group, members in groups.items():
        item = dict(zip(keys, group))
        item["n_seeds"] = len(members)
        for key, value in members[0].items():
            if key in keys or key == "seed" or not isinstance(value, (int, float)):
                continue
            values = [r[key] for r in members if r[key] is not None]
            if values:
                item[key + "_mean"] = statistics.mean(values)
                item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(item)
    return result


def paired(rows, args):
    lookup = {(r["seed"], r["selector"], r["engine"]): r for r in rows}
    result = []
    pairs = [(q, "linear_care", q, "linear_ce", "protection") for q in args.selectors]
    pairs += [(q, "residual_mlp_care", q, "linear_care", "capacity") for q in args.selectors]
    if "random" in args.selectors and "margin" in args.selectors:
        pairs += [("margin", e, "random", e, "selection") for e in ENGINES]
    metrics = ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
               "stable_macro_f1_delta", "negative_flip_rate", "positive_flip_rate", "noncollapse_new_collapses")
    for seed in args.seeds:
        for aq, ae, bq, be, question in pairs:
            a, b = lookup[seed, aq, ae], lookup[seed, bq, be]
            item = {"seed": seed, "question": question, "comparison": f"{aq}/{ae} - {bq}/{be}"}
            for key in metrics:
                item[key] = a[key] - b[key] if a[key] is not None and b[key] is not None else None
            result.append(item)
    return result


def report(args, summaries, pairs, source_metrics, support, size):
    lines = ["# CARE 小规模修复诊断", "",
             f"CPU；{len(args.seeds)} 个种子；{len(args.selectors) * len(ENGINES) * len(args.seeds)} 次拟合；"
             f"目标标注预算每组 {args.budget}，历史参考每组 {args.expected_classes * args.replay_per_class}。",
             f"共同评估 {size} 条；所有种子和选样组的查询并集已排除；评估抽样不使用真实标签。", "",
             f"Static Macro-F1={source_metrics['overall_macro_f1_before']:.6f}；"
             f"崩溃组 F1={source_metrics['collapse_macro_f1_before']:.6f}。", "",
             f"评估有支持的类别：{int((support > 0).sum())}/{len(support)}；"
             f"支持少于 20 条的类别：{int((support < 20).sum())}（含零支持）。", "",
             "| 选样 | 修复头 | Macro-F1 | 崩溃组 F1 | 非崩溃类 ΔF1 | 稳定类 ΔF1 | 负翻转率 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def val(r, key):
        v, sd = r.get(key + "_mean"), r.get(key + "_sd")
        return "NA" if v is None else f"{v:.6f}" + (f" ± {sd:.6f}" if sd is not None else "")
    for r in summaries:
        lines.append(f"| {r['selector']} | {r['engine']} | " + " | ".join(val(r, k) for k in
                     ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
                      "stable_macro_f1_delta", "negative_flip_rate")) + " |")
    lines += ["", "## 配对诊断（前者减后者）", "",
              "| 对照 | Macro-F1 Δ | 崩溃组 F1 Δ | 非崩溃类 ΔF1 差 | 负翻转率差 |",
              "|---|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {r['comparison']} | " + " | ".join(val(r, k) for k in
                     ("overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_delta",
                      "negative_flip_rate")) + " |")
    lines += ["", "## 如何解读", "",
              "- random 与 margin 对比只检验这两种选样；没有运行/重算 BADGE。",
              "- linear_care 与 linear_ce 对比检验参考 CE＋KD 这一整套保护的恢复/误伤取舍，不能单独归因于 KD。",
              "- residual_mlp_care 与 linear_care 使用同一查询、参考、优化步数和损失，检验固定特征上的头部容量。"
              "残差头从同一源预测出发，参数量和计算量更大，不是相同计算预算。",
              "- 查 by_seed.csv 的训练准确率/CE、per_class.csv 的支持和逐类损伤；"
              "若残差头仍失败，不能据此认定必须改骨干。还可能是标签覆盖、优化或容量设置不足。",
              "- 负翻转率分母为源模型原本正确的评估样本；正翻转率分母为原本错误的评估样本。",
              "- 全 178 类/指定组 Macro-F1 包含零支持类别的零值，保持原统计定义；"
              "小样本中的低支持召回、新崩溃数不稳定。可用新输出目录和 --eval-size 0 做完整 M12 评估。",
              "- M12 是开发数据。本轮不是独立确认、论文主表复现、创新性证明或固定特征可修复性的上界。"
              "三个种子共享月份/源模型，Margin 查询也相同，只变参考抽样和训练流。",
              "- 哈希校验、选样、拟合和评估耗时见各阶段记录；不包含旧特征提取成本。", ""]
    return "\n".join(lines)


def evaluate(args, info, identity):
    out = root_manifest(args, info, identity)
    with kbs.file_lock(out / ".lock"):
        choices = selections(out, identity)
        planned = [(c, e, run_path(out, c, e)) for c in choices for e in ENGINES]
        # Check every fit before loading any evaluation labels or features.
        for c, e, p in planned:
            if not kbs.is_complete(p, run_signature(identity, c, e, out)):
                raise ValueError(f"Fit incomplete: {p}; finish all training before evaluation.")
        signature = {"protocol": identity, "stage": "evaluate",
                     "runs": {str(p.relative_to(out)): kbs.file_sha(p / "complete.json") for _, _, p in planned},
                     "selection_sha256": kbs.file_sha(out / "selection/complete.json")}
        directory = out / "evaluation"
        if kbs.is_complete(directory, signature):
            print(f"Verified results: {directory / 'report.md'}", flush=True)
            return
        target = load_role(args, info, "target")
        ids = np.load(out / "selection/evaluation_rows.npy", allow_pickle=False)
        expected = evaluation_indices(info["sample_counts"]["target"], choices, args.eval_size, args.eval_seed)
        if not np.array_equal(ids, expected):
            raise ValueError("Evaluation rows differ from the frozen label-free query exclusion.")
        x, y = target["features"][ids], numpy(target["labels"][ids])
        if y.min() < 0 or y.max() >= args.expected_classes:
            raise ValueError("Evaluation labels outside the source class universe.")
        before = target["logits"][ids].argmax(1).numpy()
        del target
        mask = np.ones(len(ids), dtype=bool)
        rows, per_class, predictions = [], [], {"row_indices": ids, "labels": y, "source": before}
        source_metrics = None
        for c, e, path in planned:
            source = head_from_cache(args)
            head = ResidualHead(source, args.hidden, c["seed"]) if e == "residual_mlp_care" else source
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            started = time.perf_counter()
            after = kbs.predict(head, x, CPU, args.inference_batch_size)
            metrics, classes, _ = kbs.compare_predictions(
                y, before, after, mask, args.expected_classes, identity["collapse_classes"],
                identity["stable_classes"], .1, .05)
            source_metrics = metrics
            old_correct = before == y
            negative = int((old_correct & (after != y)).sum())
            positive = int((~old_correct & (after == y)).sum())
            trace = json.loads((path / "trace.json").read_text())
            row = {"seed": c["seed"], "selector": c["selector"], "engine": e, **metrics,
                   "negative_flips": negative, "positive_flips": positive,
                   "negative_flip_rate": negative / int(old_correct.sum()) if old_correct.any() else None,
                   "positive_flip_rate": positive / int((~old_correct).sum()) if (~old_correct).any() else None,
                   "evaluation_wall_s": time.perf_counter() - started}
            for key in ("query_train_accuracy", "query_train_ce", "trainable_parameters", "query_class_count", "train_wall_s"):
                row[key] = trace[key]
            rows.append(row)
            per_class.extend({"seed": c["seed"], "selector": c["selector"], "engine": e,
                              "low_support_lt20": r["support"] < 20, **r} for r in classes)
            predictions[f"{c['selector']}_seed{c['seed']}_{e}"] = after
            print(f"Evaluated {c['selector']} seed={c['seed']} {e}", flush=True)
        summaries, pairs = summarize(rows, ("selector", "engine")), paired(rows, args)
        pair_summary = summarize(pairs, ("question", "comparison"))
        kbs.atomic_json(directory / "resolved_config.json", signature)
        for name, values in (("by_seed.csv", rows), ("summary.csv", summaries), ("per_class.csv", per_class),
                             ("paired_by_seed.csv", pairs), ("paired_summary.csv", pair_summary)):
            kbs.write_csv(directory / name, values)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        (directory / "report.md").write_text(report(args, summaries, pair_summary, source_metrics,
                                                   np.bincount(y, minlength=args.expected_classes), len(ids)))
        kbs.finish(directory, signature, ["resolved_config.json", "by_seed.csv", "summary.csv", "per_class.csv",
                                         "paired_by_seed.csv", "paired_summary.csv", "predictions.npz", "report.md"])
        print(f"SMALL_REPAIR_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "preflight", "select", "train", "evaluate"))
    p.add_argument("--cache-dir", default=str(kbs.CORE / "outputs/kbs_supplement_v1/cache"))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--output-dir", default=str(kbs.CORE / "outputs/small_repair_diagnostic_v1"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--selectors", default="random,margin")
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--steps", type=int, default=1380)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=.0001)
    p.add_argument("--replay-per-class", type=int, default=5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--eval-size", type=int, default=100000, help="0: full query-excluded target pool")
    p.add_argument("--eval-seed", type=int, default=1729)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=460)
    p.add_argument("--expected-classes", type=int, default=178)
    p.add_argument("--expected-features", type=int, default=256)
    p.add_argument("--reference-period", default="M-2022-4")
    p.add_argument("--target-period", default="M-2022-12")
    p.add_argument("--collapse-classes", default=",".join(map(str, kbs.COLLAPSE)))
    p.add_argument("--stable-classes", default=",".join(map(str, kbs.STABLE)))
    return p


def main():
    args = parser().parse_args()
    args.seeds = kbs.int_list(args.seeds)
    args.selectors = args.selectors.split(",")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    if not args.selectors or len(set(args.selectors)) != len(args.selectors) or set(args.selectors) - {"random", "margin"}:
        raise ValueError("Selectors must be distinct choices from random,margin.")
    for key in ("budget", "steps", "batch_size", "replay_per_class", "hidden", "threads",
                "expected_classes", "expected_features", "inference_batch_size"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive.")
    if (not math.isfinite(args.lr) or not math.isfinite(args.weight_decay) or args.lr <= 0 or
            args.weight_decay < 0 or args.eval_size < 0 or args.eval_seed < 0 or args.log_every < 0):
        raise ValueError("Invalid optimizer, evaluation or logging setting.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"fits": len(args.seeds) * len(args.selectors) * len(ENGINES), "device": "cpu",
                          "seeds": args.seeds, "selectors": args.selectors, "engines": ENGINES,
                          "query_budget": args.budget, "steps_per_fit": args.steps, "eval_size": args.eval_size,
                          "output": args.output_dir}, indent=2))
        return
    started = time.perf_counter()
    info, identity = inputs(args)
    if args.command == "preflight":
        out = Path(args.output_dir).resolve()
        if (out / "diagnostic_manifest.json").exists():
            root_manifest(args, info, identity)
        elif out.exists() and any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Unregistered files exist in output; use a new output directory.")
        head_from_cache(args)
        for role in ("reference", "target"):
            payload = load_role(args, info, role, keep_labels=role == "reference")
            print(f"{role}: {tuple(payload['features'].shape)}", flush=True)
            if role == "reference":
                labels = numpy(payload["labels"])
                if labels.min() < 0 or labels.max() >= args.expected_classes:
                    raise ValueError("Historical reference labels outside the source class universe.")
                if (np.bincount(labels, minlength=args.expected_classes) < args.replay_per_class).any():
                    raise ValueError("Insufficient historical reference samples for balanced replay.")
            del payload
        if args.budget >= info["sample_counts"]["target"]:
            raise ValueError("Query budget is not smaller than the target pool.")
        print(json.dumps({"status": "ready", "device": "cpu", "periods": info["periods"],
                          "num_classes": info["num_classes"], "wall_s": time.perf_counter() - started}, indent=2))
    else:
        {"select": select, "train": train, "evaluate": evaluate}[args.command](args, info, identity)
        print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
