#!/usr/bin/env python3
"""Query-confirmed absorption relations: freeze first, audit saved repairs second.

CPU/NumPy only. No feature extraction, model fitting or new labels. Evaluation
labels are never arguments to relation construction or prediction gating.
"""
import argparse
from collections import defaultdict
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

import kbs_class_audit as audit

CORE = Path(__file__).resolve().parents[1]
METHODS = ("badge_full", "badge_kd")
RUN_FILES = ("resolved_config.json", "predictions.npz", "query_ids.csv", "replay_ids.csv",
             "training_trace.json", "metrics.json", "per_class_metrics.csv")
REPORT_FILES = ("resolved_config.json", "by_seed.csv", "summary.csv", "paired_by_seed.csv",
                "paired_summary.csv", "relation_by_seed.csv", "relation_summary.csv", "relation_per_class.csv", "per_class.csv",
                "transitions.csv", "report.md")
SCHEMA = "care-query-relation-audit-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    audit.write_text(Path(path), json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def finish(directory, signature, files):
    write_json(directory / "complete.json", {"signature": signature,
               "artifacts": {name: audit.sha(directory / name) for name in files}})


def completed(directory, signature, required, *, mandatory=False):
    if not (directory / "complete.json").is_file():
        if mandatory:
            raise ValueError(f"Missing completed input: {directory}")
        if (directory / "resolved_config.json").exists() and read_json(directory / "resolved_config.json") != signature:
            raise ValueError(f"Changed partial output; use a new output directory: {directory}")
        return False
    done = read_json(directory / "complete.json")
    if done["signature"] != signature or not set(required).issubset(done["artifacts"]):
        raise ValueError(f"Changed signature or incomplete artifact list: {directory}")
    for name, expected in done["artifacts"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid artifact path")
        path = directory / name
        if not path.is_file() or audit.sha(path) != expected:
            raise ValueError(f"Missing/modified artifact: {path}")
    if read_json(directory / "resolved_config.json") != signature:
        raise ValueError(f"Resolved signature mismatch: {directory}")
    return True


@contextlib.contextmanager
def locked(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another audit owns {out}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def integer_vector(value, name, upper=None):
    a = np.asarray(value)
    if a.ndim == 1 and a.size == 0:
        return a.astype(np.int64)
    if a.ndim != 1 or a.dtype.kind not in "iu" or (a < 0).any():
        raise ValueError(f"Invalid integer vector: {name}")
    if upper is not None and (a >= upper).any():
        raise ValueError(f"Out-of-range {name}")
    return a.astype(np.int64, copy=False)


def query_records(root, protocol, seed, selector):
    directory = root / "selections" / selector / f"seed_{seed}"
    sig = {"cache": protocol["cache"], "implementation": protocol["implementation_sha256"],
           "selector": selector, "budget": protocol["budget"], "seed": seed}
    completed(directory, sig, ("resolved_config.json", "selection.json"), mandatory=True)
    ids = integer_vector(read_json(directory / "selection.json")["row_indices"], "query IDs")
    if len(ids) != protocol["budget"] or len(np.unique(ids)) != len(ids):
        raise ValueError("Query budget/uniqueness mismatch")
    return ids


def verify_inputs(args):
    root, out = Path(args.study_dir).resolve(), Path(args.output_dir).resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Output must be separate from the completed study")
    seeds = [int(v) for v in args.seeds.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Supply distinct nonnegative seeds")
    manifest = read_json(root / "study_manifest.json")
    p = manifest["protocol"]
    engine = digest({str(path.relative_to(CORE)): audit.sha(path) for path in
                     [CORE / "scripts/kbs_supplement.py", CORE / "scripts/collapse_active_maintenance_tls22.py"]})
    if p["implementation_sha256"] != engine:
        raise ValueError("The original numerical implementation differs from the study")
    if p["num_classes"] < 2 or p["budget"] <= 0:
        raise ValueError("Invalid study class universe/budget")
    for group in ("collapse_classes", "stable_classes"):
        ids = integer_vector(p[group], group, p["num_classes"])
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicated evaluation class")
    if not p["collapse_classes"] or not 0 < p["collapse_recall_threshold"] <= 1:
        raise ValueError("Invalid collapse evaluation definition")
    extension = read_json(root / "badge_kd_extension_manifest.json")
    if extension["identity"] != {"followup_sha256": audit.sha(CORE / "scripts/kbs_badge_kd_followup.py"),
                                 "implementation_sha256": engine,
                                 "specs": [audit.expected_spec(m) for m in METHODS]}:
        raise ValueError("BADGE extension identity changed")
    hashes = {n: audit.sha(root / n) for n in ("study_manifest.json", "badge_kd_extension_manifest.json")}
    for seed in seeds:
        choices = {s: query_records(root, p, seed, s) for s in ("margin", "badge")}
        for selector in choices:
            directory = root / "selections" / selector / f"seed_{seed}"
            hashes[str((directory / "complete.json").relative_to(root))] = audit.sha(directory / "complete.json")
        saved, refs, traces = [], [], []
        for method in METHODS:
            directory = root / "runs" / method / f"seed_{seed}"
            sig = {"protocol": p, "spec": audit.expected_spec(method), "seed": seed}
            completed(directory, sig, RUN_FILES, mandatory=True)
            hashes[str((directory / "complete.json").relative_to(root))] = audit.sha(directory / "complete.json")
            q = audit.read_csv(directory / "query_ids.csv")
            if [int(r["row_index"]) for r in q] != choices["badge"].tolist():
                raise ValueError(f"BADGE query order mismatch: {directory}")
            for r in q:
                if r["sample_id"] != f"{p['cache']}:target:{r['row_index']}" or not 0 <= int(r["label"]) < p["num_classes"]:
                    raise ValueError("Query label/snapshot identity mismatch")
            saved.append(q)
            refs.append([tuple(r[k] for k in ("sample_id", "row_index", "label"))
                         for r in audit.read_csv(directory / "replay_ids.csv")])
            traces.append(read_json(directory / "training_trace.json"))
        if saved[0] != saved[1] or refs[0] != refs[1]:
            raise ValueError(f"Unpaired BADGE query/reference labels: seed {seed}")
        for key in ("optimizer_steps", "target_presentations", "reference_kd_presentations",
                    "target_stream_sha256", "reference_stream_sha256"):
            if traces[0][key] != traces[1][key]:
                raise ValueError(f"Unpaired training stream {key}: seed {seed}")
        if traces[1]["reference_ce_presentations"] != 0:
            raise ValueError("BADGE KD unexpectedly used reference CE")
    identity = {"schema": SCHEMA, "seeds": seeds, "study_protocol": p,
                "source_sha256": hashes, "script_sha256": audit.sha(Path(__file__)),
                "helper_sha256": audit.sha(Path(audit.__file__)), "numpy_version": np.__version__,
                "split": "common", "relation_min_confirmed_errors": 1,
                "control": "row-outdegree-matched uniform destinations without replacement; seed+9187",
                "status": "development diagnostic and fixed postprocessing; no training/new queries"}
    return root, out, identity


def build_relations(query_old, query_y, classes, seed):
    """Only paid query labels; rows=old predicted absorber, columns=true victim.

    An observed v->a failure allows prediction a->v. Diagonal keeps the old
    prediction. No collapse groups, complete target labels or repaired outputs.
    """
    old = integer_vector(query_old, "query predictions", classes)
    y = integer_vector(query_y, "query labels", classes)
    if len(old) != len(y):
        raise ValueError("Query prediction/label size mismatch")
    counts = np.bincount(old * classes + y, minlength=classes * classes).reshape(classes, classes)
    allowed = (counts > 0) | np.eye(classes, dtype=bool)
    control = np.eye(classes, dtype=bool)
    rng = np.random.default_rng(seed + 9187)
    for a in range(classes):
        candidates = np.delete(np.arange(classes), a)
        control[a, rng.choice(candidates, int(allowed[a].sum()) - 1, replace=False)] = True
    return counts, allowed, control


def graph_payload(old, y, classes, seed, row_ids):
    counts, allowed, control = build_relations(old, y, classes, seed)
    return {"seed": seed, "query_row_indices": list(map(int, row_ids)),
            "query_labels": list(map(int, y)), "query_static_predictions": list(map(int, old)),
            "confirmed_counts": counts.tolist(),
            "allowed": [np.flatnonzero(row).tolist() for row in allowed],
            "degree_control_allowed": [np.flatnonzero(row).tolist() for row in control]}


def freeze(args):
    root, out, identity = verify_inputs(args)
    directory = out / "frozen"
    files = ["resolved_config.json"] + [f"seed_{s}.json" for s in identity["seeds"]]
    with locked(out):
        if completed(directory, identity, files):
            print(f"Verified frozen relations: {directory}", flush=True)
            return
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "resolved_config.json", identity)
        for seed in identity["seeds"]:
            run = root / "runs/badge_full" / f"seed_{seed}"
            q = audit.read_csv(run / "query_ids.csv")
            ids = np.array([int(r["row_index"]) for r in q])
            # npz members are lazy: DO NOT read y_true or repaired_pred here.
            with np.load(run / "predictions.npz", allow_pickle=False) as z:
                old = integer_vector(z["static_pred"], "static predictions", identity["study_protocol"]["num_classes"])
                if not np.array_equal(z["row_id"], np.arange(len(old))) or (ids >= len(old)).any():
                    raise ValueError("Prediction row alignment mismatch")
                query_old = old[ids]
            payload = graph_payload(query_old, [int(r["label"]) for r in q],
                                    identity["study_protocol"]["num_classes"], seed, ids)
            write_json(directory / f"seed_{seed}.json", payload)
            n_errors = sum(a != b for a, b in zip(payload["query_static_predictions"], payload["query_labels"]))
            print(f"Frozen seed {seed}: {len(ids)} existing labels, {n_errors} confirmed errors", flush=True)
        finish(directory, identity, files)
    print("All requested relations frozen. Run evaluate in a separate process.", flush=True)


def gated_predictions(old, repaired, allowed):
    permitted = allowed[old, repaired]
    return np.where(permitted, repaired, old), ~permitted


def confusion(y, pred, classes):
    return np.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)


def class_stats(cm):
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    return recall, f1, support


def ratio(a, b):
    return float(a / b) if b else None


def metrics(y, old, pred, p):
    c = p["num_classes"]
    br, bf, support = class_stats(confusion(y, old, c))
    ar, af, _ = class_stats(confusion(y, pred, c))
    correct_old, correct_new = old == y, pred == y
    positive = ~correct_old & correct_new
    negative = correct_old & ~correct_new
    other_change = ~correct_old & ~correct_new & (old != pred)
    groups = {"overall": list(range(c)), "collapse": p["collapse_classes"],
              "noncollapse": sorted(set(range(c)) - set(p["collapse_classes"])), "stable": p["stable_classes"]}
    new = (support > 0) & (br >= p["collapse_recall_threshold"]) & (ar < p["collapse_recall_threshold"])
    nc = np.array([k for k in groups["noncollapse"] if support[k] > 0], dtype=int)
    result = {"eval_samples": len(y), "accuracy": float(correct_new.mean()),
              "positive_flips": int(positive.sum()), "negative_flips": int(negative.sum()),
              "negative_flip_rate_all": float(negative.mean()),
              "negative_flip_rate_old_correct": ratio(negative.sum(), correct_old.sum()),
              "wrong_to_different_wrong": int(other_change.sum()), "changed_predictions": int((pred != old).sum()),
              "new_collapses_all_supported": int(new.sum()), "noncollapse_new_collapses": int(new[nc].sum()),
              "noncollapse_degraded_count": int((af[nc] < bf[nc]).sum()),
              "noncollapse_drop_gt_threshold_count": int((af[nc] - bf[nc] < -p["f1_drop_threshold"]).sum()),
              "noncollapse_worst_delta_f1": float((af[nc] - bf[nc]).min()) if len(nc) else None,
              "collapse_residual_count": sum(int(support[k] > 0 and ar[k] < p["collapse_recall_threshold"]) for k in groups["collapse"])}
    for group, ids in groups.items():
        result[f"{group}_macro_f1"] = float(af[ids].mean()) if ids else None
    pcs = []
    pos_by_class = np.bincount(y[positive], minlength=c)
    neg_by_class = np.bincount(y[negative], minlength=c)
    wrong_by_class = np.bincount(y[other_change], minlength=c)
    for k in range(c):
        pcs.append({"class_id": k, "support": int(support[k]), "is_original_collapse": int(k in groups["collapse"]),
                    "before_recall": float(br[k]), "after_recall": float(ar[k]),
                    "before_f1": float(bf[k]), "after_f1": float(af[k]), "delta_f1": float(af[k] - bf[k]),
                    "new_collapse": int(new[k]), "positive_flips": int(pos_by_class[k]),
                    "negative_flips": int(neg_by_class[k]), "wrong_to_different_wrong": int(wrong_by_class[k])})
    return result, pcs


def relation_potential(y, old, allowed, counts, p):
    wrong = old != y
    recoverable = wrong & allowed[old, y]
    victim = np.isin(y, p["collapse_classes"])
    edges = int(allowed.sum() - len(allowed))
    row_degrees = allowed.sum(1) - 1
    return {"confirmed_query_errors": int(counts.sum() - np.trace(counts)),
            "confirmed_error_edges": edges, "singleton_error_edges": int(((counts == 1) & ~np.eye(len(counts), dtype=bool)).sum()),
            "active_absorber_rows": int((row_degrees > 0).sum()),
            "protected_correct_queries": int(np.trace(counts)),
            "mean_destinations_per_absorber": float(row_degrees.mean()),
            "max_destinations_per_absorber": int(row_degrees.max()),
            "off_diagonal_density": ratio(edges, len(allowed) * (len(allowed) - 1)),
            "routable_eval_fraction": float((row_degrees[old] > 0).mean()),
            "old_error_count": int(wrong.sum()), "allowed_error_count": int(recoverable.sum()),
            "allowed_error_fraction": ratio(recoverable.sum(), wrong.sum()),
            "collapse_old_error_count": int((wrong & victim).sum()),
            "collapse_allowed_error_count": int((recoverable & victim).sum()),
            "collapse_allowed_error_fraction": ratio((recoverable & victim).sum(), (wrong & victim).sum()),
            "oracle_accuracy_upper_bound": float(((~wrong) | recoverable).mean()),
            "collapse_micro_recall_upper_bound": ratio(((~wrong | recoverable) & victim).sum(), victim.sum())}


def relation_class_coverage(y, old, allowed, counts, p):
    c = p["num_classes"]
    support = np.bincount(y, minlength=c)
    errors = np.bincount(y[y != old], minlength=c)
    recoverable = np.bincount(y[(y != old) & allowed[old, y]], minlength=c)
    for k in range(c):
        yield {"class_id": k, "support": int(support[k]),
               "is_original_collapse": int(k in p["collapse_classes"]),
               "query_true_count": int(counts[:, k].sum()),
               "confirmed_query_errors": int(counts[:, k].sum() - counts[k, k]),
               "correct_query_count": int(counts[k, k]),
               "old_error_count": int(errors[k]), "allowed_error_count": int(recoverable[k]),
               "allowed_error_fraction": ratio(recoverable[k], errors[k]),
               "recall_upper_bound": ratio(support[k] - errors[k] + recoverable[k], support[k])}


def transition_rows(y, old, pred, allowed, p, class_rows):
    """All observed (truth, old prediction, new prediction) triples, all classes.

    Destination flags are post-hoc associations, not design inputs or causes.
    """
    c = p["num_classes"]
    codes, counts = np.unique((y * c + old) * c + pred, return_counts=True)
    for code, count in zip(codes, counts):
        dest, src, truth = int(code % c), int(code // c % c), int(code // (c * c))
        if src == truth:
            kind = "correct_to_correct" if dest == truth else "negative_flip"
        elif dest == truth:
            kind = "positive_flip"
        else:
            kind = "wrong_unchanged" if dest == src else "wrong_to_different_wrong"
        yield {"true_class": truth, "old_prediction": src, "new_prediction": dest, "transition": kind,
               "count": int(count), "allowed_by_confirmed_relation": int(allowed[src, dest]),
               "true_class_new_collapse": class_rows[truth]["new_collapse"],
               "destination_original_collapse": int(dest in p["collapse_classes"]),
               "destination_recall_improved": int(class_rows[dest]["after_recall"] > class_rows[dest]["before_recall"])}


def load_pair(root, p, seed):
    data = {}
    keys = ("row_id", "y_true", "static_pred", "repaired_pred", "queried", "strict_eval", "common_eval")
    for method in METHODS:
        directory = root / "runs" / method / f"seed_{seed}"
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in keys}
        n = len(a["row_id"])
        if not n or not np.array_equal(a["row_id"], np.arange(n)):
            raise ValueError("Invalid prediction row IDs")
        for key in ("y_true", "static_pred", "repaired_pred"):
            a[key] = integer_vector(a[key], key, p["num_classes"])
        for key in keys:
            if a[key].shape != (n,):
                raise ValueError(f"Invalid prediction shape: {key}")
        for key in ("queried", "strict_eval", "common_eval"):
            if a[key].dtype != np.dtype(bool):
                raise ValueError(f"Invalid evaluation mask dtype: {key}")
        ids = {s: query_records(root, p, seed, s) for s in ("margin", "badge")}
        if any((v >= n).any() for v in ids.values()):
            raise ValueError("Out-of-range query")
        queried = np.zeros(n, dtype=bool)
        queried[ids["badge"]] = True
        common = ~queried
        common[ids["margin"]] = False
        if (not common.any() or not np.array_equal(a["queried"], queried)
                or not np.array_equal(a["strict_eval"], ~queried) or not np.array_equal(a["common_eval"], common)):
            raise ValueError("Stored masks differ from the exact query union")
        q = audit.read_csv(directory / "query_ids.csv")
        if [int(r["label"]) for r in q] != a["y_true"][ids["badge"]].tolist():
            raise ValueError("Paid query labels differ from saved target labels")
        data[method] = a
    for key in keys:
        if key != "repaired_pred" and not np.array_equal(data[METHODS[0]][key], data[METHODS[1]][key]):
            raise ValueError(f"Unpaired BADGE predictions: {key}")
    return data


def reconcile(result, class_rows, directory):
    saved = read_json(directory / "metrics.json")
    for field in ("eval_samples", "noncollapse_new_collapses", "noncollapse_degraded_count",
                  "noncollapse_drop_gt_threshold_count", "noncollapse_worst_delta_f1", "collapse_residual_count"):
        expected = saved[f"common_{field}"]
        if expected is None:
            if result[field] is not None:
                raise ValueError(f"Metric mismatch: {field}")
        else:
            audit.close(result[field], expected, field)
    for group in ("overall", "collapse", "noncollapse", "stable"):
        if result[f"{group}_macro_f1"] is not None:
            audit.close(result[f"{group}_macro_f1"], saved[f"common_{group}_macro_f1_after"], group)
    rows = [r for r in audit.read_csv(directory / "per_class_metrics.csv") if r["split"] == "common"]
    if sorted(int(r["class_id"]) for r in rows) != list(range(len(class_rows))):
        raise ValueError("Missing/duplicated saved class metrics")
    for r in rows:
        actual = class_rows[int(r["class_id"])]
        for field in ("support", "before_recall", "after_recall", "before_f1", "after_f1", "delta_f1"):
            audit.close(actual[field], float(r[field]), f"class {r['class_id']} {field}")


def aggregate(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    output = []
    for group, records in groups.items():
        summary = {group_key: group, "n_seeds": len(records)}
        for key in records[0]:
            if key in (group_key, "seed"):
                continue
            values = [r[key] for r in records if r[key] is not None]
            summary[key + "_n"] = len(values)
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(summary)
    return output


def append_transitions(path, rows):
    """Stream potentially large transition tables; do not retain all seeds in RAM."""
    records = iter(rows)
    first = next(records, None)
    if first is None:
        return
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(first)
        writer.writerows(records)


def report(summaries, relations, pair_summaries, n, p):
    def f(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"
    lines = ["# 吸收关系指导修复：可行性预检", "",
             f"{n} 个查询/修复种子；每种子复用 {p['budget']} 个 BADGE 目标标签。CPU 分析，无新训练或标注。",
             "关系仅由已查询标签与旧预测构建。全部种子关系冻结后，独立评价进程才读取其余目标标签。",
             "common 评估排除该种子的 Margin∪BADGE 查询，与旧补充实验逐类对账；无需排除未参与本试验的选样探索查询。",
             "失败方向 v→a，允许纠正方向 a→v；自环保留旧预测。没有在查询中确认的方向禁止改变。",
             "degree_control 每个旧预测类别保留相同数量的候选目的类、随机打乱目的类，可能偶然包含真实边；不是多个独立控制重复。", "",
             "| 方法 | Overall F1 | 崩溃 F1 | Stable F1 | 正翻转数 | 负翻转数 | 新崩溃类数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        fields = [f(r[k + "_mean"], 3 if "f1" in k else 1) for k in
                  ("overall_macro_f1", "collapse_macro_f1", "stable_macro_f1", "positive_flips", "negative_flips", "noncollapse_new_collapses")]
        lines.append("| " + r["method"] + " | " + " | ".join(fields) + " |")
    lines += ["", "正翻转=旧错→新对；负翻转=旧对→新错。新增崩溃沿用原协议非崩溃组、有支持且 recall 从 ≥"
              f"{p['collapse_recall_threshold']} 降至阈值下；per_class.csv 另保留所有类别的新崩溃标记。", "",
              "| 与各自原修复比较 | 阻止的负翻转比例 | 保留的正翻转比例 | 崩溃 F1 差值 |",
              "|---|---:|---:|---:|"]
    for r in pair_summaries:
        if "prevented_negative_fraction_mean" in r:
            lines.append(f"| {r['comparison']} | {f(r['prevented_negative_fraction_mean'])} | "
                         f"{f(r['retained_positive_fraction_mean'])} | {f(r['collapse_macro_f1_mean'])} |")
    lines += ["", "比例先按种子计算再取均值；分母为零记为空，CSV 的 _n 列记录有效种子数。", "",
              "| 关系 | 错误边数 | 可纠正的原错误比例 | 崩溃类原错误可纠正比例 | 理想准确率上界 |",
              "|---|---:|---:|---:|---:|"]
    for r in relations:
        lines.append(f"| {r['graph']} | {f(r['confirmed_error_edges_mean'], 1)} | {f(r['allowed_error_fraction_mean'])} | "
                     f"{f(r['collapse_allowed_error_fraction_mean'])} | {f(r['oracle_accuracy_upper_bound_mean'])} |")
    lines += ["", "上界假定所有允许的旧错误都能被纠正且不伤害任何旧正确样本；不是实际算法，也不是宏 F1 上界。",
              "degree_control 表中的边数/查询证据描述共享的真实查询图，仅允许目的类被随机化。",
              "", "判读顺序：",
              "1. 先看 transitions.csv：新增崩溃流向哪里，是否指向恢复的类别。共现仅是关联，不能证明机制因果。",
              "2. 看 relation_by_seed.csv：查询图是否过稀、单次确认边是否过多，以及查询外的纠正覆盖是否足够。",
              "3. 看 paired_by_seed.csv：relation_gate 相对原修复阻止多少负翻转，同时牺牲多少正翻转。均值与逐种子一起看。",
              "4. 与 degree_control 比较真实错分方向是否提供额外价值。单纯减少更新或牺牲大量恢复，不算有用修复。",
              "5. 若限制覆盖不足或真实关系没有优势，就停止此硬限制设计；不能据此否定所有关系感知方法。",
              "", "本轮是后处理预检，尚未设计或训练新的 CARE 修复机制，不自动判成功、不宣称创新性或显著性。",
              "M12 已参与方案设计，属于开发性分析；若继续，需固定机制/参数后在未参与选择的数据上验证。",
              "五个查询/修复种子共享源模型和月份；summary.csv 的样本标准差不是五个独立部署环境的不确定性。",
              "单个确认错分对不保证该类整体崩溃，单个正确查询也不能保证目标总体保护。",
              "没有基于完整目标标签调阈值、挑最佳种子或指定失效类；不把旧源参考标签计作新增查询。", ""]
    return "\n".join(lines)


def evaluate(args):
    root, out, identity = verify_inputs(args)
    frozen = out / "frozen"
    files = ["resolved_config.json"] + [f"seed_{s}.json" for s in identity["seeds"]]
    p = identity["study_protocol"]
    with locked(out):
        completed(frozen, identity, files, mandatory=True)  # ALL seeds, before opening y_true.
        sig = {"identity": digest(identity), "frozen_sha256": audit.sha(frozen / "complete.json")}
        directory = out / "evaluation"
        derived = [f"seed_{s}_gated_predictions.npz" for s in identity["seeds"]]
        if completed(directory, sig, (*REPORT_FILES, *derived)):
            print(f"Verified, reusing report: {directory / 'report.md'}", flush=True)
            return
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "resolved_config.json", sig)
        by_seed, potential, potential_classes, all_classes, pairs = [], [], [], [], []
        transitions_temp = directory / "transitions.csv.tmp"
        transitions_temp.write_text("")
        target_identity = None
        for seed in identity["seeds"]:
            print(f"Auditing saved predictions: seed {seed}", flush=True)
            data = load_pair(root, p, seed)
            a = data[METHODS[0]]
            snapshot = {key: hashlib.sha256(a[key].tobytes()).hexdigest() for key in ("row_id", "y_true", "static_pred")}
            if target_identity is not None and snapshot != target_identity:
                raise ValueError("Target labels/order/static predictions differ across seeds")
            target_identity = snapshot
            graph = read_json(frozen / f"seed_{seed}.json")
            ids = np.array(graph["query_row_indices"], dtype=int)
            expected = graph_payload(a["static_pred"][ids], a["y_true"][ids], p["num_classes"], seed, ids)
            if expected != graph:
                raise ValueError("Frozen relation differs from paid query evidence")
            counts, allowed, control = build_relations(graph["query_static_predictions"], graph["query_labels"], p["num_classes"], seed)
            mask = a["common_eval"]
            y, old = a["y_true"][mask], a["static_pred"][mask]
            for name, g in (("confirmed", allowed), ("degree_control", control)):
                potential.append({"seed": seed, "graph": name, **relation_potential(y, old, g, counts, p)})
                potential_classes.extend({"seed": seed, "graph": name, **r}
                                         for r in relation_class_coverage(y, old, g, counts, p))
            base, pcs = metrics(y, old, old, p)
            by_seed.append({"seed": seed, "method": "source", **base})
            all_classes.extend({"seed": seed, "method": "source", **r} for r in pcs)
            predictions = {"row_id": a["row_id"], "common_eval": mask}
            for method in METHODS:
                raw_all = data[method]["repaired_pred"]
                raw = raw_all[mask]
                result, pcs = metrics(y, old, raw, p)
                reconcile(result, pcs, root / "runs" / method / f"seed_{seed}")
                by_seed.append({"seed": seed, "method": method, **result})
                all_classes.extend({"seed": seed, "method": method, **r} for r in pcs)
                append_transitions(transitions_temp, ({"seed": seed, "method": method, **r}
                                                     for r in transition_rows(y, old, raw, allowed, p, pcs)))
                raw_pos, raw_neg = (old != y) & (raw == y), (old == y) & (raw != y)
                gates = {}
                for suffix, g in (("relation_gate", allowed), ("degree_control", control)):
                    name = f"{method}_{suffix}"
                    pred_all, blocked_all = gated_predictions(a["static_pred"], raw_all, g)
                    pred, blocked = pred_all[mask], blocked_all[mask]
                    m, rows = metrics(y, old, pred, p)
                    gates[suffix] = m
                    by_seed.append({"seed": seed, "method": name, **m})
                    all_classes.extend({"seed": seed, "method": name, **r} for r in rows)
                    predictions[name] = pred_all
                    pairs.append({"seed": seed, "comparison": f"{name} minus {method}",
                                  **{k: m[k] - result[k] if m[k] is not None and result[k] is not None else None for k in m},
                                  "prevented_negative_flips": int((blocked & raw_neg).sum()),
                                  "lost_positive_flips": int((blocked & raw_pos).sum()),
                                  "prevented_negative_fraction": ratio((blocked & raw_neg).sum(), raw_neg.sum()),
                                  "retained_positive_fraction": ratio((~blocked & raw_pos).sum(), raw_pos.sum())})
                pairs.append({"seed": seed, "comparison": f"{method}_relation_gate minus {method}_degree_control",
                              **{k: gates['relation_gate'][k] - gates['degree_control'][k]
                                 if gates['relation_gate'][k] is not None and gates['degree_control'][k] is not None else None for k in result}})
            temp = directory / f"seed_{seed}_gated_predictions.tmp.npz"
            np.savez_compressed(temp, **predictions)
            os.replace(temp, directory / f"seed_{seed}_gated_predictions.npz")
        summaries, relations = aggregate(by_seed, "method"), aggregate(potential, "graph")
        pair_summaries = aggregate(pairs, "comparison")
        for name, rows in (("by_seed.csv", by_seed), ("summary.csv", summaries), ("paired_by_seed.csv", pairs),
                           ("paired_summary.csv", pair_summaries), ("relation_per_class.csv", potential_classes),
                           ("relation_by_seed.csv", potential), ("relation_summary.csv", relations),
                           ("per_class.csv", all_classes)):
            audit.write_csv(directory / name, rows)
        os.replace(transitions_temp, directory / "transitions.csv")
        audit.write_text(directory / "report.md", report(summaries, relations, pair_summaries, len(identity["seeds"]), p))
        # Recheck identities/checksums before publishing completion if source jobs changed files.
        if verify_inputs(args)[2] != identity:
            raise ValueError("Source changed during audit; output is incomplete")
        finish(directory, sig, (*REPORT_FILES, *derived))
    print(f"Results: {directory / 'report.md'}", flush=True)


def parser():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("command", choices=("plan", "preflight", "freeze", "evaluate"))
    q.add_argument("--study-dir", default="outputs/kbs_supplement_v1")
    q.add_argument("--output-dir", default="outputs/kbs_relation_audit_v1")
    q.add_argument("--seeds", default="0,1,2,3,4")
    return q


def main():
    args = parser().parse_args()
    os.chdir(CORE)
    if args.command == "plan":
        print(json.dumps({"methods": METHODS, "seeds": args.seeds, "device": "CPU/NumPy",
                          "required": "completed BADGE full/KD predictions, queries, metrics and manifests",
                          "new_labels": 0, "new_fits": 0, "feature_cache_required": False,
                          "order": ["freeze all requested seeds using paid query labels", "evaluate in separate process"]}, indent=2))
    elif args.command == "preflight":
        _, out, identity = verify_inputs(args)
        print(json.dumps({"verified_input_hashes": len(identity["source_sha256"]), "seeds": identity["seeds"],
                          "output": str(out), "new_fits": 0, "next": "run launcher with run"}, indent=2))
    elif args.command == "freeze":
        freeze(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
