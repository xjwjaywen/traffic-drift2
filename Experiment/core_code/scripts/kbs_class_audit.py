#!/usr/bin/env python3
"""Audit saved common-split class metrics without loading models or training."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

CORE = Path(__file__).resolve().parents[1]
METHODS = ("margin_ft_only", "margin_replay", "margin_kd", "margin_full", "badge_full", "badge_kd")
FILES = ("resolved_config.json", "metrics.json", "per_class_metrics.csv", "query_ids.csv", "replay_ids.csv")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_text(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)


def expected_spec(name):
    return {"name": name, "selector": "badge" if name.startswith("badge") else "margin",
            "replay_ce": name in ("margin_replay", "margin_full", "badge_full"),
            "kd_weight": 0. if name in ("margin_ft_only", "margin_replay") else .5,
            "temperature": 2., "replay_per_class": 5}


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch ({label}): {actual} != {expected}")


def load_run(root, protocol, name, seed):
    directory = root / "runs" / name / f"seed_{seed}"
    done = json.loads((directory / "complete.json").read_text())
    signature = {"protocol": protocol, "spec": expected_spec(name), "seed": seed}
    if done["signature"] != signature:
        raise ValueError(f"Run protocol/spec mismatch: {directory}")
    hashes = {}
    for filename in FILES:
        path = directory / filename
        hashes[str(path.relative_to(root))] = sha(path)
        if hashes[str(path.relative_to(root))] != done["artifacts"].get(filename):
            raise ValueError(f"Missing/modified audit input: {path}")
    hashes[str((directory / "complete.json").relative_to(root))] = sha(directory / "complete.json")
    if json.loads((directory / "resolved_config.json").read_text()) != signature:
        raise ValueError(f"Resolved configuration mismatch: {directory}")
    metrics = json.loads((directory / "metrics.json").read_text())
    if any(metrics.get(k) != v for k, v in {**expected_spec(name), "seed": seed}.items()):
        raise ValueError(f"Metrics identity mismatch: {directory}")
    selections = {role: read_csv(directory / filename) for role, filename in
                  [("query", "query_ids.csv"), ("reference", "replay_ids.csv")]}
    counts = {role: Counter(int(r["label"]) for r in rows) for role, rows in selections.items()}
    collapse, stable = set(protocol["collapse_classes"]), set(protocol["stable_classes"])
    records = []
    for row in read_csv(directory / "per_class_metrics.csv"):
        if row["split"] != "common":
            continue
        c, support = int(row["class_id"]), int(row["support"])
        if support < 0:
            raise ValueError(f"Negative support: {directory}, class {c}")
        r = {"method": name, "seed": seed, "class_id": c, "support": support,
             "is_collapse_group": c in collapse, "is_stable_group": c in stable}
        for field in ["before_recall", "after_recall", "before_f1", "after_f1", "delta_f1"]:
            r[field] = float(row[field])
            if not math.isfinite(r[field]):
                raise ValueError(f"Non-finite {field}: {directory}, class {c}")
            if field != "delta_f1" and not 0 <= r[field] <= 1:
                raise ValueError(f"Out-of-range {field}: {directory}, class {c}")
        close(r["delta_f1"], r["after_f1"] - r["before_f1"], f"{name}/{seed}/{c} delta")
        threshold = protocol["collapse_recall_threshold"]
        supported_nc = support > 0 and c not in collapse
        r.update({"query_count": counts["query"][c], "reference_count": counts["reference"][c],
                  "new_collapse": supported_nc and r["before_recall"] >= threshold > r["after_recall"],
                  "residual_collapse": support > 0 and c in collapse and r["after_recall"] < threshold,
                  "noncollapse_degraded": supported_nc and r["delta_f1"] < 0,
                  "noncollapse_drop_gt_threshold": supported_nc and r["delta_f1"] < -protocol["f1_drop_threshold"]})
        records.append(r)
    records.sort(key=lambda r: r["class_id"])
    if [r["class_id"] for r in records] != list(range(protocol["num_classes"])):
        raise ValueError(f"Missing/duplicate common-split class rows: {directory}")
    for group, chosen in [("overall", records),
                          ("collapse", [r for r in records if r["is_collapse_group"]]),
                          ("noncollapse", [r for r in records if not r["is_collapse_group"]]),
                          ("stable", [r for r in records if r["is_stable_group"]])]:
        for when in ["before", "after"]:
            if chosen:
                close(statistics.mean(r[f"{when}_f1"] for r in chosen),
                      metrics[f"common_{group}_macro_f1_{when}"], f"{name}/{seed}/{group}/{when}")
        close(sum(r["support"] > 0 for r in chosen), metrics[f"common_{group}_supported_classes"], group)
    close(sum(r["support"] for r in records), metrics["common_eval_samples"], "evaluation samples")
    for flag, metric in [("new_collapse", "noncollapse_new_collapses"),
                         ("residual_collapse", "collapse_residual_count"),
                         ("noncollapse_degraded", "noncollapse_degraded_count"),
                         ("noncollapse_drop_gt_threshold", "noncollapse_drop_gt_threshold_count")]:
        close(sum(r[flag] for r in records), metrics[f"common_{metric}"], f"{name}/{seed}/{metric}")
    noncollapse = [r for r in records if not r["is_collapse_group"] and r["support"] > 0]
    worst = sorted(noncollapse, key=lambda r: (r["delta_f1"], r["class_id"]))[:10]
    for r in records:
        r["worst_noncollapse_rank"] = next((i + 1 for i, w in enumerate(worst) if w is r), None)
    if worst:
        close(worst[0]["delta_f1"], metrics["common_noncollapse_worst_delta_f1"], "worst class")
    return records, metrics, selections, hashes


def analyze(root, seeds):
    root = Path(root).resolve()
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    if not (root / "study_manifest.json").is_file():
        raise ValueError(f"Completed study not found: {root}")
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another job is using this output directory; run the audit after it finishes.") from exc
        return analyze_locked(root, seeds)


def analyze_locked(root, seeds):
    manifest = json.loads((root / "study_manifest.json").read_text())
    protocol = manifest["protocol"]
    rows, run_metrics, cases = [], [], []
    hashes = {"study_manifest.json": sha(root / "study_manifest.json")}
    for seed in seeds:
        baseline, badge_selection = None, None
        for name in METHODS:
            records, metrics, selection, files = load_run(root, protocol, name, seed)
            before = [(r["class_id"], r["support"], r["before_recall"], r["before_f1"]) for r in records]
            if baseline is not None and before != baseline:
                raise ValueError(f"Common-split baseline is not paired: {name}, seed {seed}")
            baseline = before
            if name == "badge_full":
                badge_selection = selection
            elif name == "badge_kd":
                for role in selection:
                    keys = ("sample_id", "row_index", "label")
                    a = [tuple(r[k] for k in keys) for r in badge_selection[role]]
                    b = [tuple(r[k] for k in keys) for r in selection[role]]
                    if a != b:
                        raise ValueError(f"BADGE {role} IDs differ: seed {seed}")
            rows.extend(records)
            run_metrics.append(metrics)
            hashes.update(files)
            cases.extend(r for r in records if r["new_collapse"] or r["residual_collapse"] or
                         r["noncollapse_drop_gt_threshold"] or r["worst_noncollapse_rank"])
    summaries = []
    for name in METHODS:
        chosen = [r for r in run_metrics if r["name"] == name]
        item = {"method": name, "n_seeds": len(chosen), "recommended_n_seeds": 5,
                "meets_recommended_seed_count": len(chosen) >= 5}
        for field in ["overall_macro_f1_after", "collapse_macro_f1_after", "noncollapse_macro_f1_after",
                      "stable_macro_f1_after", "noncollapse_new_collapses", "collapse_residual_count",
                      "noncollapse_degraded_count", "noncollapse_drop_gt_threshold_count", "noncollapse_worst_delta_f1"]:
            values = [r[f"common_{field}"] for r in chosen]
            item[field + "_mean"] = statistics.mean(values)
            item[field + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(item)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["class_id"])].append(r)
    recurring = []
    for (name, c), group in groups.items():
        item = {"method": name, "class_id": c, "n_seeds": len(group),
                "support_min": min(r["support"] for r in group), "support_max": max(r["support"] for r in group),
                "before_f1_mean": statistics.mean(r["before_f1"] for r in group),
                "after_f1_mean": statistics.mean(r["after_f1"] for r in group),
                "delta_f1_mean": statistics.mean(r["delta_f1"] for r in group),
                "delta_f1_worst": min(r["delta_f1"] for r in group),
                "query_count_mean": statistics.mean(r["query_count"] for r in group),
                "reference_count_mean": statistics.mean(r["reference_count"] for r in group)}
        for flag in ["new_collapse", "residual_collapse", "noncollapse_drop_gt_threshold"]:
            affected = [str(r["seed"]) for r in group if r[flag]]
            item[flag + "_seed_count"] = len(affected)
            item[flag + "_seeds"] = ",".join(affected)
        recurring.append(item)
    index = {(r["method"], r["class_id"]): r for r in recurring}
    badge = []
    for c in range(protocol["num_classes"]):
        full, kd = [index[(name, c)] for name in ("badge_full", "badge_kd")]
        entry = {"class_id": c, "n_seeds": len(seeds), "support_min": full["support_min"], "support_max": full["support_max"]}
        for field in ["new_collapse_seed_count", "residual_collapse_seed_count", "noncollapse_drop_gt_threshold_seed_count",
                      "after_f1_mean", "delta_f1_worst", "query_count_mean"]:
            entry["badge_full_" + field], entry["badge_kd_" + field] = full[field], kd[field]
        entry["kd_minus_full_f1_mean"] = kd["after_f1_mean"] - full["after_f1_mean"]
        badge.append(entry)
    report = make_report(summaries, badge, len(seeds), protocol,
                         manifest.get("source_manifest", {}).get("periods", {}))
    # Write only derived analysis files, after every selected run has validated.
    output = root / "class_audit"
    output.mkdir(exist_ok=True)
    for filename, records in [("summary.csv", summaries), ("all_classes_by_seed.csv", rows),
                              ("cases_by_seed.csv", cases), ("class_recurrence.csv", recurring),
                              ("badge_class_comparison.csv", badge)]:
        write_csv(output / filename, records)
    write_text(output / "report.md", report)
    write_text(output / "provenance.json", json.dumps({
        "schema": "care-class-audit-v1", "analysis_script_sha256": sha(Path(__file__)),
        "seeds": seeds, "split": "common", "input_sha256": hashes,
        "verification_scope": "checksums of consumed small files; class metrics reconciled with run metrics; no prediction/model reload",
        "protocol": protocol}, indent=2) + "\n")
    print(report, flush=True)
    print(f"Class audit saved: {output}", flush=True)
    return output


def make_report(summaries, badge, n, protocol, periods):
    lines = ["# CARE 逐类保护分析", "",
             f"共同评估集 common；{len(summaries)} 种配置 × {n} 个修复种子。建议每配置 5 个种子。",
             f"源 checkpoint 固定；参考时期：{periods.get('reference', '未记录')}；目标时期：{periods.get('target', '未记录')}。",
             f"新增崩溃：原非崩溃组中有评估样本的类别，修复前 recall ≥ {protocol['collapse_recall_threshold']}、修复后低于该值。",
             "所有 F1 为 0–1 单位。类别数量为各次运行的均值；跨种子累计次数不代表不同类别数。", "",
             "| 配置 | 总体 F1 | 崩溃组 F1 | 非崩溃组 F1 | 新增崩溃类 | 残余崩溃类 | 严重退化类 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['method']} | {r['overall_macro_f1_after_mean']:.5f} | {r['collapse_macro_f1_after_mean']:.5f} | "
                     f"{r['noncollapse_macro_f1_after_mean']:.5f} | {r['noncollapse_new_collapses_mean']:.1f} | "
                     f"{r['collapse_residual_count_mean']:.1f} | {r['noncollapse_drop_gt_threshold_count_mean']:.1f} |")
    lines += ["", f"严重退化指非崩溃组有支持类别的 F1 下降超过 {protocol['f1_drop_threshold']}。样本标准差见 summary.csv。", "",
              "## BADGE：发生新增或残余崩溃的类别", "",
              "Full 为完整 BADGE，KD 为关闭参考 CE 的 BADGE。次数的分母是本次分析的种子数。", "",
              "| 类别 ID | 评估样本量范围 | Full 新增次数 | KD 新增次数 | Full 残余次数 | KD 残余次数 | KD−Full 平均 F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    affected = [r for r in badge if any(r[p + f + "_seed_count"] for p in ("badge_full_", "badge_kd_")
                                       for f in ("new_collapse", "residual_collapse"))]
    for r in affected:
        lines.append(f"| {r['class_id']} | {r['support_min']}–{r['support_max']} | "
                     f"{r['badge_full_new_collapse_seed_count']}/{n} | {r['badge_kd_new_collapse_seed_count']}/{n} | "
                     f"{r['badge_full_residual_collapse_seed_count']}/{n} | {r['badge_kd_residual_collapse_seed_count']}/{n} | "
                     f"{r['kd_minus_full_f1_mean']:+.5f} |")
    if not affected:
        lines += ["", "所选运行中未发现新增或残余崩溃类别。"]
    lines += ["", "## BADGE：KD 相对 Full 平均 F1 降低最多的类别", "",
              "| 类别 ID | 样本量范围 | KD−Full 平均 F1 | Full 最差种子修复变化 | KD 最差种子修复变化 | 平均查询样本数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in badge if r["kd_minus_full_f1_mean"] < 0), key=lambda r: r["kd_minus_full_f1_mean"])[:10]:
        lines.append(f"| {r['class_id']} | {r['support_min']}–{r['support_max']} | {r['kd_minus_full_f1_mean']:+.5f} | "
                     f"{r['badge_full_delta_f1_worst']:+.5f} | {r['badge_kd_delta_f1_worst']:+.5f} | {r['badge_kd_query_count_mean']:.1f} |")
    lines += ["", "逐种子 recall、F1、样本量、查询量及参考样本量见 cases_by_seed.csv；全部类别见 all_classes_by_seed.csv。",
              "最差种子可能因方法而异。查询量及支持数仅用于诊断，不能单独证明退化原因。", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(CORE / "outputs/kbs_supplement_v1"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    args = p.parse_args()
    analyze(args.output_dir, [int(s) for s in args.seeds.replace(",", " ").split()])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
