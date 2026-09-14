#!/usr/bin/env python3
"""Complete the fixed-query BADGE 2x2 reference-CE/KD ablation; preserve v1."""
import contextlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

import kbs_supplement as kbs
import kbs_badge_kd_followup as follow

SUITE = "badge_controls"
BASE_SPECS = kbs.make_specs
COMPARISONS = [
    ("badge_kd", "badge_ft_only"), ("badge_replay", "badge_ft_only"),
    ("badge_full", "badge_replay"), ("badge_full", "badge_kd"),
    ("badge_full", "badge_ft_only"),
]


def specs(suite=SUITE):
    if suite != SUITE:
        return BASE_SPECS(suite)
    full, kd = follow.specs()
    return [{**full, "name": "badge_ft_only", "replay_ce": False, "kd_weight": 0.},
            {**full, "name": "badge_replay", "kd_weight": 0.}, kd, full]


def signature(protocol, spec, seed):
    return {"protocol": protocol, "spec": spec, "seed": seed}


def run_dir(out, spec, seed):
    return Path(out) / "runs" / spec["name"] / f"seed_{seed}"


def identity(protocol):
    return {"runner_sha256": kbs.file_sha(Path(__file__)),
            "followup_sha256": kbs.file_sha(Path(follow.__file__)),
            "implementation_sha256": kbs.implementation_sha(),
            "protocol_sha256": kbs.digest(protocol), "specs": specs()}


def extension(out, protocol, create=False):
    path = out / f"{SUITE}_extension_manifest.json"
    expected = identity(protocol)
    if path.exists():
        if json.loads(path.read_text())["identity"] != expected:
            raise ValueError("BADGE controls implementation/protocol changed; preserve existing outputs.")
    elif create:
        kbs.atomic_json(path, {"identity": expected, "environment": kbs.runtime_info()})
    else:
        raise ValueError("BADGE controls extension manifest is required; run this extension first.")


def verify_baselines(out, seeds):
    protocol = follow.verify_baselines(out, seeds)
    for seed in seeds:
        follow.paired_metrics(out, protocol, seed)  # Require both full AND KD, not a new KD fit.
    return protocol


def check_settings(args, protocol):
    fields = {"budget": args.budget, "optimizer_steps": args.steps, "batch_size": args.batch_size,
              "lr": args.lr, "weight_decay": args.weight_decay, "threads": args.threads,
              "collapse_classes": kbs.int_list(args.collapse_classes),
              "stable_classes": kbs.int_list(args.stable_classes),
              "collapse_recall_threshold": args.collapse_recall_threshold,
              "f1_drop_threshold": args.f1_drop_threshold,
              "device": str(kbs.torch.device(args.device)),
              "torch_version": str(kbs.torch.__version__), "numpy_version": np.__version__}
    changed = [key for key, value in fields.items() if protocol[key] != value]
    source = json.loads((Path(args.output_dir) / "study_manifest.json").read_text())["source_manifest"]
    # Minimal synthetic fixtures have no data-period spec; production manifests do.
    periods = source.get("spec", {}).get("periods")
    if periods is not None and periods != {"reference": args.reference_period, "target": args.target_period}:
        changed.append("periods")
    if changed:
        raise ValueError("Study protocol changed: " + ", ".join(changed))


def checked_cache(args, protocol):
    path = Path(args.cache_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError("The existing primary feature cache is required; no extraction is performed here.")
    info = json.loads(path.read_text())
    source = json.loads((Path(args.output_dir) / "study_manifest.json").read_text())["source_manifest"]
    if info != source or info["fingerprint"] != protocol["cache"]:
        raise ValueError("Feature cache differs from the original study.")
    kbs.prepare(args)  # An existing manifest is mandatory above; validates without extracting.


def complete(out, protocol, spec, seed):
    directory = run_dir(out, spec, seed)
    if not kbs.is_complete(directory, signature(protocol, spec, seed)):
        return False
    files = json.loads((directory / "complete.json").read_text())["artifacts"]
    required = {"resolved_config.json", "repaired_head.pt", "predictions.npz", "query_ids.csv",
                "replay_ids.csv", "training_trace.json", "metrics.json", "per_class_metrics.csv"}
    if not required.issubset(files):
        raise ValueError(f"Incomplete artifact manifest: {directory}")
    return True


def paired_seed(out, protocol, seed):
    """Verify all four cells and recompute global metrics from identical holdout rows."""
    directories = [run_dir(out, spec, seed) for spec in specs()]
    if not all(complete(out, protocol, spec, seed) for spec in specs()):
        raise ValueError(f"All four completed controls are required: seed {seed}")
    tables = {name: [kbs.read_csv(d / name) for d in directories]
              for name in ["query_ids.csv", "replay_ids.csv"]}
    for filename, values in tables.items():
        ids = [[tuple(r[k] for k in ["sample_id", "row_index", "label"]) for r in rows]
               for rows in values]
        if any(rows != ids[0] for rows in ids[1:]):
            raise ValueError(f"Unpaired {filename}: seed {seed}")
    traces = [json.loads((d / "training_trace.json").read_text()) for d in directories]
    presentations = protocol["optimizer_steps"] * protocol["batch_size"]
    for spec, trace, replay in zip(specs(), traces, tables["replay_ids.csv"]):
        expected = {"optimizer_steps": protocol["optimizer_steps"], "target_presentations": presentations,
                    "reference_ce_presentations": presentations if spec["replay_ce"] else 0,
                    "reference_kd_presentations": presentations if spec["kd_weight"] else 0}
        for key, value in expected.items():
            if trace[key] != value:
                raise ValueError(f"Incorrect training {key}: {spec['name']} seed {seed}")
        for key in ["target_stream_sha256", "reference_stream_sha256"]:
            if trace[key] != traces[0][key]:
                raise ValueError(f"Unpaired training {key}: seed {seed}")
        if any(r["used_for_ce"] != str(spec["replay_ce"]) or
               r["used_for_kd"] != str(spec["kd_weight"] > 0) for r in replay):
            raise ValueError(f"Incorrect reference loss flags: {spec['name']} seed {seed}")

    metrics, classes = [], []
    with np.load(directories[-1] / "predictions.npz", allow_pickle=False) as archive:
        shared = {key: archive[key] for key in
                  ["row_id", "y_true", "static_pred", "queried", "strict_eval", "common_eval"]}
    n = len(shared["row_id"])
    if not np.array_equal(shared["row_id"], np.arange(n)):
        raise ValueError(f"Invalid row identities: seed {seed}")
    common = np.ones(n, dtype=bool)
    strict = np.ones(n, dtype=bool)
    for selector in ["margin", "badge"]:
        selected = json.loads((out / "selections" / selector / f"seed_{seed}" / "selection.json").read_text())
        idx = np.asarray(selected["row_indices"], dtype=np.int64)
        if len(idx) != protocol["budget"] or len(np.unique(idx)) != len(idx) or np.any((idx < 0) | (idx >= n)):
            raise ValueError(f"Invalid saved selection: {selector} seed {seed}")
        common[idx] = False
        if selector == "badge":
            strict[idx] = False
            if idx.tolist() != [int(r["row_index"]) for r in tables["query_ids.csv"][0]]:
                raise ValueError(f"BADGE query order differs from saved selection: seed {seed}")
    for key, value in [("strict_eval", strict), ("queried", ~strict), ("common_eval", common)]:
        if shared[key].dtype != np.bool_ or not np.array_equal(shared[key], value):
            raise ValueError(f"Incorrect exclusion {key}: seed {seed}")
    for row in tables["query_ids.csv"][0]:
        i = int(row["row_index"])
        if int(row["label"]) != shared["y_true"][i] or row["sample_id"] != f"{protocol['cache']}:target:{i}":
            raise ValueError(f"Query identity/label mismatch: seed {seed}")
    for spec, directory in zip(specs(), directories):
        metric = json.loads((directory / "metrics.json").read_text())
        if metric["seed"] != seed or any(metric[k] != v for k, v in spec.items()):
            raise ValueError(f"Metric identity mismatch: {directory}")
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            if any(not np.array_equal(archive[k], value) for k, value in shared.items()):
                raise ValueError(f"Unpaired predictions/evaluation: {directory}")
            after = archive["repaired_pred"]
        c = protocol["num_classes"]
        for values in [shared["y_true"], shared["static_pred"], after]:
            if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= c)):
                raise ValueError(f"Invalid prediction or label array: {directory}")
        for split in ["strict", "common"]:
            mask = shared[split + "_eval"]
            computed, per_class, _ = kbs.compare_predictions(
                shared["y_true"], shared["static_pred"], after, mask, c,
                protocol["collapse_classes"], protocol["stable_classes"],
                protocol["collapse_recall_threshold"], protocol["f1_drop_threshold"])
            for key, value in computed.items():
                saved = metric[split + "_" + key]
                if (saved is None) != (value is None) or (
                        value is not None and not np.isclose(saved, value, rtol=1e-10, atol=1e-12)):
                    raise ValueError(f"Metrics disagree with predictions: {directory} {split}_{key}")
            before_correct = shared["static_pred"] == shared["y_true"]
            after_correct = after == shared["y_true"]
            metric[split + "_positive_flips"] = int((mask & ~before_correct & after_correct).sum())
            metric[split + "_negative_flips"] = int((mask & before_correct & ~after_correct).sum())
            classes.extend({"name": spec["name"], "seed": seed, "split": split, **r} for r in per_class)
        metrics.append(metric)
    return metrics, classes


def aggregate(rows, identifiers):
    item = {**identifiers, "n_seeds": len(rows)}
    if rows:
        for key in rows[0]:
            if key.startswith(("strict_", "common_")) or key == "train_wall_s":
                values = [float(r[key]) for r in rows if r.get(key) is not None]
                if values:
                    item[key + "_mean"] = statistics.mean(values)
                    item[key + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
    return item


def summarize(out, seeds):
    out = Path(out)
    protocol = verify_baselines(out, seeds)
    extension(out, protocol)
    rows, classes, done, missing = [], [], [], []
    for seed in seeds:
        present = [complete(out, protocol, spec, seed) for spec in specs()]
        if not all(present):
            missing.append({"seed": seed, "configurations": [s["name"] for s, ok in zip(specs(), present) if not ok]})
            continue
        seed_rows, seed_classes = paired_seed(out, protocol, seed)
        rows.extend(seed_rows)
        classes.extend(seed_classes)
        done.append(seed)
    summaries = [aggregate([r for r in rows if r["name"] == s["name"]], s) for s in specs()]
    pairs = []
    for seed in done:
        values = {r["name"]: r for r in rows if r["seed"] == seed}
        for left, right in COMPARISONS:
            pair = {"seed": seed, "comparison": f"{left} minus {right}", "pairing_verified": True}
            pair.update({key + "_difference": value - values[right][key] for key, value in values[left].items()
                         if key.startswith(("strict_", "common_")) and value is not None})
            pairs.append(pair)
    paired_summary = [aggregate([r for r in pairs if r["comparison"] == f"{a} minus {b}"],
                                {"comparison": f"{a} minus {b}"}) for a, b in COMPARISONS]
    outputs = {"results_by_seed.csv": rows, "summary.csv": summaries, "paired_by_seed.csv": pairs,
               "paired_summary.csv": paired_summary, "per_class.csv": classes}
    # Invalidate the derived completion marker before replacing any report file.
    status_path = out / f"{SUITE}_status.json"
    status_path.unlink(missing_ok=True)
    for suffix, values in outputs.items():
        kbs.write_csv(out / f"{SUITE}_{suffix}", values)
    report = ["# BADGE 固定查询组件对照", "",
              f"已核验四组配对种子：{len(done)}/{len(seeds)}；请求 {seeds}；完成 {done}。",
              "未完成：" + (json.dumps(missing, ensure_ascii=False) if missing else "无。"), "",
              "下表仅使用四组均完成且通过核验的同一组种子，均为 common 查询并集排除评估。",
              "均值 ± 样本标准差；一个种子时标准差记为 —。", "",
              "| 配置 | 整体 F1 | 崩溃类 F1 | 非崩溃类 F1 | 稳定类 F1 | 新崩溃数 | 原崩溃残留数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    def cell(row, key):
        mean, sd = row.get(key + "_mean"), row.get(key + "_sample_sd")
        return "—" if mean is None else f"{mean:.4f} ± " + ("—" if sd is None else f"{sd:.4f}")
    fields = [f"common_{g}_macro_f1_after" for g in ["overall", "collapse", "noncollapse", "stable"]]
    fields += ["common_noncollapse_new_collapses", "common_collapse_residual_count"]
    static = " | ".join(cell(summaries[-1], k.replace("_after", "_before")) for k in fields[:4])
    report.append("| static（同一评估集） | " + static + " | — | — |")
    for row in summaries:
        report.append("| " + row["name"] + " | " + " | ".join(cell(row, k) for k in fields) + " |")
    report += ["", "配对差值为前者减后者；F1 越大越好，新增崩溃差值越小越好。", "",
               "| 对照 | 整体 F1 差值 | 崩溃类 F1 差值 | 稳定类 F1 差值 | 新崩溃数差值 |",
               "|---|---:|---:|---:|---:|"]
    for row in paired_summary:
        keys = ["common_overall_macro_f1_after", "common_collapse_macro_f1_after",
                "common_stable_macro_f1_after", "common_noncollapse_new_collapses"]
        report.append("| " + row["comparison"] + " | " + " | ".join(cell(row, k + "_difference") for k in keys) + " |")
    report += ["", "四组依次为：仅目标 CE、目标 CE＋参考 CE、目标 CE＋参考 KD、目标 CE＋参考 CE＋参考 KD。",
               "固定目标 CE 权重、学习率、步数、查询及参考行和抽样流；关闭某项损失不重新归一化其他权重。",
               "仅目标 CE 对照仍经过原引擎的参考缓存加载；不能据此声称更低的内存或 I/O 成本。",
               "", "请结合 paired_by_seed.csv / paired_summary.csv 的配对增量、per_class.csv 的逐类损伤判断；",
               "正翻转/负翻转计数和新增崩溃必须同时报告，均值提高不代表每类或每个种子都改善。",
               "训练耗时不含已缓存的特征提取和选样；此次不产生端到端效率结论。",
               "", "本轮是 M12 开发数据上的组件验证；种子共用一个源模型和月份，不是独立跨环境重复。",
               "不自动选择获胜配置、不检验显著性，也不由此认定新方法创新性。旧主表和旧实验输出保持不变。",
               ""]
    path = out / f"{SUITE}_report.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text("\n".join(report))
    os.replace(temp, path)
    files = [out / f"{SUITE}_{suffix}" for suffix in outputs] + [path]
    kbs.atomic_json(status_path, {"identity": identity(protocol), "requested_seeds": seeds,
                                "completed_seeds": done, "missing": missing, "all_requested_complete": not missing,
                                "artifacts": {p.name: kbs.file_sha(p) for p in files}})
    print(f"Verified BADGE controls: {len(done)}/{len(seeds)} seeds x 4 configurations", flush=True)


@contextlib.contextmanager
def registry(seeds):
    original_specs, original_summary = kbs.make_specs, kbs.summarize
    kbs.make_specs = specs
    kbs.summarize = lambda out, suite: summarize(out, seeds)
    try:
        yield
    finally:
        kbs.make_specs, kbs.summarize = original_specs, original_summary


def run(args, data=None):
    out, seeds = Path(args.output_dir).resolve(), kbs.int_list(args.seeds)
    if not (out / "study_manifest.json").is_file():
        raise ValueError("A completed primary and BADGE-KD study is required; set OUTPUT_DIR to it.")
    with kbs.file_lock(out / ".badge_controls.lock"):
        with kbs.file_lock(out / ".lock"):
            protocol = verify_baselines(out, seeds)
            check_settings(args, protocol)
            pending = [(s["name"], seed) for seed in seeds for s in specs()
                       if not complete(out, protocol, s, seed)]
            extension(out, protocol, create=True)
            print(f"New head fits remaining: {len(pending)} (maximum {2 * len(seeds)})", flush=True)
            if not pending:
                summarize(out, seeds)
                print("All requested controls verified; no feature cache loaded and no training performed.")
                return
        if data is None:
            checked_cache(args, protocol)
            data = kbs.load_cache(args)
        with registry(seeds):
            args.suite = SUITE
            kbs.run_study(args, data)


def main():
    args = kbs.parser().parse_args()
    if args.command not in ["plan", "preflight", "run", "summarize"] or args.suite != "primary":
        raise ValueError("Use plan, preflight, run or summarize; omit --suite.")
    args.seeds = args.seeds if args.seeds is not None else "0,1,2,3,4"
    seeds = kbs.int_list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Supply distinct non-negative seeds.")
    if any(getattr(args, key) <= 0 for key in ["budget", "steps", "batch_size", "threads", "lr"]):
        raise ValueError("Budgets, steps, batch size, threads and learning rate must be positive.")
    if args.weight_decay < 0 or args.log_every < 0:
        raise ValueError("Invalid optimizer/logging settings.")
    os.chdir(kbs.CORE)
    if args.command == "plan":
        print(json.dumps({"suite": SUITE, "seeds": seeds, "configs": specs(),
                          "required_completed_baselines": 2 * len(seeds), "maximum_new_fits": 2 * len(seeds),
                          "optimizer_steps_per_fit": args.steps,
                          "note": "Reuse full/KD and both saved query selections; train only FT-only and replay."}, indent=2))
    elif args.command == "run":
        run(args)
    elif args.command == "preflight":
        with kbs.file_lock(Path(args.output_dir) / ".lock"):
            protocol = verify_baselines(Path(args.output_dir), seeds)
            check_settings(args, protocol)
            checked_cache(args, protocol)
            print("Preflight passed: completed full/KD pairs, settings and existing cache verified; no training.")
    else:
        with kbs.file_lock(Path(args.output_dir) / ".lock"):
            summarize(args.output_dir, seeds)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
