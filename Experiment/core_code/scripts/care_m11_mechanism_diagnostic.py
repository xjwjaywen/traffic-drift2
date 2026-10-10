#!/usr/bin/env python3
"""Post-hoc M11 recovery/damage localization; no new labels, fits, or policy.

Audit all saved heads/predictions/metrics before describing paired errors.
Then inspect paid-query geometry and bounded, explicitly retrospective probes.
These observations localize mechanisms; they do not identify a causal effect.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

import care_cross_month_replication as month
import care_query_seed_robustness as robust
import care_mixed_query_ablation as ab
import kbs_supplement as kbs
import small_repair_diagnostic as diag
import small_repair_full_audit as full

SCHEMA = "care-m11-mechanism-diagnostic-v1"
CORE = ("extra_source_error_repair", "restored_balanced_damage", "extra_source_correct_damage", "lost_balanced_repair")
KINDS = (*CORE, "wrong_to_other_wrong", "both_correct", "both_wrong_unchanged")
META = ("query_seed", "seed", "class_id", "kind", "group", "selector", "relation_support")
AUDIT_FILES = {"resolved_config.json", "reconciliation.json", "paired_classes.csv", "class_by_query.csv", "class_summary.csv",
    "paired_cohorts.csv", "cohort_summary.csv", "paired_edges.csv", "edge_summary.csv", "group_contributions.csv",
    "group_summary.csv", "query_classes.csv", "query_relations.csv", "relation_cohorts.csv", "relation_summary.csv"}
GEOMETRY_FILES = {"resolved_config.json", "focus.json", "query_geometry.csv", "geometry_summary.csv", "pool_geometry.csv",
    "probe_rows.csv", "probe_classes.csv", "probe_summary.csv"}
REPORT_FILES = {"resolved_config.json", "report.md"}


def inputs(args):
    source = Path(args.month_dir).resolve()
    saved = json.loads((source / "manifest.json").read_text())["protocol"]
    ma = month.parser().parse_args(["evaluate", "--source-dir", args.source_dir, "--source-audit-dir", args.source_audit_dir,
        "--protection-dir", args.protection_dir, "--ablation-dir", args.ablation_dir, "--replication-dir", args.replication_dir,
        "--source-cache-dir", args.source_cache_dir, "--cache-dir", args.cache_dir, "--checkpoint", args.checkpoint,
        "--output-dir", str(source), "--threads", str(saved["source_protocol"]["threads"])])
    if args.data_dir:
        ma.data_dir = args.data_dir
    pc = month.parent_context(ma)
    mc = month.inputs(ma, pc)
    robust.root_manifest(mc)
    choices = month.selections(mc)
    for c in choices:
        ab.completed(robust.run_path(mc, c), robust.run_signature(mc, c), ab.RUN_ARTIFACTS)
    eval_sig = {"protocol": mc["identity"], "stage": "evaluate",
        "selection_sha256": kbs.file_sha(source / "selection/complete.json"),
        "runs": {str(robust.run_path(mc, c).relative_to(source)):
            kbs.file_sha(robust.run_path(mc, c) / "complete.json") for c in choices}}
    ab.completed(source / "evaluation", eval_sig, month.EVAL_ARTIFACTS)
    for old in (args.month_dir, args.cache_dir, args.source_cache_dir, args.checkpoint, args.source_dir,
        args.source_audit_dir, args.protection_dir, args.ablation_dir, args.replication_dir,
        mc["info"]["spec"]["data_config"]["data_dir"]):
        month.separate(args.output_dir, old)
    identity = {"schema": SCHEMA, "month_protocol": mc["identity"],
        "source_evaluation_sha256": kbs.file_sha(source / "evaluation/complete.json"), "source_runs": eval_sig["runs"],
        "device": "cpu", "threads": args.threads, "inference_batch_size": args.inference_batch_size,
        "probe_cap_per_class_outcome_fit": args.probe_cap, "probe_seed": args.probe_seed, "focus_top_k": args.top_k,
        "focus": "all M11 source low-F1 classes plus retrospective top F1/negative-flip damage classes",
        "geometry": "same source-predicted-class candidate-pool nearest squared distance on unit-normalized features",
        "primary_unit": "training-seed means within query seed; all five query-seed means; missing edge counts zero",
        "status": "retrospective diagnosis only; no new query policy, labels, fitting or causal identification",
        "code": {p.name: kbs.file_sha(p) for p in (Path(__file__), Path(__file__).with_name("run_care_m11_mechanism_diagnostic.sh"))}}
    return dict(out=Path(args.output_dir).resolve(), identity=identity, source=source, mc=mc, choices=choices,
                p=mc["protocol"], query_seeds=mc["identity"]["fresh_query_seeds"], seeds=mc["protocol"]["seeds"])


def signature(ctx, stage):
    value = {"protocol": ctx["identity"], "stage": stage}
    if stage in ("geometry", "report"):
        value["audit_sha256"] = kbs.file_sha(ctx["out"] / "audit/complete.json")
    if stage == "report":
        value["geometry_sha256"] = kbs.file_sha(ctx["out"] / "geometry/complete.json")
    return value


def root_manifest(ctx, create=False):
    ab.root_manifest(ctx, create)


def masks(y, source, balanced, coverage):
    b, c, s = balanced == y, coverage == y, source == y
    return {"extra_source_error_repair": ~s & ~b & c,
        "restored_balanced_damage": s & ~b & c,
        "extra_source_correct_damage": s & b & ~c,
        "lost_balanced_repair": ~s & b & ~c,
        "wrong_to_other_wrong": ~b & ~c & (balanced != coverage),
        "both_correct": b & c,
        "both_wrong_unchanged": ~b & ~c & (balanced == coverage)}


def pair_classes(y, source, b, c, count, low):
    bc, cc, sc = [kbs.confusion(y, x, count) for x in (b, c, source)]
    bp, br, bf, support, bpred = kbs.class_stats(bc)
    cp, cr, cf, _, cpred = kbs.class_stats(cc)
    _, _, sf, _, _ = kbs.class_stats(sc)
    bm, cm = np.diag(bc), np.diag(cc)
    cohorts = masks(y, source, b, c)
    counts = {kind: np.bincount(y[m], minlength=count) for kind, m in cohorts.items()}
    if sum(int(v.sum()) for v in counts.values()) != len(y):
        raise ValueError("Paired outcome partition is not exhaustive/disjoint.")
    rows = []
    for cid in range(count):
        nfr = int(counts["extra_source_correct_damage"][cid] - counts["restored_balanced_damage"][cid])
        pfr = int(counts["extra_source_error_repair"][cid] - counts["lost_balanced_repair"][cid])
        rows.append({"class_id": cid, "is_m11_low_f1": cid in low, "support": int(support[cid]), "source_f1": float(sf[cid]),
            "balanced_f1": float(bf[cid]), "coverage_f1": float(cf[cid]), "delta_f1": float(cf[cid] - bf[cid]),
            "delta_precision": float(cp[cid] - bp[cid]), "delta_recall": float(cr[cid] - br[cid]),
            "macro_contribution": float((cf[cid] - bf[cid]) / count),
            "delta_tp": int(cm[cid] - bm[cid]), "delta_fp": int((cpred[cid] - cm[cid]) - (bpred[cid] - bm[cid])),
            "delta_fn": int(bm[cid] - cm[cid]), "extra_negative_flips_net": nfr, "extra_positive_flips_net": pfr,
            **{k: int(v[cid]) for k, v in counts.items()}})
    return rows, cohorts


def pair_edges(y, source, b, c, count):
    changed = b != c
    codes, ns = np.unique(((y[changed].astype(np.int64) * count + source[changed]) * count + b[changed]) * count + c[changed], return_counts=True)
    rows = []
    for code, n in zip(codes, ns):
        dest, code = int(code % count), code // count
        old, code = int(code % count), code // count
        src, truth = int(code % count), int(code // count)
        if dest == truth:
            kind = "restored_balanced_damage" if src == truth else "extra_source_error_repair"
        elif old == truth:
            kind = "extra_source_correct_damage" if src == truth else "lost_balanced_repair"
        else:
            kind = "wrong_to_other_wrong"
        rows.append({"true_class": truth, "source_prediction": src, "balanced_prediction": old,
                     "coverage_prediction": dest, "kind": kind, "count": int(n)})
    return rows


def nested(rows, keys, query_seeds):
    by_query = full.summarize(rows, ("query_seed", *keys))
    metrics = sorted({k for r in rows for k, v in r.items() if k not in (*META, *keys) and
                      (v is None or isinstance(v, (int, float, np.number))) and not isinstance(v, (bool, np.bool_))})
    units = [{**{k: r[k] for k in keys}, **{k: r.get(k + "_mean") for k in metrics}} for r in by_query]
    result = full.summarize(units, keys)
    for row in result:
        row["n_query_seeds"] = row.pop("n_seeds")
        if row["n_query_seeds"] != len(query_seeds):
            raise ValueError("Nested diagnosis summary is missing a planned query unit.")
        members = [r for r in by_query if all(r[k] == row[k] for k in keys)]
        for metric in ("delta_f1", "extra_negative_flips_net"):
            if metric in metrics:
                row[metric + "_positive_query_seeds"] = sum(r.get(metric + "_mean", 0) > 0 for r in members)
                row[metric + "_negative_query_seeds"] = sum(r.get(metric + "_mean", 0) < 0 for r in members)
    return by_query, result


def sparse_edge_summary(rows, query_seeds, seeds):
    keys = ("true_class", "source_prediction", "balanced_prediction", "coverage_prediction", "kind")
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        grid = groups.setdefault(key, {})
        cell = (r["query_seed"], r["seed"])
        if cell in grid:
            raise ValueError("Duplicate paired edge cell.")
        grid[cell] = r["count"]
    result = []
    for key, grid in sorted(groups.items()):
        values = [statistics.mean(grid.get((qs, s), 0) for s in seeds) for qs in query_seeds]
        result.append({**dict(zip(keys, key)), "count_mean": statistics.mean(values),
            "count_sd": statistics.stdev(values) if len(values) > 1 else None, "n_query_seeds": len(values),
            "queries_with_edge": sum(v > 0 for v in values)})
    return result


def query_evidence(ctx, target, low):
    classes, relations, matrices = [], [], {}
    count = ctx["p"]["expected_classes"]
    for qs in ctx["query_seeds"]:
        for selector in robust.SELECTORS:
            choice = next(c for c in ctx["choices"] if c["query_seed"] == qs and c["selector"] == selector)
            ids = np.asarray(choice["rows"])
            y, pred = diag.numpy(target["labels"][ids]), target["logits"][ids].argmax(1).numpy()
            sup = np.isin(ids, choice["supplement_rows"])
            matrix = kbs.confusion(pred, y, count)  # source -> paid truth
            matrix_s = kbs.confusion(pred[sup], y[sup], count)
            matrix_r = matrix - matrix_s
            matrices[qs, selector] = (matrix, matrix_s, matrix_r)
            for cid in range(count):
                correct = (y == cid) & (pred == y)
                classes.append({"query_seed": qs, "selector": selector, "class_id": cid, "is_m11_low_f1": cid in low,
                    "paid_total": int((y == cid).sum()), "paid_source_correct": int(correct.sum()),
                    "paid_source_errors": int(((y == cid) & (pred != y)).sum()),
                    "supplement_total": int(((y == cid) & sup).sum()), "supplement_source_correct": int((correct & sup).sum()),
                    "supplement_source_errors": int(((y == cid) & (pred != y) & sup).sum()),
                    "repair_total": int(((y == cid) & ~sup).sum())})
            for a, b in np.argwhere(matrix > 0):
                relations.append({"query_seed": qs, "selector": selector, "source_prediction": int(a), "true_class": int(b),
                    "paid_count": int(matrix[a, b]), "supplement_count": int(matrix_s[a, b]), "repair_count": int(matrix_r[a, b])})
    return classes, relations, matrices


def saved_arrays(ctx):
    directory = ctx["source"] / "evaluation"
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        wanted = {"row_indices", "labels", "source"} | {f"query{c['query_seed']}_{c['selector']}_seed{c['seed']}" for c in ctx["choices"]}
        if set(data.files) != wanted:
            raise ValueError("Saved M11 prediction keys differ from the complete planned grid.")
        return {key: data[key] for key in data.files}


def validate_arrays(ctx, arrays, target):
    ids = arrays["row_indices"]
    expected = diag.evaluation_indices(ctx["mc"]["identity"]["target_samples"], ctx["choices"], 0, 1729)
    if ids.dtype.kind not in "iu" or not np.array_equal(ids, expected):
        raise ValueError("Saved M11 row order/query exclusion mismatch.")
    count, n = ctx["p"]["expected_classes"], len(ids)
    for key, a in arrays.items():
        if key != "row_indices" and (a.shape != (n,) or a.dtype.kind not in "iu" or np.any((a < 0) | (a >= count))):
            raise ValueError(f"Invalid saved M11 class predictions: {key}")
    if (not np.array_equal(arrays["labels"], diag.numpy(target["labels"][ids])) or
        not np.array_equal(arrays["source"], target["logits"][ids].argmax(1).numpy())):
        raise ValueError("Saved M11 labels/source predictions differ from verified cache.")


def audit(args, ctx):
    out = ctx["out"]
    with kbs.file_lock(out / ".lock"):
        root_manifest(ctx, True)
        directory, sig = out / "audit", signature(ctx, "audit")
        if kbs.is_complete(directory, sig):
            ab.completed(directory, sig, AUDIT_FILES)
            print("M11 paired audit verified; skipping arrays/inference.", flush=True)
            return
        target = diag.load_role(ctx["mc"]["base"], ctx["mc"]["info"], "target")
        arrays = saved_arrays(ctx)
        validate_arrays(ctx, arrays, target)
        ids, y, source = arrays["row_indices"], arrays["labels"], arrays["source"]
        p, count = ctx["p"], ctx["p"]["expected_classes"]
        _, baseline, _ = kbs.compare_predictions(y, source, source, np.ones(len(y), bool), count, p["collapse_classes"], p["stable_classes"], .1, .05)
        low = [r["class_id"] for r in baseline if r["support"] > 0 and r["before_f1"] < .1]
        groups = json.loads((ctx["source"] / "evaluation/evaluation_groups.json").read_text())
        expected_groups = {"m12_prespecified_stress_classes": p["collapse_classes"], "m12_prespecified_stable_classes": p["stable_classes"],
            "m11_low_f1_rule": "supported on common M11 evaluation and Static F1<0.1; evaluation-only", "m11_low_f1_classes": low}
        if groups != expected_groups:
            raise ValueError("Saved offline M11 groups differ from recomputed source-only rule.")
        records, class_records, query_coverage = [], [], []
        for i, c in enumerate(ctx["choices"], 1):
            print(f"Audit head/predictions {i}/{len(ctx['choices'])}: query={c['query_seed']} {c['selector']} train={c['seed']}", flush=True)
            path = robust.run_path(ctx["mc"], c)
            head = diag.head_from_cache(ctx["mc"]["base"])
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
            if any(not torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError("Non-finite source M11 head.")
            trace = json.loads((path / "trace.json").read_text())
            wanted = ab.annotate(head, copy.deepcopy(trace), target, c, count)
            old = ctx["mc"]["parent"]["identity"]["reused_parents"][f"margin/seed_{c['seed']}"]
            old_trace = json.loads((Path(old["directory"]) / "trace.json").read_text())
            if (any(trace[k] != wanted[k] for k in robust.TRACE_FIELDS) or any(trace[k] != old_trace[k] for k in month.STREAM_FIELDS) or
                trace["query_seed"] != c["query_seed"] or trace["month"] != month.TARGET or trace["new_optimizer_steps"] != p["steps"]):
                raise ValueError("Saved M11 trace/paid labels/replay/streams differ.")
            pred = arrays[f"query{c['query_seed']}_{c['selector']}_seed{c['seed']}"]
            actual = full.predict_rows(head, target["features"], ids, args.inference_batch_size)
            mismatch = int((pred != actual).sum())
            if mismatch:
                raise ValueError(f"Saved M11 head/prediction disagreement: {mismatch} rows at {path}")
            metrics, pcs = full.scores(y, source, pred, p, trace)
            chosen = [r for r in pcs if r["class_id"] in low]
            for suffix, column in (("before", "before_f1"), ("after", "after_f1"), ("delta", "delta_f1")):
                metrics["m11_low_f1_macro_" + suffix] = float(np.mean([r[column] for r in chosen])) if chosen else None
            meta = {"query_seed": c["query_seed"], "seed": c["seed"], "selector": c["selector"], "engine": "care"}
            records.append({**meta, **metrics, "new_optimizer_steps": trace["new_optimizer_steps"]})
            class_records.extend({**meta, **r} for r in full.class_flips(y, source, pred, pcs, count))
            query_coverage.extend({**meta, "class_id": cid, "queried_count": q, "source_correct_paid_count": a}
                for cid, (q, a) in enumerate(zip(trace["queried_class_histogram"], trace["source_correct_paid_histogram"])))
        eval_dir = ctx["source"] / "evaluation"
        full.reconcile_csv(eval_dir / "by_seed.csv", records, ("query_seed", "seed", "selector"))
        full.reconcile_csv(eval_dir / "per_class.csv", class_records, ("query_seed", "seed", "selector", "class_id"))
        full.reconcile_csv(eval_dir / "query_coverage.csv", query_coverage, ("query_seed", "seed", "selector", "class_id"))
        pairs = month.paired(records, ctx["mc"])
        qa, bq, arms, primary = month.summaries(records, pairs, ctx["mc"])
        for name, rows, keys in (("per_query_summary.csv", qa, ("query_seed", "selector")),
            ("paired_by_seed.csv", pairs, ("query_seed", "seed", "comparison")), ("paired_by_query.csv", bq, ("query_seed", "comparison")),
            ("fresh_arm_summary.csv", arms, ("selector",)), ("fresh_paired_summary.csv", primary, ("comparison",))):
            full.reconcile_csv(eval_dir / name, rows, keys)
        query_classes, query_relations, matrices = query_evidence(ctx, target, low)
        paired_classes, cohorts, edges, contributions, relation_cohorts = [], [], [], [], []
        for qs in ctx["query_seeds"]:
            for seed in ctx["seeds"]:
                b = arrays[f"query{qs}_mixed_balanced_seed{seed}"]
                c = arrays[f"query{qs}_mixed_coverage_seed{seed}"]
                meta = {"query_seed": qs, "seed": seed}
                pcs, flags = pair_classes(y, source, b, c, count, low)
                paired_classes.extend({**meta, **r} for r in pcs)
                cohorts.extend({**meta, "kind": kind, "count": int(mask.sum())} for kind, mask in flags.items())
                es = pair_edges(y, source, b, c, count)
                if sum(e["count"] for e in es) != int((b != c).sum()):
                    raise ValueError("Paired edge counts disagree with changed predictions.")
                edges.extend({**meta, **e} for e in es)
                for name, members in (("m11_low_f1", low), ("other_classes", sorted(set(range(count)) - set(low)))):
                    contributions.append({**meta, "group": name, "n_classes": len(members),
                        "macro_contribution": sum(pcs[cid]["macro_contribution"] for cid in members),
                        "extra_negative_flips_net": sum(pcs[cid]["extra_negative_flips_net"] for cid in members),
                        "extra_positive_flips_net": sum(pcs[cid]["extra_positive_flips_net"] for cid in members)})
                pair = next(r for r in pairs if r["query_seed"] == qs and r["seed"] == seed and r["comparison"] == "mixed_coverage - mixed_balanced")
                if not np.isclose(sum(r["macro_contribution"] for r in pcs), pair["overall_macro_f1_after"], atol=1e-12):
                    raise ValueError("Class contributions do not sum to original paired Macro-F1 difference.")
                if not np.isclose(sum(r["extra_negative_flips_net"] for r in pcs) / int((source == y).sum()), pair["negative_flip_rate"], atol=1e-12):
                    raise ValueError("Class damage counts do not reconcile with original negative-flip difference.")
                bm, cm = matrices[qs, "mixed_balanced"][1], matrices[qs, "mixed_coverage"][1]
                for kind in ("extra_source_error_repair", "extra_source_correct_damage"):
                    bins = {name: 0 for name in ("both", "coverage_only", "balanced_only", "neither")}
                    for e in es:
                        if e["kind"] != kind:
                            continue
                        a, dest = e["source_prediction"], e["coverage_prediction"]
                        sb, sc = a != dest and bm[a, dest] > 0, a != dest and cm[a, dest] > 0
                        label = "both" if sb and sc else "coverage_only" if sc else "balanced_only" if sb else "neither"
                        bins[label] += e["count"]
                    relation_cohorts.extend({**meta, "kind": kind, "relation_support": name, "count": n} for name, n in bins.items())
        class_by_query, class_summary = nested(paired_classes, ("class_id",), ctx["query_seeds"])
        outputs = {"paired_classes.csv": paired_classes, "class_by_query.csv": class_by_query, "class_summary.csv": class_summary,
            "paired_cohorts.csv": cohorts, "cohort_summary.csv": nested(cohorts, ("kind",), ctx["query_seeds"])[1],
            "paired_edges.csv": edges, "edge_summary.csv": sparse_edge_summary(edges, ctx["query_seeds"], ctx["seeds"]),
            "group_contributions.csv": contributions, "group_summary.csv": nested(contributions, ("group",), ctx["query_seeds"])[1],
            "query_classes.csv": query_classes, "query_relations.csv": query_relations,
            "relation_cohorts.csv": relation_cohorts, "relation_summary.csv": nested(relation_cohorts, ("kind", "relation_support"), ctx["query_seeds"])[1]}
        kbs.atomic_json(directory / "resolved_config.json", sig)
        kbs.atomic_json(directory / "reconciliation.json", {"heads_checked": len(ctx["choices"]), "prediction_mismatches": 0,
            "evaluation_rows": len(ids), "m11_low_f1_classes": low, "csv_metrics_reconciled": True,
            "primary_pairs": primary, "group_macro_contribution_sum_checked": True, "negative_flip_count_sum_checked": True})
        for name, rows in outputs.items():
            kbs.write_csv(directory / name, rows)
        kbs.finish(directory, sig, sorted(AUDIT_FILES))
        print("M11 paired audit complete: saved heads/predictions/metrics reconciled.", flush=True)


def read_numeric(path):
    rows = kbs.read_csv(path)
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = None
            elif v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
    return rows


def unit(x):
    return F.normalize(x.double(), dim=1).numpy()


def nearest_squared(x, anchors):
    if not len(anchors):
        return None
    # Exact bounded query/candidate distances; clip roundoff at zero.
    return np.maximum(0, ((x * x).sum(1)[:, None] + (anchors * anchors).sum(1)[None, :] - 2 * x @ anchors.T).min(1))


def query_geometry(ctx, target, low):
    registries = json.loads((ctx["source"] / "selection/candidate_pools.json").read_text())
    rows, pool_rows = [], []
    for qs in ctx["query_seeds"]:
        reg = registries[str(qs)]
        for selector in robust.SELECTORS:
            choice = next(c for c in ctx["choices"] if c["query_seed"] == qs and c["selector"] == selector)
            ids, repair = np.asarray(choice["supplement_rows"]), np.asarray(choice["repair_rows"])
            y = diag.numpy(target["labels"][ids])
            logits = target["logits"][ids]
            pred = logits.argmax(1).numpy()
            top = logits.topk(2, dim=1).values
            all_y = diag.numpy(target["labels"][choice["rows"]])
            repair_y = set(diag.numpy(target["labels"][repair]).tolist())
            matrices = kbs.confusion(pred, y, ctx["p"]["expected_classes"])
            margins, maxdist, distances = [], [], []
            for cid, pool in reg["pools"].items():
                cid = int(cid)
                selected = ids[pred == cid]
                d = nearest_squared(unit(target["features"][pool]), unit(target["features"][selected]))
                if d is not None:
                    distances.extend(d.tolist())
                    margins.append(float(d.mean()))
                    maxdist.append(float(d.max()))
                pool_rows.append({"query_seed": qs, "selector": selector, "class_id": cid, "candidate_count": len(pool),
                    "quota": reg["quotas"][str(cid)], "pool_mean_nearest_d2": float(d.mean()) if d is not None else None,
                    "pool_p90_nearest_d2": float(np.quantile(d, .9)) if d is not None else None,
                    "pool_max_nearest_d2": float(d.max()) if d is not None else None})
            rows.append({"query_seed": qs, "selector": selector, "supplement_count": len(ids),
                "supplement_true_classes": len(set(y.tolist())), "all_paid_true_classes": len(set(all_y.tolist())),
                "new_true_classes_vs_repair": len(set(y.tolist()) - repair_y),
                "supplement_source_errors": int((pred != y).sum()), "supplement_source_correct": int((pred == y).sum()),
                "supplement_low_f1_labels": int(np.isin(y, low).sum()),
                "supplement_low_f1_source_errors": int(((pred != y) & np.isin(y, low)).sum()),
                "supplement_error_relations": int(((matrices > 0) & ~np.eye(len(matrices), dtype=bool)).sum()),
                "supplement_mean_raw_margin": float((top[:, 0] - top[:, 1]).mean()),
                "pool_mean_nearest_d2_class_balanced": float(np.mean(margins)) if margins else None,
                "pool_mean_nearest_d2_row_weighted": float(np.mean(distances)) if distances else None,
                "pool_mean_max_d2_class_balanced": float(np.mean(maxdist)) if maxdist else None,
                "candidate_classes_with_quota": len(margins), "candidate_classes_without_quota": len(reg["pools"]) - len(margins)})
        print(f"M11 paid-query geometry: query_seed={qs} complete.", flush=True)
    summary = full.summarize([{k: v for k, v in r.items() if k != "query_seed"} for r in rows], ("selector",))
    for r in summary:
        r["n_query_seeds"] = r.pop("n_seeds")
    return rows, pool_rows, summary


def boundary_terms(x, truth, rival, balanced, coverage):
    """Exact algebra on frozen linear heads; no causal attribution to query rows."""
    bw, cw = balanced["weight"].double().numpy(), coverage["weight"].double().numpy()
    bb, cb = balanced["bias"].double().numpy(), coverage["bias"].double().numpy()
    xx = x.double().numpy()
    gap_b = (xx * (bw[truth] - bw[rival])).sum(1) + bb[truth] - bb[rival]
    gap_c = (xx * (cw[truth] - cw[rival])).sum(1) + cb[truth] - cb[rival]
    feature = (xx * ((cw[truth] - cw[rival]) - (bw[truth] - bw[rival]))).sum(1)
    bias = (cb[truth] - cb[rival]) - (bb[truth] - bb[rival])
    if not np.allclose(gap_c - gap_b, feature + bias, atol=1e-10, rtol=1e-10):
        raise ValueError("Frozen linear boundary decomposition mismatch.")
    return dict(balanced_gap=gap_b, coverage_gap=gap_c, gap_shift=gap_c - gap_b, feature_term=feature, bias_term=bias)


def geometry(args, ctx):
    root_manifest(ctx)
    ab.completed(ctx["out"] / "audit", signature(ctx, "audit"), AUDIT_FILES)
    with kbs.file_lock(ctx["out"] / ".lock"):
        directory, sig = ctx["out"] / "geometry", signature(ctx, "geometry")
        if kbs.is_complete(directory, sig):
            ab.completed(directory, sig, GEOMETRY_FILES)
            print("M11 geometry diagnostic verified; skipping tensors.", flush=True)
            return
        audit_info = json.loads((ctx["out"] / "audit/reconciliation.json").read_text())
        class_summary = read_numeric(ctx["out"] / "audit/class_summary.csv")
        low = audit_info["m11_low_f1_classes"]
        harmed = sorted((r for r in class_summary if r["delta_f1_mean"] < 0), key=lambda r: (r["delta_f1_mean"], r["class_id"]))[:args.top_k]
        flipped = sorted((r for r in class_summary if r["extra_negative_flips_net_mean"] > 0),
                         key=lambda r: (-r["extra_negative_flips_net_mean"], r["class_id"]))[:args.top_k]
        focus = sorted(set(low) | {r["class_id"] for r in harmed + flipped})
        target = diag.load_role(ctx["mc"]["base"], ctx["mc"]["info"], "target")
        arrays = saved_arrays(ctx)
        validate_arrays(ctx, arrays, target)
        qrows, pool_rows, qsummary = query_geometry(ctx, target, low)
        ids, y, source = arrays["row_indices"], arrays["labels"], arrays["source"]
        evidence, probe_classes = [], []
        for qs in ctx["query_seeds"]:
            choices = {sel: next(c for c in ctx["choices"] if c["query_seed"] == qs and c["selector"] == sel) for sel in robust.SELECTORS}
            neighbors = {}
            for sel, choice in choices.items():
                for part in ("rows", "supplement_rows"):
                    query_ids = np.asarray(choice[part])
                    qy = diag.numpy(target["labels"][query_ids])
                    for cid in focus:
                        neighbors[sel, part, cid] = unit(target["features"][query_ids[qy == cid]])
            for seed in ctx["seeds"]:
                b, c = arrays[f"query{qs}_mixed_balanced_seed{seed}"], arrays[f"query{qs}_mixed_coverage_seed{seed}"]
                flags = masks(y, source, b, c)
                states = {sel: torch.load(robust.run_path(ctx["mc"], {"query_seed": qs, "seed": seed, "selector": sel}) / "head.pt",
                    map_location="cpu", weights_only=True) for sel in robust.SELECTORS}
                for cid in focus:
                    for kind_i, kind in enumerate(CORE):
                        candidates = np.flatnonzero((y == cid) & flags[kind])
                        rng = np.random.default_rng(args.probe_seed + qs * 17 + seed * 100003 + cid * 997 + kind_i * 13)
                        positions = np.sort(rng.choice(candidates, min(len(candidates), args.probe_cap), replace=False))
                        row_ids = ids[positions]
                        x = target["features"][row_ids]
                        xx = unit(x)
                        rival = b[positions] if kind in CORE[:2] else c[positions]
                        terms = boundary_terms(x, y[positions], rival, states["mixed_balanced"], states["mixed_coverage"])
                        distances = {}
                        for part, label in (("rows", "paid"), ("supplement_rows", "supplement")):
                            for sel, short in (("mixed_balanced", "balanced"), ("mixed_coverage", "coverage")):
                                distances[short + "_" + label + "_same_truth_d2"] = nearest_squared(xx, neighbors[sel, part, cid])
                            a, z = distances["balanced_" + label + "_same_truth_d2"], distances["coverage_" + label + "_same_truth_d2"]
                            distances[label + "_d2_difference"] = z - a if a is not None and z is not None else None
                        sampled = []
                        for i, row_id in enumerate(row_ids):
                            r = {"query_seed": qs, "seed": seed, "class_id": cid, "kind": kind, "row_index": int(row_id),
                                "source_prediction": int(source[positions[i]]), "balanced_prediction": int(b[positions[i]]),
                                "coverage_prediction": int(c[positions[i]]), "rival_class": int(rival[i]),
                                **{k: float(v[i]) for k, v in terms.items()},
                                **{k: float(v[i]) if v is not None else None for k, v in distances.items()}}
                            sampled.append(r)
                            evidence.append(r)
                        continuous = list(terms) + list(distances)
                        probe_classes.append({"query_seed": qs, "seed": seed, "class_id": cid, "kind": kind,
                            "population_count": len(candidates), "sample_count": len(sampled),
                            "balanced_paid_same_truth_count": len(neighbors["mixed_balanced", "rows", cid]),
                            "coverage_paid_same_truth_count": len(neighbors["mixed_coverage", "rows", cid]),
                            "balanced_supplement_same_truth_count": len(neighbors["mixed_balanced", "supplement_rows", cid]),
                            "coverage_supplement_same_truth_count": len(neighbors["mixed_coverage", "supplement_rows", cid]),
                            **{k: statistics.mean(r[k] for r in sampled) if sampled and sampled[0][k] is not None else None for k in continuous}})
                print(f"M11 retrospective probes: query={qs} train={seed}, focus classes={len(focus)} complete.", flush=True)
        _, probe_summary = nested(probe_classes, ("class_id", "kind"), ctx["query_seeds"])
        kbs.atomic_json(directory / "resolved_config.json", sig)
        kbs.atomic_json(directory / "focus.json", {"classes": focus, "all_source_low_f1": low,
            "top_f1_damage": [r["class_id"] for r in harmed], "top_extra_negative_flips": [r["class_id"] for r in flipped],
            "selection": "retrospective diagnostic focus; not a query policy or a representative sample of all classes",
            "sampling": "uniform within each class/outcome/fit, capped; continuous geometry summaries conditional on observed cohorts",
            "nearest_neighbor": "same true class among already paid labels; absent neighbor is undefined, not zero",
            "sample_id_definition": ctx["mc"]["info"]["sample_id_definition"], "cache_fingerprint": ctx["mc"]["info"]["fingerprint"]})
        for name, rows in (("query_geometry.csv", qrows), ("pool_geometry.csv", pool_rows), ("geometry_summary.csv", qsummary),
            ("probe_rows.csv", evidence), ("probe_classes.csv", probe_classes), ("probe_summary.csv", probe_summary)):
            kbs.write_csv(directory / name, rows)
        kbs.finish(directory, sig, sorted(GEOMETRY_FILES))
        print("M11 query geometry/boundary probes complete; no fitting or new labels.", flush=True)


def render_report(ctx):
    audit_dir, geo_dir = ctx["out"] / "audit", ctx["out"] / "geometry"
    rec = json.loads((audit_dir / "reconciliation.json").read_text())
    classes = read_numeric(audit_dir / "class_summary.csv")
    groups = read_numeric(audit_dir / "group_summary.csv")
    cohorts = read_numeric(audit_dir / "cohort_summary.csv")
    edges = read_numeric(audit_dir / "edge_summary.csv")
    query_classes = read_numeric(audit_dir / "query_classes.csv")
    geometry_rows = read_numeric(geo_dir / "geometry_summary.csv")
    relations = read_numeric(audit_dir / "relation_summary.csv")
    probes = read_numeric(geo_dir / "probe_summary.csv")
    def fmt(value, places=6):
        return "NA" if value is None else f"{value:.{places}f}"
    pair = next(r for r in rec["primary_pairs"] if r["comparison"] == "mixed_coverage - mixed_balanced")
    lines = ["# CARE M11 特征覆盖：恢复与误伤诊断", "",
        f"CPU；新增训练 0，新增查询 0；对账 {rec['heads_checked']} 个头、{rec['evaluation_rows']:,} 个共同评价行，预测不一致 {rec['prediction_mismatches']}。",
        f"coverage−balanced Macro-F1={fmt(pair['overall_macro_f1_after_mean'])} ± {fmt(pair['overall_macro_f1_after_sd'])}；负翻转率差={fmt(pair['negative_flip_rate_mean'])}。",
        "以下主数值先平均同一查询种子的训练重复，再汇总全部五个查询种子；± 为查询种子 SD。", "",
        "## 总体增益来自哪里", "", "| 分组 | 类数 | 对总体 Macro-F1 的贡献 | 净增加负翻转样本 | 净增加正翻转样本 |",
        "|---|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| {r['group']} | {r['n_classes_mean']:.0f} | {fmt(r['macro_contribution_mean'])} | {fmt(r['extra_negative_flips_net_mean'], 1)} | {fmt(r['extra_positive_flips_net_mean'], 1)} |")
    lines += ["", "## 配对恢复与损伤：样本数", "", "| 类别 | 平均样本数 ± SD |", "|---|---:|"]
    names = {"extra_source_error_repair": "额外救回源模型错分", "restored_balanced_damage": "恢复均衡头造成的误伤",
        "extra_source_correct_damage": "新增源正确样本误伤", "lost_balanced_repair": "丢失均衡头已救回样本",
        "wrong_to_other_wrong": "两头都错且错分去向改变", "both_correct": "两头都正确", "both_wrong_unchanged": "两头都错且预测相同"}
    for r in cohorts:
        lines.append(f"| {names[r['kind']]} | {fmt(r['count_mean'], 1)} ± {fmt(r['count_sd'], 1)} |")
    qlookup = {(r["query_seed"], r["selector"], r["class_id"]): r for r in query_classes}
    def paid(cid, selector, key):
        return statistics.mean(qlookup[qs, selector, cid][key] for qs in ctx["query_seeds"])
    def class_table(title, values):
        lines.extend(["", title, "", "| 类ID | 源 F1 | 均衡 F1 | 覆盖 F1 | F1差 | ΔRecall | ΔPrecision | ΔTP | ΔFP | 补充标签 B/C | F1正/负种子 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for r in values:
            cid = r["class_id"]
            lines.append(f"| {cid} | {fmt(r['source_f1_mean'], 3)} | {fmt(r['balanced_f1_mean'], 3)} | {fmt(r['coverage_f1_mean'], 3)} | " +
                " | ".join(fmt(r[k + "_mean"], 3 if k.startswith("delta_") and k not in ("delta_tp", "delta_fp") else 1)
                    for k in ("delta_f1", "delta_recall", "delta_precision", "delta_tp", "delta_fp")) +
                f" | {paid(cid, 'mixed_balanced', 'supplement_total'):.1f}/{paid(cid, 'mixed_coverage', 'supplement_total'):.1f} | {r['delta_f1_positive_query_seeds']}/{r['delta_f1_negative_query_seeds']} |")
    k = ctx["identity"]["focus_top_k"]
    class_table("## 低 F1 类：按额外恢复排序", sorted((r for r in classes if r["class_id"] in rec["m11_low_f1_classes"]), key=lambda r: -r["delta_f1_mean"]))
    class_table("## F1 损伤最大的类别", sorted((r for r in classes if r["delta_f1_mean"] < 0), key=lambda r: r["delta_f1_mean"])[:k])
    lines += ["", "## 新增负翻转集中在哪些真实类", "", "| 真实类ID | 净增加负翻转 | 新增误伤 | 恢复旧误伤 | F1差 | ΔFP |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in classes if r["extra_negative_flips_net_mean"] > 0), key=lambda r: -r["extra_negative_flips_net_mean"])[:k]:
        lines.append(f"| {r['class_id']} | " + " | ".join(fmt(r[x + "_mean"], 3 if x == "delta_f1" else 1) for x in
            ("extra_negative_flips_net", "extra_source_correct_damage", "restored_balanced_damage", "delta_f1", "delta_fp")) + " |")
    lines += ["", "## 新增源正确误伤：主要错分去向", "", "| 真实类/源预测 | 均衡预测 | 覆盖预测 | 平均样本数 | 出现查询数 |",
              "|---|---:|---:|---:|---:|"]
    for r in sorted((r for r in edges if r["kind"] == "extra_source_correct_damage"), key=lambda r: -r["count_mean"])[:k]:
        lines.append(f"| {r['true_class']} | {r['balanced_prediction']} | {r['coverage_prediction']} | {r['count_mean']:.1f} | {r['queries_with_edge']}/5 |")
    lines += ["", "## 补充查询与候选池几何（五个查询种子均值）", "",
        "| 查询 | 补充真实类数 | 新增真实类数 | 源错分标签 | 低F1标签 | 低F1错分标签 | 池平均最近距离² | 池平均最大距离² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in geometry_rows:
        lines.append(f"| {r['selector']} | " + " | ".join(fmt(r.get(x + "_mean"), 4 if x.startswith("pool_") else 1) for x in
            ("supplement_true_classes", "new_true_classes_vs_repair", "supplement_source_errors", "supplement_low_f1_labels",
             "supplement_low_f1_source_errors", "pool_mean_nearest_d2_class_balanced", "pool_mean_max_d2_class_balanced")) + " |")
    lines += ["", "## 已付费补充查询的错误关系支持", "", "| 配对事件 | 支持情况 | 平均样本数 |", "|---|---|---:|"]
    for r in relations:
        lines.append(f"| {names[r['kind']]} | {r['relation_support']} | {r['count_mean']:.1f} |")
    lines += ["", "## 抽样边界与同真类近邻：低 F1 类额外救回", "",
        "| 类ID | 每头平均事件数 | 每头平均抽样数 | 边界移动 | 特征项 | 偏置项 | 到付费近邻距离差 | 有定义查询数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in probes:
        if r["class_id"] in rec["m11_low_f1_classes"] and r["kind"] == "extra_source_error_repair" and r["population_count_mean"] > 0:
            lines.append(f"| {r['class_id']} | " + " | ".join(fmt(r.get(x + "_mean"), 4 if x not in ("population_count", "sample_count") else 1)
                for x in ("population_count", "sample_count", "gap_shift", "feature_term", "bias_term", "paid_d2_difference")) +
                f" | {r.get('paid_d2_difference_valid_seeds', 0)}/5 |")
    lines += ["", "## 判读与数据范围", "",
        "- F1差=覆盖−均衡。ΔTP/ΔFP/ΔRecall/ΔPrecision 联合判断；真实类被翻错与错误流入该类造成的精度下降要分开。",
        "- 净新增负翻转=新增源正确误伤−恢复均衡误伤；净新增正翻转=额外源错分修复−丢失均衡已修复。",
        "- 类贡献是该类 F1差÷全部源类别数；低F1组与其余类贡献之和对账到总体 Macro-F1差。",
        "- 补充查询的 both/coverage_only/balanced_only/neither 表示相应源预测→覆盖预测的付费错误关系是否被两臂补充查询观察到；相关支持不是因果归因。",
        "- 几何池仅为各预测类的高间隔 capped reservoir，不能推断整个目标分布；距离在单位特征上计算，越小覆盖越近。",
        "- 探针聚焦事后发现的低F1与损伤类，每类/事件/拟合均匀抽样并限量；保存每个行号及母体/样本数，不是全体样本的几何统计。",
        "- 线性边界移动精确分为特征权重项与偏置项，只说明冻结头如何改变样本决策，不证明某个查询导致该变化。",
        "- 同真类近邻使用已付费查询；没有同类邻居时距离为 NA，不填零。不一致事件的条件近邻统计有选择偏差。",
        "- 稀疏错分边在未出现的训练/查询种子补零；连续探针量无事件时保持 NA，有效重复数另列。",
        "- 只使用已完成 M11 的共同评价集；全量未查询真值仅作事后诊断，不构建/训练新策略或选择最优参数。",
        "- 类别以已绑定的整数ID报告，没有经核验的应用名称映射；M12 预定类组不等同于 M11 崩溃/稳定类。",
        "- 这些是机制线索；要证明查询设计因果有效，后续仍需预算/池/训练配对的干预消融。", ""]
    return "\n".join(lines)


def report(args, ctx):
    root_manifest(ctx)
    ab.completed(ctx["out"] / "audit", signature(ctx, "audit"), AUDIT_FILES)
    ab.completed(ctx["out"] / "geometry", signature(ctx, "geometry"), GEOMETRY_FILES)
    with kbs.file_lock(ctx["out"] / ".lock"):
        directory, sig = ctx["out"] / "report", signature(ctx, "report")
        if kbs.is_complete(directory, sig):
            ab.completed(directory, sig, REPORT_FILES)
            print(f"CARE_M11_MECHANISM_DIAGNOSTIC_COMPLETE: {directory / 'report.md'} (verified; skipped)", flush=True)
            return
        kbs.atomic_json(directory / "resolved_config.json", sig)
        (directory / "report.md").write_text(render_report(ctx))
        kbs.finish(directory, sig, sorted(REPORT_FILES))
        print(f"CARE_M11_MECHANISM_DIAGNOSTIC_COMPLETE: {directory / 'report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "preflight", "audit", "geometry", "report"))
    for name, default in (("source-dir", "small_repair_diagnostic_v1"), ("source-audit-dir", "small_repair_full_audit_v1"),
        ("protection-dir", "care_target_protection_v1"), ("ablation-dir", "care_mixed_query_ablation_v1"),
        ("replication-dir", "care_query_seed_robustness_v1"), ("source-cache-dir", "kbs_supplement_v1/cache"),
        ("cache-dir", "care_m11_cache_v1"), ("month-dir", "care_m11_replication_v1"), ("output-dir", "care_m11_mechanism_diagnostic_v1")):
        p.add_argument("--" + name, default=str(kbs.CORE / "outputs" / default))
    p.add_argument("--checkpoint", default=str(kbs.CORE / "outputs/tls22_cnn/best_model.pt"))
    p.add_argument("--data-dir")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=4096)
    p.add_argument("--probe-cap", type=int, default=64)
    p.add_argument("--probe-seed", type=int, default=8621)
    p.add_argument("--top-k", type=int, default=10)
    return p


def main():
    args = parser().parse_args()
    if min(args.threads, args.inference_batch_size, args.probe_cap, args.top_k) <= 0 or args.probe_seed < 0:
        raise ValueError("Invalid diagnostic settings.")
    torch.set_num_threads(args.threads)
    if args.command == "plan":
        print(json.dumps({"device": "cpu", "new_queries": 0, "new_fits": 0,
            "stages": ["audit all saved heads/predictions/metrics", "paid-query geometry and bounded retrospective probes", "report"],
            "probe_cap_per_class_outcome_fit": args.probe_cap, "status": "mechanism localization, not causal identification"}, indent=2))
        return
    started = time.perf_counter()
    ctx = inputs(args)
    if args.command == "preflight":
        if (ctx["out"] / "manifest.json").exists():
            root_manifest(ctx)
        print(json.dumps({"status": "source M11 stages/hashes verified", "device": "cpu", "heads": len(ctx["choices"]),
            "new_queries": 0, "new_fits": 0}, indent=2))
    else:
        {"audit": audit, "geometry": geometry, "report": report}[args.command](args, ctx)
    print(f"Stage {args.command} wall_s={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
