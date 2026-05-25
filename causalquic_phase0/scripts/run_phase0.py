#!/usr/bin/env python3
"""Phase-0 validation pipeline for QUIC drift-source attribution.

This script is intentionally conservative:
- Source attributions are weak, confidence-scored labels.
- Policy evaluation is held out from attribution by using action outcomes only.
- Unknown remains a first-class outcome when probes are not separable.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse


HANDSHAKE_KEYS = (
    "QUIC",
    "TLS",
    "SNI",
    "USER_AGENT",
    "TOKEN",
    "VERSION",
    "HANDSHAKE",
    "CERT",
)
PROVIDER_KEYS = ("ASN", "COUNTRY", "HOST", "SUBNET", "DST", "CDN", "PROVIDER")
NETWORK_KEYS = ("PPI", "DURATION", "BYTES", "PACKETS", "PKTS", "HIST", "IAT", "RTT", "TIME")
CANDIDATE_COLUMNS = [
    "service",
    "time_bin",
    "prev_time_bin",
    "drift_score",
    "placebo_pvalue",
    "weak_source",
    "confidence",
    "handshake_shift",
    "provider_shift",
    "network_shift",
    "other_shift",
]


def parse_time_column(series: pd.Series) -> pd.Series:
    """Parse timestamps while handling CESNET Unix-second flow times."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        median = float(numeric.dropna().abs().median())
        if median > 1e17:
            return pd.to_datetime(numeric, unit="ns", errors="coerce")
        if median > 1e14:
            return pd.to_datetime(numeric, unit="us", errors="coerce")
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e8:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--input-glob")
    parser.add_argument("--out-dir")
    parser.add_argument("--time-col")
    parser.add_argument("--service-col")
    parser.add_argument("--target-col")
    parser.add_argument("--provider-col")
    parser.add_argument("--country-col")
    parser.add_argument("--events-csv")
    parser.add_argument("--bin-size", default="1D")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--min-flows-per-bin", type=int, default=100)
    parser.add_argument("--max-candidate-windows", type=int, default=500)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--placebo-pvalue-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        for key, value in cfg.items():
            attr = key.replace("-", "_")
            if getattr(args, attr, None) in (None, 0) and value is not None:
                setattr(args, attr, value)

    required = ["input_glob", "out_dir", "time_col", "service_col"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    return args


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported file type: {path}")


def load_data(input_glob: str, sample_rows: int, seed: int) -> pd.DataFrame:
    paths = sorted(glob.glob(input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no files matched {input_glob!r}")

    frames: list[pd.DataFrame] = []
    remaining = sample_rows if sample_rows and sample_rows > 0 else None
    rng = np.random.default_rng(seed)

    for path in paths:
        df = read_table(path)
        if remaining is not None and len(df) > remaining:
            df = df.sample(n=remaining, random_state=int(rng.integers(0, 1_000_000)))
        frames.append(df)
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break

    return pd.concat(frames, ignore_index=True)


def infer_groups(columns: Iterable[str], exclude: set[str]) -> dict[str, list[str]]:
    groups = {"handshake": [], "provider": [], "network": [], "other": []}
    for col in columns:
        if col in exclude:
            continue
        upper = col.upper()
        if any(key in upper for key in HANDSHAKE_KEYS):
            groups["handshake"].append(col)
        elif any(key in upper for key in PROVIDER_KEYS):
            groups["provider"].append(col)
        elif any(key in upper for key in NETWORK_KEYS):
            groups["network"].append(col)
        else:
            groups["other"].append(col)
    return groups


def numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def top_distribution(values: pd.Series, limit: int = 30) -> dict[str, float]:
    clean = values.dropna().astype(str)
    if clean.empty:
        return {}
    counts = clean.value_counts().head(limit)
    total = counts.sum()
    return {str(k): float(v / total) for k, v in counts.items()}


def dist_shift(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    pa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    pa = pa / max(pa.sum(), 1e-12)
    pb = pb / max(pb.sum(), 1e-12)
    return float(jensenshannon(pa, pb, base=2.0) ** 2)


def summarize_window(df: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for col in cols:
        if col not in df:
            continue
        numeric = numeric_safe(df[col])
        if numeric.notna().mean() >= 0.8:
            out[f"{col}__mean"] = float(numeric.mean())
            out[f"{col}__std"] = float(numeric.std(ddof=0))
        else:
            out[f"{col}__dist"] = top_distribution(df[col])
    return out


def build_window_metrics(
    df: pd.DataFrame,
    time_col: str,
    service_col: str,
    groups: dict[str, list[str]],
    bin_size: str,
    min_flows: int,
) -> pd.DataFrame:
    df = df.copy()
    df["_time"] = parse_time_column(df[time_col])
    df = df.dropna(subset=["_time", service_col])
    df["_bin"] = df["_time"].dt.floor(bin_size)

    rows = []
    group_cols = [service_col, "_bin"]
    for (service, time_bin), sub in df.groupby(group_cols, dropna=False):
        if len(sub) < min_flows:
            continue
        row: dict[str, object] = {"service": service, "time_bin": time_bin, "n_flows": len(sub)}
        for group, cols in groups.items():
            row[f"{group}_n_cols"] = len(cols)
            row[f"{group}_summary"] = summarize_window(sub, cols)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["service", "time_bin", "n_flows"])
    return pd.DataFrame(rows).sort_values(["service", "time_bin"]).reset_index(drop=True)


def group_shift(prev_summary: dict[str, object], cur_summary: dict[str, object]) -> float:
    scores = []
    for key in sorted(set(prev_summary) | set(cur_summary)):
        a = prev_summary.get(key)
        b = cur_summary.get(key)
        if key.endswith("__dist"):
            scores.append(dist_shift(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}))
        else:
            try:
                av = float(a)
                bv = float(b)
            except (TypeError, ValueError):
                continue
            denom = max(abs(av), abs(bv), 1.0)
            scores.append(min(abs(bv - av) / denom, 10.0))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def empirical_pvalue(score: float, placebo: list[float]) -> float:
    if not placebo:
        return 1.0
    ge = sum(1 for value in placebo if value >= score)
    return float((ge + 1) / (len(placebo) + 1))


def source_from_scores(
    scores: dict[str, float],
    pvalue: float,
    confidence_threshold: float,
    pvalue_threshold: float,
) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] <= 0:
        return "unknown", 0.0
    top_source, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    separability = max(top - second, 0.0) / max(top, 1e-9)
    evidence = min(-math.log10(max(pvalue, 1e-12)) / 3.0, 1.0)
    confidence = float(0.65 * separability + 0.35 * evidence)
    if pvalue > pvalue_threshold or confidence < 0.35:
        return "unknown", confidence
    if confidence < confidence_threshold:
        return "multi_or_weak", confidence
    return top_source, confidence


def find_candidate_windows(
    metrics: pd.DataFrame,
    max_candidates: int,
    confidence_threshold: float,
    pvalue_threshold: float,
) -> pd.DataFrame:
    if metrics.empty or "service" not in metrics.columns:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    rows = []
    placebo_by_service: dict[object, list[float]] = {}

    for service, sub in metrics.groupby("service"):
        sub = sub.sort_values("time_bin").reset_index(drop=True)
        placebo_scores = []
        for i in range(len(sub) - 3):
            a = sub.iloc[i]
            b = sub.iloc[i + 3]
            scores = {
                group: group_shift(a.get(f"{group}_summary", {}), b.get(f"{group}_summary", {}))
                for group in ("handshake", "provider", "network", "other")
            }
            placebo_scores.append(max(scores.values()) if scores else 0.0)
        placebo_by_service[service] = placebo_scores

    for service, sub in metrics.groupby("service"):
        sub = sub.sort_values("time_bin").reset_index(drop=True)
        placebo = placebo_by_service.get(service, [])
        for i in range(1, len(sub)):
            prev = sub.iloc[i - 1]
            cur = sub.iloc[i]
            scores = {
                group: group_shift(prev.get(f"{group}_summary", {}), cur.get(f"{group}_summary", {}))
                for group in ("handshake", "provider", "network", "other")
            }
            drift_score = max(scores.values()) if scores else 0.0
            pvalue = empirical_pvalue(drift_score, placebo)
            source, confidence = source_from_scores(scores, pvalue, confidence_threshold, pvalue_threshold)
            rows.append(
                {
                    "service": service,
                    "time_bin": cur["time_bin"],
                    "prev_time_bin": prev["time_bin"],
                    "drift_score": drift_score,
                    "placebo_pvalue": pvalue,
                    "weak_source": source,
                    "confidence": confidence,
                    **{f"{k}_shift": v for k, v in scores.items()},
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return (
        out.sort_values(["confidence", "drift_score"], ascending=False)
        .head(max_candidates)
        .reset_index(drop=True)
    )


def records_for_hashing(df: pd.DataFrame, cols: list[str]) -> list[dict[str, float]]:
    records = []
    for _, row in df[cols].iterrows():
        rec: dict[str, float] = {}
        for col, value in row.items():
            if pd.isna(value):
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                rec[col] = float(numeric)
            else:
                rec[f"{col}={str(value)[:80]}"] = 1.0
        records.append(rec)
    return records


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    hasher = FeatureHasher(n_features=2**18, input_type="dict", alternate_sign=False)
    x_train = hasher.transform(records_for_hashing(train, cols))
    x_test = hasher.transform(records_for_hashing(test, cols))

    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def evaluate_actions(
    df: pd.DataFrame,
    time_col: str,
    service_col: str,
    target_col: str,
    groups: dict[str, list[str]],
    candidates: pd.DataFrame,
    bin_size: str,
    seed: int,
) -> pd.DataFrame:
    if not target_col or target_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["_time"] = parse_time_column(work[time_col])
    work = work.dropna(subset=["_time", target_col, service_col])
    work["_bin"] = work["_time"].dt.floor(bin_size)
    if work["_bin"].nunique() < 4 or work[target_col].nunique() < 2:
        return pd.DataFrame()

    split_time = sorted(work["_bin"].dropna().unique())[int(work["_bin"].nunique() * 0.6)]
    train = work[work["_bin"] <= split_time]
    test = work[work["_bin"] > split_time]
    if train.empty or test.empty:
        return pd.DataFrame()

    seen_labels = set(train[target_col].astype(str))
    test = test[test[target_col].astype(str).isin(seen_labels)]
    if test.empty:
        return pd.DataFrame()

    exclude = {time_col, service_col, target_col, "_time", "_bin"}
    all_feature_cols = [col for col in work.columns if col not in exclude]
    action_drops = {
        "full": [],
        "no_handshake": groups["handshake"],
        "no_provider": groups["provider"],
        "no_network": groups["network"],
    }
    action_map = {
        "handshake": "no_handshake",
        "provider": "no_provider",
        "network": "no_network",
        "other": "full",
        "multi_or_weak": "full",
        "unknown": "full",
    }

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[target_col].astype(str))
    y_test = encoder.transform(test[target_col].astype(str))

    action_predictions: dict[str, np.ndarray] = {}
    action_summary = []
    for action, drop_cols in action_drops.items():
        cols = [col for col in all_feature_cols if col not in set(drop_cols)]
        if not cols:
            continue
        x_train, x_test = build_feature_matrix(train, test, cols)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1200,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        action_predictions[action] = pred
        action_summary.append(
            {
                "row_type": "action_summary",
                "action": action,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            }
        )

    if not action_predictions:
        return pd.DataFrame(action_summary)

    eval_rows = []
    test_index = test.reset_index(drop=True)
    y_test_series = pd.Series(y_test)
    for _, cand in candidates.iterrows():
        time_bin = pd.Timestamp(cand["time_bin"])
        service = str(cand["service"])
        mask = (test_index["_bin"] == time_bin) & (test_index[service_col].astype(str) == service)
        if mask.sum() < 10:
            continue

        action_scores = {}
        idx = np.where(mask.to_numpy())[0]
        truth = y_test_series.iloc[idx].to_numpy()
        for action, pred in action_predictions.items():
            action_scores[action] = float((pred[idx] == truth).mean())

        oracle = max(action_scores.values())
        best_static_action = max(
            action_predictions,
            key=lambda action: f1_score(y_test, action_predictions[action], average="macro", zero_division=0),
        )
        selected_action = action_map.get(str(cand["weak_source"]), "full")
        if selected_action not in action_scores:
            selected_action = "full"

        eval_rows.append(
            {
                "row_type": "candidate_policy",
                "service": service,
                "time_bin": time_bin,
                "weak_source": cand["weak_source"],
                "confidence": cand["confidence"],
                "selected_action": selected_action,
                "selected_score": action_scores[selected_action],
                "oracle_score": oracle,
                "regret": oracle - action_scores[selected_action],
                "best_static_action": best_static_action,
                "best_static_score": action_scores[best_static_action],
                "static_regret": oracle - action_scores[best_static_action],
                **{f"score_{action}": score for action, score in action_scores.items()},
            }
        )

    return pd.DataFrame(action_summary + eval_rows)


def align_events(candidates: pd.DataFrame, events_csv: str | None) -> pd.DataFrame:
    if not events_csv:
        return pd.DataFrame()
    events = pd.read_csv(events_csv)
    required = {"service", "time", "source"}
    if not required.issubset(events.columns):
        raise ValueError(f"events CSV must contain columns: {sorted(required)}")
    events = events.copy()
    events["event_time"] = pd.to_datetime(events["time"], errors="coerce")
    candidates = candidates.copy()
    if "time_bin" not in candidates.columns:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    candidates["time_bin"] = pd.to_datetime(candidates["time_bin"], errors="coerce")

    rows = []
    for lag in range(0, 8):
        shifted = events.assign(aligned_time=events["event_time"] + pd.to_timedelta(lag, unit="D"))
        for _, event in shifted.iterrows():
            match = candidates[
                (candidates["service"].astype(str) == str(event["service"]))
                & (abs(candidates["time_bin"] - event["aligned_time"]) <= pd.Timedelta(days=1))
            ]
            rows.append(
                {
                    "lag_days": lag,
                    "service": event["service"],
                    "source": event["source"],
                    "event_key": event.get("event_key", ""),
                    "matched_candidates": len(match),
                    "best_candidate_confidence": float(match["confidence"].max()) if len(match) else 0.0,
                    "best_candidate_source": match.iloc[0]["weak_source"] if len(match) else "none",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    service_col: str,
    confidence_threshold: float,
    groups: dict[str, list[str]],
    candidates: pd.DataFrame,
    policy: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    if candidates.empty or "confidence" not in candidates.columns:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    high = candidates[(candidates["confidence"] >= confidence_threshold) & (candidates["weak_source"] != "unknown")]
    non_google = high[~high["service"].astype(str).str.contains("google", case=False, na=False)]
    source_counts = Counter(non_google["weak_source"].astype(str))

    policy_candidates = policy[policy.get("row_type", pd.Series(dtype=str)) == "candidate_policy"] if not policy.empty else pd.DataFrame()
    report = [
        "# Phase-0 Gate Report",
        "",
        "## Data",
        f"- Rows loaded: {len(df):,}",
        f"- Services: {df[service_col].nunique() if service_col in df.columns else 'n/a'}",
        "",
        "## Inferred Feature Groups",
    ]
    for group, cols in groups.items():
        preview = ", ".join(cols[:12])
        suffix = " ..." if len(cols) > 12 else ""
        report.append(f"- {group}: {len(cols)} columns ({preview}{suffix})")

    report.extend(
        [
            "",
            "## Drift Windows",
            f"- Candidate windows: {len(candidates):,}",
            f"- High-confidence non-Google windows: {len(non_google):,}",
            f"- Non-Google source counts: {dict(source_counts)}",
            "",
            "## Policy",
        ]
    )
    if policy_candidates.empty:
        report.append("- Policy evaluation skipped or had no candidate windows in the held-out split.")
    else:
        report.append(f"- Evaluated candidate windows: {len(policy_candidates):,}")
        report.append(f"- Mean source-aware regret: {policy_candidates['regret'].mean():.4f}")
        report.append(f"- Mean best-static regret: {policy_candidates['static_regret'].mean():.4f}")
        if "score_no_handshake" in policy_candidates:
            diff = policy_candidates["selected_score"] - policy_candidates["score_no_handshake"]
            report.append(f"- Mean selected minus no-handshake score: {diff.mean():.4f}")

    report.append("")
    report.append("## Optional Event Alignment")
    if events.empty:
        report.append("- No external events CSV was provided.")
    else:
        lag_summary = events.groupby("lag_days")["matched_candidates"].sum().to_dict()
        report.append(f"- Matched event candidates by lag: {lag_summary}")

    report.extend(
        [
            "",
            "## Provisional Decision",
            "- Pass only if high-confidence windows are probe-separable and policy gains survive held-out evaluation.",
            "- Treat unknown as conservative. Do not force attribution for coverage.",
        ]
    )

    (out_dir / "gate_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input_glob, args.sample_rows, args.seed)
    exclude = {args.time_col, args.service_col}
    if args.target_col:
        exclude.add(args.target_col)
    groups = infer_groups(df.columns, exclude)

    metrics = build_window_metrics(
        df=df,
        time_col=args.time_col,
        service_col=args.service_col,
        groups=groups,
        bin_size=args.bin_size,
        min_flows=args.min_flows_per_bin,
    )
    candidates = find_candidate_windows(
        metrics,
        args.max_candidate_windows,
        args.confidence_threshold,
        args.placebo_pvalue_threshold,
    )
    policy = evaluate_actions(
        df=df,
        time_col=args.time_col,
        service_col=args.service_col,
        target_col=args.target_col,
        groups=groups,
        candidates=candidates,
        bin_size=args.bin_size,
        seed=args.seed,
    )
    events = align_events(candidates, args.events_csv)

    metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    candidates.to_csv(out_dir / "candidate_windows.csv", index=False)
    policy.to_csv(out_dir / "action_policy_eval.csv", index=False)
    if not events.empty:
        events.to_csv(out_dir / "event_alignment.csv", index=False)
    write_report(out_dir, df, args.service_col, args.confidence_threshold, groups, candidates, policy, events)

    print(f"wrote outputs to {out_dir}")
    print(f"candidate windows: {len(candidates):,}")
    if not candidates.empty:
        high = candidates[(candidates["confidence"] >= 0.55) & (candidates["weak_source"] != "unknown")]
        print(f"high-confidence non-unknown windows: {len(high):,}")


if __name__ == "__main__":
    main()
