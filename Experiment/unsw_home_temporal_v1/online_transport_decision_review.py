"""Score simple next-day routes from the completed transport diagnostic.

Read-only post-hoc analysis of already examined development transitions.
The sign rule is an exploratory diagnostic, not a validated policy. No
archive, model, or final-window file is opened.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def review(rows):
    required = {"home", "seed", "today", "tomorrow", "today_full_delta_pp",
                "today_audit_delta_pp", "tomorrow_full_delta_pp"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Transport rows missing columns: {sorted(missing)}")
    rows = rows.copy()
    tomorrow = rows.tomorrow_full_delta_pp
    rows["source_gain_pp"] = 0.0
    rows["always_care_gain_pp"] = tomorrow
    rows["audit_sign_route_gain_pp"] = np.where(
        rows.today_audit_delta_pp > 0, tomorrow, 0.0)
    rows["full_today_sign_route_gain_pp"] = np.where(
        rows.today_full_delta_pp > 0, tomorrow, 0.0)
    rows["oracle_next_gain_pp"] = np.maximum(tomorrow, 0.0)
    rows["audit_uses_care"] = rows.today_audit_delta_pp > 0
    rows["full_today_uses_care"] = rows.today_full_delta_pp > 0
    rows["tomorrow_care_better"] = tomorrow > 0
    rows["audit_regret_pp"] = (rows.oracle_next_gain_pp -
                               rows.audit_sign_route_gain_pp)
    rows["full_today_regret_pp"] = (rows.oracle_next_gain_pp -
                                    rows.full_today_sign_route_gain_pp)
    by_home = (rows.groupby("home", as_index=False)
               .agg(transitions=("tomorrow", "count"),
                    tomorrow_care_better_rate=("tomorrow_care_better", "mean"),
                    audit_uses_care_rate=("audit_uses_care", "mean"),
                    source_gain_pp=("source_gain_pp", "mean"),
                    always_care_gain_pp=("always_care_gain_pp", "mean"),
                    audit_sign_route_gain_pp=("audit_sign_route_gain_pp", "mean"),
                    full_today_sign_route_gain_pp=("full_today_sign_route_gain_pp", "mean"),
                    oracle_next_gain_pp=("oracle_next_gain_pp", "mean"),
                    audit_regret_pp=("audit_regret_pp", "mean"),
                    full_today_regret_pp=("full_today_regret_pp", "mean")))
    by_home["best_fixed_hindsight_gain_pp"] = np.maximum(
        by_home.always_care_gain_pp, 0.0)
    return rows, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_transport_feasibility_v1/per_transition.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_transport_decision_review_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    rows = pd.read_csv(args.transport.resolve(strict=True))
    scored, by_home = review(rows)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "per_transition.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    mean = by_home.select_dtypes(include="number").drop(
        columns="transitions").mean()
    report = "\n".join([
        "Read-only, post-hoc next-day route review on examined UNSW development transitions.",
        "The exploratory sign route uses CARE tomorrow iff yesterday's 1000-label audit estimated positive gain.",
        "Full-today sign uses all yesterday labels; oracle-next and best-fixed-hindsight use future labels and are contaminated ceilings.",
        "All gains are macro-F1 percentage points relative to frozen source predictions on tomorrow's day.",
        "Overlapping seeds and adjacent days are correlated; homes, not rows, are independent units.",
        "No archive, model, or final-window file was opened.",
        "Per-home means:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        mean.round(4).to_string(),
        "Homes where audit route beats source: " +
        f"{int((by_home.audit_sign_route_gain_pp > 0).sum())}/{len(by_home)}",
        "Homes where audit route beats always CARE: " +
        f"{int((by_home.audit_sign_route_gain_pp > by_home.always_care_gain_pp).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
