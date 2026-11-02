"""Compare daily switching to cheaper fixed and smoothed audit decisions.

Read-only, post-hoc analysis of examined development transitions. All route
rules use only paid audit estimates available by the prior day. No archive,
model, or final-window file is opened. These rules were specified after
viewing exploratory development summaries, so none is a validated method.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_PER_DAY = 1000


def review(frame):
    required = {"home", "seed", "today", "tomorrow",
                "today_audit_delta_pp", "tomorrow_full_delta_pp"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Transport rows missing columns: {sorted(missing)}")
    rows = frame.sort_values(["home", "seed", "today"]).copy()
    groups = rows.groupby(["home", "seed"], sort=False)
    rows["decision_number"] = groups.cumcount() + 1
    rows["first_audit_delta_pp"] = groups.today_audit_delta_pp.transform("first")
    rows["cumulative_audit_delta_pp"] = groups.today_audit_delta_pp.transform(
        lambda x: x.expanding().mean())
    rows["rolling_two_audit_delta_pp"] = groups.today_audit_delta_pp.transform(
        lambda x: x.rolling(2, min_periods=1).mean())
    tomorrow_gain = rows.tomorrow_full_delta_pp
    routes = {
        "source": np.zeros(len(rows), dtype=bool),
        "always_care": np.ones(len(rows), dtype=bool),
        "daily_sign": rows.today_audit_delta_pp > 0,
        "first_day_fixed": rows.first_audit_delta_pp > 0,
        "cumulative_sign": rows.cumulative_audit_delta_pp > 0,
        "rolling_two_sign": rows.rolling_two_audit_delta_pp > 0,
        "oracle_next": tomorrow_gain > 0,
    }
    for name, use_care in routes.items():
        rows[f"{name}_gain_pp"] = np.where(use_care, tomorrow_gain, 0.0)
        rows[f"{name}_use_care"] = use_care
    rows["daily_labels_to_date"] = rows.decision_number * LABELS_PER_DAY
    rows["fixed_labels_to_date"] = LABELS_PER_DAY
    names = list(routes)
    by_home = rows.groupby("home", as_index=False).agg(
        transitions=("tomorrow", "count"),
        audit_labels_per_seed_daily=("decision_number",
                                     lambda x: int(x.max() * LABELS_PER_DAY)),
        audit_labels_per_seed_fixed=("decision_number",
                                     lambda x: LABELS_PER_DAY),
        **{f"{name}_gain_pp":(f"{name}_gain_pp", "mean") for name in names},
        **{f"{name}_use_care_rate":(f"{name}_use_care", "mean")
           for name in names if name != "source"},
    )
    by_home["daily_minus_first_fixed_pp"] = (
        by_home.daily_sign_gain_pp - by_home.first_day_fixed_gain_pp)
    by_home["cumulative_minus_first_fixed_pp"] = (
        by_home.cumulative_sign_gain_pp - by_home.first_day_fixed_gain_pp)
    return rows, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_transport_feasibility_v1/per_transition.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_online_transport_budget_review_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    raw = pd.read_csv(args.transport.resolve(strict=True))
    rows, by_home = review(raw)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "per_transition.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    key = ["source_gain_pp", "always_care_gain_pp", "daily_sign_gain_pp",
           "first_day_fixed_gain_pp", "cumulative_sign_gain_pp",
           "rolling_two_sign_gain_pp", "oracle_next_gain_pp",
           "daily_minus_first_fixed_pp", "audit_labels_per_seed_daily",
           "audit_labels_per_seed_fixed"]
    report = "\n".join([
        "Read-only exploratory route/budget comparison on examined UNSW development transitions.",
        "The first-day-fixed rule spends 1000 audit labels once; daily, cumulative, and rolling-two rules spend 1000 per transition day.",
        "All route decisions use only earlier paid audit estimates; source and CARE predictions are frozen.",
        "Oracle-next uses tomorrow's labels and is a contaminated ceiling, not a deployable method.",
        "The route rules were written after viewing development summaries; no independent validation is claimed.",
        "Overlapping seeds and adjacent days are correlated; homes are the independent units.",
        "No archive, model, or final-window file was opened.",
        "Per-home means:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        by_home[key].mean().round(4).to_string(),
        "Homes where daily beats first-day fixed: " +
        f"{int((by_home.daily_minus_first_fixed_pp > 0).sum())}/{len(by_home)}",
        "Homes where cumulative beats first-day fixed: " +
        f"{int((by_home.cumulative_minus_first_fixed_pp > 0).sum())}/{len(by_home)}",
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
