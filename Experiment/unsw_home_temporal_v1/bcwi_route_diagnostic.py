"""Read-only attribution of paid BCWI/source gate errors on development days.

Separates paid-sample estimation error on today from full-today to full-
tomorrow transport error. All three outcomes are post-hoc diagnostics;
only the paid estimate was available to the frozen route. No archive opened.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import PRIMARY


def diagnostic(root):
    records = []
    for seed in range(5):
        folder = root / f"seed_{seed}"
        scores = pd.read_csv(folder / "evaluation_all" / "per_day.csv")
        if (set(scores.home) != set(PRIMARY) or set(scores.seed) != {seed} or
                set(scores.method) != {"source", "bcwi_050", "updated_care",
                                       "paid_sign_route"} or
                scores.duplicated(["home", "transition", "method"]).any()):
            raise ValueError(f"Completed route score layout differs: {folder}")
        delta = scores.pivot(index=["home", "transition"], columns="method",
                             values="delta_vs_source_pp")
        for home in PRIMARY:
            gate = pd.read_csv(folder / home / "decisions.csv")
            days = delta.loc[home].sort_index()
            if (len(gate) != len(days) or
                    not np.array_equal(gate.transition, np.arange(len(days))) or
                    not np.array_equal(days.index, np.arange(len(days))) or
                    gate.loc[0, "choice"] != "bcwi_050"):
                raise ValueError(f"Gate/day alignment differs: {home} seed={seed}")
            for t in range(1, len(days)):
                decision = gate.iloc[t]
                paid = float(decision.audit_delta_pp)
                today = float(days.loc[t - 1, "bcwi_050"])
                tomorrow = float(days.loc[t, "bcwi_050"])
                if (not np.isfinite([paid, today, tomorrow]).all() or
                        decision.choice != ("bcwi_050" if paid > 0 else "source") or
                        int(decision.audit_paid_rows) != 1000):
                    raise ValueError(f"Gate value/rule differs: {home} seed={seed} t={t}")
                paid_choice = paid > 0
                today_choice = today > 0
                tomorrow_choice = tomorrow > 0
                records.append(dict(
                    home=home, seed=seed, transition=t,
                    audit_present_classes=int(decision.audit_present_classes),
                    paid_delta_pp=paid, full_today_delta_pp=today,
                    full_tomorrow_delta_pp=tomorrow,
                    paid_chooses_bcwi=paid_choice,
                    full_today_chooses_bcwi=today_choice,
                    oracle_tomorrow_chooses_bcwi=tomorrow_choice,
                    paid_matches_today=paid_choice == today_choice,
                    full_today_matches_tomorrow=today_choice == tomorrow_choice,
                    paid_matches_tomorrow=paid_choice == tomorrow_choice,
                    paid_route_gain_pp=tomorrow if paid_choice else 0.0,
                    full_today_route_gain_pp=tomorrow if today_choice else 0.0,
                    oracle_next_gain_pp=max(tomorrow, 0.0)))
    frame = pd.DataFrame(records)
    frame["paid_regret_pp"] = (frame.oracle_next_gain_pp -
                               frame.paid_route_gain_pp)
    frame["full_today_regret_pp"] = (frame.oracle_next_gain_pp -
                                     frame.full_today_route_gain_pp)
    by_home = frame.groupby("home", as_index=False).agg(
        transitions=("transition", "count"),
        mean_paid_present_classes=("audit_present_classes", "mean"),
        paid_matches_today=("paid_matches_today", "mean"),
        full_today_matches_tomorrow=("full_today_matches_tomorrow", "mean"),
        paid_matches_tomorrow=("paid_matches_tomorrow", "mean"),
        paid_uses_bcwi=("paid_chooses_bcwi", "mean"),
        tomorrow_bcwi_better=("oracle_tomorrow_chooses_bcwi", "mean"),
        mean_paid_regret_pp=("paid_regret_pp", "mean"),
        mean_full_today_regret_pp=("full_today_regret_pp", "mean"))
    return frame, by_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_paid_route_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data/xjw/tma-care-data/unsw_bcwi_paid_route_v1/transport_diagnostic_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        parser.error(f"Output must be new and empty: {output}")
    frame, by_home = diagnostic(args.root.resolve(strict=True))
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_transition.csv", index=False)
    by_home.to_csv(output / "per_home.csv", index=False)
    home07 = frame[frame.home == "home07"][[
        "seed", "transition", "audit_present_classes", "paid_delta_pp",
        "full_today_delta_pp", "full_tomorrow_delta_pp",
        "paid_chooses_bcwi", "paid_regret_pp"]]
    report = "\n".join([
        "Read-only post-hoc BCWI/source paid-route error attribution.",
        "Paid-vs-full-today mismatch estimates sample error for the same frozen model.",
        "Full-today-vs-tomorrow mismatch estimates transport failure across both day and model update.",
        "The full-day directions and oracle regret use examined development labels; they are not available to the route.",
        "First predicted day has no prior BCWI audit and is excluded from this diagnostic.",
        "Five seeds and adjacent days overlap; homes are independent units. Final window unopened.",
        "Per-home rates and mean daily regret:",
        by_home.round(4).to_string(index=False),
        "Across-home means:",
        by_home.drop(columns="home").mean(numeric_only=True).round(4).to_string(),
        "Home07 transitions:", home07.round(4).to_string(index=False),
    ]) + "\n"
    (output / "REPORT.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
