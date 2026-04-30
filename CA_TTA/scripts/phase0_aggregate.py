"""
CA-TTA Phase 0 — aggregation + decision.

Reads:
  outputs/phase0_cert_acc_per_month/<dataset>_cert_acc_sigma*.json   (vanilla)
  outputs/phase0_tent_then_certify/<dataset>_tent_cert_acc_sigma*.json (Tent)

Prints two tables (vanilla, Tent-adapted) and a verdict for go/no-go on
the CA-TTA direction.

Decision criteria:
  Step 0.3 (vanilla cert acc drop):
    > 20% drop from earliest to latest month  -> motivation for adaptation
  Step 0.4 (Tent harms certification):
    Tent cert acc < vanilla cert acc on >= 50% of periods at any radius
    -> motivation for Certification-Aware TTA
"""
import json
import os
from glob import glob


_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "outputs"))


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(title, summary):
    if summary is None or not summary.get("periods"):
        print(f"\n{title}: (no data)")
        return
    radii = summary["radii"]
    print(f"\n{title}")
    print("=" * (14 + 12 * len(radii)))
    print(f"{'Period':<14}" + "".join(f"  r={r:<.2f}".rjust(12) for r in radii))
    print("-" * (14 + 12 * len(radii)))
    for period, p_data in summary["periods"].items():
        row = f"{period:<14}"
        for r in radii:
            v = p_data["certified_accuracy"].get(str(r),
                  p_data["certified_accuracy"].get(r, None))
            row += f"  {v:>10.4f}" if v is not None else "  " + " " * 8 + "N/A"
        print(row)
    print("=" * (14 + 12 * len(radii)))


def first_last_drop(summary, radius):
    """Compute (first_period_acc, last_period_acc, drop_pct)."""
    periods = list(summary["periods"].keys())
    if len(periods) < 2:
        return None, None, None
    f = summary["periods"][periods[0]]["certified_accuracy"].get(str(radius),
            summary["periods"][periods[0]]["certified_accuracy"].get(radius))
    l = summary["periods"][periods[-1]]["certified_accuracy"].get(str(radius),
            summary["periods"][periods[-1]]["certified_accuracy"].get(radius))
    drop = (f - l) / f if f and f > 0 else 0.0
    return f, l, drop


def tent_vs_vanilla(vanilla, tent):
    """For each period, count how many radii Tent is worse than vanilla."""
    if vanilla is None or tent is None:
        return None
    radii = vanilla["radii"]
    period_results = []
    for period in vanilla["periods"]:
        if period not in tent["periods"]:
            continue
        v = vanilla["periods"][period]["certified_accuracy"]
        t = tent["periods"][period]["certified_accuracy"]
        worse_count = 0
        for r in radii:
            v_acc = v.get(str(r), v.get(r))
            t_acc = t.get(str(r), t.get(r))
            if t_acc is not None and v_acc is not None and t_acc < v_acc:
                worse_count += 1
        period_results.append({
            "period": period,
            "tent_worse_at_radii": worse_count,
            "total_radii": len(radii),
        })
    return period_results


def main():
    for dataset in ("quic22", "tls22"):
        print("\n" + "#" * 80)
        print(f"# Dataset: {dataset}")
        print("#" * 80)

        # Find all sigma values present
        v_paths = sorted(glob(os.path.join(
            ROOT, "phase0_cert_acc_per_month",
            f"{dataset}_cert_acc_sigma*.json")))
        for v_path in v_paths:
            sigma = v_path.split("_sigma")[-1].replace(".json", "")
            t_path = os.path.join(
                ROOT, "phase0_tent_then_certify",
                f"{dataset}_tent_cert_acc_sigma{sigma}.json")

            vanilla = load_summary(v_path)
            tent = load_summary(t_path)

            print_table(f"-- Vanilla source model (sigma={sigma}) --", vanilla)
            print_table(f"-- Tent-adapted model (sigma={sigma}) --", tent)

            # Step 0.3 decision
            if vanilla:
                print(f"\n[Step 0.3] Cert acc drop (first vs last period):")
                for r in vanilla["radii"]:
                    f, l, drop = first_last_drop(vanilla, r)
                    if f is not None:
                        print(f"  r={r:.2f}: {f:.4f} -> {l:.4f}  "
                              f"drop = {drop*100:+.1f}%")

            # Step 0.4 decision
            tvv = tent_vs_vanilla(vanilla, tent)
            if tvv:
                print(f"\n[Step 0.4] Tent vs vanilla — # radii where Tent is worse:")
                for row in tvv:
                    print(f"  {row['period']}: "
                          f"{row['tent_worse_at_radii']}/{row['total_radii']}")

            # Verdict
            if vanilla and tent:
                # Average drop across radii
                drops = []
                for r in vanilla["radii"]:
                    _, _, d = first_last_drop(vanilla, r)
                    if d is not None:
                        drops.append(d)
                avg_drop = sum(drops) / len(drops) if drops else 0
                tent_worse_total = sum(
                    row["tent_worse_at_radii"] for row in tvv) if tvv else 0
                tent_total = sum(row["total_radii"] for row in tvv) if tvv else 1
                tent_worse_frac = tent_worse_total / tent_total

                print(f"\n[Verdict — sigma={sigma}, dataset={dataset}]")
                print(f"  Average cert-acc drop over time: {avg_drop*100:+.1f}%")
                print(f"  Tent-worse fraction:             {tent_worse_frac*100:.1f}%")
                if avg_drop > 0.20 and tent_worse_frac > 0.5:
                    verdict = "STRONG MOTIVATION — proceed to CA-TTA Phase 1"
                elif avg_drop > 0.20:
                    verdict = "PARTIAL — drift hurts cert-acc, but Tent is not the right baseline; consider alternative TTA methods"
                elif tent_worse_frac > 0.5:
                    verdict = "PARTIAL — Tent harms cert-acc but drift effect is small; consider sharper sigma or other dataset"
                else:
                    verdict = "WEAK — neither drift nor Tent significantly affects cert-acc; CA-TTA direction may not be motivated"
                print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
