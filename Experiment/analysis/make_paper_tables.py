"""
Generate LaTeX and CSV tables from TTA-TC experiment results.

Produces:
  - Table 1: Main comparison (accuracy + AURC, QUIC22)
  - Table 2: Long-term results (TLS-Year22, monthly ARR)
  - Table 3–7: Ablation study tables (A1–A5)

Usage (from Experiment/core_code/):
    python ../analysis/make_paper_tables.py --outputs-dir outputs/
    python ../analysis/make_paper_tables.py --outputs-dir outputs/ \
        --save-dir ../analysis/tables/
"""
import argparse
import json
import os


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt(v, digits=4, bold_thresh=None, all_vals=None):
    """Format a float; bold the best value in a column if all_vals given."""
    if v is None:
        return "-"
    s = f"{v:.{digits}f}"
    if all_vals is not None:
        best = max(x for x in all_vals if x is not None)
        if abs(v - best) < 1e-9:
            return r"\textbf{" + s + "}"
    return s


METHOD_ORDER = ["static", "bn_adapt", "tent", "eata", "cotta", "sar", "note", "tta_tc"]
METHOD_LABELS = {
    "static":   r"Static (no adapt.)",
    "bn_adapt": r"BN-Adapt~\cite{schneider2020}",
    "tent":     r"Tent~\cite{wang2021tent}",
    "eata":     r"EATA~\cite{niu2022eata}",
    "cotta":    r"CoTTA~\cite{wang2022cotta}",
    "sar":      r"SAR~\cite{niu2023sar}",
    "note":     r"NOTE~\cite{gong2022note}",
    "tta_tc":   r"\textbf{TTA-TC (Ours)}",
}


# --------------------------------------------------------------------------- #
#  Table 1: Main comparison (QUIC22)
# --------------------------------------------------------------------------- #

def make_table1(single, seq_quic22, save_dir):
    """
    Main comparison table:
    Method | Acc@W45 | F1@W45 | Acc@W46 | Acc@W47 | AURC
    """
    # Collect values
    rows = []
    for method in METHOD_ORDER:
        row = {"method": method}

        if method in single:
            row["acc_w45"] = single[method].get("accuracy")
            row["f1_w45"] = single[method].get("macro_f1")

        if method in seq_quic22:
            periods = seq_quic22[method].get("periods", {})
            for period in ["W-2022-45", "W-2022-46", "W-2022-47"]:
                key = f"acc_{period.replace('-', '').replace('W', 'w').replace('2022', '')}"
                row[key] = periods.get(period, {}).get("accuracy")
            row["aurc"] = seq_quic22[method].get("aurc")

        rows.append(row)

    # Determine best values per numeric column
    cols = ["acc_w45", "f1_w45", "acc_w4545", "acc_w4546", "acc_w4547", "aurc"]
    col_vals = {}
    for col in cols:
        col_vals[col] = [r.get(col) for r in rows if r.get(col) is not None]

    # Build LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of TTA methods on CESNET-QUIC22. "
        r"Acc@W45 is single-period accuracy; "
        r"AURC is the area under the ARR curve over W-45 to W-47.}",
        r"\label{tab:main_comparison}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Acc@W-45 $\uparrow$ & F1@W-45 $\uparrow$ & "
        r"Acc@W-46 $\uparrow$ & Acc@W-47 $\uparrow$ & AURC $\uparrow$ \\",
        r"\midrule",
    ]

    for row in rows:
        m = row["method"]
        label = METHOD_LABELS.get(m, m)

        acc45 = _fmt(row.get("acc_w45"), all_vals=col_vals.get("acc_w45"))
        f145  = _fmt(row.get("f1_w45"),  all_vals=col_vals.get("f1_w45"))
        acc46 = _fmt(row.get("acc_w4545"), all_vals=col_vals.get("acc_w4545"))
        acc47 = _fmt(row.get("acc_w4546"), all_vals=col_vals.get("acc_w4546"))
        aurc  = _fmt(row.get("aurc"),    all_vals=col_vals.get("aurc"))

        if m == "tta_tc":
            lines.append(r"\midrule")
        lines.append(f"{label} & {acc45} & {f145} & {acc46} & {acc47} & {aurc} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    _save(tex, os.path.join(save_dir, "table1_comparison_quic22.tex"))
    return tex


# --------------------------------------------------------------------------- #
#  Table 2: Long-term TLS-Year22
# --------------------------------------------------------------------------- #

def make_table2(seq_tls22, save_dir):
    """
    Long-term table:
    Method | M4 | M5 | M6 | M7 | M8 | M9 | AURC
    """
    methods_present = [m for m in METHOD_ORDER if m in seq_tls22]
    if not methods_present:
        print("  [skip] No TLS-Year22 data for Table 2")
        return ""

    period_keys = [
        "M-2022-4", "M-2022-5", "M-2022-6",
        "M-2022-7", "M-2022-8", "M-2022-9",
    ]
    short = ["M4", "M5", "M6", "M7", "M8", "M9"]

    # Collect ARR values per period
    col_arrs = {p: [] for p in period_keys}
    aurc_vals = []
    for method in methods_present:
        periods = seq_tls22[method].get("periods", {})
        for p in period_keys:
            v = periods.get(p, {}).get("arr")
            if v is not None:
                col_arrs[p].append(v)
        aurc = seq_tls22[method].get("aurc")
        if aurc is not None:
            aurc_vals.append(aurc)

    header_cols = " & ".join(f"ARR@{s}" for s in short)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Long-term ARR on CESNET-TLS-Year22. "
        r"Models trained on M-2022-1 to M-2022-3, tested continuously "
        r"from M-2022-4 to M-2022-9 (6 months shown).}",
        r"\label{tab:longterm_tls22}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (len(short) + 1) + "}",
        r"\toprule",
        f"Method & {header_cols} & AURC \\\\",
        r"\midrule",
    ]

    for method in methods_present:
        label = METHOD_LABELS.get(method, method)
        periods = seq_tls22[method].get("periods", {})
        cells = []
        for p in period_keys:
            v = periods.get(p, {}).get("arr")
            cells.append(_fmt(v, digits=3, all_vals=col_arrs[p]))
        aurc = _fmt(seq_tls22[method].get("aurc"), digits=3, all_vals=aurc_vals)

        if method == "tta_tc":
            lines.append(r"\midrule")
        lines.append(f"{label} & {' & '.join(cells)} & {aurc} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    tex = "\n".join(lines)
    _save(tex, os.path.join(save_dir, "table2_longterm_tls22.tex"))
    return tex


# --------------------------------------------------------------------------- #
#  Tables 3–7: Ablations
# --------------------------------------------------------------------------- #

ABLATION_META = {
    "ssl_tasks": {
        "caption": (
            "Ablation A1: effect of SSL task selection on TTA-TC. "
            "MPFP alone achieves the largest gain; combining all three tasks performs best."
        ),
        "label": "tab:abl_ssl_tasks",
        "col": "accuracy",
    },
    "mask_ratio": {
        "caption": "Ablation A2: effect of MPFP masking ratio on W-2022-45 accuracy.",
        "label": "tab:abl_mask_ratio",
        "col": "accuracy",
    },
    "adapt_depth": {
        "caption": "Ablation A3: encoder adaptation depth vs. accuracy/compute trade-off.",
        "label": "tab:abl_adapt_depth",
        "col": "accuracy",
    },
    "anti_forgetting": {
        "caption": (
            "Ablation A5: anti-forgetting mechanisms. "
            "Fisher regularization and stochastic restoration together prevent error accumulation."
        ),
        "label": "tab:abl_anti_forgetting",
        "col": "accuracy",
    },
    "norm_type": {
        "caption": "Ablation A8: normalization type. GroupNorm outperforms BatchNorm under non-i.i.d. streams.",
        "label": "tab:abl_norm_type",
        "col": "accuracy",
    },
}


def make_ablation_table(abl_data, name, save_dir, table_num=3):
    if not abl_data:
        print(f"  [skip] No data for ablation {name}")
        return ""

    meta = ABLATION_META.get(name, {
        "caption": f"Ablation: {name}.",
        "label": f"tab:abl_{name}",
        "col": "accuracy",
    })
    col = meta["col"]
    settings = list(abl_data.keys())
    values = [abl_data[s].get(col) for s in settings]
    f1s    = [abl_data[s].get("macro_f1") for s in settings]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{meta['caption']}}}",
        f"\\label{{{meta['label']}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Setting & Accuracy $\uparrow$ & Macro-F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for s, v, f in zip(settings, values, f1s):
        acc_str = _fmt(v, all_vals=[x for x in values if x is not None])
        f1_str  = _fmt(f, all_vals=[x for x in f1s if x is not None])
        lines.append(f"{s.replace('_', r'\_')} & {acc_str} & {f1_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    _save(tex, os.path.join(save_dir, f"table{table_num}_abl_{name}.tex"))
    return tex


# --------------------------------------------------------------------------- #
#  Save helper
# --------------------------------------------------------------------------- #

def _save(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="TTA-TC paper table generation")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tables"
    )
    os.makedirs(save_dir, exist_ok=True)
    out = args.outputs_dir

    def _try(path):
        if os.path.exists(path):
            return _load_json(path)
        print(f"  [not found] {path}")
        return {}

    single    = _try(os.path.join(out, "eval_quic22_single", "results_single.json"))
    seq_q22   = _try(os.path.join(out, "eval_quic22_sequential", "results_sequential.json"))
    seq_tls22 = _try(os.path.join(out, "eval_tls22_sequential", "results_sequential.json"))

    ablations = {}
    for name in ABLATION_META:
        d = _try(os.path.join(out, "ablations", name, "ablation_results.json"))
        ablations[name] = d

    print(f"\nSaving LaTeX tables to {save_dir}/")
    make_table1(single, seq_q22, save_dir)
    make_table2(seq_tls22, save_dir)
    for i, (name, data) in enumerate(ablations.items(), start=3):
        make_ablation_table(data, name, save_dir, table_num=i)

    # Write combined include file
    includes = "\n".join(
        f"\\input{{{os.path.basename(f)}}}"
        for f in sorted(os.listdir(save_dir)) if f.endswith(".tex")
    )
    inc_path = os.path.join(save_dir, "all_tables.tex")
    with open(inc_path, "w") as f:
        f.write(includes + "\n")
    print(f"  Saved: {inc_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
