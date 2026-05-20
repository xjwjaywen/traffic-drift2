"""
Create one advisor-facing figure that summarizes all major method lines.

This complements the focused TTA / normalization plots. It is intentionally a
static summary figure: the numbers come from the experiment inventory and the
server-run summaries already produced in this project.

Usage from Experiment/core_code/:
    python scripts/visualize_method_inventory_summary.py \
      --output-dir outputs/teacher_result_visuals
"""
import argparse
import csv
import os
import tempfile
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tta_tc_matplotlib_cache"),
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


ROWS = [
    {
        "line": "Generic TTA",
        "methods": "Static, BN-Adapt, Tent, EATA, CoTTA, SAR, NOTE, TTA-TC",
        "best_signal": "TLS22 M12 macro-F1: 0.6286 -> 0.6308; collapsed F1: 0.0255 -> 0.0565",
        "conclusion": "Marginal overall gain; collapsed classes remain mostly unrecovered",
    },
    {
        "line": "Norm / AdaBN",
        "methods": "GN, IN, BN, LN, BN + AdaBN",
        "best_signal": "IN helps gradual recall (0.0246 -> 0.0601), but hurts abrupt recall (0.0102 -> 0.0069)",
        "conclusion": "Useful diagnostic; not a stable collapse solution",
    },
    {
        "line": "Static correction",
        "methods": "Global / quantile correction, prototype recalibration, oracle pair recalibration",
        "best_signal": "Oracle pair macro-F1: 0.628647 -> 0.628840",
        "conclusion": "Very low upper bound for post-hoc score correction",
    },
    {
        "line": "Target prototypes",
        "methods": "CAPS confidence-gated target prototype update",
        "best_signal": "M12 macro-F1: 0.6286 -> 0.6319; bad F1: 0.1544 -> 0.1639",
        "conclusion": "Weak but real signal; helps only part of bad classes",
    },
    {
        "line": "Representation adaptation",
        "methods": "CAPS++ target adapter",
        "best_signal": "Best run degraded to macro-F1 about 0.1839",
        "conclusion": "Unstable; direct target-side adapter is risky",
    },
    {
        "line": "Training-time attempts",
        "methods": "Pooled ERM, class-balanced ERM, risk-weighted ERM, temporal prototype loss",
        "best_signal": "Warm-start pooled ERM around M12 macro-F1 0.646; temporal proto did not beat pooled ERM",
        "conclusion": "More historical data helps, but tested regularizer is not enough",
    },
    {
        "line": "SSL / active maintenance",
        "methods": "MPFP / POP / FSR verify; AL sweep with labeled samples",
        "best_signal": "SSL verify output near random; AL raw outputs not fully summarized here",
        "conclusion": "Do not claim SSL/TTA success; AL is a possible separate maintenance story",
    },
]


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "methods", "best_signal", "conclusion"])
        writer.writeheader()
        writer.writerows(rows)


def render_with_matplotlib(out_png):
    wrapped_rows = []
    for row in ROWS:
        wrapped_rows.append([
            wrap(row["line"], 18),
            wrap(row["methods"], 34),
            wrap(row["best_signal"], 44),
            wrap(row["conclusion"], 36),
        ])

    columns = ["Experiment line", "Methods tried", "Best observed signal", "Current conclusion"]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title(
        "Method Inventory: What Has Been Tried for TLS22 Temporal Drift",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=wrapped_rows,
        colLabels=columns,
        colColours=["#2F5597"] * len(columns),
        colLoc="center",
        cellLoc="left",
        loc="center",
        colWidths=[0.16, 0.28, 0.30, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row_idx == 0:
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
            cell.set_height(0.07)
        else:
            cell.set_facecolor("#F7F9FC" if row_idx % 2 else "white")
            cell.set_text_props(va="center")

    footer = (
        "Takeaway: the evidence supports a collapse-aware diagnosis. "
        "Generic TTA, normalization, and static correction provide only marginal or partial relief; "
        "abrupt class collapse remains unsolved."
    )
    fig.text(0.5, 0.03, footer, ha="center", va="center", fontsize=12, color="#333333")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_pil_font(size, bold=False):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_wrapped(draw, text, xy, width_chars, font, fill, line_gap=6):
    x, y = xy
    lines = textwrap.wrap(text, width=width_chars, break_long_words=False)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap


def render_with_pil(out_png):
    from PIL import Image, ImageDraw

    width, height = 2600, 1500
    margin = 60
    title_h = 90
    header_h = 70
    row_h = 165
    footer_h = 90
    table_top = margin + title_h
    col_widths = [360, 690, 760, 670]
    x_positions = [margin]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_pil_font(38, bold=True)
    header_font = load_pil_font(24, bold=True)
    cell_font = load_pil_font(24, bold=False)
    footer_font = load_pil_font(23, bold=False)

    title = "Method Inventory: What Has Been Tried for TLS22 Temporal Drift"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, margin), title, font=title_font, fill="#111111")

    headers = ["Experiment line", "Methods tried", "Best observed signal", "Current conclusion"]
    y = table_top
    for x, w, header in zip(x_positions, col_widths, headers):
        draw.rectangle([x, y, x + w, y + header_h], fill="#2F5597", outline="#D9D9D9")
        draw.text((x + 16, y + 20), header, font=header_font, fill="white")

    y += header_h
    wrap_widths = [19, 35, 45, 36]
    for idx, row in enumerate(ROWS):
        fill = "#F7F9FC" if idx % 2 == 0 else "white"
        values = [row["line"], row["methods"], row["best_signal"], row["conclusion"]]
        for x, w, value, wrap_width in zip(x_positions, col_widths, values, wrap_widths):
            draw.rectangle([x, y, x + w, y + row_h], fill=fill, outline="#D9D9D9")
            draw_wrapped(draw, value, (x + 16, y + 18), wrap_width, cell_font, "#111111")
        y += row_h

    footer = (
        "Takeaway: generic TTA, normalization, and static correction provide only marginal or partial relief; "
        "abrupt class collapse remains unsolved."
    )
    draw_wrapped(draw, footer, (margin, height - footer_h + 10), 150, footer_font, "#333333")
    img.save(out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/teacher_result_visuals")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    out_png = os.path.join(args.output_dir, "method_inventory_summary.png")
    out_csv = os.path.join(args.output_dir, "method_inventory_summary.csv")
    if HAS_MATPLOTLIB:
        render_with_matplotlib(out_png)
    else:
        render_with_pil(out_png)
    write_csv(out_csv, ROWS)
    print(f"Saved method inventory figure: {out_png}")
    print(f"Saved method inventory CSV: {out_csv}")


if __name__ == "__main__":
    main()
