"""Denial rate against pre-attack retrieval quality — the mechanism figure.

Reads results/retrieval_binning.csv (produced by retrieval_binning.py) and plots
denial rate against the bin the gold passage occupied *before* the attack.

The visual argument is a divergence, not a level. Within each panel the aligned
line climbs as retrieval degrades while the base line stays flat on the floor:
retrieval sensitivity is a property of the aligned model, not of RAG. The third
panel is the control — both sides of the R1 pair are instruction-tuned, and both
climb, which is what the mechanism predicts and what a "base models are simply
different" account does not.

Colour carries the model side; line style carries whether an attack was present.
That keeps the palette at the two validated categorical slots already used by
forest_genuine.png, so the two figures read as one system.

Saved to results/retrieval_gradient.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CSV_PATH = RESULTS_DIR / "retrieval_binning.csv"
OUT_PATH = RESULTS_DIR / "retrieval_gradient.png"

# Categorical slots 1 and 2, validated all-pairs on the light surface — the same
# two the forest plot uses, so "base" and "aligned" mean one colour across figures.
C_BASE = "#2a78d6"
C_ALIGNED = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

BIN_ORDER = ["rank 0", "rank 1-2", "rank 3-4", "absent"]
BIN_LABELS = ["rank 0\n(gold on top)", "rank 1–2", "rank 3–4", "absent\n(not retrieved)"]

# (csv group, panel title, subtitle). The R1 pair goes last: it is the control,
# and it reads as one only after the first two have established the pattern.
PANELS = [
    ("NQ (excl. llama-r1)", "Natural Questions",
     "37 runs · 3 base → instruct pairs"),
    ("HotpotQA", "HotpotQA",
     "12 runs · multi-hop"),
    ("NQ llama-r1 only", "Control: R1 pair",
     "13 runs · both sides instruction-tuned"),
]

# Side labels differ on the R1 pair, where neither side is a base model.
SIDE_LABELS = {
    "default": ("Base model", "Aligned model"),
    "NQ llama-r1 only": ("Llama-3.1 Instruct", "R1-Distill"),
}

# Cells too thin to carry a per-run mean; pooled figures only. Marked, not dropped.
THIN_CELL_MAX_RUNS = 2


def load() -> dict[str, list[dict]]:
    rows = list(csv.DictReader(open(CSV_PATH)))
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: BIN_ORDER.index(r["bin"]))
    return by_group


def series(rows: list[dict], field: str) -> list[float | None]:
    out = []
    for r in rows:
        v = r.get(field)
        out.append(None if v in (None, "") else float(v))
    return out


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found — run scripts/retrieval_binning.py --csv first")
    by_group = load()

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.3), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    x = list(range(len(BIN_ORDER)))

    for ax, (group, title, subtitle) in zip(axes, PANELS):
        rows = by_group.get(group)
        if not rows:
            raise SystemExit(f"group {group!r} missing from {CSV_PATH.name}")
        ax.set_facecolor(SURFACE)

        base_lo, aligned_lo = SIDE_LABELS.get(group, SIDE_LABELS["default"])

        specs = [
            (series(rows, "base_clean_denial"), C_BASE, "--", "o", 5.5, False),
            (series(rows, "uncond_attacked_denial_base"), C_BASE, "-", "o", 7.0, True),
            (series(rows, "aligned_clean_denial"), C_ALIGNED, "--", "s", 5.5, False),
            (series(rows, "uncond_attacked_denial_aligned"), C_ALIGNED, "-", "s", 7.0, True),
        ]
        for ys, colour, style, marker, msize, filled in specs:
            pts = [(xi, y) for xi, y in zip(x, ys) if y is not None]
            if not pts:
                continue
            xs, yy = zip(*pts)
            ax.plot(xs, yy, color=colour, linestyle=style, linewidth=2.0,
                    zorder=3, alpha=1.0 if filled else 0.85)
            # Filled marks take a surface ring so overlapping series stay legible;
            # hollow marks invert it (surface face, coloured edge) to read as "no
            # attack" without spending a second hue.
            ax.plot(xs, yy, linestyle="none", marker=marker, markersize=msize,
                    color=colour if filled else SURFACE,
                    markeredgecolor=SURFACE if filled else colour,
                    markeredgewidth=2.0, zorder=4)

        # Thin cells carry pooled figures only — say so on the figure, not just in
        # the caption, so the point cannot be quoted as if it had 12 runs behind it.
        for xi, r in zip(x, rows):
            if int(r["n_runs"]) <= THIN_CELL_MAX_RUNS:
                ax.axvspan(xi - 0.34, xi + 0.34, color=GRID, alpha=0.55, zorder=0)
                # Below the tick labels, where nothing else competes for the space.
                ax.annotate(f"pooled only\nn={int(r['pooled_n_queries']):,}",
                            xy=(xi, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -38), textcoords="offset points",
                            fontsize=7.4, color=INK_MUTED, ha="center", va="top",
                            linespacing=1.2)

        # Direct-label the aligned-under-attack endpoint: the series the argument
        # rests on, labelled once rather than at every point.
        # Value labels wear ink, not the series colour — the mark beside them
        # carries identity, and coloured text competes with the lines.
        for field in ("uncond_attacked_denial_aligned", "uncond_attacked_denial_base"):
            end = series(rows, field)[-1]
            if end is None:
                continue
            # Above the marker by default. Near the ceiling there is no room, so
            # drop under it — below the floor the x tick labels are in the way.
            dy = -17 if end > 0.85 else 11
            ax.annotate(f"{end:.2f}", xy=(x[-1], end),
                        xytext=(-4, dy), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", color=INK, ha="right")

        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=26, loc="left")
        ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_SECONDARY, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, fontsize=8.8, color=INK_SECONDARY)
        ax.set_xlim(-0.45, len(x) - 0.55)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(colors=INK_MUTED, length=0)

        # Per-panel side legend — the R1 panel names different models, so a single
        # figure-level legend would be wrong there.
        ax.legend(handles=[
            Line2D([0], [0], color=C_BASE, linewidth=2.0, marker="o", markersize=6.5,
                   markeredgecolor=C_BASE, label=base_lo),
            Line2D([0], [0], color=C_ALIGNED, linewidth=2.0, marker="s", markersize=6.5,
                   markeredgecolor=C_ALIGNED, label=aligned_lo),
        ], loc="upper left", fontsize=8.8, frameon=False, labelcolor=INK,
            handlelength=1.9, borderpad=0.2, labelspacing=0.35)

    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                            fontsize=9.5, color=INK_SECONDARY)
    axes[0].set_ylabel("Denial rate", fontsize=10.5, color=INK_SECONDARY, labelpad=10)

    fig.suptitle(
        "Aligned models refuse more as retrieval degrades; base models do not",
        fontsize=15, fontweight="bold", color=INK, x=0.008, ha="left", y=0.985,
    )
    fig.text(
        0.008, 0.925,
        "Denial rate by the rank the gold passage held BEFORE the attack. "
        "Dashed + hollow = no attack; solid + filled = under attack.",
        fontsize=10, color=INK_SECONDARY, ha="left",
    )
    fig.text(
        0.008, 0.028,
        "Every query counted once per attack, pooled within a bin. Shaded bins have "
        f"≤{THIN_CELL_MAX_RUNS} runs clearing the 25-answerable threshold and show pooled figures only.\n"
        "The gap is present with no attack at all (dashed lines), which is what makes this a "
        "property of the model rather than of the attack.   Source: results/retrieval_binning.csv",
        fontsize=8.6, color=INK_MUTED, ha="left", linespacing=1.5,
    )

    fig.subplots_adjust(left=0.055, right=0.988, top=0.775, bottom=0.235, wspace=0.075)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
