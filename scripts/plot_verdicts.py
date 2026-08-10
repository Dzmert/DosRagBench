"""Verdict breakdown of all kept DoSRAGBench runs.

Reads results/avi_significance.json and renders the filtering story: of the kept
runs, how many point in the paradox direction, and what survives the conjunction
of the FDR-corrected Fisher test and the within-model McNemar test.

Deliberately split by direction first. The four substantive verdicts do not all
live under "aligned denies more" — `protective` is by definition a run where the
*base* model denies more, so it belongs to the other group. Presenting all four
as a breakdown of the paradox-direction runs double-counts the split.

Saved to results/verdict_breakdown.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Which dataset arm to plot. Defaults to NQ (the headline); set DATASET=FiQA to
# render the generalisation arm to a suffixed file, leaving NQ figures untouched.
DATASET = os.environ.get("DATASET", "NQ")
SUFFIX = "" if DATASET == "NQ" else f"_{DATASET.lower()}"

# Categorical slots 1-4 for the substantive verdicts; the null bucket takes a
# neutral gray so "no effect" stays recessive rather than reading as a finding.
COLORS = {
    "GENUINE paradox": "#2a78d6",
    "floor artifact": "#eb6834",
    "attack, indep.": "#1baf7a",
    "protective": "#eda100",
    "n.s.": "#d5d4cd",
}
LABELS = {
    "GENUINE paradox": "Genuine paradox",
    "floor artifact": "Floor artifact",
    "attack, indep.": "Attack, alignment-independent",
    "protective": "Alignment protective",
    "n.s.": "Not significant",
}
BLURBS = {
    "GENUINE paradox": "passes Fisher + McNemar",
    "floor artifact": "Fisher only — baseline refusal floor,\nnot the attack",
    "attack, indep.": "attack works, hits base and\naligned equally",
    "protective": "base denies significantly more",
    "n.s.": "no significant difference",
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"


def main() -> None:
    data = json.loads((RESULTS_DIR / "avi_significance.json").read_text())
    # NQ-only: the 12 HotpotQA runs were withdrawn (two-hop retrieval confound,
    # 24.3% joint gold recall@5 -- refusals there are correct, not adversarial).
    rows = [r for r in data["rows"] if r["dataset"] == DATASET]
    kept = len(rows)

    up = [r for r in rows if r["aligned_asr"] > r["base_asr"]]
    down = [r for r in rows if r["aligned_asr"] <= r["base_asr"]]

    def tally(group, order):
        return [(v, sum(1 for r in group if r["verdict"] == v)) for v in order]

    bars = [
        (f"Aligned denies more\n{len(up)} of {kept} runs",
         tally(up, ["GENUINE paradox", "floor artifact", "attack, indep.", "n.s."])),
        (f"Base denies more (or equal)\n{len(down)} of {kept} runs",
         tally(down, ["protective", "n.s."])),
    ]

    fig, ax = plt.subplots(figsize=(13, 4.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_h = 0.42
    ypos = [1.0, 0.0]
    gap = 0.16  # surface gap between segments, in data units (~2px at this scale)

    for yv, (label, segs) in zip(ypos, bars):
        x = 0.0
        for verdict, n in segs:
            if n == 0:
                continue
            ax.barh(yv, n - gap, left=x, height=bar_h,
                    color=COLORS[verdict], zorder=2, linewidth=0)
            # Direct label on every segment — this is the relief channel for the
            # sub-3:1 fills, so no value depends on reading the color alone.
            seg_ink = "#ffffff" if verdict in ("GENUINE paradox", "floor artifact") else INK
            ax.text(x + (n - gap) / 2, yv, str(n), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=seg_ink, zorder=3)
            x += n
        ax.text(-0.5, yv, label, ha="right", va="center", fontsize=10.5,
                color=INK, linespacing=1.5)

    # Callout for the headline claim. Derived from the NQ-only rows above, not
    # data["counts"] (which still totals the HotpotQA-inclusive 62-run set).
    n_genuine = sum(1 for r in rows if r["verdict"] == "GENUINE paradox")
    ax.annotate(
        f"{n_genuine} survive both tests",
        xy=(n_genuine / 2, 1.0 + bar_h / 2), xycoords="data",
        xytext=(n_genuine / 2, 1.72), textcoords="data",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.0,
                        shrinkA=0, shrinkB=3),
    )

    ax.set_xlim(-0.5, len(up) + 0.5)
    ax.set_ylim(-0.62, 2.05)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    order = ["GENUINE paradox", "floor artifact", "attack, indep.", "protective", "n.s."]
    legend = ax.legend(
        handles=[Patch(facecolor=COLORS[v], label=f"{LABELS[v]} — {BLURBS[v]}")
                 for v in order],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
        ncol=3, fontsize=9.5, handlelength=1.5, handleheight=1.1,
        labelspacing=1.0, columnspacing=2.0,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle(
        "Not every apparent paradox is real"
        f"{'' if DATASET == 'NQ' else f'  ·  {DATASET}'}",
        fontsize=15, fontweight="bold", color=INK, x=0.5, y=0.98,
    )
    fig.text(
        0.5, 0.905,
        f"{len(up)} of {kept} kept runs point in the paradox direction, but only "
        f"{n_genuine} pass both the FDR-corrected Fisher test and the "
        f"within-model McNemar test.",
        fontsize=10.5, color=INK_SECONDARY, ha="center",
    )
    fig.text(
        0.5, 0.012,
        f"Verdicts partition all {kept} kept runs. 'Protective' cannot appear in the "
        "paradox-direction group — it is defined by the base model denying more.",
        fontsize=8.5, color=INK_MUTED, ha="center", style="italic",
    )

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.87])
    out = RESULTS_DIR / f"verdict_breakdown{SUFFIX}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")
    for label, segs in bars:
        print(f"  {label.splitlines()[0]}: " +
              ", ".join(f"{LABELS[v]}={n}" for v, n in segs if n))


if __name__ == "__main__":
    main()
