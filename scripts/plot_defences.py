"""Defence results in one figure: attributable ASR, undefended -> defended.

Reads results_defence/defence_report.json (produced by compare_defences.py) and
draws one dumbbell per (attack, model pair), one panel per defence arm. The
hollow grey dot is the undefended run, the filled dot the defended one.

Colour carries the verdict, not the series: blue where the defence lowers the
attacked denial rate significantly (McNemar p < 0.05), red where it raises it,
grey where it changes nothing. Blue and red are the two diverging poles of the
reference palette and pass the CVD validator as a pair (ΔE 21.6 protan). The
direct value label on every defended dot means colour is never the only cue.

Saved to results_defence/defence_asr.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "results_defence" / "defence_report.json"
OUT_PATH = REPO_ROOT / "results_defence" / "defence_asr.png"

# Tokens shared with plot_retrieval_gradient.py and sweep_cluster_filter.py.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
C_LOWER = "#2a78d6"   # diverging pole: defence helps
C_HIGHER = "#e34948"  # diverging pole: defence hurts
C_NULL = "#b5b4ad"

ALPHA = 0.05
ARMS = [
    ("def-cluster", "Cluster filter"),
    ("def-rerank", "Rerank"),
    ("def-cluster+rerank", "Cluster filter + rerank"),
    ("prompt-chat-no-refusal", "Prompt without refusal licence"),
]
ATTACKS = ["D2", "B3", "A2", "C1"]
PAIRS = [("llama-3.1-8b", "Llama"), ("qwen-2.5-7b", "Qwen")]


def main() -> None:
    rows = json.loads(REPORT.read_text(encoding="utf-8"))
    cell = {(r["family"], r["attack"], r["arm"]): r for r in rows if r["side"] == "aligned"}

    # Top-to-bottom: attacks in order, the two pairs adjacent within each attack.
    slots = [(a, fam, short) for a in ATTACKS for fam, short in PAIRS]
    y = {s[:2]: len(slots) - 1 - i - 0.6 * ATTACKS.index(s[0]) for i, s in enumerate(slots)}

    fig, axes = plt.subplots(1, len(ARMS), figsize=(13.5, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (arm, title) in zip(axes, ARMS):
        ax.set_facecolor(SURFACE)
        for attack, fam, _ in slots:
            yy = y[(attack, fam)]
            r = cell.get((fam, attack, arm))
            if r is None:
                ax.text(1, yy, "not run (walltime)", va="center", fontsize=7, color=INK_MUTED)
                continue
            u, d = 100 * r["undefended"]["asr"], 100 * r["defended"]["asr"]
            sig = r["p_att_denial"] < ALPHA
            colour = C_NULL if not sig else (C_LOWER if d < u else C_HIGHER)
            ax.plot([u, d], [yy, yy], color=colour, linewidth=2, solid_capstyle="round", zorder=2)
            # The undefended ring is hollow and drawn last, so it stays visible
            # when the defence barely moves the value and the dots overlap.
            ax.scatter([d], [yy], s=46, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter([u], [yy], s=38, facecolor="none", edgecolor=INK_MUTED, linewidth=1.4, zorder=4)
            # Label the defended value on the side away from the undefended dot.
            right = d >= u
            ax.text(d + (1.6 if right else -1.6), yy, f"{d:.0f}%", va="center",
                    ha="left" if right else "right", fontsize=7.5, color=INK_SECONDARY, zorder=5)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlim(-6, 64)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xticklabels(["0%", "20%", "40%", "60%"])
        ax.set_xlabel("attributable ASR", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)

    axes[0].set_yticks([y[s[:2]] for s in slots])
    axes[0].set_yticklabels([f"{a} · {short}" for a, _, short in slots], fontsize=8.5, color=INK)

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=INK_MUTED, markersize=6.5, label="undefended"),
        Line2D([], [], color=C_LOWER, marker="o", linewidth=2, markersize=6.5,
               markeredgecolor=SURFACE, label="defended: lower (p < 0.05)"),
        Line2D([], [], color=C_HIGHER, marker="o", linewidth=2, markersize=6.5,
               markeredgecolor=SURFACE, label="defended: higher (p < 0.05)"),
        Line2D([], [], color=C_NULL, marker="o", linewidth=2, markersize=6.5,
               markeredgecolor=SURFACE, label="no significant change"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=4,
               frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.text(0.005, 0.005,
             "Aligned side, NQ, 1,000 queries per run. p = exact McNemar on attacked denials, "
             "paired per query with the undefended run. Source: results_defence/defence_report.json.",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
