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

A second figure, results_defence/defence_accuracy.png, shows what ASR cannot: the
base side barely refuses, so its attack losses are wrong answers, not denials.
It plots accuracy under attack on the NQ gold subset, undefended -> cluster
filter, for both sides of the Llama pair, against each side's clean accuracy.
Colour there carries the side (the two categorical slots plot_retrieval_gradient.py
uses for base and aligned), and "n.s." marks a change McNemar does not support.
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
ACC_OUT_PATH = REPO_ROOT / "results_defence" / "defence_accuracy.png"

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
C_BASE = "#2a78d6"     # categorical slot 1, as in plot_retrieval_gradient.py
C_ALIGNED = "#eb6834"  # categorical slot 2

ALPHA = 0.05
ARMS = [
    ("def-cluster", "Cluster filter"),
    ("def-cluster-t0.83-n3", "Cluster filter, τ 0.83 / ≥3"),
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

    fig, axes = plt.subplots(1, len(ARMS), figsize=(16.5, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (arm, title) in zip(axes, ARMS):
        ax.set_facecolor(SURFACE)
        for attack, fam, _ in slots:
            yy = y[(attack, fam)]
            r = cell.get((fam, attack, arm))
            if r is None:
                why = "not run (walltime)" if arm.startswith("prompt") else "not run"
                ax.text(1, yy, why, va="center", fontsize=7, color=INK_MUTED)
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
    plot_accuracy(rows)


def plot_accuracy(rows: list) -> None:
    fam = "llama-3.1-8b"
    cell = {(r["side"], r["attack"]): r for r in rows
            if r["family"] == fam and r["arm"] == "def-cluster"}
    sides = [("base", "Base model", C_BASE), ("aligned", "Aligned model", C_ALIGNED)]
    if not all((side, a) in cell for side, _, _ in sides for a in ATTACKS):
        print("  skip defence_accuracy.png: base-side cluster runs missing")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    gap = 0.32
    for i, attack in enumerate(ATTACKS):
        for j, (side, _, colour) in enumerate(sides):
            r = cell[(side, attack)]
            yy = len(ATTACKS) - 1 - i + (gap / 2 if j == 0 else -gap / 2)
            u, d = 100 * r["undefended"]["acc_att"], 100 * r["defended"]["acc_att"]
            ax.plot([u, d], [yy, yy], color=colour, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([d], [yy], s=46, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter([u], [yy], s=38, facecolor="none", edgecolor=INK_MUTED, linewidth=1.4, zorder=4)
            note = "" if r["p_att_acc"] < ALPHA else "  n.s."
            ax.text(max(u, d) + 1.0, yy, f"{d:.0f}%{note}", va="center", ha="left",
                    fontsize=8, color=INK_SECONDARY, zorder=5)
    # Clean accuracy per side: the level a perfect defence would restore.
    for side, label, colour in sides:
        clean = 100 * cell[(side, ATTACKS[0])]["undefended"]["acc_clean"]
        ax.axvline(clean, color=colour, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
        # Aligned clean sits left of base clean; label each on its outer side.
        left = side == "aligned"
        ax.text(clean + (-0.4 if left else 0.4), len(ATTACKS) - 0.45,
                f"{label.split()[0].lower()} clean {clean:.0f}%",
                ha="right" if left else "left", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    ax.set_yticks(range(len(ATTACKS)))
    ax.set_yticklabels(list(reversed(ATTACKS)), fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(ATTACKS) - 0.1)
    ax.set_xlim(15, 62)
    ax.set_xticks([20, 30, 40, 50, 60])
    ax.set_xticklabels(["20%", "30%", "40%", "50%", "60%"])
    ax.set_xlabel("accuracy under attack (NQ gold subset, n = 656)", fontsize=8.5, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_title("The cluster filter restores correct answers on both sides",
                 fontsize=10.5, color=INK, loc="left", pad=22)

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=INK_MUTED, markersize=6.5, label="undefended"),
        Line2D([], [], color=C_BASE, marker="o", linewidth=2, markersize=6.5,
               markeredgecolor=SURFACE, label="base, cluster filter"),
        Line2D([], [], color=C_ALIGNED, marker="o", linewidth=2, markersize=6.5,
               markeredgecolor=SURFACE, label="aligned, cluster filter"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=3,
               frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.text(0.005, 0.005,
             "Llama-3.1-8B, τ 0.90 / ≥2. Base denial stays at 0.4–3.3% throughout, so its losses "
             "are silent wrong answers.\nn.s. = exact McNemar p ≥ 0.05. Dashed: clean accuracy. "
             "Source: results_defence/defence_report.json.",
             fontsize=7, color=INK_MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(ACC_OUT_PATH, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {ACC_OUT_PATH}")


if __name__ == "__main__":
    main()
