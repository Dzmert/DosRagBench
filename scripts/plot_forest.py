"""Forest plot of base-vs-aligned ASR for the GENUINE paradox runs.

Reads results/avi_significance.json (produced by compute_significance.py) and
renders one row per genuine paradox: base and aligned attack-attributable ASR
with 95% Wilson confidence intervals, grouped by model family, annotated with
the risk difference (ASR_aligned - ASR_base, pp) and the aligned-model McNemar
discordant counts. NQ-only; the 12 HotpotQA runs were withdrawn.

The visual argument is the CI separation: where the base interval and the
aligned interval do not touch, the paradox is not a point-estimate artifact.

Saved to results/forest_genuine.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Categorical slots 1 and 2 (validated all-pairs, light surface).
C_BASE = "#2a78d6"
C_ALIGNED = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FAMILY_LABELS = {
    "llama-3.1-8b": "Llama-3.1-8B   base → instruct",
    "mistral-7b": "Mistral-7B   base → instruct",
    "qwen-2.5-7b": "Qwen-2.5-7B   base → instruct",
    "llama-r1-8b": "Llama-3.1-8B   instruct → R1-distill",
}
FAMILY_ORDER = ["llama-3.1-8b", "mistral-7b", "qwen-2.5-7b", "llama-r1-8b"]
ATTACK_NAMES = {
    "A1": "Guardrail Triggering",
    "A1_instructional": "Guardrail Triggering (instr.)",
    "A2": "Contradiction Flooding",
    "A3": "Authority Spoofing",
    "B1": "Context Window Saturation",
    "B2": "Generation Loop Induction",
    "B3": "Multi-Retrieval Amplification",
    "C1": "Embedding Space Clustering",
    "C2": "Index Pollution",
    "C3": "Adversarial Embedding Perturbation",
    "D1": "Logical Contradiction Traps",
    "D2": "Circular Reference Chains",
    "D3": "Epistemic Uncertainty Amplification",
    "D4": "Infinite Qualification Traps",
}


def main() -> None:
    data = json.loads((RESULTS_DIR / "avi_significance.json").read_text())
    # NQ-only: the 12 HotpotQA runs were withdrawn (two-hop retrieval confound).
    genuine = [r for r in data["rows"]
               if r["verdict"] == "GENUINE paradox" and r["dataset"] == "NQ"]

    # Group by family, strongest effect first within each group. Rows are laid
    # out top-down, so build the list reversed for matplotlib's bottom-up y-axis.
    grouped: list[tuple[str, list[dict]]] = []
    for fam in FAMILY_ORDER:
        runs = [r for r in genuine if r["family"] == fam]
        if runs:
            runs.sort(key=lambda r: r["aligned_asr"], reverse=True)
            grouped.append((fam, runs))

    # Assign y positions: one slot per run, one blank slot per group header.
    ypos: list[float] = []
    ylabels: list[str] = []
    header_y: list[tuple[float, str, int]] = []
    y = 0.0
    plot_rows: list[dict] = []
    for fam, runs in reversed(grouped):
        for r in reversed(runs):
            ypos.append(y)
            ylabels.append(f"{r['attack']}  {ATTACK_NAMES.get(r['attack'], r['attack'])}")
            plot_rows.append(r)
            y += 1.0
        header_y.append((y, FAMILY_LABELS.get(fam, fam), len(runs)))
        y += 1.0

    fig, (ax, axt) = plt.subplots(
        1, 2, figsize=(13.5, 8.2), gridspec_kw={"width_ratios": [3.05, 1.0], "wspace": 0.02},
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axt):
        a.set_facecolor(SURFACE)

    offset = 0.17
    for yv, r in zip(ypos, plot_rows):
        for key_asr, key_ci, color, sign in (
            ("base_asr", "base_ci", C_BASE, +1),
            ("aligned_asr", "aligned_ci", C_ALIGNED, -1),
        ):
            asr = r[key_asr] * 100
            lo, hi = r[key_ci][0] * 100, r[key_ci][1] * 100
            yy = yv + sign * offset
            ax.plot([lo, hi], [yy, yy], color=color, linewidth=2.0,
                    solid_capstyle="round", zorder=2)
            # 2px surface ring keeps the marker legible where intervals overlap.
            ax.plot([asr], [yy], marker="o", markersize=8.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)

    # Row separators, recessive.
    for yv in ypos:
        ax.axhline(yv + 0.5, color=GRID, linewidth=0.6, zorder=0)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=9.5, color=INK)
    ax.set_ylim(-0.8, y - 0.4)
    # Fit the full spread: the strongest aligned CIs reach ~62% (D2 family), so a
    # 25% limit clipped 14 of 32 rows -- including the poster-child D2 markers.
    ax.set_xlim(-1.5, 65)
    ax.set_xlabel("Attack-attributable denial rate, ASR (%)   —   95% Wilson CI",
                  fontsize=10, color=INK_SECONDARY, labelpad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    # Family group headers, spanning both panels.
    for yv, label, n in header_y:
        ax.text(-0.6, yv, f"{label}   ({n})", fontsize=10, fontweight="bold",
                color=INK, va="center", ha="left")
        ax.axhline(yv - 0.5, color="#c3c2b7", linewidth=1.0, zorder=1)
        axt.axhline(yv - 0.5, color="#c3c2b7", linewidth=1.0, zorder=1)

    # ---- Right panel: the numbers a forest plot is expected to carry ----
    axt.set_ylim(ax.get_ylim())
    axt.set_xlim(0, 1)
    axt.axis("off")
    top = y - 0.4
    axt.text(0.14, top - 0.35, "Risk Δ pp", fontsize=9.5, fontweight="bold",
             color=INK_SECONDARY, ha="center", va="top")
    axt.text(0.52, top - 0.35, "McNemar c/b", fontsize=9.5, fontweight="bold",
             color=INK_SECONDARY, ha="center", va="top")
    axt.text(0.90, top - 0.35, "FDR q", fontsize=9.5, fontweight="bold",
             color=INK_SECONDARY, ha="center", va="top")
    for yv, r in zip(ypos, plot_rows):
        axt.text(0.14, yv, f"{r['risk_diff'] * 100:+.0f}", fontsize=9.5, color=INK,
                 ha="center", va="center", fontfamily="monospace")
        axt.text(0.52, yv, f"{r['aligned_mcnemar_c']}/{r['aligned_mcnemar_b']}",
                 fontsize=9.5, color=INK, ha="center", va="center",
                 fontfamily="monospace")
        axt.text(0.90, yv, f"{r['fisher_q']:.0e}", fontsize=9, color=INK_SECONDARY,
                 ha="center", va="center", fontfamily="monospace")

    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=C_BASE, marker="o", markersize=8.5, linewidth=2.0,
                   markeredgecolor=SURFACE, markeredgewidth=2.0, label="Base model"),
            Line2D([0], [0], color=C_ALIGNED, marker="o", markersize=8.5, linewidth=2.0,
                   markeredgecolor=SURFACE, markeredgewidth=2.0, label="Aligned model"),
        ],
        loc="upper right", frameon=False, fontsize=10, ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.suptitle(
        f"Base vs. aligned denial rate — the {len(plot_rows)} genuine alignment paradoxes",
        fontsize=14, fontweight="bold", color=INK, x=0.5, y=0.975,
    )
    fig.text(
        0.5, 0.937,
        "Runs passing both FDR-corrected Fisher (aligned ≠ base) and McNemar "
        "(the attack breaks queries that worked clean).  Non-overlapping intervals = "
        "the separation is not a point-estimate artifact.",
        fontsize=9.5, color=INK_SECONDARY, ha="center",
    )

    fig.tight_layout(rect=[0, 0.0, 1, 0.925])
    out = RESULTS_DIR / "forest_genuine.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}  ({len(plot_rows)} genuine runs)")


if __name__ == "__main__":
    main()
