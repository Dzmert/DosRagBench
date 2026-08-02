"""Visual summary of the DoSRAGBench alignment-paradox results.

Reads results/avi_report_clean.json (produced by clean_avi_report.py) and renders
a two-panel figure:
  - Left:  aligned-model ASR (%) per attack x model pair  -> attack strength.
  - Right: AVI regime per attack x model pair (paradox / independent / protective).

Saved to results/avi_summary.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

FAMILIES = ["llama-3.1-8b", "mistral-7b", "qwen-2.5-7b", "llama-r1-8b"]
FAMILY_LABELS = {
    "llama-3.1-8b": "Llama-3.1\nBase→Instruct",
    "mistral-7b": "Mistral-7B\nBase→Instruct",
    "qwen-2.5-7b": "Qwen-2.5\nBase→Instruct",
    "llama-r1-8b": "Llama-R1\nInstruct→R1-distill",
}
ATTACK_ORDER = [
    "A1", "A2", "A3", "B1", "B2", "B3",
    "C1", "C2", "C3", "D1", "D2", "D3", "D4",
]


def main() -> None:
    data = json.loads((RESULTS_DIR / "avi_report_clean.json").read_text())
    kept = data["kept"]
    by = {(e["family"], e["attack_category"]): e for e in kept}
    attacks = [a for a in ATTACK_ORDER if any((f, a) in by for f in FAMILIES)]

    asr = np.full((len(attacks), len(FAMILIES)), np.nan)
    avi = np.full((len(attacks), len(FAMILIES)), np.nan)
    for i, a in enumerate(attacks):
        for j, f in enumerate(FAMILIES):
            e = by.get((f, a))
            if e is None:
                continue
            asr[i, j] = e["aligned_asr"] * 100
            avi[i, j] = e["avi_asr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
    xlabels = [FAMILY_LABELS[f] for f in FAMILIES]

    # ---- Panel 1: aligned ASR heatmap ----
    im1 = ax1.imshow(asr, cmap="Reds", aspect="auto", vmin=0, vmax=np.nanmax(asr))
    ax1.set_title("Aligned-model ASR (%)\nattack-attributable denial rate", fontsize=12, pad=10)
    for i in range(len(attacks)):
        for j in range(len(FAMILIES)):
            if np.isnan(asr[i, j]):
                ax1.text(j, i, "–", ha="center", va="center", color="gray")
                continue
            v = asr[i, j]
            ax1.text(j, i, f"{v:.1f}", ha="center", va="center",
                     color="white" if v > 0.55 * np.nanmax(asr) else "black", fontsize=9)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Aligned ASR (%)")

    # ---- Panel 2: AVI regime heatmap (categorical) ----
    # 0 protective (<0.8), 1 independent (0.8-1.5), 2 moderate (1.5-3), 3 strong (>=3)
    regime = np.full_like(avi, np.nan)
    for i in range(len(attacks)):
        for j in range(len(FAMILIES)):
            v = avi[i, j]
            if np.isnan(v):
                continue
            regime[i, j] = 0 if v < 0.8 else 1 if v < 1.5 else 2 if v < 3.0 else 3
    cmap = ListedColormap(["#2e7d32", "#8fbf6f", "#f0ad4e", "#c0392b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im2 = ax2.imshow(regime, cmap=cmap, norm=norm, aspect="auto")
    ax2.set_title("AVI regime  (=ASR_aligned / ASR_base)\nalignment paradox map", fontsize=12, pad=10)
    for i in range(len(attacks)):
        for j in range(len(FAMILIES)):
            v = avi[i, j]
            if np.isnan(v):
                ax2.text(j, i, "–", ha="center", va="center", color="gray")
                continue
            label = f"{v:.2f}" if v < 100 else "∞"
            ax2.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar2.ax.set_yticklabels(["protective\n<0.8", "independent\n0.8–1.5",
                              "moderate\n1.5–3", "strong\n≥3"])

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks, fontsize=9)
        ax.set_xticks(np.arange(-.5, len(FAMILIES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(attacks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    fig.suptitle(
        "DoSRAGBench — Alignment Paradox Summary (thin-sample runs excluded)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS_DIR / "avi_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
