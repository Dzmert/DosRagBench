"""Visual summary of the DoSRAGBench alignment results (NQ-only, no AVI).

Reads results/avi_report_clean.json (produced by clean_avi_report.py) and renders
a two-panel figure over the NQ runs:
  - Left:  aligned-model ASR (%) per attack x model pair  -> attack strength.
  - Right: risk difference (pp), ASR_aligned - ASR_base   -> the alignment effect.

The risk difference replaces the old AVI ratio: AVI explodes when base ASR is
near zero (most runs), so the field-standard reporting is the difference of the
two attack success rates, not their ratio. Diverging colour: red = aligned denies
more (the paradox), blue = base denies more (protective).

The 12 HotpotQA runs are excluded -- that arm was withdrawn (two-hop retrieval
confound: 24.3% joint gold recall@5, so refusals there are correct not adversarial).

Saved to results/asr_risk_summary.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Which dataset arm to plot. Defaults to NQ (the headline); set DATASET=FiQA to
# render the generalisation arm to a suffixed file, leaving NQ figures untouched.
DATASET = os.environ.get("DATASET", "NQ")
SUFFIX = "" if DATASET == "NQ" else f"_{DATASET.lower()}"

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
    # Defaults to NQ (headline); DATASET=FiQA renders the generalisation arm.
    # The 12 HotpotQA runs were withdrawn.
    kept = [e for e in data["kept"] if e["dataset"] == DATASET]
    by = {(e["family"], e["attack_category"]): e for e in kept}
    attacks = [a for a in ATTACK_ORDER if any((f, a) in by for f in FAMILIES)]

    asr = np.full((len(attacks), len(FAMILIES)), np.nan)
    risk = np.full((len(attacks), len(FAMILIES)), np.nan)
    for i, a in enumerate(attacks):
        for j, f in enumerate(FAMILIES):
            e = by.get((f, a))
            if e is None:
                continue
            asr[i, j] = e["aligned_asr"] * 100
            risk[i, j] = (e["aligned_asr"] - e["base_asr"]) * 100

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

    # ---- Panel 2: risk-difference heatmap (diverging, centred at 0) ----
    lim = np.nanmax(np.abs(risk))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im2 = ax2.imshow(risk, cmap="RdBu_r", norm=norm, aspect="auto")
    ax2.set_title("Risk difference (pp)\nASR_aligned − ASR_base", fontsize=12, pad=10)
    for i in range(len(attacks)):
        for j in range(len(FAMILIES)):
            v = risk[i, j]
            if np.isnan(v):
                ax2.text(j, i, "–", ha="center", va="center", color="gray")
                continue
            ax2.text(j, i, f"{v:+.0f}", ha="center", va="center",
                     color="white" if abs(v) > 0.55 * lim else "black", fontsize=9)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Risk difference (pp)   red = aligned denies more")

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
        f"DoSRAGBench — Alignment effect on {DATASET} (thin-sample runs excluded)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS_DIR / f"asr_risk_summary{SUFFIX}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}  ({len(kept)} NQ runs)")


if __name__ == "__main__":
    main()
