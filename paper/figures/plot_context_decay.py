#!/usr/bin/env python3
"""
Plot distance-from-seed AUROC decay curves for the seed-length experiment.

Reads context_decay_auroc.csv produced by context_decay_auroc.py and draws
one curve per seed length showing how AUROC evolves as a function of distance
from the seed end (analogous to Fig. 8C-D of the paper).

Usage:
    python3 plot_context_decay.py \
        --results  /path/to/results/context_decay/context_decay_auroc.csv \
        --outdir   /path/to/figures
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS_ORDERED = ["seed3k", "seed10k", "seed20k"]
COLORS = {"seed3k": "#2196F3", "seed10k": "#FF9800", "seed20k": "#4CAF50"}
LABELS = {"seed3k": "3 kb seed", "seed10k": "10 kb seed", "seed20k": "20 kb seed"}
MARKERS = {"seed3k": "o", "seed10k": "s", "seed20k": "^"}


def main():
    ap = argparse.ArgumentParser(
        description="Plot distance-from-seed AUROC curves")
    ap.add_argument("--results", required=True,
                    help="Path to context_decay_auroc.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data: config -> bin_start_kb -> {mean, std}
    data = defaultdict(dict)
    with open(args.results, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["auroc_mean"]:
                data[row["config"]][float(row["bin_start_kb"])] = {
                    "mean": float(row["auroc_mean"]),
                    "std":  float(row["auroc_std"]) if row["auroc_std"] else 0.0,
                }

    configs_present = [c for c in CONFIGS_ORDERED if c in data]
    if not configs_present:
        print("[warn] no data found in results CSV — nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cfg in configs_present:
        bk_sorted = sorted(data[cfg])
        x = np.array(bk_sorted)
        y = np.array([data[cfg][bk]["mean"] for bk in bk_sorted])
        yerr = np.array([data[cfg][bk]["std"]  for bk in bk_sorted])

        color = COLORS[cfg]
        marker = MARKERS[cfg]
        ax.plot(x, y, marker + "-", color=color, label=LABELS[cfg],
                lw=2, ms=5, zorder=3)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)

    ax.axhline(0.5, color="0.55", linestyle="--", lw=1.2,
               label="Chance (AUROC = 0.5)", zorder=2)

    ax.set_xlabel("Distance from seed end (kb)", fontsize=11)
    ax.set_ylabel("AUROC (synthetic vs. natural, 6-mer LogReg)", fontsize=11)
    ax.set_title(
        "Context decay: detectability vs. distance from conditioning seed\n"
        "(Human, 5 windows, 5-fold CV)",
        fontsize=10)
    ax.set_ylim(0.44, 1.03)
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = outdir / f"context_decay_curves.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[done] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
