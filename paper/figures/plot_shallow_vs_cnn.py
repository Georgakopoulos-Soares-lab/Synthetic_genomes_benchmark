#!/usr/bin/env python3
"""
Figure + table for Reviewer #1 comment #2: shallow baseline vs CNN.

Reads revisions/results/shallow_baseline_results.json and plots, per domain,
the AUROC of shallow models (k-mer LogReg, Markov-1, GC-only) against the CNN
AUROC reported in the manuscript. Produces:
  revisions/figures/shallow_vs_cnn_auroc.png
  revisions/results/shallow_vs_cnn_table.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(_ROOT)
RES = PROJ / "revisions" / "results"
FIG = PROJ / "revisions" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# CNN AUROC reported in the manuscript (aggregated / >=8 chunks).
CNN_AUROC = {"euk": 0.97, "prok": 0.82, "vir": 0.69}

DOMAIN_LABEL = {"euk": "Eukaryotes", "prok": "Prokaryotes", "vir": "Viruses"}
FEAT_LABEL = {
    "kmer": "k-mer freq (k=4), LogReg",
    "markov1": "Markov-1, LogReg",
    "gc": "GC + monont, LogReg",
}
FEAT_COLOR = {"kmer": "#1f77b4", "markov1": "#ff7f0e", "gc": "#2ca02c"}


def main() -> int:
    with open(RES / "shallow_baseline_results.json") as fh:
        summaries = json.load(fh)

    # keep logistic regression rows (linsvm nearly identical)
    rows = [s for s in summaries if s["model"] == "logreg"]
    df = pd.DataFrame(rows)

    table_rows = []
    for domain in ["euk", "prok", "vir"]:
        for feat in ["kmer", "markov1", "gc"]:
            r = df[(df.domain == domain) & (df.feature == feat)]
            if r.empty:
                continue
            table_rows.append({
                "domain": domain,
                "feature": feat,
                "shallow_auroc_mean": float(r.auroc_mean.iloc[0]),
                "shallow_auroc_std": float(r.auroc_std.iloc[0]),
                "cnn_auroc_manuscript": CNN_AUROC[domain],
                "shallow_minus_cnn": float(r.auroc_mean.iloc[0]) - CNN_AUROC[domain],
            })
    tab = pd.DataFrame(table_rows)
    tab.to_csv(RES / "shallow_vs_cnn_table.csv", index=False)

    # --- grouped bar chart ---
    domains = ["euk", "prok", "vir"]
    feats = ["kmer", "markov1", "gc"]
    x = np.arange(len(domains))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
    for i, feat in enumerate(feats):
        means = [df[(df.domain == d) & (df.feature == feat)].auroc_mean.iloc[0]
                 for d in domains]
        stds = [df[(df.domain == d) & (df.feature == feat)].auroc_std.iloc[0]
                for d in domains]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=FEAT_LABEL[feat], color=FEAT_COLOR[feat], alpha=0.9)

    # CNN reference markers (black diamonds + dashed connectors)
    cnn_vals = [CNN_AUROC[d] for d in domains]
    ax.scatter(x, cnn_vals, marker="D", s=70, color="black", zorder=5,
               label="CNN (manuscript)")
    for xi, cv in zip(x, cnn_vals):
        ax.hlines(cv, xi - 1.5 * width, xi + 1.5 * width, colors="black",
                  linestyles="dashed", linewidth=1)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.text(len(domains) - 0.5, 0.505, "chance", color="grey", fontsize=8,
            ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in domains])
    ax.set_ylabel("Leave-one-tag-out AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Shallow compositional classifiers vs. CNN\n"
                 "(natural vs. synthetic discrimination)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = FIG / "shallow_vs_cnn_auroc.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[ok] {out}")
    print(f"[ok] {RES / 'shallow_vs_cnn_table.csv'}")
    print("\n=== shallow vs CNN (LogReg) ===")
    print(tab.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
