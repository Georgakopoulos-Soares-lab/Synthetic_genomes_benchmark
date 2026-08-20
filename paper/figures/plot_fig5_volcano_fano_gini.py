#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Figure 5: TFBS volcano (top) + Fano / Gini hotspot boxplots (bottom).

Layout (portrait, 8.5 in wide x 11 in tall):
  - Top row (full width): TFBS volcano plot
  - Bottom row: Fano factor boxplot (left), Gini coefficient boxplot (right)
No suptitle, no correlation scatter panels.

Inputs:
  volcano : results/metrics/Publish_Human/tfbs_aggregate_stats.csv
  boxplots: results/metrics/Publish_Human/tfbs_clustering_stats.csv

CPU-only.
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

try:
    from adjustText import adjust_text
except ImportError as e:
    raise SystemExit(
        "Missing dependency: adjustText\n"
        "  pip install --user adjustText\n"
        "  conda install -c conda-forge adjusttext\n"
    ) from e

BASE = _ROOT
AGG_CSV = f"{BASE}/results/metrics/Publish_Human/tfbs_aggregate_stats.csv"
CLUST_CSV = f"{BASE}/results/metrics/Publish_Human/tfbs_clustering_stats.csv"
FIG_DIR = f"{BASE}/revisions/figures/"
OUT = FIG_DIR + "fig5_volcano_fano_gini.png"
os.makedirs(FIG_DIR, exist_ok=True)

# --- volcano config (matches plot_tfbs_volcano_from_csv_adjusttext.py) ---
VOLCANO_P_THRESH = 0.05
VOLCANO_LOG2FC_THRESH = 1.0
TOP_N_LABELS = 10
POINT_SIZE = 10
TOP_POINT_SIZE = 45
ALPHA = 0.7

# --- boxplot config (matches plot_fig5B_tfbs_hotspot.py) ---
METRICS = [
    ("fano", "Fano factor\n(variance/mean)"),
    ("gini", "Gini coefficient"),
]
ORIG_C, SYN_C = "#1b9e77", "#d95f02"


def draw_volcano(ax):
    df = pd.read_csv(AGG_CSV)
    if "neglog10_pvalue" not in df.columns:
        p = np.clip(df["pvalue_fisher"].astype(float).to_numpy(), 1e-300, 1.0)
        df["neglog10_pvalue"] = -np.log10(p)
    df["log2FC_orig_over_syn"] = pd.to_numeric(df["log2FC_orig_over_syn"], errors="coerce")
    df["pvalue_fisher"] = pd.to_numeric(df["pvalue_fisher"], errors="coerce")
    df["neglog10_pvalue"] = pd.to_numeric(df["neglog10_pvalue"], errors="coerce")
    df = df.dropna(subset=["log2FC_orig_over_syn", "pvalue_fisher", "neglog10_pvalue"])

    x = df["log2FC_orig_over_syn"].to_numpy()
    y = df["neglog10_pvalue"].to_numpy()
    pvals = df["pvalue_fisher"].to_numpy()

    base_colors = []
    for xi, pi in zip(x, pvals):
        if (pi < VOLCANO_P_THRESH) and (abs(xi) > VOLCANO_LOG2FC_THRESH):
            base_colors.append("red" if xi > 0 else "blue")
        else:
            base_colors.append("grey")

    ax.scatter(x, y, s=POINT_SIZE, c=base_colors, alpha=ALPHA, linewidths=0, zorder=1)
    ax.axvline(VOLCANO_LOG2FC_THRESH, linestyle="--")
    ax.axvline(-VOLCANO_LOG2FC_THRESH, linestyle="--")
    ax.axhline(-math.log10(VOLCANO_P_THRESH), linestyle="--")
    ax.set_xlabel("log2(Natural / Synthetic)")
    ax.set_ylabel("-log10(Fisher p-value)")

    top_df = df.nsmallest(TOP_N_LABELS, "pvalue_fisher").copy()
    cmap = plt.get_cmap("tab10")
    texts = []
    for i, row in enumerate(top_df.itertuples(index=False)):
        xi = float(row.log2FC_orig_over_syn)
        yi = float(row.neglog10_pvalue)
        ax.scatter([xi], [yi], s=TOP_POINT_SIZE, color=cmap(i % 10),
                   edgecolor="black", linewidths=0.5, zorder=3)
        texts.append(ax.text(xi, yi, str(row.motif_key), fontsize=7, zorder=4))

    if texts:
        adjust_text(
            texts,
            x=top_df["log2FC_orig_over_syn"].to_numpy(),
            y=top_df["neglog10_pvalue"].to_numpy(),
            expand_text=(1.05, 1.2),
            expand_points=(1.2, 1.4),
            force_text=(0.2, 0.5),
            force_points=(0.2, 0.5),
            arrowprops=dict(arrowstyle="-", lw=0.6),
            ax=ax,
        )


def draw_boxplot(ax, df, key, label):
    o = df[f"orig_{key}"].values.astype(float)
    s = df[f"syn_{key}"].values.astype(float)
    mask = np.isfinite(o) & np.isfinite(s)
    o, s = o[mask], s[mask]
    try:
        _, p = wilcoxon(s, o)
    except ValueError:
        p = np.nan

    bp = ax.boxplot([o, s], positions=[1, 2], widths=0.5, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], [ORIG_C, SYN_C]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("black")
    for xi, yi in zip(o, s):
        ax.plot([1, 2], [xi, yi], color="grey", alpha=0.3, lw=0.6, zorder=1)
    ax.scatter(np.ones_like(o), o, color=ORIG_C, s=14, zorder=2)
    ax.scatter(2 * np.ones_like(s), s, color=SYN_C, s=14, zorder=2)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Natural", "Synthetic"])
    ax.set_ylabel(label)
    sig = "ns" if (np.isnan(p) or p >= 0.05) else (
        "****" if p < 1e-4 else
        ("***" if p < 1e-3 else ("**" if p < 1e-2 else "*")))
    ptxt = "p=n/a" if np.isnan(p) else f"p={p:.1e}"
    ax.set_title(f"{ptxt} ({sig})", fontsize=10)


def main():
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          hspace=0.28, wspace=0.28)

    ax_volcano = fig.add_subplot(gs[0, :])
    ax_fano = fig.add_subplot(gs[1, 0])
    ax_gini = fig.add_subplot(gs[1, 1])

    draw_volcano(ax_volcano)

    clust = pd.read_csv(CLUST_CSV)
    draw_boxplot(ax_fano, clust, *METRICS[0])
    draw_boxplot(ax_gini, clust, *METRICS[1])

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
