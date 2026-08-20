#!/usr/bin/env python3
"""
Regenerate Figure 3 (nullomer content, natural vs synthetic) from the
CANONICALISED nullomer fractions (Reviewer #2, Major #3).

The original figure used a 4^k denominator with KMC's default *canonical*
counting, which floored the nullomer fraction near 0.5 for odd k. Here we use
the correct denominator (number of canonical k-mer classes) column
`nullomer_fraction_canonical` from nullomers_canonical_combined.csv.

Panels:
  A  : per-species O/E = natural / synthetic canonical nullomer COUNT at k=11
       (log scale; O/E > 1 => synthetic depleted in nullomers), all eukaryotes.
  B-I: grouped bars of canonical nullomer fraction, natural (ORIG) vs synthetic
       (SYN), k = 9-13, for the 8 focal species.

Significance (per species x k): conservative McNemar-style chi-square on the
canonical nullomer counts, chi2 = (No - Ns)^2 / (No + Ns), df = 1
(uses the maximal discordance bound b + c = No + Ns, i.e. conservative).
Stars: * p<0.05, ** p<0.01, *** p<0.001, **** p<1e-4.

CPU. Run with system python3 from /tmp (needs user-site pandas/scipy).
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from matplotlib.colors import Normalize

REV = Path(f"{_ROOT}/revisions")
CSV = REV / "results" / "nullomers_canonical_combined.csv"
OUT = REV / "figures" / "fig3_nullomers_canonical.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

KS = [9, 10, 11, 12, 13]
PANEL_A_K = 11
ORIG_C = "#2b2b2b"
SYN_C = "#b0b0b0"

FOCAL = [
    ("B", "Publish_Human", "Homo sapiens"),
    ("C", "Publish_Mus", "Mus musculus"),
    ("D", "Publish_Canis", "Canis lupus familiaris"),
    ("E", "Publish_Bos", "Bos taurus"),
    ("F", "Publish_Gallus", "Gallus gallus"),
    ("G", "Publish_Xenopus", "Xenopus tropicalis"),
    ("H", "Publish_Triticum", "Triticum aestivum"),
    ("I", "Publish_Zea", "Zea mays"),
]

EUK_LATIN = {
    "Publish_Aedes": "Aedes aegypti", "Publish_Apis": "Apis mellifera",
    "Publish_Arabidopsis": "Arabidopsis thaliana", "Publish_Bos": "Bos taurus",
    "Publish_Branchiostoma": "Branchiostoma floridae",
    "Publish_Caenorhabditis": "Caenorhabditis elegans",
    "Publish_Canis": "Canis lupus familiaris", "Publish_Danio": "Danio rerio",
    "Publish_Drosophila": "Drosophila melanogaster", "Publish_Gallus": "Gallus gallus",
    "Publish_Gossypium": "Gossypium hirsutum", "Publish_Human": "Homo sapiens",
    "Publish_Mus": "Mus musculus", "Publish_Nematostella": "Nematostella vectensis",
    "Publish_Oryza": "Oryza sativa", "Publish_Saccharina": "Saccharina japonica",
    "Publish_Saccharomyces": "Saccharomyces cerevisiae",
    "Publish_Takifugu": "Takifugu rubripes", "Publish_Triticum": "Triticum aestivum",
    "Publish_Xenopus": "Xenopus tropicalis", "Publish_Zea": "Zea mays",
}


def stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def mcnemar_p(No, Ns):
    """Conservative McNemar chi-square on canonical nullomer counts."""
    denom = No + Ns
    if denom <= 0:
        return 1.0
    stat = (No - Ns) ** 2 / denom
    return float(chi2.sf(stat, 1))


def get(df, tag, which, k, col):
    r = df[(df.tag == tag) & (df.which == which) & (df.k == k)]
    return None if r.empty else float(r.iloc[0][col])


def main():
    df = pd.read_csv(CSV)
    df = df[["tag", "which", "k", "nullomer_fraction_canonical",
             "nullomer_count_canonical"]].copy()

    fig = plt.figure(figsize=(8.5, 11))
    # row 1 is a spacer so Panel A's long rotated labels clear the B-E row.
    gs = gridspec.GridSpec(4, 4, height_ratios=[1.15, 0.55, 1, 1],
                           hspace=0.42, wspace=0.45)

    # ---------- Panel A ----------
    axA = fig.add_subplot(gs[0, :])
    tags = [t for t in EUK_LATIN if not df[(df.tag == t) & (df.k == PANEL_A_K)].empty]
    oe = {}
    for t in tags:
        No = get(df, t, "orig", PANEL_A_K, "nullomer_count_canonical")
        Ns = get(df, t, "syn", PANEL_A_K, "nullomer_count_canonical")
        if No is None or Ns is None:
            continue
        oe[t] = (No + 1.0) / (Ns + 1.0)
    order = sorted(oe, key=lambda t: oe[t])
    vals = [oe[t] for t in order]
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    colors = cm.viridis(norm(vals))
    axA.scatter(range(len(order)), vals, c=colors, s=110, edgecolor="k",
                linewidth=0.5, zorder=3)
    axA.axhline(1.0, ls="--", color="grey", lw=1)
    axA.set_yscale("log")
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels([EUK_LATIN[t] for t in order], rotation=90,
                        fontsize=11, style="italic")
    axA.tick_params(axis="y", labelsize=12)
    axA.set_ylabel(f"O/E nullomer count (k={PANEL_A_K})\nnatural / synthetic",
                   fontsize=13)
    axA.annotate("A", xy=(0, 1), xycoords="axes fraction", xytext=(-42, 22),
                 textcoords="offset points", fontweight="bold", fontsize=18,
                 va="bottom", ha="right", annotation_clip=False)
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axA, fraction=0.025, pad=0.01)
    cb.set_label("O/E ratio", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    # ---------- Panels B-I ----------
    positions = [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    x = np.arange(len(KS))
    w = 0.38
    for (letter, tag, latin), (r, c) in zip(FOCAL, positions):
        ax = fig.add_subplot(gs[r, c])
        orig = [get(df, tag, "orig", k, "nullomer_fraction_canonical") or 0 for k in KS]
        syn = [get(df, tag, "syn", k, "nullomer_fraction_canonical") or 0 for k in KS]
        ax.bar(x - w / 2, orig, w, color=ORIG_C)
        ax.bar(x + w / 2, syn, w, color=SYN_C)
        for i, k in enumerate(KS):
            No = get(df, tag, "orig", k, "nullomer_count_canonical")
            Ns = get(df, tag, "syn", k, "nullomer_count_canonical")
            if No is None or Ns is None:
                continue
            s = stars(mcnemar_p(No, Ns))
            if s:
                ax.text(i, max(orig[i], syn[i]) + 0.02, s, ha="center",
                        va="bottom", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in KS], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        # only outer panels get axis labels (avoids inner-label overlaps)
        if r == 3:
            ax.set_xlabel("k-mer length (k)", fontsize=12)
        if c == 0:
            ax.set_ylabel("Nullomer fraction", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-30, 6), textcoords="offset points",
                    fontweight="bold", fontsize=16, va="bottom", ha="right",
                    annotation_clip=False)

    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(f"[done] wrote {OUT}")
    # also dump the corrected values used
    rows = []
    for _, tag, latin in FOCAL:
        for k in KS:
            rows.append(dict(species=latin, k=k,
                             orig=get(df, tag, "orig", k, "nullomer_fraction_canonical"),
                             syn=get(df, tag, "syn", k, "nullomer_fraction_canonical"),
                             p=mcnemar_p(get(df, tag, "orig", k, "nullomer_count_canonical"),
                                         get(df, tag, "syn", k, "nullomer_count_canonical"))))
    pd.DataFrame(rows).to_csv(REV / "results" / "fig3_nullomers_canonical_values.csv",
                              index=False)
    print("[done] wrote fig3_nullomers_canonical_values.csv")


if __name__ == "__main__":
    main()
