#!/usr/bin/env python3
"""
Figure 4A replot (Reviewer #1 #3 + Reviewer #2 resolution/font).

Keeps the manuscript's original three-block layout — eukaryotic species (top,
full width), viral groups (bottom-left) and prokaryotic domains (bottom-right) —
but (a) enlarges all labels / raises resolution and (b) visually SEPARATES the
prokaryotic block (gap + title + framed border) to improve scannability.

Cell = median over paired windows of log2((orig+eps)/(syn+eps)) of non-B DNA
motif base-pair coverage. Positive (red) = depletion in Evo 2 synthetic genomes.
Significance = paired Wilcoxon signed-rank with Benjamini-Hochberg FDR within
each block (min MIN_PAIRS pairs). Data: results/metrics/<TAG>/{g4hunter,zseeker,
nonbgfa}.metrics.csv per config/three_heatmaps.yaml.

CPU. Run with system python3 from /tmp (needs user-site scipy).
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon

BASE = Path(f"{_ROOT}/results/metrics")
OUT = Path(f"{_ROOT}/revisions/figures/fig4A_heatmap.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

EPS = 1.0
MIN_PAIRS = 10
MOTIFS = ["G4", "ZDNA", "DR", "MR", "IR", "STR"]
VMIN, VMAX = -1.5, 1.5

EUK = {
    "Publish_Human": "Homo sapiens", "Publish_Mus": "Mus musculus",
    "Publish_Gallus": "Gallus gallus", "Publish_Xenopus": "Xenopus tropicalis",
    "Publish_Oryza": "Oryza sativa", "Publish_Zea": "Zea mays",
    "Publish_Triticum": "Triticum aestivum", "Publish_Takifugu": "Takifugu rubripes",
    "Publish_Apis": "Apis mellifera", "Publish_Aedes": "Aedes aegypti",
    "Publish_Nematostella": "Nematostella vectensis",
}
VIR = {
    "Kitrinoviricota": "Kitrinoviricota", "Nucleocytoviricota": "Nucleocytoviricota",
    "Peploviricota": "Peploviricota", "Preplasmiviricota": "Preplasmiviricota",
    "Uroviricota": "Uroviricota",
}
PROK = {
    "Publish_Chlamydiota": "Chlamydiota", "Publish_Pseudomonadota": "Pseudomonadota",
    "Publish_Mycoplasmatota": "Mycoplasmatota", "Publish_Archaea": "Archaea",
}


def load_tag(tag_dir):
    frames = []
    for fname, motif in (("g4hunter.metrics.csv", "G4"),
                         ("zseeker.metrics.csv", "ZDNA")):
        f = tag_dir / fname
        if f.exists():
            d = pd.read_csv(f)[["pair_id", "which", "bp_covered"]].copy()
            d["motif"] = motif
            frames.append(d)
    nb = tag_dir / "nonbgfa.metrics.csv"
    if nb.exists():
        d = pd.read_csv(nb)
        d = d[d["motif"].isin(["DR", "MR", "IR", "STR"])][
            ["pair_id", "which", "bp_covered", "motif"]].copy()
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["pair_id", "which", "motif", "bp_covered"])
    return pd.concat(frames, ignore_index=True)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def stars(q):
    if np.isnan(q):
        return ""
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    if q < 0.10:
        return "\u2022"
    return ""


def build(tags):
    """Return (species_names, vals[motif x species], qmat[motif x species])."""
    names = list(tags.keys())
    vals = np.full((len(MOTIFS), len(names)), np.nan)
    raw_p = np.full((len(MOTIFS), len(names)), np.nan)
    for j, tag in enumerate(names):
        df = load_tag(BASE / tag)
        if df.empty:
            print(f"[warn] no data for {tag}", file=sys.stderr)
            continue
        for i, mot in enumerate(MOTIFS):
            sub = df[df["motif"] == mot]
            piv = sub.pivot_table(index="pair_id", columns="which",
                                  values="bp_covered", aggfunc="first")
            if "orig" not in piv.columns or "syn" not in piv.columns:
                continue
            piv = piv.dropna(subset=["orig", "syn"])
            if len(piv) == 0:
                continue
            o = piv["orig"].to_numpy(float)
            s = piv["syn"].to_numpy(float)
            vals[i, j] = np.median(np.log2((o + EPS) / (s + EPS)))
            if len(piv) >= MIN_PAIRS and np.any(o != s):
                try:
                    raw_p[i, j] = wilcoxon(o, s, zero_method="wilcox").pvalue
                except ValueError:
                    pass
    flat = raw_p.flatten()
    mask = ~np.isnan(flat)
    q = np.full_like(flat, np.nan)
    if mask.any():
        q[mask] = bh_fdr(flat[mask])
    return [tags[t] for t in names], vals, q.reshape(raw_p.shape)


def wrap2(name):
    """Put two-word (Genus species) names on two lines to save vertical space."""
    parts = name.split()
    return "\n".join(parts) if len(parts) == 2 else name


def draw(ax, names, vals, qmat, norm, title, motif_labels=True, xtick_fontsize=7.5):
    im = ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([wrap2(n) for n in names], rotation=90, ha="center",
                       va="top", fontsize=xtick_fontsize,
                       style="italic" if title == "Eukaryotic species" else "normal")
    if motif_labels:
        ax.set_yticks(range(len(MOTIFS)))
        ax.set_yticklabels(MOTIFS, fontsize=9, fontweight="bold")
    else:
        ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MOTIFS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=5)
    for i in range(len(MOTIFS)):
        for j in range(len(names)):
            m = stars(qmat[i, j])
            if m:
                v = vals[i, j]
                col = "white" if (not np.isnan(v) and abs(v) > 0.9) else "black"
                ax.text(j, i, m, ha="center", va="center", fontsize=8,
                        color=col, fontweight="bold")
    return im


def main():
    e_names, e_vals, e_q = build(EUK)
    v_names, v_vals, v_q = build(VIR)
    p_names, p_vals, p_q = build(PROK)
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)

    fig = plt.figure(figsize=(8.5, 4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[5, 4],
                          hspace=1.35, wspace=0.28, left=0.08, right=0.86,
                          top=0.90, bottom=0.14)
    ax_e = fig.add_subplot(gs[0, :])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])

    im = draw(ax_e, e_names, e_vals, e_q, norm, "Eukaryotic species",
              xtick_fontsize=9.5)
    draw(ax_v, v_names, v_vals, v_q, norm, "Viral groups")
    draw(ax_p, p_names, p_vals, p_q, norm, "Prokaryotic domains")

    cax = fig.add_axes([0.885, 0.45, 0.015, 0.42])
    cb = fig.colorbar(im, cax=cax)
    ticks = [-1.5, -1.0, 0.0, 1.0, 1.5]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{2**t:.2g}\u00d7" for t in ticks])
    cb.set_label("median fold change\n(natural / synthetic)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.985, 0.30,
             "red = depletion\nin synthetic\n\n** q<0.01\n* q<0.05\n\u2022 q<0.1",
             ha="right", va="top", fontsize=8)
    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
