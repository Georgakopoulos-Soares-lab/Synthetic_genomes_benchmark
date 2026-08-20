#!/usr/bin/env python3
"""
Reviewer #1 (#3) + Reviewer #2 (resolution/font minor): replot Figure 4A.

The original Figure 4A packed eukaryotic species, viral groups and prokaryotic
domains into a single dense heatmap. Here we render the three groups as
distinct, well-separated sub-panels with enlarged labels, larger significance
annotations and high resolution, improving scannability.

Cell value = median over paired windows of log2((orig+eps)/(syn+eps)) of
non-B DNA motif base-pair coverage (positive = depleted in synthetic).
Per-cell significance = paired Wilcoxon signed-rank (orig vs syn) with
Benjamini-Hochberg FDR within each panel; cells with < MIN_PAIRS pairs are not
tested. Matches the analysis settings in config/three_heatmaps.yaml.

CPU-only. Run with system python3 from /tmp (needs user-site scipy).
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
OUT = Path(f"{_ROOT}/revisions/figures/fig4A_subpanels.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

EPS = 1.0
MIN_PAIRS = 10
MOTIFS = ["G4", "ZDNA", "DR", "MR", "IR", "STR"]
VMIN, VMAX = -1.5, 1.5

GROUPS = {
    "Eukaryotic species": {
        "Publish_Human": "Homo sapiens", "Publish_Mus": "Mus musculus",
        "Publish_Gallus": "Gallus gallus", "Publish_Xenopus": "Xenopus tropicalis",
        "Publish_Oryza": "Oryza sativa", "Publish_Zea": "Zea mays",
        "Publish_Triticum": "Triticum aestivum", "Publish_Takifugu": "Takifugu rubripes",
        "Publish_Apis": "Apis mellifera", "Publish_Aedes": "Aedes aegypti",
        "Publish_Nematostella": "Nematostella vectensis",
    },
    "Viral groups": {
        "Kitrinoviricota": "Kitrinoviricota", "Nucleocytoviricota": "Nucleocytoviricota",
        "Peploviricota": "Peploviricota", "Preplasmiviricota": "Preplasmiviricota",
        "Uroviricota": "Uroviricota",
    },
    "Prokaryotic domains": {
        "Publish_Chlamydiota": "Chlamydiota", "Publish_Pseudomonadota": "Pseudomonadota",
        "Publish_Mycoplasmatota": "Mycoplasmatota", "Publish_Archaea": "Archaea",
    },
}


def load_tag(tag_dir: Path) -> pd.DataFrame:
    """Return tidy [pair_id, which, motif, bp_covered] for the six motifs."""
    frames = []
    g4 = tag_dir / "g4hunter.metrics.csv"
    z = tag_dir / "zseeker.metrics.csv"
    nb = tag_dir / "nonbgfa.metrics.csv"
    if g4.exists():
        d = pd.read_csv(g4)[["pair_id", "which", "bp_covered"]].copy()
        d["motif"] = "G4"
        frames.append(d)
    if z.exists():
        d = pd.read_csv(z)[["pair_id", "which", "bp_covered"]].copy()
        d["motif"] = "ZDNA"
        frames.append(d)
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


def build_panel(tags: dict):
    rows = list(tags.keys())
    vals = np.full((len(rows), len(MOTIFS)), np.nan)
    raw_p = np.full((len(rows), len(MOTIFS)), np.nan)
    for i, tag in enumerate(rows):
        df = load_tag(BASE / tag)
        if df.empty:
            print(f"[warn] no data for {tag}", file=sys.stderr)
            continue
        for j, mot in enumerate(MOTIFS):
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
    # BH-FDR within panel over tested cells
    flat = raw_p.flatten()
    mask = ~np.isnan(flat)
    q = np.full_like(flat, np.nan)
    if mask.any():
        q[mask] = bh_fdr(flat[mask])
    qmat = q.reshape(raw_p.shape)
    names = [tags[t] for t in rows]
    return names, vals, qmat


def main():
    panels = {name: build_panel(tags) for name, tags in GROUPS.items()}
    heights = [len(panels[n][0]) for n in panels]
    fig = plt.figure(figsize=(9, 16))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[h + 1.2 for h in heights],
                          hspace=0.28)
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)
    im = None
    for ax_i, (title, (names, vals, qmat)) in zip(range(len(panels)), panels.items()):
        ax = fig.add_subplot(gs[ax_i])
        im = ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(MOTIFS)))
        ax.set_xticklabels(MOTIFS, fontsize=15, weight="bold")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=14, style="italic"
                           if title == "Eukaryotic species" else "normal")
        ax.set_title(title, fontsize=18, weight="bold", pad=10)
        ax.set_xticks(np.arange(-.5, len(MOTIFS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for i in range(len(names)):
            for j in range(len(MOTIFS)):
                mark = stars(qmat[i, j])
                if mark:
                    v = vals[i, j]
                    col = "white" if (not np.isnan(v) and abs(v) > 0.9) else "black"
                    ax.text(j, i, mark, ha="center", va="center",
                            fontsize=17, color=col, weight="bold")
    cbar = fig.colorbar(im, ax=fig.axes, orientation="vertical",
                        fraction=0.025, pad=0.04, shrink=0.5)
    cbar.set_label("median log\u2082(original / synthetic) bp coverage",
                   fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.suptitle("Figure 4A replot: non-B DNA motif coverage shift, by domain "
                 "(\u2022 q<0.1  * q<0.05  ** q<0.01)",
                 fontsize=15, weight="bold", y=0.995)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
