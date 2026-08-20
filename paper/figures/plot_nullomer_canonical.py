#!/usr/bin/env python3
"""
Reviewer #2 (major #3): replot nullomer fractions with the corrected canonical
denominator (companion to fix_nullomer_canonical.py / Figure 3).

Reads revisions/results/nullomers_canonical_combined.csv and, per domain, plots
orig vs syn nullomer fraction across k under BOTH conventions:
  * old: 4**k denominator (the floored ~0.5 artifact at odd k)
  * new: canonical-class denominator (correct)

Output: revisions/figures/nullomer_canonical_vs_4k_<domain>.png
        revisions/results/nullomer_canonical_domain_summary.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

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

DOMAIN_TAGS = {
    "Eukaryotes": [
        "Publish_Aedes", "Publish_Apis", "Publish_Arabidopsis", "Publish_Bos",
        "Publish_Branchiostoma", "Publish_Caenorhabditis", "Publish_Canis",
        "Publish_Danio", "Publish_Drosophila", "Publish_Gallus", "Publish_Gossypium",
        "Publish_Mus", "Publish_Nematostella", "Publish_Oryza", "Publish_Saccharina",
        "Publish_Takifugu", "Publish_Triticum", "Publish_Xenopus", "Publish_Zea",
        "Publish_Human", "Publish_Saccharomyces",
    ],
    "Prokaryotes": ["Publish_Archaea", "Publish_Chlamydiota",
                    "Publish_Mycoplasmatota", "Publish_Pseudomonadota"],
    "Viruses": ["Kitrinoviricota", "Nucleocytoviricota", "Peploviricota",
                "Preplasmiviricota", "Uroviricota"],
}


def tag_to_domain(tag: str) -> str | None:
    base = str(tag)
    for dom, tags in DOMAIN_TAGS.items():
        if base in tags:
            return dom
    # per-id viral/prok tags may carry suffixes; match by prefix
    for dom, tags in DOMAIN_TAGS.items():
        for t in tags:
            if base.startswith(t):
                return dom
    return None


def main() -> int:
    df = pd.read_csv(RES / "nullomers_canonical_combined.csv")
    df["domain"] = df["tag"].map(tag_to_domain)
    df = df[df["domain"].notna()].copy()

    summary_rows = []
    for dom in ["Eukaryotes", "Prokaryotes", "Viruses"]:
        sub = df[df["domain"] == dom]
        if sub.empty:
            continue
        ks = sorted(sub["k"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, sharex=True)
        for ax, col, title in (
            (axes[0], "nullomer_fraction_orig_4k_denom", "Old: 4$^k$ denominator"),
            (axes[1], "nullomer_fraction_canonical", "Corrected: canonical denominator"),
        ):
            for which, color in (("orig", "#1f77b4"), ("syn", "#ff7f0e")):
                means, los, his = [], [], []
                for k in ks:
                    vals = sub[(sub["k"] == k) & (sub["which"] == which)][col].dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    if len(vals) > 1:
                        se = vals.std(ddof=1) / np.sqrt(len(vals))
                    else:
                        se = 0.0
                    los.append(means[-1] - se)
                    his.append(means[-1] + se)
                ax.plot(ks, means, "-o", color=color,
                        label=("natural" if which == "orig" else "synthetic"))
                ax.fill_between(ks, los, his, color=color, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("k-mer length (k)")
            ax.set_ylabel("Nullomer fraction")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
        fig.suptitle(f"Nullomer fraction — {dom}  (mean ± SE across species)")
        fig.tight_layout()
        out = FIG / f"nullomer_canonical_vs_4k_{dom.lower()}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {out}")

        for k in ks:
            for which in ("orig", "syn"):
                v = sub[(sub["k"] == k) & (sub["which"] == which)]
                summary_rows.append({
                    "domain": dom, "k": k, "which": which,
                    "n_species": int(v["tag"].nunique()),
                    "frac_old_4k_mean": float(v["nullomer_fraction_orig_4k_denom"].mean()),
                    "frac_canonical_mean": float(v["nullomer_fraction_canonical"].mean()),
                })

    sm = pd.DataFrame(summary_rows)
    sm.to_csv(RES / "nullomer_canonical_domain_summary.csv", index=False)
    print(f"[ok] {RES / 'nullomer_canonical_domain_summary.csv'}")
    print("\n=== k=9 corrected fractions by domain ===")
    print(sm[sm["k"] == 9].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
