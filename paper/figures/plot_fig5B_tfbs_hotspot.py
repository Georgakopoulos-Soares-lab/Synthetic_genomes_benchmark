#!/usr/bin/env python3
"""
Replacement for Figure 5B (Reviewer #2 comment #4): paired boxplots + paired
scatter of TFBS hotspot organisation metrics (Fano factor, Gini coefficient,
lag-1 spatial autocorrelation) for original vs synthetic windows, with paired
Wilcoxon signed-rank tests.

Data: results/metrics/Publish_Human/tfbs_clustering_stats.csv (40 paired
windows; Fano / Gini / lag-1 autocorr already computed per window by
scripts/tfbs_clustering.py).

NOTE ON DIRECTION: the data show natural windows have significantly HIGHER Fano
and Gini than synthetic (synthetic TFBS distributions are more uniform / less
patchy), i.e. the opposite of the current manuscript sentence. Flagged for the
authors' text revision.

CPU-only. Run with system python3 from /tmp.
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

STATS = f"{_ROOT}/results/metrics/Publish_Human/tfbs_clustering_stats.csv"
FIG = f"{_ROOT}/revisions/figures/"
OUTRES = f"{_ROOT}/revisions/results/"
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUTRES, exist_ok=True)

METRICS = [
    ("fano", "Fano factor\n(variance/mean)"),
    ("gini", "Gini coefficient"),
    ("lag1_autocorr", "Lag-1 spatial\nautocorrelation"),
]
ORIG_C, SYN_C = "#1b9e77", "#d95f02"


def main():
    df = pd.read_csv(STATS)
    n = len(df)
    stat_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for j, (key, label) in enumerate(METRICS):
        o = df[f"orig_{key}"].values.astype(float)
        s = df[f"syn_{key}"].values.astype(float)
        mask = np.isfinite(o) & np.isfinite(s)
        o, s = o[mask], s[mask]
        try:
            w, p = wilcoxon(s, o)
        except ValueError:
            w, p = np.nan, np.nan
        direction = "syn > orig" if np.median(s) > np.median(o) else "syn < orig"
        stat_rows.append({
            "metric": key, "n_pairs": int(mask.sum()),
            "orig_median": float(np.median(o)), "syn_median": float(np.median(s)),
            "orig_mean": float(np.mean(o)), "syn_mean": float(np.mean(s)),
            "direction": direction, "wilcoxon_W": float(w), "wilcoxon_p": float(p),
        })

        # --- top row: paired boxplot with connecting lines ---
        axb = axes[0, j]
        bp = axb.boxplot([o, s], positions=[1, 2], widths=0.5, patch_artist=True,
                         showfliers=False)
        for patch, c in zip(bp["boxes"], [ORIG_C, SYN_C]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
        for xi, yi in zip(o, s):
            axb.plot([1, 2], [xi, yi], color="grey", alpha=0.3, lw=0.6, zorder=1)
        axb.scatter(np.ones_like(o), o, color=ORIG_C, s=14, zorder=2)
        axb.scatter(2 * np.ones_like(s), s, color=SYN_C, s=14, zorder=2)
        axb.set_xticks([1, 2]); axb.set_xticklabels(["Natural", "Synthetic"])
        axb.set_ylabel(label)
        sig = "ns" if (np.isnan(p) or p >= 0.05) else (
            "***" if p < 1e-3 else ("**" if p < 1e-2 else "*"))
        ptxt = "p=n/a" if np.isnan(p) else (f"p={p:.1e}")
        axb.set_title(f"{ptxt} ({sig})", fontsize=10)

        # --- bottom row: paired scatter natural vs synthetic ---
        axs = axes[1, j]
        axs.scatter(o, s, color="#404040", s=20, alpha=0.7)
        lo = min(o.min(), s.min()); hi = max(o.max(), s.max())
        axs.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        axs.set_xlabel(f"Natural {key}")
        axs.set_ylabel(f"Synthetic {key}")
        axs.legend(fontsize=8, loc="best")

    fig.suptitle("TFBS hotspot organisation: natural vs synthetic windows "
                 f"(Homo sapiens, n={n} paired windows)", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG + "fig5B_tfbs_hotspot_metrics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")

    res = pd.DataFrame(stat_rows)
    res_path = OUTRES + "fig5B_tfbs_hotspot_stats.csv"
    res.to_csv(res_path, index=False)
    print(res.to_string(index=False))
    print(f"\n[wrote] {res_path}")
    print("\n[FLAG for manuscript text] Natural windows show significantly HIGHER "
          "Fano and Gini than synthetic (synthetic TFBS are more uniform/less "
          "patchy) — opposite of the current sentence's stated direction.")


if __name__ == "__main__":
    main()
