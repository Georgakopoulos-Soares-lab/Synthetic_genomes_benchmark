#!/usr/bin/env python3
"""
R3.7 comparison figure: do the paper's failure modes persist under alternative
decoding? Plots the per-window synthetic-vs-natural metrics for each decoding
config (lowtemp, nucleus, and baseline if present) against the intrinsic
natural-natural variability band.

Reads revisions/results/sweep_metrics_per_window.csv (from run_sweep_metrics.py).
Outputs revisions/figures/sweep_metrics_comparison.png +
        revisions/results/sweep_metrics_config_summary.csv

CPU. Run with system python3 from /tmp.
"""

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

RES = Path(f"{_ROOT}/revisions/results")
FIG = Path(f"{_ROOT}/revisions/figures")
FIG.mkdir(parents=True, exist_ok=True)

PER = RES / "sweep_metrics_per_window.csv"
NATNAT = RES / "natnat_fcgr_summary_k8.csv"

METRICS = [
    ("fcgr_l1_k8", "FCGR L1 (k=8)\nsynthetic vs natural"),
    ("kmer_jsd_k6", "k-mer JSD (k=6)\nsynthetic vs natural"),
    ("nullfrac_delta_k9", "\u0394 nullomer fraction (k=9)\nsynthetic \u2212 natural"),
]
COLORS = {"baseline": "#888888", "lowtemp": "#1b9e77", "nucleus": "#7570b3"}


def natnat_band():
    if not NATNAT.exists():
        return None
    df = pd.read_csv(NATNAT)
    for c in df.columns:
        if "median" in c.lower() or "l1" in c.lower():
            try:
                return float(df[c].median())
            except Exception:
                pass
    return None


def main():
    if not PER.exists():
        print(f"[error] {PER} not found; run run_sweep_metrics.py first")
        return 1
    df = pd.read_csv(PER)
    configs = [c for c in ["baseline", "lowtemp", "nucleus"] if c in set(df.config)]
    configs += [c for c in sorted(set(df.config)) if c not in configs]

    # per-config summary
    num = [m for m, _ in METRICS if m in df.columns]
    summ = df.groupby("config")[num].median().reset_index()
    summ.to_csv(RES / "sweep_metrics_config_summary.csv", index=False)
    print("=== median metrics by config (n windows each) ===")
    print(df.groupby("config").size().to_string())
    print(summ.to_string(index=False))

    band = natnat_band()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, METRICS):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = [df[df.config == c][col].dropna().values for c in configs]
        bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], configs):
            patch.set_facecolor(COLORS.get(c, "#cccccc"))
            patch.set_alpha(0.75)
        # scatter individual windows
        for i, c in enumerate(configs):
            y = df[df.config == c][col].dropna().values
            x = np.random.normal(i + 1, 0.05, size=len(y))
            ax.scatter(x, y, s=14, color="k", alpha=0.5, zorder=3)
        if col == "fcgr_l1_k8" and band is not None:
            ax.axhline(band, ls="--", color="red", lw=1.2,
                       label=f"natural-natural median ({band:.3f})")
            ax.legend(fontsize=9, frameon=False)
        if col == "nullfrac_delta_k9":
            ax.axhline(0, ls=":", color="grey", lw=1)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(col, fontsize=10)
        ax.tick_params(axis="x", labelsize=11)
    fig.suptitle("R3.7: failure modes persist across decoding configurations "
                 "(Evo 2 7B, 300 kb eukaryotic windows)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG / "sweep_metrics_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
