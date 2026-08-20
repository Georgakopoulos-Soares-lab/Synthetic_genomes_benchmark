#!/usr/bin/env python3
"""
Combined natural-natural baseline summary figure (Reviewer #2 comment #2).

Ties together the three natural-natural baselines into one publication panel:
the fraction of species (or species x motif tests) for which the synthetic
deviation from natural exceeds the intrinsic natural-natural variation, plus the
median effect size, for FCGR (k=8), nullomers (canonical k=9) and the five
non-B DNA motif classes.

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

R = f"{_ROOT}/revisions/results/"
FIG = f"{_ROOT}/revisions/figures/"
os.makedirs(FIG, exist_ok=True)

# --- FCGR ---
fcgr = pd.read_csv(R + "natnat_fcgr_summary_k8.csv")
fcgr_sig = float(((fcgr.q_bh < 0.05) & (fcgr.syn_nat_over_nat_nat_ratio > 1)).mean())
fcgr_ratio = float(fcgr.syn_nat_over_nat_nat_ratio.median())

# --- Nullomers ---
nul = pd.read_csv(R + "natnat_nullomer_summary_k9.csv")
nul_sig = float((nul.mwu_q < 0.05).mean())
nul_outside = float(nul.syn_median_outside_nat_iqr.mean())
# effect size: median |delta| relative to natural median
nul_ratio = float((1 + (nul.delta_syn_minus_nat.abs() / nul.nat_median)).median())

# --- non-B per motif ---
nonb = pd.read_csv(R + "natnat_nonb_motif_summary.csv")

labels, fracs, ratios, fam = [], [], [], []
labels.append("FCGR\n(k=8)"); fracs.append(fcgr_sig); ratios.append(fcgr_ratio); fam.append("comp")
labels.append("Nullomers\n(k=9)"); fracs.append(nul_sig); ratios.append(nul_ratio); fam.append("comp")
for _, row in nonb.iterrows():
    labels.append(f"non-B\n{row.motif}")
    fracs.append(row.n_sig_q05 / row.n_species)
    ratios.append(row.median_ratio)
    fam.append("nonb")

colors = ["#2c7fb8" if f == "comp" else "#d95f0e" for f in fam]
x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

ax1.bar(x, [100 * f for f in fracs], color=colors, edgecolor="black", linewidth=0.6)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("% species with synthetic deviation\n> natural-natural variation (q<0.05)")
ax1.set_ylim(0, 105)
ax1.axhline(50, color="grey", ls=":", lw=0.8)
ax1.set_title("A. Synthetic exceeds natural baseline", loc="left", fontsize=11, weight="bold")
for xi, f in zip(x, fracs):
    ax1.text(xi, 100 * f + 1.5, f"{100*f:.0f}", ha="center", va="bottom", fontsize=8)

ax2.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.6)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Median effect size\n(syn-nat / nat-nat)")
ax2.axhline(1.0, color="red", ls="--", lw=1.0, label="no excess (=1)")
ax2.set_title("B. Effect size vs natural variation", loc="left", fontsize=11, weight="bold")
ax2.legend(fontsize=8, loc="upper left")
for xi, rr in zip(x, ratios):
    ax2.text(xi, rr + 0.03, f"{rr:.2f}", ha="center", va="bottom", fontsize=8)

from matplotlib.patches import Patch
handles = [Patch(facecolor="#2c7fb8", edgecolor="black", label="Compositional (FCGR, nullomers)"),
           Patch(facecolor="#d95f0e", edgecolor="black", label="non-B DNA motifs")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = FIG + "natnat_baseline_summary.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"[done] wrote {out}")

# also a small machine-readable table
tab = pd.DataFrame({"metric": [l.replace("\n", " ") for l in labels],
                    "pct_species_sig": [100 * f for f in fracs],
                    "median_effect_ratio": ratios})
tab.to_csv(R + "natnat_baseline_summary_table.csv", index=False)
print(tab.to_string(index=False))
print(f"\n[note] FCGR sig {fcgr_sig:.2f}, nullomer sig {nul_sig:.2f} "
      f"(outside-IQR {nul_outside:.2f})")
