#!/usr/bin/env python3
"""
Natural-natural baseline for non-B DNA motif coverage (Reviewer #2 comment #2,
part 3 of 3).

For each species and each non-B motif class (DR, GQ, IR, MR, STR) we ask whether
synthetic sequences deviate from natural sequences MORE than natural windows
deviate among themselves. Mirrors the FCGR and nullomer natural-natural
baselines.

Design (per species x motif):
  natural windows give coverage values {orig_i} (bp_covered per window).
  - nat-nat distances : all pairwise |orig_i - orig_j|  (i < j)
  - syn-nat distances : all pairwise |syn_i  - orig_j|
  Mann-Whitney U, one-sided (syn-nat > nat-nat). BH-FDR across all tests.

Both distance sets are all-pairs over the SAME set of window lengths, so raw
bp_covered is a fair metric (no length normalisation needed for the test). A
relative effect size (median ratio vs natural variation) is also reported.

CPU-only. Run with system python3 from /tmp (needs user-site pyarrow).
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import glob
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations, product
from scipy.stats import mannwhitneyu

ROOT = f"{_ROOT}/results/harmonized"
OUTDIR = f"{_ROOT}/revisions/results"
MOTIFS = ["DR", "GQ", "IR", "MR", "STR"]
MIN_WINDOWS = 4  # need enough natural windows for a meaningful baseline


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "*", "nonbgfa.harmonized.parquet")))
    per_rows = []
    for f in files:
        tag = os.path.basename(os.path.dirname(f))
        df = pd.read_parquet(f)
        for motif in MOTIFS:
            sub = df[df.motif == motif]
            orig = sub[sub.which == "orig"].set_index("pair_id")["bp_covered"]
            syn = sub[sub.which == "syn"].set_index("pair_id")["bp_covered"]
            orig = orig[orig.notna()]
            syn = syn[syn.notna()]
            if len(orig) < MIN_WINDOWS or len(syn) < MIN_WINDOWS:
                continue
            ov = orig.values.astype(float)
            sv = syn.values.astype(float)
            natnat = np.array([abs(a - b) for a, b in combinations(ov, 2)], float)
            synnat = np.array([abs(a - b) for a, b in product(sv, ov)], float)
            if natnat.size == 0 or synnat.size == 0:
                continue
            try:
                U, p = mannwhitneyu(synnat, natnat, alternative="greater")
            except ValueError:
                continue
            med_nat = float(np.median(natnat)) if natnat.size else np.nan
            med_syn = float(np.median(synnat)) if synnat.size else np.nan
            ratio = (med_syn / med_nat) if med_nat > 0 else np.nan
            per_rows.append({
                "species": tag,
                "motif": motif,
                "n_orig_windows": int(len(orig)),
                "n_syn_windows": int(len(syn)),
                "orig_median_bp": float(np.median(ov)),
                "syn_median_bp": float(np.median(sv)),
                "natnat_median_dist": med_nat,
                "synnat_median_dist": med_syn,
                "median_ratio_syn_vs_nat": ratio,
                "U": float(U),
                "pval": float(p),
            })
    res = pd.DataFrame(per_rows)
    if res.empty:
        print("[error] no testable species/motif combinations", file=sys.stderr)
        sys.exit(1)
    res["qval"] = bh_fdr(res["pval"].values)
    res["sig_q05"] = res["qval"] < 0.05
    res = res.sort_values(["motif", "qval"]).reset_index(drop=True)
    per_path = os.path.join(OUTDIR, "natnat_nonb_per_species_motif.csv")
    res.to_csv(per_path, index=False)

    # per-motif summary
    summ = (res.groupby("motif")
            .agg(n_species=("species", "nunique"),
                 n_sig_q05=("sig_q05", "sum"),
                 median_ratio=("median_ratio_syn_vs_nat", "median"))
            .reset_index())
    summ_path = os.path.join(OUTDIR, "natnat_nonb_motif_summary.csv")
    summ.to_csv(summ_path, index=False)

    n_tests = len(res)
    n_sig = int(res["sig_q05"].sum())
    print(f"[done] {n_tests} species x motif tests; "
          f"{n_sig} significant at q<0.05 "
          f"({100 * n_sig / n_tests:.0f}%)")
    print("\nPer-motif summary (syn-nat deviation > nat-nat):")
    print(summ.to_string(index=False))
    print(f"\nWrote:\n  {per_path}\n  {summ_path}")
    # headline examples
    print("\nMost significant per motif:")
    for m in MOTIFS:
        mm = res[res.motif == m]
        if not mm.empty:
            top = mm.iloc[0]
            print(f"  {m}: {top.species} q={top.qval:.2e} "
                  f"ratio={top.median_ratio_syn_vs_nat:.2f} "
                  f"(orig {top.orig_median_bp:.0f}bp vs syn {top.syn_median_bp:.0f}bp)")


if __name__ == "__main__":
    main()
