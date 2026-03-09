#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmer_spectra_summary.py  —  Cross-bin summary of k-mer spectra results

Reads the per-bin support_stats.csv files produced by kmer_spectra.py and
aggregates directional patterns across length bins.

For each bin the script computes:
  - delta_coverage   : CDF endpoint difference (syn_cdf_end − nat_cdf_end)
  - early_area       : mean ΔCDF for x ≤ T  (shape in the low-abundance regime)
  - frac_pos_early   : fraction of x ≤ T points where ΔCDF > 0
  - peak_pos / neg   : maximum and minimum ΔCDF values and where they occur
  - auc_total        : mean ΔCDF across all abundances

A final "general patterns" block summarises directional consistency across bins.

Example
-------
python kmer_spectra_summary.py \\
    --indir  results/kmer_spectra \\
    --tag    my_phages \\
    --k      6 \\
    --T      50 \\
    --out    results/kmer_spectra/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class BinSummary:
    path: str
    tag: str
    k: Optional[int]
    bin_index: Optional[int]
    L_lo: Optional[int]
    L_hi: Optional[int]
    n_points: int
    xmax: int
    delta_coverage: float
    early_area: float
    frac_pos_early: float
    peak_pos: float
    x_peak_pos: int
    peak_neg: float
    x_peak_neg: int
    auc_total: float
    early_pos_end_neg: bool


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_filename(fname: str) -> Tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract (tag, k, bin_index, L_lo, L_hi) from a support_stats filename.

    Expected pattern:
      <tag>.k<k>.bin<N>_L<lo>_to_<hi>.support_stats.csv
    """
    base = os.path.basename(fname)

    m = re.match(
        r"^(?P<tag>.+)\.k(?P<k>\d+)\.bin(?P<bin>\d+)_L(?P<lo>\d+)_to_(?P<hi>\d+)\.support_stats\.csv$",
        base,
    )
    if m:
        return (
            m.group("tag"),
            int(m.group("k")),
            int(m.group("bin")),
            int(m.group("lo")),
            int(m.group("hi")),
        )

    # Fallback: bin present but no length range
    m = re.match(r"^(?P<tag>.+)\.k(?P<k>\d+)\.bin(?P<bin>\d+).+\.support_stats\.csv$", base)
    if m:
        return m.group("tag"), int(m.group("k")), int(m.group("bin")), None, None

    # Fallback: only k present
    m = re.match(r"^(?P<tag>.+)\.k(?P<k>\d+).+\.support_stats\.csv$", base)
    if m:
        return m.group("tag"), int(m.group("k")), None, None, None

    return os.path.splitext(base)[0], None, None, None, None


def _read_support_stats(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (abundances, delta_cdf, nat_cdf, syn_cdf) arrays from a
    support_stats.csv file.
    """
    xs, d_cdf, nat_cdf, syn_cdf = [], [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"abundance", "nat_cdf_mean", "syn_cdf_mean", "delta_cdf_mean"}
        missing  = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        for row in reader:
            xs.append(int(row["abundance"]))
            nat_cdf.append(float(row["nat_cdf_mean"]))
            syn_cdf.append(float(row["syn_cdf_mean"]))
            d_cdf.append(float(row["delta_cdf_mean"]))

    return (
        np.array(xs,      dtype=int),
        np.array(d_cdf,   dtype=float),
        np.array(nat_cdf, dtype=float),
        np.array(syn_cdf, dtype=float),
    )


def _summarise_file(path: str, T: int) -> BinSummary:
    tag, k, bin_idx, L_lo, L_hi = _parse_filename(path)
    xs, d, nat, syn = _read_support_stats(path)

    n_points = int(xs.size)
    xmax     = int(xs.max()) if n_points else 0

    delta_coverage = float(syn[-1] - nat[-1]) if n_points else float("nan")

    early_mask = xs <= T
    if np.any(early_mask):
        early_d    = d[early_mask]
        early_area = float(np.mean(early_d))
        frac_pos   = float(np.mean(early_d > 0))
    else:
        early_area = float("nan")
        frac_pos   = float("nan")

    if n_points:
        i_pos  = int(np.argmax(d))
        i_neg  = int(np.argmin(d))
        peak_pos  = float(d[i_pos]);  x_peak_pos = int(xs[i_pos])
        peak_neg  = float(d[i_neg]);  x_peak_neg = int(xs[i_neg])
        auc_total = float(np.mean(d))
    else:
        peak_pos = peak_neg = auc_total = float("nan")
        x_peak_pos = x_peak_neg = 0

    return BinSummary(
        path=path, tag=tag, k=k, bin_index=bin_idx, L_lo=L_lo, L_hi=L_hi,
        n_points=n_points, xmax=xmax,
        delta_coverage=delta_coverage,
        early_area=early_area, frac_pos_early=frac_pos,
        peak_pos=peak_pos, x_peak_pos=x_peak_pos,
        peak_neg=peak_neg, x_peak_neg=x_peak_neg,
        auc_total=auc_total,
        early_pos_end_neg=(np.isfinite(early_area) and early_area > 0 and delta_coverage < 0),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarise cross-bin k-mer spectra patterns from kmer_spectra.py outputs."
    )
    ap.add_argument("--indir", required=True, help="Directory containing *.support_stats.csv files.")
    ap.add_argument("--tag",   default=None,  help="Only process files whose name starts with this tag.")
    ap.add_argument("--k",     type=int, default=None, help="Only process files with .k<K> in the name.")
    ap.add_argument("--T",     type=int, default=50,
                    help="Abundance threshold for the 'early region' shape summary. (default: 50)")
    ap.add_argument("--out",   default=None,  help="Write per-bin summary to this CSV path.")
    args = ap.parse_args()

    # Discover files
    files = glob.glob(os.path.join(args.indir, "**", "*.support_stats.csv"), recursive=True)
    if args.tag:
        files = [p for p in files if os.path.basename(p).startswith(args.tag + ".")]
    if args.k is not None:
        files = [p for p in files if f".k{args.k}." in os.path.basename(p)]

    if not files:
        raise SystemExit("No support_stats.csv files found with the given filters.")

    rows: List[BinSummary] = []
    for p in sorted(files):
        try:
            rows.append(_summarise_file(p, T=args.T))
        except Exception as exc:
            print(f"[WARN] skipping {p}: {exc}")

    rows.sort(key=lambda r: (
        r.tag,
        r.k if r.k is not None else -1,
        r.bin_index if r.bin_index is not None else 10**9,
    ))

    # ------------------------------------------------------------------
    # Per-bin table
    # ------------------------------------------------------------------
    col_headers = [
        "tag", "k", "bin", "L_lo", "L_hi", "xmax", "n_points",
        "delta_coverage",
        f"early_area_x<=T({args.T})",
        f"frac_pos_x<=T({args.T})",
        "peak_pos", "x_peak_pos",
        "peak_neg", "x_peak_neg",
        "auc_total",
        "early_pos_end_neg",
    ]

    print("\n# Per-bin summaries")
    print("\t".join(col_headers))
    for r in rows:
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}" if np.isfinite(v) else "nan"
            return str(v)

        print("\t".join(_fmt(v) for v in [
            r.tag, r.k, r.bin_index, r.L_lo, r.L_hi, r.xmax, r.n_points,
            r.delta_coverage,
            r.early_area,
            r.frac_pos_early,
            r.peak_pos, r.x_peak_pos,
            r.peak_neg, r.x_peak_neg,
            r.auc_total,
            int(r.early_pos_end_neg),
        ]))

    # ------------------------------------------------------------------
    # General direction summary
    # ------------------------------------------------------------------
    delta_cov  = np.array([r.delta_coverage for r in rows], dtype=float)
    early_area = np.array([r.early_area     for r in rows], dtype=float)

    ok_cov   = np.isfinite(delta_cov)
    ok_early = np.isfinite(early_area)

    n          = len(rows)
    n_cov_neg  = int(np.sum(delta_cov[ok_cov] < 0))
    n_cov_pos  = int(np.sum(delta_cov[ok_cov] > 0))
    n_ea_pos   = int(np.sum(early_area[ok_early] > 0))
    n_ea_neg   = int(np.sum(early_area[ok_early] < 0))
    n_split    = sum(1 for r in rows if r.early_pos_end_neg)

    print("\n# General patterns")
    print(f"  bins analysed           : {n}")
    print(f"  coverage (Δcoverage)    : neg={n_cov_neg}  pos={n_cov_pos}  (of {int(ok_cov.sum())} valid)")
    print(f"  early shape (x ≤ {args.T:3d})  : pos={n_ea_pos}  neg={n_ea_neg}  (of {int(ok_early.sum())} valid)")
    print(f"  split pattern (early+, end−): {n_split} / {n}")

    if ok_cov.any():
        frac_neg = n_cov_neg / int(ok_cov.sum())
        if frac_neg >= 0.7:
            print("  -> Synthetic tends to have LOWER k-mer coverage (more absent k-mers) across bins.")
        elif frac_neg <= 0.3:
            print("  -> Synthetic tends to have HIGHER k-mer coverage across bins.")
        else:
            print("  -> k-mer coverage direction is MIXED across bins.")

    if ok_early.any():
        frac_pos = n_ea_pos / int(ok_early.sum())
        if frac_pos >= 0.7:
            print(f"  -> In the low-abundance regime (x ≤ {args.T}), synthetic has POSITIVE ΔCDF (more low-count mass).")
        elif frac_pos <= 0.3:
            print(f"  -> In the low-abundance regime (x ≤ {args.T}), synthetic has NEGATIVE ΔCDF (less low-count mass).")
        else:
            print(f"  -> Low-abundance ΔCDF direction is MIXED (x ≤ {args.T}).")

    # ------------------------------------------------------------------
    # Optional CSV output
    # ------------------------------------------------------------------
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(col_headers + ["path"])
            for r in rows:
                w.writerow([
                    r.tag, r.k, r.bin_index, r.L_lo, r.L_hi, r.xmax, r.n_points,
                    r.delta_coverage, r.early_area, r.frac_pos_early,
                    r.peak_pos, r.x_peak_pos,
                    r.peak_neg, r.x_peak_neg,
                    r.auc_total,
                    int(r.early_pos_end_neg),
                    r.path,
                ])
        print(f"\n  Summary CSV: {args.out}")


if __name__ == "__main__":
    main()
