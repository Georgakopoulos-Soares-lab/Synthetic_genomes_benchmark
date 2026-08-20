#!/usr/bin/env python3
"""
Reviewer #2, comment #3 fix.

KMC was run with DEFAULT canonical k-mer counting (no -b flag), which collapses
each k-mer and its reverse complement into a single canonical class. The original
pipeline then divided the canonical "observed_distinct_kmers" by the FULL k-mer
space (4**k). For odd k there are no reverse-complement palindromes, so the number
of canonical classes is exactly 4**k / 2; the observed count can therefore never
exceed 4**k / 2 and the reported nullomer fraction is floored at ~0.5 (this is the
"~0.5 at k=9" artifact the reviewer flagged in Figure 3).

This script recomputes the nullomer counts/fractions from the EXISTING KMC
observed-distinct values using the correct canonical denominator:

    canonical_classes(k) = (4**k + P(k)) / 2
    P(k) = number of reverse-complement palindromes
         = 0            if k is odd  (a DNA k-mer can only equal its revcomp for even k)
         = 4**(k/2)     if k is even

    nullomer_count_corrected    = canonical_classes(k) - observed_distinct_kmers
    nullomer_fraction_corrected = nullomer_count_corrected / canonical_classes(k)

No KMC rerun is required: observed_distinct_kmers is already the canonical count.
We keep the original columns for provenance and add corrected columns + the
denominator convention used.

Outputs:
  * For every results/**/nullomers.metrics.csv -> sibling nullomers.metrics.canonical.csv
  * A single combined table revisions/results/nullomers_canonical_combined.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import sys
from pathlib import Path

import pandas as pd


def canonical_classes(k: int) -> int:
    """Number of canonical (min(kmer, revcomp)) classes for DNA k-mers of length k."""
    if k % 2 == 0:
        palindromes = 4 ** (k // 2)
    else:
        palindromes = 0
    return (4 ** k + palindromes) // 2


def correct_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["canonical_classes"] = df["k"].astype(int).map(canonical_classes)
    df["full_kmer_space"] = df["k"].astype(int).map(lambda k: 4 ** k)
    # Provenance: keep the (incorrect) original fraction under a clear name.
    if "nullomer_fraction" in df.columns:
        df = df.rename(columns={"nullomer_fraction": "nullomer_fraction_orig_4k_denom"})
    if "nullomer_count" in df.columns:
        df = df.rename(columns={"nullomer_count": "nullomer_count_orig_4k_denom"})
    obs = df["observed_distinct_kmers"].astype(float)
    df["nullomer_count_canonical"] = (df["canonical_classes"] - obs).clip(lower=0)
    df["nullomer_fraction_canonical"] = (
        df["nullomer_count_canonical"] / df["canonical_classes"]
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        default=f"{_ROOT}/results",
        help="Root directory to search for nullomers.metrics.csv files.",
    )
    ap.add_argument(
        "--combined-out",
        default=f"{_ROOT}/revisions/results/nullomers_canonical_combined.csv",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    files = sorted(root.rglob("nullomers.metrics.csv"))
    if not files:
        print(f"[err] no nullomers.metrics.csv under {root}", file=sys.stderr)
        return 1

    combined = []
    n_written = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip {f}: {exc}", file=sys.stderr)
            continue
        if "observed_distinct_kmers" not in df.columns or "k" not in df.columns:
            print(f"[warn] unexpected schema, skip {f}", file=sys.stderr)
            continue
        out = correct_frame(df)
        out["source_file"] = str(f.relative_to(root))
        out_path = f.with_name("nullomers.metrics.canonical.csv")
        out.to_csv(out_path, index=False)
        n_written += 1
        combined.append(out)

    combined_df = pd.concat(combined, ignore_index=True)
    Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(args.combined_out, index=False)

    print(f"[ok] wrote {n_written} per-tag canonical CSVs")
    print(f"[ok] combined -> {args.combined_out}  ({len(combined_df)} rows)")

    # Quick sanity summary at k=9 (the flagged value).
    k9 = combined_df[combined_df["k"] == 9]
    if not k9.empty:
        print("\n[sanity] k=9 nullomer fraction (orig 4^k denom vs canonical):")
        print(
            k9.groupby("which")[
                ["nullomer_fraction_orig_4k_denom", "nullomer_fraction_canonical"]
            ]
            .mean()
            .round(5)
            .to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
