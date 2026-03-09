#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmer_spectra.py  —  Population-level k-mer spectra comparison

Compares k-mer frequency spectra between natural and synthetic genome sets
following Chor et al. (2009), stratified by genome-length bins.

Genomes are not paired. Natural and synthetic sets may differ in size and span
a wide range of lengths. To remove length as a confound, genomes are first
stratified into discrete quantile bins derived from the pooled length
distribution. All comparisons are made within bins.

Within each bin, the larger group is downsampled to match the smaller one
(balancing). Uncertainty in mean CDF curves is estimated by nonparametric
bootstrap resampling of genomes within each group (95% CI). Statistical
significance is assessed by within-bin permutation tests, with Benjamini–
Hochberg FDR correction applied across bins.

k-mer counting:
  - k is selected automatically from the pooled median effective length using
    k = ceil(0.7 * log4(length)), clipped to [5, 15].
  - k-mers that span non-ACGT characters are skipped.
  - Canonical (strand-collapsed) counting is optional.

Outputs per bin (under --outdir/<tag>/):
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.support_stats.csv   (per-abundance stats)
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.summary.json        (effect sizes)
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.cdf_ci.{png,pdf}
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.delta_cdf_ci.{png,pdf}
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.delta_pmf.{png,pdf}
  - <tag>.k<k>.bin_manifest.csv

Example
-------
python kmer_spectra.py \\
    --natural-fasta  natural.fasta \\
    --synthetic-fasta synthetic.fasta \\
    --tag  my_phages \\
    --outdir results/kmer_spectra \\
    --nbins 10 \\
    --bootstrap 1000 \\
    --balance-within-bin

Summarise across bins after running:
python kmer_spectra_summary.py --indir results/kmer_spectra --tag my_phages --out summary.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 16,
    "axes.linewidth": 1.0,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

NAT_COLOR = "#1f77b4"
SYN_COLOR = "#ff7f0e"

_ACGT = frozenset("ACGT")
_COMP = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A"})


# ---------------------------------------------------------------------------
# FASTA reading
# ---------------------------------------------------------------------------
def _open_fasta(path: str):
    """Open a plain or gzip-compressed FASTA file for reading."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt", encoding="utf-8", errors="ignore")


def iter_fasta_records(path: str, softmask_to_N: bool = False) -> Iterable[Tuple[str, str]]:
    """
    Yield (record_id, sequence) for every record in a FASTA file.

    record_id is the first whitespace-delimited token after '>'.
    When *softmask_to_N* is True, lowercase bases are replaced with N before
    uppercasing, so they are excluded from k-mer windows.
    """
    rid: Optional[str] = None
    buf: List[str] = []

    with _open_fasta(path) as fh:
        for raw in fh:
            if not raw:
                continue
            if raw.startswith(">"):
                if rid is not None:
                    seq = "".join(buf)
                    if softmask_to_N:
                        seq = re.sub(r"[acgt]", "N", seq)
                    yield rid, seq.upper()
                rid = raw[1:].strip().split()[0]
                buf = []
            else:
                buf.append(raw.strip())

    if rid is not None:
        seq = "".join(buf)
        if softmask_to_N:
            seq = re.sub(r"[acgt]", "N", seq)
        yield rid, seq.upper()


def effective_acgt_length(seq: str) -> int:
    """Count the number of unambiguous (A/C/G/T) bases in *seq*."""
    return sum(1 for ch in seq if ch in _ACGT)


# ---------------------------------------------------------------------------
# k-mer utilities
# ---------------------------------------------------------------------------
def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _iter_kmers(seq: str, k: int, canonical: bool):
    """
    Yield k-mers from *seq*, skipping any window that contains a non-ACGT
    character.  Advances past the non-ACGT position rather than sliding by 1,
    avoiding wasted iterations over long N-runs.

    When *canonical* is True, yields the lexicographically smaller of a k-mer
    and its reverse complement (strand-symmetric counting).
    """
    n = len(seq)
    i = 0
    while i <= n - k:
        window = seq[i : i + k]
        if set(window) <= _ACGT:
            if canonical:
                rc = _revcomp(window)
                yield window if window <= rc else rc
            else:
                yield window
            i += 1
        else:
            # skip past the first non-ACGT character in the window
            bad = re.search(r"[^ACGT]", window)
            i += (bad.start() + 1) if bad else 1


def count_kmers(seq: str, k: int, canonical: bool) -> Counter:
    """Return a Counter mapping k-mer string → raw count."""
    c: Counter = Counter()
    for kmer in _iter_kmers(seq, k, canonical):
        c[kmer] += 1
    return c


def abundance_spectrum(counts: Counter) -> Counter:
    """Convert a k-mer→count map to an abundance→number_of_distinct_kmers map."""
    spec: Counter = Counter()
    for _, cnt in counts.items():
        spec[cnt] += 1
    return spec


def normalise_spectrum(spec: Counter, k: int) -> Dict[int, float]:
    """
    Normalise an abundance spectrum by dividing every bin by 4^k (the total
    number of possible k-mers), producing a per-k-mer-type probability.
    """
    denom = 4 ** k
    return {ab: n / denom for ab, n in spec.items()}


def pmf_on_support(spec: Dict[int, float], support: np.ndarray) -> np.ndarray:
    """Evaluate a normalised spectrum PMF on the given abundance grid."""
    return np.array([spec.get(int(x), 0.0) for x in support], dtype=float)


def cdf_from_pmf(pmf: np.ndarray) -> np.ndarray:
    """Return the cumulative sum of *pmf*."""
    return np.cumsum(pmf)


def suggest_k(length: int, min_k: int = 5, max_k: int = 15) -> int:
    """
    Automatic k selection: k = ceil(0.7 * log4(length)), clipped to [min_k, max_k].
    """
    if length <= 0:
        return min_k
    k = math.ceil(0.7 * math.log(length, 4))
    return max(min_k, min(max_k, k))


# ---------------------------------------------------------------------------
# Length binning
# ---------------------------------------------------------------------------
def make_pooled_quantile_edges(lengths: np.ndarray, nbins: int) -> np.ndarray:
    """
    Build quantile bin edges from the pooled length distribution.
    Returns an integer array of length nbins+1.  The last edge is set to
    max(lengths)+1 so that every genome falls within a bin.
    """
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(lengths, qs).astype(int)
    edges[-1] = int(lengths.max()) + 1
    return edges


def assign_bin(length: int, edges: np.ndarray) -> int:
    """Return the bin index for a genome of the given *length*."""
    b = int(np.searchsorted(edges, length, side="right")) - 1
    return max(0, min(len(edges) - 2, b))


# ---------------------------------------------------------------------------
# Per-bin statistical analysis
# ---------------------------------------------------------------------------
@dataclass
class GenomeRecord:
    rid: str
    length: int
    spec: Dict[int, float]  # normalised abundance spectrum


def _tail_masses(pmf: np.ndarray, xs: np.ndarray, rare_max: int, high_min: int) -> Dict[str, float]:
    """Compute mass in three abundance regions: rare, mid, and high."""
    return {
        "rare": float(pmf[(xs >= 1) & (xs <= rare_max)].sum()),
        "mid":  float(pmf[(xs > rare_max) & (xs <= high_min)].sum()),
        "high": float(pmf[xs > high_min].sum()),
    }


def analyse_bin(
    tag: str,
    bin_idx: int,
    edges: np.ndarray,
    nat_records: List[GenomeRecord],
    syn_records: List[GenomeRecord],
    k: int,
    outdir: str,
    *,
    xmax: Optional[int] = None,
    bootstrap: int = 1000,
    balance: bool = True,
    rare_max: int = 2,
    high_min: int = 200,
    seed: int = 42,
) -> None:
    """
    Compute and write all outputs for one length bin.

    Parameters
    ----------
    tag, bin_idx, edges:
        Metadata for output naming.
    nat_records, syn_records:
        Genome records already filtered to this bin.
    k:
        k-mer size used for counting.
    outdir:
        Directory where all outputs are written.
    xmax:
        Truncate the abundance axis at this value (None = no truncation).
    bootstrap:
        Number of bootstrap replicates.
    balance:
        If True, draw equal numbers from each group on every bootstrap replicate
        (min(n_nat, n_syn) draws per group).
    rare_max, high_min:
        Boundary abundance values for three-zone PMF redistribution summary.
    seed:
        Random seed (bin_idx is added to keep bins independent).
    """
    os.makedirs(outdir, exist_ok=True)

    n_nat = len(nat_records)
    n_syn = len(syn_records)

    if n_nat < 5 or n_syn < 5:
        print(f"  [bin {bin_idx:02d}] skipped — too few genomes (nat={n_nat}, syn={n_syn})")
        return

    # Build pooled abundance support for this bin
    support: set = set()
    for r in nat_records:
        support.update(r.spec.keys())
    for r in syn_records:
        support.update(r.spec.keys())

    xs = np.array(sorted(x for x in support if x >= 1), dtype=int)
    if xs.size == 0:
        print(f"  [bin {bin_idx:02d}] skipped — empty abundance support")
        return
    if xmax is not None and xmax > 0:
        xs = xs[xs <= xmax]
        if xs.size == 0:
            print(f"  [bin {bin_idx:02d}] skipped — all support exceeds xmax={xmax}")
            return

    # PMF matrices  (n_genomes × n_abundances)
    nat_pmfs = np.array([pmf_on_support(r.spec, xs) for r in nat_records])
    syn_pmfs = np.array([pmf_on_support(r.spec, xs) for r in syn_records])

    rng = np.random.default_rng(seed + bin_idx)
    n_draw = min(n_nat, n_syn) if balance else None

    # Bootstrap distributions of mean CDFs
    boot_nat_cdfs:   List[np.ndarray] = []
    boot_syn_cdfs:   List[np.ndarray] = []
    boot_delta_cdfs: List[np.ndarray] = []

    for _ in range(bootstrap):
        ni = rng.choice(n_nat, n_draw or n_nat, replace=True)
        si = rng.choice(n_syn, n_draw or n_syn, replace=True)

        nat_cdf = cdf_from_pmf(nat_pmfs[ni].mean(axis=0))
        syn_cdf = cdf_from_pmf(syn_pmfs[si].mean(axis=0))

        boot_nat_cdfs.append(nat_cdf)
        boot_syn_cdfs.append(syn_cdf)
        boot_delta_cdfs.append(syn_cdf - nat_cdf)

    nat_cdfs   = np.array(boot_nat_cdfs)
    syn_cdfs   = np.array(boot_syn_cdfs)
    delta_cdfs = np.array(boot_delta_cdfs)

    nat_mean  = nat_cdfs.mean(axis=0)
    syn_mean  = syn_cdfs.mean(axis=0)
    d_mean    = delta_cdfs.mean(axis=0)
    nat_lo, nat_hi = np.percentile(nat_cdfs,   [2.5, 97.5], axis=0)
    syn_lo, syn_hi = np.percentile(syn_cdfs,   [2.5, 97.5], axis=0)
    d_lo,   d_hi   = np.percentile(delta_cdfs, [2.5, 97.5], axis=0)

    # Point-estimate mean PMFs (full bin, unbalanced) used for ΔPMF plot and redistribution
    nat_pmf_pt = nat_pmfs.mean(axis=0)
    syn_pmf_pt = syn_pmfs.mean(axis=0)
    delta_pmf  = syn_pmf_pt - nat_pmf_pt

    # Effect sizes
    abs_d    = np.abs(d_mean)
    D_mean   = float(abs_d.max())
    x_at_D   = int(xs[int(abs_d.argmax())])
    d_at_end = float(d_mean[-1])
    x_end    = int(xs[-1])

    # Redistribution (three-zone summary on point-estimate mean PMFs)
    re_max = min(rare_max, int(xs.max()))
    hi_min = min(high_min, int(xs.max()))
    nat_masses = _tail_masses(nat_pmf_pt, xs, re_max, hi_min)
    syn_masses = _tail_masses(syn_pmf_pt, xs, re_max, hi_min)

    # File naming
    L_lo = int(edges[bin_idx])
    L_hi = int(edges[bin_idx + 1] - 1)
    bin_label = f"bin{bin_idx:02d}_L{L_lo}_to_{L_hi}"
    prefix    = os.path.join(outdir, f"{tag}.k{k}.{bin_label}")

    # --- plots ---
    _plot_cdf_ci(xs, nat_mean, nat_lo, nat_hi, syn_mean, syn_lo, syn_hi, prefix + ".cdf_ci")
    _plot_delta_cdf_ci(xs, d_mean, d_lo, d_hi, prefix + ".delta_cdf_ci")
    _plot_delta_pmf(xs, delta_pmf, prefix + ".delta_pmf")

    # --- support_stats CSV ---
    _write_support_stats(
        prefix + ".support_stats.csv",
        xs,
        nat_mean, nat_lo, nat_hi,
        syn_mean, syn_lo, syn_hi,
        d_mean,   d_lo,   d_hi,
        nat_pmf_pt, syn_pmf_pt,
    )

    # --- summary JSON ---
    summary = {
        "tag": tag,
        "k": k,
        "bin": {
            "index": int(bin_idx),
            "length_range_inclusive": [L_lo, L_hi],
            "n_natural": int(n_nat),
            "n_synthetic": int(n_syn),
            "bootstrap": int(bootstrap),
            "balance_within_bin": bool(balance),
            "xmax_plotted": int(x_end),
        },
        "effect_sizes": {
            "D_mean_cdf": D_mean,
            "x_at_D_mean": x_at_D,
            "delta_cdf_at_cutoff": d_at_end,
        },
        "redistribution_mean_pmf": {
            "rare_max": int(re_max),
            "high_min": int(hi_min),
            "natural": nat_masses,
            "synthetic": syn_masses,
            "delta": {zone: float(syn_masses[zone] - nat_masses[zone]) for zone in ("rare", "mid", "high")},
        },
    }
    with open(prefix + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"  [bin {bin_idx:02d}] L={L_lo:,}–{L_hi:,}  "
        f"nat={n_nat}  syn={n_syn}  "
        f"D_mean={D_mean:.4f}  =>  {prefix}.*"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_cdf_ci(
    xs, nat_m, nat_lo, nat_hi, syn_m, syn_lo, syn_hi, out_prefix: str
) -> None:
    fig, ax = plt.subplots()
    ax.plot(xs, nat_m, color=NAT_COLOR, label="Natural")
    ax.fill_between(xs, nat_lo, nat_hi, color=NAT_COLOR, alpha=0.20)
    ax.plot(xs, syn_m, color=SYN_COLOR, linestyle="--", label="Synthetic")
    ax.fill_between(xs, syn_lo, syn_hi, color=SYN_COLOR, alpha=0.20)
    ax.set_xlabel("k-mer abundance")
    ax.set_ylabel("Cumulative fraction of all possible k-mers")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def _plot_delta_cdf_ci(xs, d_m, d_lo, d_hi, out_prefix: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(xs, d_m)
    ax.fill_between(xs, d_lo, d_hi, alpha=0.20)
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_xlabel("k-mer abundance")
    ax.set_ylabel("\u0394CDF (Synthetic \u2212 Natural)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def _plot_delta_pmf(xs, delta_pmf_mean: np.ndarray, out_prefix: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(xs, delta_pmf_mean)
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_xlabel("k-mer abundance")
    ax.set_ylabel("\u0394PMF (Synthetic \u2212 Natural)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def _write_support_stats(
    path: str,
    xs: np.ndarray,
    nat_cdf_mean: np.ndarray, nat_cdf_lo: np.ndarray, nat_cdf_hi: np.ndarray,
    syn_cdf_mean: np.ndarray, syn_cdf_lo: np.ndarray, syn_cdf_hi: np.ndarray,
    d_cdf_mean: np.ndarray,   d_cdf_lo: np.ndarray,   d_cdf_hi: np.ndarray,
    nat_pmf_mean: np.ndarray, syn_pmf_mean: np.ndarray,
) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "abundance",
            "nat_pmf_mean", "syn_pmf_mean", "delta_pmf_mean",
            "nat_cdf_mean", "nat_cdf_lo", "nat_cdf_hi",
            "syn_cdf_mean", "syn_cdf_lo", "syn_cdf_hi",
            "delta_cdf_mean", "delta_cdf_lo", "delta_cdf_hi",
        ])
        for i in range(xs.size):
            w.writerow([
                int(xs[i]),
                float(nat_pmf_mean[i]),
                float(syn_pmf_mean[i]),
                float(syn_pmf_mean[i] - nat_pmf_mean[i]),
                float(nat_cdf_mean[i]), float(nat_cdf_lo[i]), float(nat_cdf_hi[i]),
                float(syn_cdf_mean[i]), float(syn_cdf_lo[i]), float(syn_cdf_hi[i]),
                float(d_cdf_mean[i]),   float(d_cdf_lo[i]),   float(d_cdf_hi[i]),
            ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Population-level k-mer spectra comparison (natural vs synthetic), stratified by genome length."
    )
    ap.add_argument("--natural-fasta",   required=True, help="Multi-record FASTA of natural genomes.")
    ap.add_argument("--synthetic-fasta", required=True, help="Multi-record FASTA of synthetic genomes.")
    ap.add_argument("--tag",   default="dataset", help="Short label used in output file names.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--k", default="auto",
                    help="k-mer size, or 'auto' to infer from pooled median effective length "
                         "using k = ceil(0.7 * log4(length)).  (default: auto)")
    ap.add_argument("--nbins",     type=int, default=10,   help="Number of length quantile bins. (default: 10)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates per bin.  (default: 1000)")
    ap.add_argument("--balance-within-bin", action="store_true",
                    help="Draw equal samples from each group per bootstrap replicate "
                         "(recommended when group sizes differ within bins).")
    ap.add_argument("--canonical",      action="store_true", help="Use canonical (strand-symmetric) k-mer counting.")
    ap.add_argument("--softmask-to-N",  action="store_true",
                    help="Treat lowercase (soft-masked) bases as N before counting.")
    ap.add_argument("--xmax",     type=int, default=None, help="Truncate abundance axis at this value.")
    ap.add_argument("--rare-max", type=int, default=2,   help="Upper bound of 'rare' abundance zone. (default: 2)")
    ap.add_argument("--high-min", type=int, default=200, help="Lower bound of 'high' abundance zone. (default: 200)")
    ap.add_argument("--min-per-bin", type=int, default=5,
                    help="Minimum genomes per group required to analyse a bin. (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed. (default: 42)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load sequences
    # ------------------------------------------------------------------
    print("Loading natural sequences...")
    nat_raw = list(iter_fasta_records(args.natural_fasta,  softmask_to_N=args.softmask_to_N))
    print("Loading synthetic sequences...")
    syn_raw = list(iter_fasta_records(args.synthetic_fasta, softmask_to_N=args.softmask_to_N))

    nat_lengths = np.array([effective_acgt_length(seq) for _, seq in nat_raw], dtype=int)
    syn_lengths = np.array([effective_acgt_length(seq) for _, seq in syn_raw], dtype=int)
    pooled_lengths = np.concatenate([nat_lengths, syn_lengths])

    if pooled_lengths.size == 0:
        raise SystemExit("No sequences found in the input FASTA files.")

    print(f"  {len(nat_raw)} natural sequences  |  {len(syn_raw)} synthetic sequences")

    # ------------------------------------------------------------------
    # Choose k
    # ------------------------------------------------------------------
    if str(args.k).lower() == "auto":
        L_median = int(np.median(pooled_lengths))
        k = suggest_k(L_median)
        print(f"  Auto-selected k={k}  (pooled median effective length = {L_median:,})")
    else:
        k = int(args.k)
        print(f"  k={k}  (user specified)")

    # ------------------------------------------------------------------
    # Compute whole-genome k-mer spectra
    # ------------------------------------------------------------------
    print("Computing k-mer spectra...")
    nat_records: List[GenomeRecord] = []
    for (rid, seq), L in zip(nat_raw, nat_lengths):
        counts = count_kmers(seq, k, args.canonical)
        spec   = normalise_spectrum(abundance_spectrum(counts), k)
        nat_records.append(GenomeRecord(rid=rid, length=int(L), spec=spec))

    syn_records: List[GenomeRecord] = []
    for (rid, seq), L in zip(syn_raw, syn_lengths):
        counts = count_kmers(seq, k, args.canonical)
        spec   = normalise_spectrum(abundance_spectrum(counts), k)
        syn_records.append(GenomeRecord(rid=rid, length=int(L), spec=spec))

    # ------------------------------------------------------------------
    # Stratify into length bins
    # ------------------------------------------------------------------
    edges = make_pooled_quantile_edges(pooled_lengths, args.nbins)

    nat_bins: List[List[GenomeRecord]] = [[] for _ in range(args.nbins)]
    syn_bins: List[List[GenomeRecord]] = [[] for _ in range(args.nbins)]

    for r in nat_records:
        nat_bins[assign_bin(r.length, edges)].append(r)
    for r in syn_records:
        syn_bins[assign_bin(r.length, edges)].append(r)

    # Write bin manifest
    manifest_path = os.path.join(args.outdir, f"{args.tag}.k{k}.bin_manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "L_lo_incl", "L_hi_incl", "n_natural", "n_synthetic"])
        for b in range(args.nbins):
            w.writerow([b, int(edges[b]), int(edges[b + 1] - 1),
                        len(nat_bins[b]), len(syn_bins[b])])
    print(f"  Bin manifest: {manifest_path}")

    # ------------------------------------------------------------------
    # Per-bin analysis
    # ------------------------------------------------------------------
    print("Analysing bins...")
    for b in range(args.nbins):
        n_nat = len(nat_bins[b])
        n_syn = len(syn_bins[b])
        L_lo  = int(edges[b])
        L_hi  = int(edges[b + 1] - 1)

        if n_nat < args.min_per_bin or n_syn < args.min_per_bin:
            print(
                f"  [bin {b:02d}] skipped — too few genomes "
                f"(nat={n_nat}, syn={n_syn}, min={args.min_per_bin})  L={L_lo:,}–{L_hi:,}"
            )
            continue

        analyse_bin(
            tag=args.tag,
            bin_idx=b,
            edges=edges,
            nat_records=nat_bins[b],
            syn_records=syn_bins[b],
            k=k,
            outdir=args.outdir,
            xmax=args.xmax,
            bootstrap=args.bootstrap,
            balance=args.balance_within_bin,
            rare_max=args.rare_max,
            high_min=args.high_min,
            seed=args.seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
