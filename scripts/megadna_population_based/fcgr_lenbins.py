#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fcgr_lenbins.py  —  Population-level FCGR comparison

Compares Frequency Chaos Game Representation (FCGR) matrices between natural
and synthetic genome sets, stratified by genome-length bins.

Genomes are not paired. Natural and synthetic sets may differ in size and span
a wide range of lengths. To remove length as a confound, genomes are stratified
into discrete quantile bins derived from the pooled length distribution, and all
comparisons are performed within bins.

FCGR construction (Almeida & Vinga, 2002):
  - Order k produces a 2^k × 2^k probability matrix.
  - Each cell counts k-mers that map to a specific grid coordinate via the
    standard 2-bit corner encoding:
        A → (0, 0),  C → (0, 1),  G → (1, 0),  T → (1, 1)
  - k-mers that span non-ACGT characters (e.g. N) are skipped.  The rolling
    window is reset at each non-ACGT character.
  - The raw count matrix is normalised to a probability matrix by dividing
    by the total number of valid k-mers counted.

Length binning and balancing:
  - Bin edges are computed as quantiles of the *pooled* length distribution
    so that natural and synthetic genomes share the same bin boundaries.
  - Within each bin, the larger group is downsampled (without replacement) to
    match the smaller one (--balance-within-bin, recommended).

Per-bin outputs (written to --outdir):
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.tripanel.{png,pdf}
      Three-panel figure: mean natural FCGR / mean synthetic FCGR / log2 ratio.
      Natural and synthetic panels share a common colour scale (log10(P+eps)).
      The ratio panel uses symmetric scaling centred at zero.
  - <tag>.k<k>.bin<N>_L<lo>_to_<hi>.summary.json
      L1 distance between group mean FCGRs + 95% bootstrap CI (genome-level
      bootstrap within each group).
  - <tag>.k<k>.bins_summary.tsv
      One-row-per-bin table aggregating results across all bins.

Example
-------
python fcgr_lenbins.py \\
    --natural-fasta  natural.fasta \\
    --synthetic-fasta synthetic.fasta \\
    --tag  my_phages \\
    --outdir results/fcgr \\
    --k 8 --nbins 10 --balance-within-bin \\
    --bootstrap 1000 --seed 7
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (16, 5),
    "font.size": 14,
    "axes.linewidth": 1.0,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# FASTA reading
# ---------------------------------------------------------------------------
def _open_fasta(path: str):
    """Open a plain or gzip-compressed FASTA file for reading."""
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.endswith(".gz")
        else open(path, "rt", encoding="utf-8", errors="replace")
    )


def iter_fasta_records(path: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (record_id, sequence) for every record in a FASTA file.
    record_id is the first whitespace-delimited token after '>'.
    Sequence is uppercased before yielding.
    """
    rid: Optional[str] = None
    buf: List[str] = []

    with _open_fasta(path) as fh:
        for raw in fh:
            if not raw:
                continue
            if raw.startswith(">"):
                if rid is not None:
                    yield rid, "".join(buf).upper()
                rid = raw[1:].strip().split()[0]
                buf = []
            else:
                buf.append(raw.strip())

    if rid is not None:
        yield rid, "".join(buf).upper()


# ---------------------------------------------------------------------------
# FCGR computation
# ---------------------------------------------------------------------------
# Standard 2-bit corner encoding: A→(0,0), C→(0,1), G→(1,0), T→(1,1)
_FCGR_BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}


def compute_fcgr(seq: str, k: int) -> np.ndarray:
    """
    Compute the FCGR count matrix of shape (2^k, 2^k) using a rolling
    bit-accumulation approach.

    For each nucleotide the x-coordinate accumulates the first bit and the
    y-coordinate the second bit of its 2-bit code.  After k consecutive
    valid (ACGT) nucleotides the cell (y, x) is incremented.  Any non-ACGT
    character resets the rolling window so that no k-mer spans an ambiguous
    position.

    Returns
    -------
    np.ndarray of dtype uint32, shape (2^k, 2^k).
    """
    side = 1 << k
    mat  = np.zeros((side, side), dtype=np.uint32)
    mask = side - 1
    x = y = run = 0

    for ch in seq:
        bits = _FCGR_BITS.get(ch)
        if bits is None:
            x = y = run = 0
            continue
        bx, by = bits
        x   = ((x << 1) & mask) | bx
        y   = ((y << 1) & mask) | by
        run += 1
        if run >= k:
            mat[y, x] += 1

    return mat


def normalise_fcgr(mat: np.ndarray) -> np.ndarray:
    """Normalise a count matrix to a probability matrix (sums to ≤ 1)."""
    s = float(mat.sum())
    return mat.astype(np.float32) if s <= 0 else (mat / s).astype(np.float32)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """L1 (Manhattan) distance between two matrices."""
    return float(np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# Length binning
# ---------------------------------------------------------------------------
def make_pooled_quantile_bins(lengths: np.ndarray, nbins: int) -> List[Tuple[int, int]]:
    """
    Build *nbins* inclusive integer length ranges from the pooled distribution.

    Edges are derived from quantiles and rounded to integer values.  When
    multiple quantile edges coincide (e.g. if many genomes share the same
    length) bins are widened so that every bin has a distinct, non-overlapping
    integer range.

    Returns
    -------
    List of (lo, hi) tuples (both inclusive) of length *nbins*.
    """
    if lengths.size == 0:
        return [(1, 1)]

    qs    = np.linspace(0, 1, nbins + 1)
    edges = np.round(np.quantile(lengths, qs, method="linear")).astype(int)

    # Enforce non-decreasing
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]

    bins: List[Tuple[int, int]] = []
    for i in range(nbins):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        # Avoid overlap with the previous bin's upper edge
        if bins and lo <= bins[-1][1]:
            lo = bins[-1][1] + 1
        if hi < lo:
            hi = lo
        bins.append((lo, hi))

    return bins


def indices_in_bin(lengths: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return indices of genomes whose length falls within [lo, hi]."""
    return np.where((lengths >= lo) & (lengths <= hi))[0]


# ---------------------------------------------------------------------------
# Bootstrap CI on L1 distance between group mean FCGRs
# ---------------------------------------------------------------------------
def bootstrap_l1_ci(
    nat_fcgrs: np.ndarray,
    syn_fcgrs: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Estimate a (1-alpha) CI for the L1 distance between group mean FCGRs by
    resampling genomes with replacement within each group independently.

    Parameters
    ----------
    nat_fcgrs, syn_fcgrs : arrays of shape (N, 2^k, 2^k)
    n_boot : number of bootstrap replicates
    rng    : numpy random Generator
    alpha  : error rate (default 0.05 → 95% CI)

    Returns
    -------
    (lo, hi) percentile bounds.
    """
    n_nat = nat_fcgrs.shape[0]
    n_syn = syn_fcgrs.shape[0]

    if n_nat == 0 or n_syn == 0 or n_boot <= 0:
        return float("nan"), float("nan")

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ii = rng.integers(0, n_nat, size=n_nat)
        jj = rng.integers(0, n_syn, size=n_syn)
        boot[b] = l1_distance(nat_fcgrs[ii].mean(axis=0), syn_fcgrs[jj].mean(axis=0))

    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_tripanel(
    mean_nat: np.ndarray,
    mean_syn: np.ndarray,
    out_prefix: str,
    k: int,
    eps: float = 1e-10,
    pct_lo: float = 1.0,
    pct_hi: float = 99.9,
    ratio_pct: float = 99.5,
) -> None:
    """
    Save a three-panel figure:
      (1) mean natural FCGR — log10(P + eps)
      (2) mean synthetic FCGR — log10(P + eps), same colour scale as (1)
      (3) log2((natural + eps) / (synthetic + eps)) — symmetric colour scale

    Parameters
    ----------
    out_prefix : path without extension; both .png (300 dpi) and .pdf are saved.
    eps        : small constant added before log to prevent log(0).
    pct_lo/hi  : percentiles used to clip the shared colour scale.
    ratio_pct  : percentile of |ratio| used for symmetric ratio scaling.
    """
    log_nat = np.log10(mean_nat + eps)
    log_syn = np.log10(mean_syn + eps)

    both = np.concatenate([log_nat.ravel(), log_syn.ravel()])
    vmin = float(np.percentile(both, pct_lo))
    vmax = float(np.percentile(both, pct_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(both.min()), float(both.max())

    ratio = np.log2((mean_nat + eps) / (mean_syn + eps))
    dv = max(float(np.percentile(np.abs(ratio.ravel()), ratio_pct)), 1e-6)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    im1 = ax1.imshow(log_nat, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax1.set_title(f"Natural mean FCGR  (k={k})\nlog10(P + ε)")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    im2 = ax2.imshow(log_syn, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax2.set_title(f"Synthetic mean FCGR  (k={k})\nlog10(P + ε)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    im3 = ax3.imshow(ratio, vmin=-dv, vmax=dv, cmap="RdBu_r", interpolation="nearest", aspect="equal")
    ax3.set_title(f"log2 ratio  (natural / synthetic)\nlog2((nat + ε) / (syn + ε))")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_prefix + ".pdf",           bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Genome record
# ---------------------------------------------------------------------------
@dataclass
class GenomeRecord:
    rid: str
    length: int
    fcgr: np.ndarray  # normalised FCGR, shape (2^k, 2^k), dtype float32


def load_genomes(fasta: str, k: int, max_records: int = 0) -> List[GenomeRecord]:
    """
    Load all genomes from *fasta*, compute their FCGR at order *k*, and return
    a list of GenomeRecord objects.

    Parameters
    ----------
    fasta       : path to a (possibly multi-record) FASTA file.
    k           : FCGR order.
    max_records : if > 0, stop after reading this many records (useful for testing).
    """
    records: List[GenomeRecord] = []
    for idx, (rid, seq) in enumerate(iter_fasta_records(fasta), 1):
        if max_records and idx > max_records:
            break
        mat = compute_fcgr(seq, k)
        records.append(GenomeRecord(rid=rid, length=len(seq), fcgr=normalise_fcgr(mat)))
    return records


# ---------------------------------------------------------------------------
# Per-bin analysis
# ---------------------------------------------------------------------------
def analyse_bin(
    tag: str,
    bin_idx: int,
    L_lo: int,
    L_hi: int,
    nat_records: List[GenomeRecord],
    syn_records: List[GenomeRecord],
    k: int,
    outdir: str,
    *,
    bootstrap: int = 1000,
    rng: np.random.Generator,
    eps: float = 1e-10,
) -> dict:
    """
    Analyse one length bin: compute group mean FCGRs, L1 distance, bootstrap CI,
    produce the tripanel plot, and write a per-bin JSON summary.

    Parameters
    ----------
    nat_records, syn_records : genomes already filtered to this bin and balanced.
    rng : shared Generator (seeded externally so bins stay independent).

    Returns
    -------
    dict with scalar metrics for the TSV summary row.
    """
    os.makedirs(outdir, exist_ok=True)

    n_nat = len(nat_records)
    n_syn = len(syn_records)

    nat_fcgrs = np.stack([r.fcgr for r in nat_records], axis=0).astype(np.float32)
    syn_fcgrs = np.stack([r.fcgr for r in syn_records], axis=0).astype(np.float32)

    mean_nat = nat_fcgrs.mean(axis=0)
    mean_syn = syn_fcgrs.mean(axis=0)

    d_l1         = l1_distance(mean_nat, mean_syn)
    ci_lo, ci_hi = bootstrap_l1_ci(nat_fcgrs, syn_fcgrs, n_boot=bootstrap, rng=rng)

    # File naming
    bin_label  = f"bin{bin_idx:02d}_L{L_lo}_to_{L_hi}"
    out_prefix = os.path.join(outdir, f"{tag}.k{k}.{bin_label}")

    # Tripanel plot
    plot_tripanel(mean_nat, mean_syn, out_prefix=out_prefix + ".tripanel", k=k, eps=eps)

    # Per-bin JSON
    summary = {
        "tag": tag,
        "k": k,
        "bin": {
            "index": int(bin_idx),
            "length_range_inclusive": [int(L_lo), int(L_hi)],
            "n_natural": int(n_nat),
            "n_synthetic": int(n_syn),
            "bootstrap": int(bootstrap),
            "eps": float(eps),
        },
        "distance": {
            "l1_mean_fcgr": float(d_l1),
            "l1_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        },
        "notes": [
            "mean_nat and mean_syn are averages of per-genome normalised FCGR matrices.",
            "Bootstrap CI is computed by resampling genomes within each group and recomputing "
            "the L1 distance between the resampled group means.",
            "The tripanel uses a shared colour scale for the log10(P+eps) panels and symmetric "
            "scaling (centred at zero) for the log2 ratio panel.",
        ],
    }

    with open(out_prefix + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"  [bin {bin_idx:02d}] L={L_lo:,}–{L_hi:,}  "
        f"nat={n_nat}  syn={n_syn}  "
        f"L1={d_l1:.4g}  CI95=[{ci_lo:.4g}, {ci_hi:.4g}]  =>  {out_prefix}.*"
    )

    return {
        "tag": tag, "k": k, "bin": bin_idx,
        "L_lo": L_lo, "L_hi": L_hi,
        "n_natural": n_nat, "n_synthetic": n_syn,
        "l1_mean_fcgr": d_l1,
        "l1_boot_lo": ci_lo, "l1_boot_hi": ci_hi,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Population-level FCGR comparison (natural vs synthetic), stratified by genome length."
    )
    ap.add_argument("--natural-fasta",   required=True, help="Multi-record FASTA of natural genomes.")
    ap.add_argument("--synthetic-fasta", required=True, help="Multi-record FASTA of synthetic genomes.")
    ap.add_argument("--tag",    default="dataset", help="Short label used in output file names.")
    ap.add_argument("--outdir", required=True,     help="Output directory.")
    ap.add_argument("--k",    type=int, default=8,    help="FCGR order k (matrix size 2^k × 2^k). (default: 8)")
    ap.add_argument("--nbins", type=int, default=10,  help="Number of quantile length bins. (default: 10)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Bootstrap replicates per bin for the L1-distance CI. (default: 1000)")
    ap.add_argument("--balance-within-bin", action="store_true",
                    help="Downsample the larger group to match the smaller one within each bin "
                         "(recommended when group sizes differ).")
    ap.add_argument("--min-per-bin", type=int, default=5,
                    help="Skip bins where either group has fewer than this many genomes. (default: 5)")
    ap.add_argument("--eps",  type=float, default=1e-10,
                    help="Small constant added before log in visualisation. (default: 1e-10)")
    ap.add_argument("--max-records", type=int, default=0,
                    help="If > 0, load only the first N records from each FASTA. (default: 0 = all)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed. (default: 7)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load genomes and compute FCGRs
    # ------------------------------------------------------------------
    print(f"Loading natural genomes from: {args.natural_fasta}")
    nat = load_genomes(args.natural_fasta,   args.k, max_records=args.max_records)
    print(f"Loading synthetic genomes from: {args.synthetic_fasta}")
    syn = load_genomes(args.synthetic_fasta, args.k, max_records=args.max_records)

    if not nat or not syn:
        raise SystemExit("One or both FASTA files contain no valid records.")

    print(f"  {len(nat)} natural genomes  |  {len(syn)} synthetic genomes  |  k={args.k}")

    nat_lengths = np.array([r.length for r in nat], dtype=int)
    syn_lengths = np.array([r.length for r in syn], dtype=int)
    pooled      = np.concatenate([nat_lengths, syn_lengths])

    # ------------------------------------------------------------------
    # Build pooled quantile length bins
    # ------------------------------------------------------------------
    bins = make_pooled_quantile_bins(pooled, args.nbins)
    print(f"  Created {len(bins)} length bins from pooled distribution")

    # ------------------------------------------------------------------
    # Per-bin analysis
    # ------------------------------------------------------------------
    tsv_path = os.path.join(args.outdir, f"{args.tag}.k{args.k}.bins_summary.tsv")
    tsv_rows: List[dict] = []

    print("Analysing bins...")
    for b_idx, (L_lo, L_hi) in enumerate(bins):
        nat_idx = indices_in_bin(nat_lengths, L_lo, L_hi)
        syn_idx = indices_in_bin(syn_lengths, L_lo, L_hi)

        n_nat = int(nat_idx.size)
        n_syn = int(syn_idx.size)

        if n_nat < args.min_per_bin or n_syn < args.min_per_bin:
            print(
                f"  [bin {b_idx:02d}] skipped — too few genomes "
                f"(nat={n_nat}, syn={n_syn}, min={args.min_per_bin})  L={L_lo:,}–{L_hi:,}"
            )
            continue

        # Within-bin balancing: downsample larger group
        if args.balance_within_bin:
            m = min(n_nat, n_syn)
            nat_idx = rng.choice(nat_idx, size=m, replace=False)
            syn_idx = rng.choice(syn_idx, size=m, replace=False)
            n_nat = n_syn = int(m)

        bin_nat = [nat[i] for i in nat_idx]
        bin_syn = [syn[i] for i in syn_idx]

        row = analyse_bin(
            tag=args.tag,
            bin_idx=b_idx,
            L_lo=L_lo,
            L_hi=L_hi,
            nat_records=bin_nat,
            syn_records=bin_syn,
            k=args.k,
            outdir=args.outdir,
            bootstrap=args.bootstrap,
            rng=rng,
            eps=args.eps,
        )
        tsv_rows.append(row)

    # ------------------------------------------------------------------
    # Write cross-bin TSV summary
    # ------------------------------------------------------------------
    if tsv_rows:
        cols = ["tag", "k", "bin", "L_lo", "L_hi",
                "n_natural", "n_synthetic",
                "l1_mean_fcgr", "l1_boot_lo", "l1_boot_hi"]
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in tsv_rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\n  Cross-bin summary TSV: {tsv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
