#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmer_spectra.py

Compare k-mer abundance spectra between ORIGINAL and SYNTHETIC sequences using
Chor-style normalization:

  PMF(x) = n_x / 4^k
  where n_x = # distinct k-mers observed with abundance x (x >= 1)

Notes
-----
- Contig-aware: never counts k-mers across FASTA record boundaries.
- Optional canonical k-mers: k-mer and reverse complement collapsed.
- Optional softmask-to-N: lowercase a/c/g/t treated as 'N' so they break k-mers.
- Auto-k based on effective ACGT length (min of orig/syn).

Subcommands
-----------
1) pair
   - Per-pair spectra, CDF, Q-Q plots
   - Per-pair effect sizes (KS-like on PMF-CDF, JSD, EMD)
   - metrics CSV

2) aggregate
   - Aggregates many window-pairs (e.g., 300kbp windows) into:
       * mean CDF ± 95% CI
       * mean ΔCDF ± 95% CI
       * mean ΔPMF
       * support_stats.csv (mean+CI per support point)
       * summary.json (effect sizes + PMF-bin mass shifts)
   - Aggregate outputs are PNG.

Manifest format
---------------
CSV with at least columns:
  id, orig, syn

The id can be any string. If missing, it will be inferred as pair_###.

Examples
--------
Per-pair plots:
  python kmer_spectra.py pair --k auto --manifest pairs.csv --outdir results/spectra --softmask-to-N --smooth-win 5 --xmax 500

Aggregate:
  python kmer_spectra.py aggregate --k auto --manifest pairs.csv --outdir results/cdf_ci --tag Human --canonical --softmask-to-N --xmax 1200 --rare-max 2 --high-min 200
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Plot style (consistent, paper-friendly)
# -------------------------

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 13,
    "axes.linewidth": 1.0,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ORIG_COLOR = "#1f77b4"   # blue
SYN_COLOR  = "#ff7f0e"   # orange


# -------------------------
# FASTA reading (contig-aware)
# -------------------------

_ACGT = set("ACGT")
_comp = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A"})


def iter_fasta_records(path: str, softmask_to_N: bool) -> Iterable[str]:
    """
    Yield uppercase sequence strings for each FASTA record separately.
    If softmask_to_N is True, convert lowercase a/c/g/t to 'N' so they break k-mers.
    """
    def _open(p: str):
        if p.endswith(".gz"):
            return gzip.open(p, "rt")
        return open(p, "rt", encoding="utf-8", errors="ignore")

    seq: List[str] = []
    with _open(path) as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    s = "".join(seq)
                    if softmask_to_N:
                        s = re.sub(r"[acgt]", "N", s)
                    yield s.upper()
                    seq = []
                continue
            seq.append(line.strip())

        if seq:
            s = "".join(seq)
            if softmask_to_N:
                s = re.sub(r"[acgt]", "N", s)
            yield s.upper()


# -------------------------
# k-mer utilities
# -------------------------

def revcomp(s: str) -> str:
    return s.translate(_comp)[::-1]


def iter_kmers_in_seq(seq: str, k: int, canonical: bool) -> Iterable[str]:
    """
    Yield (optionally canonical) k-mers from one sequence, skipping windows with non-ACGT.
    """
    n = len(seq)
    i = 0
    while i <= n - k:
        window = seq[i:i + k]
        if set(window) <= _ACGT:
            if canonical:
                rc = revcomp(window)
                yield window if window <= rc else rc
            else:
                yield window
            i += 1
        else:
            # skip past the first invalid base in the window
            m = re.search(r"[^ACGT]", window)
            i += (m.start() + 1) if m else 1


def count_kmers_fasta(path: str, k: int, canonical: bool, softmask_to_N: bool) -> Tuple[Counter, int]:
    """
    Count k-mers across all records WITHOUT crossing record boundaries.
    Returns (counts, total_bases_acgt).
    """
    c = Counter()
    total_acgt = 0
    for rec in iter_fasta_records(path, softmask_to_N=softmask_to_N):
        total_acgt += sum(1 for ch in rec if ch in _ACGT)
        for kmer in iter_kmers_in_seq(rec, k, canonical=canonical):
            c[kmer] += 1
    return c, total_acgt


# -------------------------
# Spectra + normalization
# -------------------------

def spectrum_from_counts(counts: Counter) -> Counter:
    """Map abundance x -> number of DISTINCT k-mers with that abundance."""
    spec = Counter()
    for _, cnt in counts.items():
        spec[cnt] += 1
    return spec


def normalize_spectrum(spec: Counter, k: int) -> Dict[int, float]:
    """
    Chor-style normalization: PMF(x) = n_x / 4^k over x >= 1.
    """
    if not spec:
        return {}
    denom = 4 ** k
    return {abundance: n / denom for abundance, n in spec.items()}


def support_union(a: Dict[int, float], b: Dict[int, float]) -> np.ndarray:
    return np.array(sorted(set(a.keys()) | set(b.keys())), dtype=int)


def pmf_on_support(spec: Dict[int, float], support: np.ndarray) -> np.ndarray:
    return np.array([spec.get(int(x), 0.0) for x in support], dtype=float)


def cdf_from_pmf(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(pmf)


# -------------------------
# Auto-k heuristic
# -------------------------

def suggest_k_from_length(L: int, min_k: int = 5, max_k: int = 15) -> int:
    if L <= 0:
        return min_k
    k = math.ceil(0.7 * (math.log(L, 4)))  # log base 4
    return max(min_k, min(max_k, int(k)))


def resolve_k(k_opt: Union[str, int], orig_path: str, syn_path: str, softmask_to_N: bool) -> int:
    if str(k_opt).lower() != "auto":
        return int(k_opt)

    _, L_o = count_kmers_fasta(orig_path, 1, canonical=False, softmask_to_N=softmask_to_N)
    _, L_s = count_kmers_fasta(syn_path, 1, canonical=False, softmask_to_N=softmask_to_N)
    L_eff = min(L_o, L_s)
    k = suggest_k_from_length(L_eff)
    print(f"  -> auto-k: L_orig(ACGT)={L_o:,}, L_syn(ACGT)={L_s:,}, L_eff={L_eff:,} => k={k}")
    return k


# -------------------------
# Metrics (KS-like on PMF-CDF, JSD, EMD)
# -------------------------

def ks_distance_from_pmf(p: np.ndarray, q: np.ndarray) -> float:
    """KS-like distance computed on the CDFs of the PMFs."""
    return float(np.max(np.abs(cdf_from_pmf(p) - cdf_from_pmf(q))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(p, eps, 1)
    Q = np.clip(q, eps, 1)
    P /= P.sum() if P.sum() > 0 else 1.0
    Q /= Q.sum() if Q.sum() > 0 else 1.0
    M = 0.5 * (P + Q)

    def kl(a, b):
        return float(np.sum(a * (np.log2(a) - np.log2(b))))

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(cdf_from_pmf(p) - cdf_from_pmf(q))))


# -------------------------
# Plot helpers
# -------------------------

def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or y.size == 0:
        return y
    if win % 2 == 0:
        win += 1
    pad = win // 2
    z = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(z, kernel, mode="valid")


def _dense_hist_from_pmf(support: np.ndarray, pmf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sparse PMF into dense arrays on x = 1..x_max.
    """
    if support.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    x_max = int(support.max())
    xs = np.arange(1, x_max + 1, dtype=int)
    probs = np.zeros_like(xs, dtype=float)
    idx = support.astype(int) - 1
    probs[idx] = pmf
    return xs, probs


# -------------------------
# Pair mode: plots + metrics
# -------------------------

@dataclass
class PairResult:
    pair_id: str
    k_used: int
    ks_stat: float
    jsd_bits: float
    emd_l1cdf: float
    L_orig_acgt: int
    L_syn_acgt: int


def spectra_from_path(path: str, k: int, canonical: bool, softmask_to_N: bool) -> Tuple[Dict[int, float], int]:
    counts, L_acgt = count_kmers_fasta(path, k, canonical=canonical, softmask_to_N=softmask_to_N)
    spec = spectrum_from_counts(counts)
    spec_norm = normalize_spectrum(spec, k)
    return spec_norm, L_acgt


def plot_pair_bundle(
    pair_id: str,
    k: int,
    support: np.ndarray,
    p_norm: np.ndarray,
    q_norm: np.ndarray,
    outdir: str,
    use_hist: bool,
    smooth_win: int,
    xmax: Optional[int],
    save_pdf: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Apply cutoff for plotting
    if xmax is not None and xmax > 0:
        mask = support <= xmax
        support_plot = support[mask]
        p_plot = p_norm[mask]
        q_plot = q_norm[mask]
    else:
        support_plot, p_plot, q_plot = support, p_norm, q_norm

    # --- Spectra ---
    if use_hist:
        xo, po = _dense_hist_from_pmf(support_plot, p_plot)
        xs, ps = _dense_hist_from_pmf(support_plot, q_plot)
        if smooth_win and smooth_win > 1:
            po = _moving_average(po, smooth_win)
            ps = _moving_average(ps, smooth_win)

        plt.figure()
        plt.bar(xo, po, width=1.0, align="center", alpha=0.7, label="Original", color=ORIG_COLOR, linewidth=0)
        plt.bar(xs, ps, width=1.0, align="center", alpha=0.5, label="Synthetic", color=SYN_COLOR, linewidth=0)
    else:
        plt.figure()
        plt.step(support_plot, p_plot, where="mid", label="Original", linewidth=1.6, color=ORIG_COLOR)
        plt.step(support_plot, q_plot, where="mid", label="Synthetic", linewidth=1.6, linestyle="--", color=SYN_COLOR)

    plt.xlabel("k-mer abundance")
    plt.ylabel("Frequency of appearance")
    if xmax is not None and xmax > 0:
        plt.xlim(0, xmax)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    png = os.path.join(outdir, f"{pair_id}.k{k}.spectra.png")
    plt.savefig(png, dpi=300)
    if save_pdf:
        plt.savefig(os.path.join(outdir, f"{pair_id}.k{k}.spectra.pdf"))
    plt.close()

    # --- CDF ---
    plt.figure()
    Fp = cdf_from_pmf(p_plot)
    Fq = cdf_from_pmf(q_plot)
    plt.step(support_plot, Fp, where="post", label="Original", linewidth=1.6, color=ORIG_COLOR)
    plt.step(support_plot, Fq, where="post", label="Synthetic", linewidth=1.6, linestyle="--", color=SYN_COLOR)
    plt.xlabel("k-mer abundance")
    plt.ylabel("Cumulative fraction of all possible k-mers")
    if xmax is not None and xmax > 0:
        plt.xlim(0, xmax)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    png = os.path.join(outdir, f"{pair_id}.k{k}.cdf.png")
    plt.savefig(png, dpi=300)
    if save_pdf:
        plt.savefig(os.path.join(outdir, f"{pair_id}.k{k}.cdf.pdf"))
    plt.close()

    # --- Q-Q ---
    plt.figure(figsize=(4.2, 4.2))
    cum = np.linspace(0.01, 0.99, 99)

    def quantile_from_cdf(cdf: np.ndarray, xs: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return np.interp(probs, cdf, xs)

    qp = quantile_from_cdf(Fp, support_plot, cum)
    qqv = quantile_from_cdf(Fq, support_plot, cum)

    lim_max = float(xmax) if (xmax is not None and xmax > 0) else max(qp.max(initial=1.0), qqv.max(initial=1.0)) * 1.05
    lim_max = max(lim_max, 1.0)

    plt.plot(qp, qqv, marker=".", linestyle="none", alpha=0.8)
    plt.plot([0, lim_max], [0, lim_max], linewidth=1.0, color="black", linestyle="--")
    plt.xlabel("Original quantiles (counts)")
    plt.ylabel("Synthetic quantiles (counts)")
    plt.xlim(0, lim_max)
    plt.ylim(0, lim_max)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    png = os.path.join(outdir, f"{pair_id}.k{k}.qq.png")
    plt.savefig(png, dpi=300)
    if save_pdf:
        plt.savefig(os.path.join(outdir, f"{pair_id}.k{k}.qq.pdf"))
    plt.close()


def compare_pair(
    pair_id: str,
    orig_path: str,
    syn_path: str,
    k_opt: Union[str, int],
    outdir: str,
    canonical: bool,
    softmask_to_N: bool,
    use_hist: bool,
    smooth_win: int,
    xmax: Optional[int],
    save_pdf: bool,
) -> PairResult:
    k = resolve_k(k_opt, orig_path, syn_path, softmask_to_N=softmask_to_N)

    spec_o, L_o = spectra_from_path(orig_path, k, canonical=canonical, softmask_to_N=softmask_to_N)
    spec_s, L_s = spectra_from_path(syn_path,  k, canonical=canonical, softmask_to_N=softmask_to_N)
    print(f"  -> k={k} (canonical={canonical}) : L_orig(ACGT)={L_o:,}, L_syn(ACGT)={L_s:,}")

    supp = support_union(spec_o, spec_s)
    pmf_o = pmf_on_support(spec_o, supp)
    pmf_s = pmf_on_support(spec_s, supp)

    if supp.size and supp.min() < 1:
        raise AssertionError("Support should start at 1 (no zero-abundance bins).")

    ks_stat = ks_distance_from_pmf(pmf_o, pmf_s)
    jsd = js_divergence(pmf_o, pmf_s)
    emd = emd_1d(pmf_o, pmf_s)

    # plots
    plot_pair_bundle(
        pair_id=pair_id,
        k=k,
        support=supp,
        p_norm=pmf_o,
        q_norm=pmf_s,
        outdir=outdir,
        use_hist=use_hist,
        smooth_win=smooth_win,
        xmax=xmax,
        save_pdf=save_pdf,
    )

    return PairResult(pair_id, k, ks_stat, jsd, emd, L_o, L_s)




# -------------------------
# Aggregate mode: mean CDF ± CI, ΔCDF ± CI, ΔPMF
# -------------------------

def mass_in_range(pmf: np.ndarray, xs: np.ndarray, lo: int, hi: int) -> float:
    mask = (xs >= lo) & (xs <= hi)
    return float(pmf[mask].sum())


def tail_masses(orig_pmf_mean: np.ndarray, syn_pmf_mean: np.ndarray, xs: np.ndarray, rare_max: int, high_min: int) -> Dict[str, Dict[str, float]]:
    x_max = int(xs.max())
    rare_o = mass_in_range(orig_pmf_mean, xs, 1, rare_max)
    rare_s = mass_in_range(syn_pmf_mean,  xs, 1, rare_max)

    mid_o  = mass_in_range(orig_pmf_mean, xs, rare_max + 1, high_min)
    mid_s  = mass_in_range(syn_pmf_mean,  xs, rare_max + 1, high_min)

    high_o = mass_in_range(orig_pmf_mean, xs, high_min + 1, x_max)
    high_s = mass_in_range(syn_pmf_mean,  xs, high_min + 1, x_max)

    return {
        "rare": {"orig": rare_o, "syn": rare_s, "delta": (rare_s - rare_o)},
        "mid":  {"orig": mid_o,  "syn": mid_s,  "delta": (mid_s  - mid_o)},
        "high": {"orig": high_o, "syn": high_s, "delta": (high_s - high_o)},
    }


def write_support_csv(
    path: str,
    xs: np.ndarray,
    orig_pmf_mean: np.ndarray, orig_pmf_lo: np.ndarray, orig_pmf_hi: np.ndarray,
    syn_pmf_mean: np.ndarray,  syn_pmf_lo: np.ndarray,  syn_pmf_hi: np.ndarray,
    orig_cdf_mean: np.ndarray, orig_cdf_lo: np.ndarray, orig_cdf_hi: np.ndarray,
    syn_cdf_mean: np.ndarray,  syn_cdf_lo: np.ndarray,  syn_cdf_hi: np.ndarray,
    delta_cdf_mean: np.ndarray, delta_cdf_lo: np.ndarray, delta_cdf_hi: np.ndarray,
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "abundance",
            "orig_pmf_mean","orig_pmf_lo","orig_pmf_hi",
            "syn_pmf_mean","syn_pmf_lo","syn_pmf_hi",
            "delta_pmf_mean",
            "orig_cdf_mean","orig_cdf_lo","orig_cdf_hi",
            "syn_cdf_mean","syn_cdf_lo","syn_cdf_hi",
            "delta_cdf_mean","delta_cdf_lo","delta_cdf_hi"
        ])
        for i in range(xs.size):
            w.writerow([
                int(xs[i]),
                float(orig_pmf_mean[i]), float(orig_pmf_lo[i]), float(orig_pmf_hi[i]),
                float(syn_pmf_mean[i]),  float(syn_pmf_lo[i]),  float(syn_pmf_hi[i]),
                float(syn_pmf_mean[i] - orig_pmf_mean[i]),
                float(orig_cdf_mean[i]), float(orig_cdf_lo[i]), float(orig_cdf_hi[i]),
                float(syn_cdf_mean[i]),  float(syn_cdf_lo[i]),  float(syn_cdf_hi[i]),
                float(delta_cdf_mean[i]), float(delta_cdf_lo[i]), float(delta_cdf_hi[i]),
            ])


def plot_cdf_ci_png(
    xs: np.ndarray,
    orig_mean: np.ndarray, orig_lo: np.ndarray, orig_hi: np.ndarray,
    syn_mean: np.ndarray,  syn_lo: np.ndarray,  syn_hi: np.ndarray,
    out_png: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, orig_mean, color=ORIG_COLOR, label="Original")
    plt.fill_between(xs, orig_lo, orig_hi, color=ORIG_COLOR, alpha=0.20)
    plt.plot(xs, syn_mean, color=SYN_COLOR, linestyle="--", label="Synthetic")
    plt.fill_between(xs, syn_lo, syn_hi, color=SYN_COLOR, alpha=0.20)
    plt.xlabel("k-mer abundance")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_delta_cdf_ci_png(
    xs: np.ndarray,
    delta_mean: np.ndarray,
    delta_lo: np.ndarray,
    delta_hi: np.ndarray,
    out_png: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, delta_mean)
    plt.fill_between(xs, delta_lo, delta_hi, alpha=0.20)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k-mer abundance")
    plt.ylabel("ΔCDF (Synthetic − Original)")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_delta_pmf_png(xs: np.ndarray, delta_pmf_mean: np.ndarray, out_png: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, delta_pmf_mean)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k-mer abundance")
    plt.ylabel("ΔPMF (Synthetic − Original)")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def aggregate_from_manifest(
    manifest_path: str,
    k_opt: Union[str, int],
    outdir: str,
    tag: str,
    canonical: bool,
    softmask_to_N: bool,
    xmax: Optional[int],
    rare_max: int,
    high_min: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    rows: List[dict] = []
    with open(manifest_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)
    if not rows:
        raise SystemExit("Manifest is empty.")

    # decide k using first pair (consistent with your earlier script)
    first = rows[0]
    k = resolve_k(k_opt, first["orig"], first["syn"], softmask_to_N=softmask_to_N)
    print(f"[aggregate] tag={tag} k={k} n_pairs={len(rows)}")

    orig_specs: List[Dict[int, float]] = []
    syn_specs:  List[Dict[int, float]] = []
    global_support: set[int] = set()

    for i, row in enumerate(rows, 1):
        op = row["orig"]
        sp = row["syn"]
        print(f"[{i}/{len(rows)}] counting k={k} :: {op} vs {sp}")

        counts_o, _ = count_kmers_fasta(op, k, canonical=canonical, softmask_to_N=softmask_to_N)
        counts_s, _ = count_kmers_fasta(sp, k, canonical=canonical, softmask_to_N=softmask_to_N)

        spec_o = normalize_spectrum(spectrum_from_counts(counts_o), k)
        spec_s = normalize_spectrum(spectrum_from_counts(counts_s), k)

        orig_specs.append(spec_o)
        syn_specs.append(spec_s)
        global_support.update(spec_o.keys())
        global_support.update(spec_s.keys())

    support = np.array(sorted(global_support), dtype=int)
    if support.size == 0:
        raise SystemExit("No k-mers found (support empty).")
    if support.min() < 1:
        raise AssertionError("Support should start at 1 (no zero-abundance bins).")

    # align PMFs
    orig_pmfs = np.array([pmf_on_support(spec, support) for spec in orig_specs])  # (n, m)
    syn_pmfs  = np.array([pmf_on_support(spec, support) for spec in syn_specs])

    orig_cdfs = np.cumsum(orig_pmfs, axis=1)
    syn_cdfs  = np.cumsum(syn_pmfs,  axis=1)

    # mean + empirical CI across windows
    orig_pmf_mean = orig_pmfs.mean(axis=0)
    syn_pmf_mean  = syn_pmfs.mean(axis=0)
    orig_cdf_mean = orig_cdfs.mean(axis=0)
    syn_cdf_mean  = syn_cdfs.mean(axis=0)

    orig_pmf_lo, orig_pmf_hi = np.percentile(orig_pmfs, [2.5, 97.5], axis=0)
    syn_pmf_lo,  syn_pmf_hi  = np.percentile(syn_pmfs,  [2.5, 97.5], axis=0)
    orig_cdf_lo, orig_cdf_hi = np.percentile(orig_cdfs, [2.5, 97.5], axis=0)
    syn_cdf_lo,  syn_cdf_hi  = np.percentile(syn_cdfs,  [2.5, 97.5], axis=0)

    delta_cdfs = syn_cdfs - orig_cdfs
    delta_cdf_mean = delta_cdfs.mean(axis=0)
    delta_cdf_lo, delta_cdf_hi = np.percentile(delta_cdfs, [2.5, 97.5], axis=0)

    # apply xmax mask (plot + reported effect sizes on plotted domain)
    if xmax is not None and xmax > 0:
        mask = support <= xmax
        xs = support[mask]
        orig_pmf_mean_m, syn_pmf_mean_m = orig_pmf_mean[mask], syn_pmf_mean[mask]
        orig_cdf_mean_m, syn_cdf_mean_m = orig_cdf_mean[mask], syn_cdf_mean[mask]
        orig_pmf_lo_m, orig_pmf_hi_m = orig_pmf_lo[mask], orig_pmf_hi[mask]
        syn_pmf_lo_m,  syn_pmf_hi_m  = syn_pmf_lo[mask],  syn_pmf_hi[mask]
        orig_cdf_lo_m, orig_cdf_hi_m = orig_cdf_lo[mask], orig_cdf_hi[mask]
        syn_cdf_lo_m,  syn_cdf_hi_m  = syn_cdf_lo[mask],  syn_cdf_hi[mask]
        delta_cdf_mean_m = delta_cdf_mean[mask]
        delta_cdf_lo_m, delta_cdf_hi_m = delta_cdf_lo[mask], delta_cdf_hi[mask]
    else:
        xs = support
        orig_pmf_mean_m, syn_pmf_mean_m = orig_pmf_mean, syn_pmf_mean
        orig_cdf_mean_m, syn_cdf_mean_m = orig_cdf_mean, syn_cdf_mean
        orig_pmf_lo_m, orig_pmf_hi_m = orig_pmf_lo, orig_pmf_hi
        syn_pmf_lo_m,  syn_pmf_hi_m  = syn_pmf_lo,  syn_pmf_hi
        orig_cdf_lo_m, orig_cdf_hi_m = orig_cdf_lo, orig_cdf_hi
        syn_cdf_lo_m,  syn_cdf_hi_m  = syn_cdf_lo,  syn_cdf_hi
        delta_cdf_mean_m = delta_cdf_mean
        delta_cdf_lo_m, delta_cdf_hi_m = delta_cdf_lo, delta_cdf_hi

    # effect sizes on mean CDF
    delta_mean = syn_cdf_mean_m - orig_cdf_mean_m
    abs_delta = np.abs(delta_mean)
    D_mean = float(abs_delta.max()) if xs.size else float("nan")
    x_at_D = int(xs[int(abs_delta.argmax())]) if xs.size else None
    delta_at_cutoff = float(delta_mean[-1]) if xs.size else float("nan")
    cutoff_x = int(xs[-1]) if xs.size else None

    # bin masses on mean PMF
    if xs.size:
        rare_max_eff = min(rare_max, int(xs.max()))
        high_min_eff = min(high_min, int(xs.max()))
    else:
        rare_max_eff, high_min_eff = rare_max, high_min

    bins = tail_masses(orig_pmf_mean_m, syn_pmf_mean_m, xs, rare_max_eff, high_min_eff) if xs.size else {}

    summary = {
        "tag": tag,
        "k": int(k),
        "n_windows": int(orig_pmfs.shape[0]),
        "xmax_plotted": int(cutoff_x) if cutoff_x is not None else None,
        "effect_sizes": {
            "D_mean_cdf": D_mean,
            "x_at_D_mean": x_at_D,
            "delta_cdf_at_cutoff": delta_at_cutoff,
        },
        "bins": {
            "rare_max": int(rare_max_eff),
            "high_min": int(high_min_eff),
            "masses": bins,
        },
        "normalization": "Chor-style PMF(x)=n_x/4^k over x>=1 (x=0 omitted).",
    }

    base = f"{tag}.k{k}"

    # plots (PNG only)
    plot_cdf_ci_png(
        xs,
        orig_cdf_mean_m, orig_cdf_lo_m, orig_cdf_hi_m,
        syn_cdf_mean_m,  syn_cdf_lo_m,  syn_cdf_hi_m,
        out_png=os.path.join(outdir, f"{base}.cdf_ci.png"),
        ylabel="Cumulative fraction of all possible k-mers",
    )
    plot_delta_cdf_ci_png(
        xs,
        delta_cdf_mean_m, delta_cdf_lo_m, delta_cdf_hi_m,
        out_png=os.path.join(outdir, f"{base}.delta_cdf_ci.png"),
    )
    plot_delta_pmf_png(
        xs,
        syn_pmf_mean_m - orig_pmf_mean_m,
        out_png=os.path.join(outdir, f"{base}.delta_pmf.png"),
    )

    # CSV + JSON
    csv_path = os.path.join(outdir, f"{base}.support_stats.csv")
    write_support_csv(
        csv_path,
        xs,
        orig_pmf_mean_m, orig_pmf_lo_m, orig_pmf_hi_m,
        syn_pmf_mean_m,  syn_pmf_lo_m,  syn_pmf_hi_m,
        orig_cdf_mean_m, orig_cdf_lo_m, orig_cdf_hi_m,
        syn_cdf_mean_m,  syn_cdf_lo_m,  syn_cdf_hi_m,
        delta_cdf_mean_m, delta_cdf_lo_m, delta_cdf_hi_m,
    )

    json_path = os.path.join(outdir, f"{base}.summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Aggregate Summary ===")
    print(f"tag={tag}  k={k}  n_windows={orig_pmfs.shape[0]}")
    if cutoff_x is not None:
        print(f"plotted cutoff xmax={cutoff_x}")
    print(f"D_mean_cdf = {D_mean:.6g}  at abundance x={x_at_D}")
    print(f"ΔCDF(at cutoff) = {delta_at_cutoff:.6g}  (Synthetic − Original)")
    if bins:
        print("Bin masses (mean PMF over plotted support):")
        for name, d in bins.items():
            print(f"  {name:>4}: orig={d['orig']:.6g}  syn={d['syn']:.6g}  delta={d['delta']:.6g}")
    print(f"Wrote: {os.path.join(outdir, base)}.* (PNG + CSV + JSON)\n")


# -------------------------
# Manifest utilities
# -------------------------

def load_manifest(manifest: str) -> List[dict]:
    rows: List[dict] = []
    with open(manifest, newline="") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr, 1):
            if not row.get("id"):
                row["id"] = f"pair_{i:03d}"
            rows.append(row)
    return rows


def select_pairs(rows: List[dict], selectors: Optional[Sequence[str]]) -> List[Tuple[int, dict]]:
    if not selectors:
        return list(enumerate(rows, start=1))

    selected_idx = set()
    selected_ids = set()
    for tok in selectors:
        if tok.isdigit():
            selected_idx.add(int(tok))
        else:
            selected_ids.add(tok)

    out: List[Tuple[int, dict]] = []
    for i, row in enumerate(rows, start=1):
        pid = row["id"]
        if (i in selected_idx) or (pid in selected_ids):
            out.append((i, row))
    return out


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="k-mer spectra benchmark (pair + aggregate).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # pair mode
    p_pair = sub.add_parser("pair", help="Per-pair spectra/CDF/Q-Q plots + metrics CSV.")
    p_pair.add_argument("--k", required=True, help="k or 'auto'.")
    p_pair.add_argument("--orig", help="Original FASTA(.gz) (single-pair mode).")
    p_pair.add_argument("--syn", help="Synthetic FASTA(.gz) (single-pair mode).")
    p_pair.add_argument("--manifest", help="CSV with columns: id,orig,syn (multi-pair mode).")
    p_pair.add_argument("--pairs", nargs="+", help="Subset: 1-based row indices and/or pair IDs.")
    p_pair.add_argument("--outdir", required=True, help="Output directory for plots/metrics.")
    p_pair.add_argument("--canonical", action="store_true", help="Use canonical k-mers (revcomp collapsed).")
    p_pair.add_argument("--softmask-to-N", action="store_true", help="Treat lowercase as N.")
    p_pair.add_argument("--no-hist", action="store_true", help="Use step lines instead of filled hist.")
    p_pair.add_argument("--smooth-win", type=int, default=5, help="Moving-average window for histogram (odd).")
    p_pair.add_argument("--xmax", type=int, default=None, help="Optional max abundance for plotting.")
    p_pair.add_argument("--no-pdf", action="store_true", help="Do not write PDFs (PNG only).")

    # aggregate mode
    p_agg = sub.add_parser("aggregate", help="Aggregate many pairs: mean CDF±CI, ΔCDF±CI, ΔPMF + CSV/JSON (PNG only).")
    p_agg.add_argument("--k", required=True, help="k or 'auto'.")
    p_agg.add_argument("--manifest", required=True, help="CSV with columns: id,orig,syn.")
    p_agg.add_argument("--outdir", required=True, help="Output directory for aggregate plots/CSV/JSON.")
    p_agg.add_argument("--tag", default="ALL", help="Tag used in filenames.")
    p_agg.add_argument("--canonical", action="store_true", help="Use canonical k-mers (revcomp collapsed).")
    p_agg.add_argument("--softmask-to-N", action="store_true", help="Treat lowercase as N.")
    p_agg.add_argument("--xmax", type=int, default=None, help="Optional max abundance for plotting/effect sizes.")
    p_agg.add_argument("--rare-max", type=int, default=2, help="Rare bin upper bound (inclusive).")
    p_agg.add_argument("--high-min", type=int, default=200, help="High bin threshold (exclusive).")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "pair":
        os.makedirs(args.outdir, exist_ok=True)
        use_hist = not args.no_hist
        save_pdf = not args.no_pdf

        results: List[PairResult] = []
        metrics_path = os.path.join(args.outdir, f"metrics.k{args.k}.csv")

        if args.manifest:
            rows = load_manifest(args.manifest)
            chosen = select_pairs(rows, args.pairs)
            for i, row in chosen:
                pid = row["id"]
                out_sub = os.path.join(args.outdir, pid)
                print(f"[{i}] k={args.k} :: {pid}")
                res = compare_pair(
                    pair_id=pid,
                    orig_path=row["orig"],
                    syn_path=row["syn"],
                    k_opt=args.k,
                    outdir=out_sub,
                    canonical=args.canonical,
                    softmask_to_N=args.softmask_to_N,
                    use_hist=use_hist,
                    smooth_win=args.smooth_win,
                    xmax=args.xmax,
                    save_pdf=save_pdf,
                )
                results.append(res)
        else:
            if not (args.orig and args.syn):
                raise SystemExit("Provide --orig and --syn OR --manifest.")
            pid = os.path.splitext(os.path.basename(args.orig))[0]
            out_sub = os.path.join(args.outdir, pid)
            print(f"[1] k={args.k} :: {pid}")
            res = compare_pair(
                pair_id=pid,
                orig_path=args.orig,
                syn_path=args.syn,
                k_opt=args.k,
                outdir=out_sub,
                canonical=args.canonical,
                softmask_to_N=args.softmask_to_N,
                use_hist=use_hist,
                smooth_win=args.smooth_win,
                xmax=args.xmax,
                save_pdf=save_pdf,
            )
            results.append(res)

        with open(metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id","k_used","ks_stat","jsd_bits","emd_l1cdf","L_orig_ACGT","L_syn_ACGT"])
            for r in results:
                w.writerow([r.pair_id, r.k_used, f"{r.ks_stat:.6g}", f"{r.jsd_bits:.6g}", f"{r.emd_l1cdf:.6g}", r.L_orig_acgt, r.L_syn_acgt])


        print(f"\nWrote metrics: {metrics_path}\nDone.")
        return

    if args.cmd == "aggregate":
        aggregate_from_manifest(
            manifest_path=args.manifest,
            k_opt=args.k,
            outdir=args.outdir,
            tag=args.tag,
            canonical=args.canonical,
            softmask_to_N=args.softmask_to_N,
            xmax=args.xmax,
            rare_max=args.rare_max,
            high_min=args.high_min,
        )
        print("Done.")
        return


if __name__ == "__main__":
    main()
