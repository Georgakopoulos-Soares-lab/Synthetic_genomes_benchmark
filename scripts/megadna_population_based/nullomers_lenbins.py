#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nullomers_lenbins.py  —  Population-level nullomer fraction comparison

For each genome, KMC counts all distinct k-mers.  The *nullomer fraction*
measures how sparsely the genome covers k-mer space:

  nullomer_fraction = (4^k − distinct_k-mers) / 4^k

A synthetic genome that reproduces natural k-mer diversity should have a
nullomer fraction matching (or close to) its natural counterpart of the same
length.  Because nullomer fraction depends strongly on genome length, we
stratify comparisons into pooled quantile length bins before testing.

Multiple k-mer orders can be evaluated in a single run (--k-min/--k-max or
an explicit --ks list).  Each k yields its own Δ-vs-bin plot, while all k
values contribute rows to the per-genome output.

Statistical design (population-level, no pairing)
-------------------------------------------------
Identical to the other lenbins scripts:
  • Pooled quantile bin edges shared by both groups
  • Optional within-bin downsampling (--balance-within-bin)
  • Δ mean (synthetic − natural) ± 95% bootstrap CI
  • Two-sided Mann–Whitney U test per bin per k
  • Benjamini–Hochberg FDR correction across all (bin × k) combinations

Dependencies
------------
  kmc        https://github.com/refresh-bio/KMC
  kmc_tools  (bundled with KMC)
  numpy, pandas, matplotlib, scipy (optional, for MW test)

Outputs (under --outdir)
------------------------
  metrics/nullomers.per_genome.csv
      group, genome_id, length, k,
      observed_distinct_kmers, kmer_space, nullomer_count, nullomer_fraction

  metrics/nullomers.per_bin.tsv
      per-bin Δ mean + bootstrap CI (lo/hi) + p + q, one row per (k, bin)

  metrics/nullomers.significance_summary.csv
      per-k overall significance (pooling across bins)

  plots/delta_vs_bin.k<K>.{png,pdf}   — one panel per k
  plots/delta_vs_bin.all_k.{png,pdf}  — all k on one figure

  calls/<group>/<genome_id>/k<K>.done  — resume sentinels

Example
-------
python nullomers_lenbins.py \\
    --natural-fasta  natural.fasta \\
    --synthetic-fasta synthetic.fasta \\
    --tag  my_phages \\
    --outdir results/nullomers \\
    --k-min 7 --k-max 9 \\
    --nbins 10 --balance-within-bin \\
    --bootstrap 2000 --threads 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Plot style (matches other lenbins scripts)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# FASTA utilities
# ---------------------------------------------------------------------------
def _open_fasta(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if str(path).endswith(".gz")
        else path.open("rt", encoding="utf-8", errors="replace")
    )


def _clean_id(s: str) -> str:
    tok = s.strip().split()[0] if s.strip() else "NA"
    return "".join(ch if ch.isalnum() or ch in "._-|" else "_" for ch in tok) or "NA"


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (record_id, sequence) for every record in *path*."""
    rid: Optional[str] = None
    buf: List[str] = []
    with _open_fasta(path) as fh:
        for raw in fh:
            if raw.startswith(">"):
                if rid is not None:
                    yield rid, "".join(buf).upper()
                rid = _clean_id(raw[1:])
                buf = []
            else:
                buf.append(raw.strip())
    if rid is not None:
        yield rid, "".join(buf).upper()


def write_single_fasta(path: Path, rid: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f">{rid}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i : i + 60] + "\n")


# ---------------------------------------------------------------------------
# KMC wrappers
# ---------------------------------------------------------------------------
def _require_exe(name: str) -> str:
    p = shutil.which(name)
    if p is None:
        raise SystemExit(
            f"Required executable '{name}' not found on PATH. "
            f"Install KMC (https://github.com/refresh-bio/KMC) and make sure "
            f"both 'kmc' and 'kmc_tools' are available."
        )
    return p


def _kmc_build_db(
    fasta: Path,
    db_prefix: Path,
    tmp_dir: Path,
    k: int,
    threads: int,
) -> None:
    """Build a KMC k-mer database from *fasta* at *db_prefix*."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _require_exe("kmc"),
            f"-k{k}",
            "-fm",       # FASTA multi-record input
            "-ci1",      # count all k-mers appearing ≥ 1 time
            f"-t{threads}",
            str(fasta),
            str(db_prefix),
            str(tmp_dir),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kmc_histogram(db_prefix: Path, hist_path: Path) -> None:
    """Write the KMC database histogram (count-frequency table) to *hist_path*."""
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _require_exe("kmc_tools"),
            "transform",
            str(db_prefix),
            "histogram",
            str(hist_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _parse_histogram(hist_path: Path) -> int:
    """
    Return the number of *distinct* k-mers in a KMC histogram file.

    Format: each line is  <frequency>  <count>
      where <count> = number of distinct k-mers appearing <frequency> times.
    Summing all <count> values gives total distinct k-mers.
    """
    total = 0
    with hist_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                total += int(parts[1])
            except ValueError:
                continue
    return total


def _kmer_space(k: int) -> int:
    """Total number of possible distinct k-mers in a 4-letter alphabet."""
    return 4 ** k


# ---------------------------------------------------------------------------
# Per-genome KMC run (with resume)
# ---------------------------------------------------------------------------
def _run_genome_kmc(
    fasta_path: Path,
    gid: str,
    seq: str,
    group: str,
    calls_dir: Path,
    tmp_dir: Path,
    k: int,
    threads: int,
) -> Dict:
    """
    Run KMC for one genome at one k and return a metrics dict.
    Writes a DONE sentinel JSON file for resume.  If it already exists, reads
    metrics from it and returns without re-running.
    """
    genome_dir = calls_dir / group / gid
    genome_dir.mkdir(parents=True, exist_ok=True)
    done_file = genome_dir / f"k{k}.done"

    if done_file.exists():
        try:
            return json.loads(done_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass  # corrupt sentinel — re-run

    # Write temporary FASTA
    tmp_fa = tmp_dir / group / f"{gid}.k{k}.fa"
    write_single_fasta(tmp_fa, gid, seq)

    db_prefix = genome_dir / f"k{k}.kmc"
    kmc_tmp   = tmp_dir / group / f"{gid}.k{k}.kmc_tmp"
    hist_path = genome_dir / f"k{k}.hist.txt"

    try:
        _kmc_build_db(tmp_fa, db_prefix, kmc_tmp, k=k, threads=threads)
        _kmc_histogram(db_prefix, hist_path)
        distinct = _parse_histogram(hist_path)
        space    = _kmer_space(k)
        nullc    = space - distinct
        nullf    = nullc / space if space > 0 else float("nan")
    finally:
        try:
            tmp_fa.unlink()
        except OSError:
            pass
        try:
            shutil.rmtree(kmc_tmp, ignore_errors=True)
        except OSError:
            pass

    metrics = {
        "group": group,
        "genome_id": gid,
        "length": len(seq),
        "k": k,
        "observed_distinct_kmers": distinct,
        "kmer_space": space,
        "nullomer_count": nullc,
        "nullomer_fraction": nullf,
    }
    done_file.write_text(json.dumps(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Process all genomes × all ks
# ---------------------------------------------------------------------------
def process_all(
    records: List[Tuple[str, str, str]],
    ks: List[int],
    calls_dir: Path,
    tmp_dir: Path,
    threads: int,
) -> pd.DataFrame:
    """
    Run KMC for every (genome, k) combination and return a flat DataFrame.
    Already-computed (group, genome_id, k) entries are loaded from their
    sentinel files without re-running KMC.
    """
    rows: List[Dict] = []
    n_total = len(records) * len(ks)
    done_so_far = 0

    for group, gid, seq in records:
        for k in ks:
            done_so_far += 1
            if done_so_far % max(1, n_total // 20) == 0 or done_so_far == n_total:
                pct = 100 * done_so_far / n_total
                print(f"  KMC: {done_so_far}/{n_total} ({pct:.0f}%)  "
                      f"group={group} genome_id={gid} k={k}")
            rows.append(
                _run_genome_kmc(
                    fasta_path=None,
                    gid=gid,
                    seq=seq,
                    group=group,
                    calls_dir=calls_dir,
                    tmp_dir=tmp_dir,
                    k=k,
                    threads=threads,
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Length binning (shared pattern with other lenbins scripts)
# ---------------------------------------------------------------------------
def _make_bin_edges(lengths: np.ndarray, nbins: int) -> np.ndarray:
    lengths = lengths[np.isfinite(lengths)]
    qs    = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(lengths, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        mn, mx = float(lengths.min()), float(lengths.max())
        edges  = np.array([mn, mx + 1], dtype=float)
    else:
        edges[-1] += 1
    return edges


def _assign_bins(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    df  = df.copy()
    cut = pd.cut(df["length"], bins=edges, include_lowest=True, right=True, duplicates="drop")
    df["len_bin"] = cut.astype(str)
    df["L_lo"]    = [int(b.left)  if not pd.isna(b) else np.nan for b in cut]
    df["L_hi"]    = [int(b.right) if not pd.isna(b) else np.nan for b in cut]
    return df.dropna(subset=["len_bin"])


# ---------------------------------------------------------------------------
# Statistics (shared pattern)
# ---------------------------------------------------------------------------
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    m     = len(pvals)
    order = np.argsort(pvals)
    q     = np.empty(m, dtype=float)
    prev  = 1.0
    for i in range(m - 1, -1, -1):
        prev  = min(prev, pvals[order[i]] * m / (i + 1))
        q[i]  = prev
    out        = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _mw_pvalue(nat: np.ndarray, syn: np.ndarray) -> float:
    if not _HAS_SCIPY or len(nat) == 0 or len(syn) == 0:
        return float("nan")
    try:
        return float(mannwhitneyu(syn, nat, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _bootstrap_delta_mean(
    nat: np.ndarray, syn: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if len(nat) == 0 or len(syn) == 0:
        return float("nan"), float("nan"), float("nan")
    obs     = float(syn.mean() - nat.mean())
    deltas  = (syn[rng.integers(0, len(syn), (n_boot, len(syn)))].mean(axis=1)
               - nat[rng.integers(0, len(nat), (n_boot, len(nat)))].mean(axis=1))
    return obs, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _balance_bin(sub: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    nat = sub[sub["group"] == "natural"]
    syn = sub[sub["group"] == "synthetic"]
    n   = min(len(nat), len(syn))
    if n == 0:
        return sub.iloc[0:0].copy()
    seed = int(rng.integers(0, 2**31))
    return pd.concat([
        nat.sample(n=n, replace=False, random_state=seed),
        syn.sample(n=n, replace=False, random_state=seed + 1),
    ], ignore_index=True)


def compute_per_bin_stats(
    tidy: pd.DataFrame,
    ks: List[int],
    bootstrap: int,
    balance: bool,
    min_per_group: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in ks:
        d_k = tidy[tidy["k"] == k]
        for (bin_lbl, L_lo, L_hi), sub in d_k.groupby(
            ["len_bin", "L_lo", "L_hi"], dropna=True
        ):
            nat_r = sub[sub["group"] == "natural"]
            syn_r = sub[sub["group"] == "synthetic"]
            if len(nat_r) < min_per_group or len(syn_r) < min_per_group:
                continue
            used = _balance_bin(sub, rng) if balance else sub
            nat  = used[used["group"] == "natural"]["nullomer_fraction"].to_numpy(float)
            syn  = used[used["group"] == "synthetic"]["nullomer_fraction"].to_numpy(float)
            dmean, dlo, dhi = _bootstrap_delta_mean(nat, syn, bootstrap, rng)
            p               = _mw_pvalue(nat, syn)
            rows.append({
                "k": k,
                "len_bin": str(bin_lbl), "L_lo": int(L_lo), "L_hi": int(L_hi),
                "n_natural": int(len(nat)), "n_synthetic": int(len(syn)),
                "delta_mean": dmean, "delta_lo": dlo, "delta_hi": dhi, "p": p,
            })

    if not rows:
        return pd.DataFrame()

    out   = pd.DataFrame(rows).sort_values(["k", "L_lo"])
    pvals = out["p"].to_numpy(dtype=float)
    finite = np.isfinite(pvals)
    out["q"] = np.nan
    if finite.sum() > 0:
        out.loc[finite, "q"] = _bh_fdr(pvals[finite])
    return out


def compute_overall_significance(
    tidy: pd.DataFrame,
    ks: List[int],
    bootstrap: int,
    balance: bool,
    min_per_group: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in ks:
        d_k = tidy[tidy["k"] == k]
        pieces = []
        for _, sub in d_k.groupby("len_bin", dropna=True):
            if (sub["group"] == "natural").sum() < min_per_group:
                continue
            if (sub["group"] == "synthetic").sum() < min_per_group:
                continue
            pieces.append(_balance_bin(sub, rng) if balance else sub)
        if not pieces:
            continue
        used  = pd.concat(pieces, ignore_index=True)
        nat   = used[used["group"] == "natural"]["nullomer_fraction"].to_numpy(float)
        syn   = used[used["group"] == "synthetic"]["nullomer_fraction"].to_numpy(float)
        dmean, dlo, dhi = _bootstrap_delta_mean(nat, syn, bootstrap, rng)
        p = _mw_pvalue(nat, syn)
        rows.append({
            "k": k,
            "n_natural": int(len(nat)), "n_synthetic": int(len(syn)),
            "delta_mean": dmean, "delta_lo": dlo, "delta_hi": dhi, "p": p,
        })

    if not rows:
        return pd.DataFrame()

    out   = pd.DataFrame(rows).sort_values("k")
    pvals = out["p"].to_numpy(dtype=float)
    finite = np.isfinite(pvals)
    out["q"] = np.nan
    if finite.sum() > 0:
        out.loc[finite, "q"] = _bh_fdr(pvals[finite])
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_CMAP = plt.get_cmap("tab10")


def _bin_labels(perbin: pd.DataFrame, k: int) -> Tuple[np.ndarray, List[str]]:
    d = perbin[perbin["k"] == k].sort_values("L_lo")
    x    = np.arange(len(d), dtype=float)
    labs = [f"{int(r.L_lo):,}–{int(r.L_hi):,}" for _, r in d.iterrows()]
    return x, labs, d


def plot_delta_vs_bin_per_k(perbin: pd.DataFrame, plots_dir: Path, tag: str) -> None:
    """One delta-vs-bin figure per k-mer order."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(perbin["k"].unique())
    for ki, k in enumerate(ks):
        x, labs, d = _bin_labels(perbin, k)
        if len(x) == 0:
            continue
        y  = d["delta_mean"].to_numpy(float)
        lo = d["delta_lo"].to_numpy(float)
        hi = d["delta_hi"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, y, marker="o", color=_CMAP(ki % 10))
        ax.fill_between(x, lo, hi, alpha=0.20, color=_CMAP(ki % 10))
        ax.axhline(0, linewidth=0.8, color="black", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_xlabel("Length bin (bp)")
        ax.set_ylabel("Δ mean nullomer fraction\n(synthetic − natural)")
        ax.set_title(f"{tag}: Δ nullomer fraction vs length bin (k={k})")
        fig.tight_layout()
        stem = plots_dir / f"delta_vs_bin.k{k}"
        fig.savefig(str(stem) + ".png", dpi=300)
        fig.savefig(str(stem) + ".pdf")
        plt.close(fig)


def plot_delta_vs_bin_all_k(perbin: pd.DataFrame, plots_dir: Path, tag: str) -> None:
    """All k-mer orders on one figure (useful for quick comparison)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(perbin["k"].unique())
    if not ks:
        return

    # Use bin indices from the first k that has data as x-axis; label with ranges
    first_k = ks[0]
    _, labs, _ = _bin_labels(perbin, first_k)
    x_range  = np.arange(len(labs), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    for ki, k in enumerate(ks):
        x, _, d = _bin_labels(perbin, k)
        y  = d["delta_mean"].to_numpy(float)
        lo = d["delta_lo"].to_numpy(float)
        hi = d["delta_hi"].to_numpy(float)
        ax.plot(x, y, marker="o", label=f"k={k}", color=_CMAP(ki % 10))
        ax.fill_between(x, lo, hi, alpha=0.12, color=_CMAP(ki % 10))

    ax.axhline(0, linewidth=0.8, color="black", linestyle="--")
    ax.set_xticks(x_range)
    ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_xlabel("Length bin (bp)")
    ax.set_ylabel("Δ mean nullomer fraction\n(synthetic − natural)")
    ax.set_title(f"{tag}: Δ nullomer fraction vs length bin (all k)")
    ax.legend(frameon=False, ncol=max(1, len(ks) // 3))
    fig.tight_layout()
    stem = plots_dir / "delta_vs_bin.all_k"
    fig.savefig(str(stem) + ".png", dpi=300)
    fig.savefig(str(stem) + ".pdf")
    plt.close(fig)


def plot_boxplots_by_bin(tidy_binned: pd.DataFrame, plots_dir: Path, tag: str) -> None:
    """Side-by-side boxplots (natural vs synthetic) per length bin, one figure per k."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(tidy_binned["k"].unique())

    from matplotlib.patches import Patch

    for k in ks:
        d = tidy_binned[tidy_binned["k"] == k].sort_values(["L_lo", "L_hi"])
        bins = d.drop_duplicates(["len_bin", "L_lo"]).sort_values("L_lo")
        n_bins = len(bins)
        if n_bins == 0:
            continue

        nat_data, syn_data, labels = [], [], []
        for _, br in bins.iterrows():
            sub = d[d["len_bin"] == br["len_bin"]]
            nat_data.append(sub[sub["group"] == "natural"]["nullomer_fraction"].to_numpy(float))
            syn_data.append(sub[sub["group"] == "synthetic"]["nullomer_fraction"].to_numpy(float))
            labels.append(f"{int(br.L_lo):,}–{int(br.L_hi):,}")

        x = np.arange(n_bins, dtype=float)
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(7, n_bins * 1.2), 4))
        ax.boxplot(nat_data, positions=x - w / 2, widths=w,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor="#1f77b4", alpha=0.6),
                   medianprops=dict(color="black"),
                   flierprops=dict(marker=".", markersize=3))
        ax.boxplot(syn_data, positions=x + w / 2, widths=w,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor="#ff7f0e", alpha=0.6),
                   medianprops=dict(color="black"),
                   flierprops=dict(marker=".", markersize=3))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Length bin (bp)")
        ax.set_ylabel("nullomer_fraction")
        ax.set_title(f"{tag}: nullomer fraction by length bin (k={k})")
        ax.legend(
            handles=[Patch(facecolor="#1f77b4", alpha=0.6, label="Natural"),
                     Patch(facecolor="#ff7f0e", alpha=0.6, label="Synthetic")],
            frameon=False,
        )
        fig.tight_layout()
        stem = plots_dir / f"boxplots_by_bin.k{k}"
        fig.savefig(str(stem) + ".png", dpi=300)
        fig.savefig(str(stem) + ".pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Population-level nullomer fraction comparison (natural vs synthetic), "
            "stratified by genome length."
        )
    )
    ap.add_argument("--natural-fasta",   required=True, help="Multi-record FASTA of natural genomes.")
    ap.add_argument("--synthetic-fasta", required=True, help="Multi-record FASTA of synthetic genomes.")
    ap.add_argument("--outdir",  required=True, help="Output directory.")
    ap.add_argument("--tag",     default="dataset", help="Short label for outputs. (default: dataset)")

    kmers = ap.add_argument_group("k-mer settings")
    kmers.add_argument("--k-min",  type=int, default=7,
                       help="Smallest k-mer order to evaluate. (default: 7)")
    kmers.add_argument("--k-max",  type=int, default=9,
                       help="Largest k-mer order to evaluate. (default: 9)")
    kmers.add_argument("--ks",     type=int, nargs="+", default=None,
                       help="Explicit list of k values; overrides --k-min/--k-max.")
    kmers.add_argument("--threads", type=int, default=4,
                       help="CPU threads passed to KMC (-t). (default: 4)")

    stats = ap.add_argument_group("statistical settings")
    stats.add_argument("--nbins",         type=int, default=10,
                       help="Number of pooled quantile length bins. (default: 10)")
    stats.add_argument("--bootstrap",     type=int, default=2000,
                       help="Bootstrap replicates per bin. (default: 2000)")
    stats.add_argument("--balance-within-bin", action="store_true",
                       help="Downsample the larger group to match the smaller one within each bin.")
    stats.add_argument("--min-per-bin",   type=int, default=10,
                       help="Minimum genomes per group to include a bin. (default: 10)")
    stats.add_argument("--seed",          type=int, default=7,
                       help="Random seed. (default: 7)")

    misc = ap.add_argument_group("miscellaneous")
    misc.add_argument("--max-genomes", type=int, default=0,
                      help="If > 0, load only the first N genomes per group. (default: 0 = all)")
    args = ap.parse_args()

    # Validate KMC availability early
    _require_exe("kmc")
    _require_exe("kmc_tools")

    if not _HAS_SCIPY:
        print("[WARN] scipy not available; Mann–Whitney p-values will be NaN. "
              "Install scipy for significance testing.", file=sys.stderr)

    # Determine k values
    ks: List[int] = (
        sorted(set(args.ks))
        if args.ks
        else list(range(args.k_min, args.k_max + 1))
    )
    if not ks:
        raise SystemExit("No k values to evaluate.")

    outdir      = Path(args.outdir).resolve()
    calls_dir   = outdir / "calls"
    metrics_dir = outdir / "metrics"
    plots_dir   = outdir / "plots"
    tmp_dir     = outdir / "tmp"
    for d in (calls_dir, metrics_dir, plots_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load genomes
    # ------------------------------------------------------------------
    print("Loading genomes...")
    records: List[Tuple[str, str, str]] = []  # (group, genome_id, seq)
    for group, fasta_path in [
        ("natural",   Path(args.natural_fasta)),
        ("synthetic", Path(args.synthetic_fasta)),
    ]:
        n = 0
        for gid, seq in iter_fasta_records(fasta_path):
            records.append((group, gid, seq))
            n += 1
            if args.max_genomes and n >= args.max_genomes:
                break
        print(f"  {group}: {n} genomes from {fasta_path.name}")

    if not records:
        raise SystemExit("No sequences loaded from either FASTA file.")

    print(f"  k values: {ks}")

    # ------------------------------------------------------------------
    # Run KMC for each genome × k
    # ------------------------------------------------------------------
    print("\nRunning KMC...")
    tidy = process_all(records, ks, calls_dir, tmp_dir, threads=args.threads)

    tidy["group"]  = tidy["group"].str.lower()
    tidy["length"] = tidy["length"].astype(int)
    tidy["nullomer_fraction"] = pd.to_numeric(
        tidy["nullomer_fraction"], errors="coerce"
    )

    per_genome_csv = metrics_dir / "nullomers.per_genome.csv"
    tidy.to_csv(per_genome_csv, index=False)
    print(f"\n  Per-genome metrics: {per_genome_csv}")

    # ------------------------------------------------------------------
    # Length binning (one set of edges, shared across all ks)
    # ------------------------------------------------------------------
    # Use one length value per genome (deduplicate across k rows)
    pooled = (
        tidy.drop_duplicates(subset=["group", "genome_id"])["length"]
        .to_numpy(dtype=float)
    )
    edges  = _make_bin_edges(pooled, args.nbins)
    tidy_b = _assign_bins(tidy, edges)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print("Computing per-bin statistics...")
    perbin = compute_per_bin_stats(
        tidy_b, ks=ks,
        bootstrap=args.bootstrap,
        balance=args.balance_within_bin,
        min_per_group=args.min_per_bin,
        rng=rng,
    )
    if not perbin.empty:
        perbin.to_csv(metrics_dir / "nullomers.per_bin.tsv", sep="\t", index=False)
        print(f"  Per-bin summary: {metrics_dir / 'nullomers.per_bin.tsv'}")
    else:
        print("  [WARN] No bins met the minimum-per-group threshold; per-bin TSV not written.")

    overall = compute_overall_significance(
        tidy_b, ks=ks,
        bootstrap=args.bootstrap,
        balance=args.balance_within_bin,
        min_per_group=args.min_per_bin,
        rng=rng,
    )
    if not overall.empty:
        overall.to_csv(metrics_dir / "nullomers.significance_summary.csv", index=False)
        print(f"  Significance summary: {metrics_dir / 'nullomers.significance_summary.csv'}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not perbin.empty:
        print("Generating plots...")
        plot_delta_vs_bin_per_k(perbin, plots_dir, args.tag)
        plot_delta_vs_bin_all_k(perbin, plots_dir, args.tag)
        plot_boxplots_by_bin(tidy_b, plots_dir, args.tag)
        print(f"  Plots: {plots_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
