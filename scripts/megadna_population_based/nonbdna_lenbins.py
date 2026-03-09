#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonbdna_lenbins.py  —  Population-level non-B DNA motif comparison

Compares five classes of non-B DNA structural motifs between natural and
synthetic genome sets, stratified by genome-length bins.

Motif classes
-------------
  GQ    — G-quadruplexes, detected with G4Hunter
  Z-DNA — Z-DNA-forming regions, detected with ZSeeker
  DR    — Direct repeats     ⎫
  IR    — Inverted repeats   ⎬  detected with Non-B GFA (nonbgfa)
  MR    — Mirror repeats     ⎟
  STR   — Short tandem repeats ⎭

All three external tools must be installed and available on PATH:
  g4hunter   https://github.com/AnimaTardeb/G4Hunter
  ZSeeker    https://github.com/BlanchetteLab/ZSeeker
  <gfa-bin>  Non-B GFA binary (supply path with --gfa-bin)

Statistical design (population-level, no pairing)
-------------------------------------------------
Genomes are not paired. Natural and synthetic sets may differ in size and span
a wide range of lengths. Bin edges are derived from the *pooled* length
distribution so both groups share the same boundaries (see kmer_spectra.py for
a detailed rationale).

Within each bin the larger group is randomly downsampled to match the smaller
one (--balance-within-bin, recommended). Effect sizes (Δ mean bp/kb or
hits/kb) are estimated with nonparametric bootstrap (genome-level resampling,
95% CI). Statistical significance per bin is assessed with two-sided
Mann–Whitney U tests. p-values are then adjusted across all (bin × motif)
combinations using the Benjamini–Hochberg FDR procedure.

Two normalised metrics are reported for each motif per genome:
  bp_per_kb   — base-pairs covered by motif hits per kilobase of genome
  hits_per_kb — number of distinct hits per kilobase of genome

Outputs (under --outdir)
------------------------
  metrics/per_genome.nonbdna.csv      per-genome raw metrics
  metrics/per_bin.nonbdna.tsv         per-bin Δ mean + CI + p + q
  metrics/significance_summary.csv    motif-level overall significance
  plots/delta_vs_bin.<motif>.<measure>.{png,pdf}
  plots/boxplots_by_bin.<motif>.<measure>.{png,pdf}
  calls/g4hunter/<group>/<genome_id>/  raw g4hunter outputs  (resumable)
  calls/zseeker/<group>/<genome_id>/   raw ZSeeker outputs   (resumable)
  calls/nonbgfa/<group>/<genome_id>/   raw nonbgfa outputs   (resumable)

Example
-------
python nonbdna_lenbins.py \\
    --natural-fasta  natural.fasta \\
    --synthetic-fasta synthetic.fasta \\
    --tag  my_phages \\
    --outdir results/nonbdna \\
    --gfa-bin /path/to/gfa \\
    --nbins 10 --balance-within-bin \\
    --bootstrap 2000 --seed 7

Resume support
--------------
Each genome's calling step is skipped if a DONE sentinel file already exists
under calls/<tool>/<group>/<genome_id>/. Re-run the script freely after an
interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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
# Constants
# ---------------------------------------------------------------------------
MOTIF_ORDER = ["GQ", "Z-DNA", "DR", "IR", "MR", "STR"]
_GFA_MOTIFS = ["DR", "IR", "MR", "STR"]

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
            if not raw:
                continue
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


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# G4Hunter
# ---------------------------------------------------------------------------
def _run_g4hunter(fasta: Path, work_dir: Path) -> Path:
    """
    Run g4hunter on *fasta*, return path to the output TSV.
    Raises FileNotFoundError if the expected output is absent.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g4hunter", str(fasta), "--outdir", str(work_dir)], check=True)
    cands = sorted(work_dir.glob("*_pG4s.g4_hunter.tsv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"G4Hunter produced no TSV in {work_dir}")
    return cands[-1]


def _harmonize_g4hunter(tsv: Path, gid: str, group: str, genome_len: int) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if not {"seqID", "start", "end"}.issubset(df.columns):
        raise ValueError(f"G4Hunter TSV missing required columns in {tsv}")
    out = pd.DataFrame({
        "group": group, "genome_id": gid, "length": genome_len, "motif": "GQ",
        "start": pd.to_numeric(df["start"], errors="coerce"),
        "end":   pd.to_numeric(df["end"],   errors="coerce"),
    }).dropna(subset=["start", "end"])
    out["hit_len"] = (out["end"] - out["start"]).clip(lower=0).astype(int)
    return out


def process_g4hunter(
    records: List[Tuple[str, str, str]],
    calls_dir: Path,
    tmp_dir: Path,
    save_hits: bool,
) -> pd.DataFrame:
    """
    Run G4Hunter on every genome (natural + synthetic) in *records*.
    Skips genomes whose DONE sentinel file already exists.
    Returns a per-genome metrics DataFrame (one row per genome).
    """
    metrics_rows = []
    hits_all: List[pd.DataFrame] = []

    for group, gid, seq in records:
        work_dir = calls_dir / group / gid
        done     = work_dir / "DONE.tsv"
        fa       = tmp_dir / group / f"{gid}.{_md5(gid + group)[:8]}.fa"

        if done.exists() and done.stat().st_size > 0:
            tsv = done
        else:
            write_single_fasta(fa, gid, seq)
            raw_tsv = _run_g4hunter(fa, work_dir)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_bytes(raw_tsv.read_bytes())
            tsv = done
            try:
                fa.unlink()
            except OSError:
                pass

        h = _harmonize_g4hunter(tsv, gid, group, len(seq))
        if save_hits:
            hits_all.append(h)

        metrics_rows.append({
            "group": group, "genome_id": gid, "length": len(seq),
            "motif": "GQ",
            "n_hits":    int(len(h)),
            "bp_covered": int(h["hit_len"].sum()) if len(h) else 0,
        })

    print(f"  G4Hunter: processed {len(metrics_rows)} genomes")
    return pd.DataFrame(metrics_rows), pd.concat(hits_all, ignore_index=True) if hits_all else pd.DataFrame()


# ---------------------------------------------------------------------------
# ZSeeker
# ---------------------------------------------------------------------------
def _read_zseeker_table(path: Path) -> pd.DataFrame:
    with path.open("r", errors="ignore") as fh:
        first = fh.readline()
    sep = "," if first.count(",") > first.count("\t") else "\t"
    return pd.read_csv(path, sep=sep)


def _run_zseeker(fasta: Path, work_dir: Path, n_jobs: int) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "zdna_extractions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ZSeeker", "--fasta", str(fasta), "--n_jobs", str(n_jobs)],
                   cwd=str(work_dir), check=True)
    cands = sorted(out_dir.glob("*zdna_score.*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"ZSeeker produced no output under {out_dir}")
    return cands[-1]


def _harmonize_zseeker(tab: Path, gid: str, group: str, genome_len: int) -> pd.DataFrame:
    df = _read_zseeker_table(tab).dropna(subset=["Start", "End"])
    out = pd.DataFrame({
        "group": group, "genome_id": gid, "length": genome_len, "motif": "Z-DNA",
        "start": pd.to_numeric(df["Start"], errors="coerce"),
        "end":   pd.to_numeric(df["End"],   errors="coerce"),
    }).dropna(subset=["start", "end"])
    out["hit_len"] = (out["end"] - out["start"]).clip(lower=0).astype(int)
    return out


def process_zseeker(
    records: List[Tuple[str, str, str]],
    calls_dir: Path,
    tmp_dir: Path,
    n_jobs: int,
    save_hits: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    hits_all: List[pd.DataFrame] = []

    for group, gid, seq in records:
        work_dir = calls_dir / group / gid
        done     = work_dir / "DONE.tab"
        fa       = tmp_dir / group / f"{gid}.{_md5(gid + group)[:8]}.fa"

        if done.exists() and done.stat().st_size > 0:
            tab = done
        else:
            write_single_fasta(fa, gid, seq)
            raw_tab = _run_zseeker(fa, work_dir, n_jobs)
            done.parent.mkdir(parents=True, exist_ok=True)
            try:
                done.write_bytes(raw_tab.read_bytes())
            except Exception:
                done.write_text(raw_tab.read_text(errors="replace"))
            tab = done
            try:
                fa.unlink()
            except OSError:
                pass

        h = _harmonize_zseeker(tab, gid, group, len(seq))
        if save_hits:
            hits_all.append(h)

        metrics_rows.append({
            "group": group, "genome_id": gid, "length": len(seq),
            "motif": "Z-DNA",
            "n_hits":     int(len(h)),
            "bp_covered": int(h["hit_len"].sum()) if len(h) else 0,
        })

    print(f"  ZSeeker: processed {len(metrics_rows)} genomes")
    return pd.DataFrame(metrics_rows), pd.concat(hits_all, ignore_index=True) if hits_all else pd.DataFrame()


# ---------------------------------------------------------------------------
# Non-B GFA  (DR / IR / MR / STR)
# ---------------------------------------------------------------------------
def _run_gfa(gfa_bin: Path, fasta: Path, out_prefix: str, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(gfa_bin), "-skipWGET", "-skipAPR", "-skipZ",
           "-seq", str(fasta), "-out", out_prefix]
    # GFA can be noisy; do not hard-fail on non-zero exit
    subprocess.run(cmd, cwd=str(work_dir), check=False)


def _infer_motif(path: Path) -> Optional[str]:
    name = path.name.upper()
    for m in _GFA_MOTIFS:
        if re.search(rf"(?:^|[_.]){m}(?:[_.]|$)", name):
            return m
    return None


def _read_interval_table(path: Path) -> pd.DataFrame:
    """
    Read a GFA output table and return a DataFrame with columns
    chrom, start, end  (best-effort, supports GFF, TSV, BED, plain text).
    """
    suf = path.suffix.lower()

    # GFF3-style: columns 0=chrom, 3=start, 4=end (1-based)
    if suf == ".gff":
        try:
            df = pd.read_csv(path, sep="\t", comment="#", header=None, engine="python")
            if df.shape[1] >= 5:
                return pd.DataFrame({
                    "chrom": df.iloc[:, 0].astype(str),
                    "start": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
                    "end":   pd.to_numeric(df.iloc[:, 4], errors="coerce"),
                }).dropna(subset=["start", "end"])
        except Exception:
            pass

    # TSV with header
    if suf in (".tsv", ".bed", ".txt", ".out", ".dat", ".tab"):
        try:
            df = pd.read_csv(path, sep="\t", comment="#", header=0, engine="python")
            lc = {c.lower(): c for c in df.columns}
            def _pick(*names):
                for n in names:
                    if n in df.columns:
                        return n
                    if n.lower() in lc:
                        return lc[n.lower()]
                return None
            c_start = _pick("start", "Start", "begin")
            c_end   = _pick("end",   "End",   "stop")
            c_chrom = _pick("chrom", "Chrom", "chr", "sequence", "seqid", "contig")
            if c_start and c_end:
                return pd.DataFrame({
                    "chrom": df[c_chrom].astype(str) if c_chrom else "NA",
                    "start": pd.to_numeric(df[c_start], errors="coerce"),
                    "end":   pd.to_numeric(df[c_end],   errors="coerce"),
                }).dropna(subset=["start", "end"])
        except Exception:
            pass

    # Last-resort: headerless whitespace/tab, first 3 columns
    for sep in ["\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python")
            if df.shape[1] >= 3:
                return pd.DataFrame({
                    "chrom": df.iloc[:, 0].astype(str),
                    "start": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                    "end":   pd.to_numeric(df.iloc[:, 2], errors="coerce"),
                }).dropna(subset=["start", "end"])
        except Exception:
            continue

    return pd.DataFrame(columns=["chrom", "start", "end"])


def _parse_gfa_outputs(work_dir: Path) -> pd.DataFrame:
    """
    Walk *work_dir* and return per-hit rows for all motif files found.
    """
    rows: List[dict] = []
    valid_suffixes = {".tsv", ".csv", ".txt", ".bed", ".out", ".dat", ".gff", ".tab"}

    for f in sorted(work_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in valid_suffixes:
            continue
        motif = _infer_motif(f)
        if motif is None:
            continue
        ivs = _read_interval_table(f)
        if ivs.empty:
            continue
        for _, row in ivs.iterrows():
            length = max(0, int(row["end"]) - int(row["start"]))
            rows.append({"motif": motif, "hit_len": length})

    if not rows:
        return pd.DataFrame(columns=["motif", "hit_len"])
    return pd.DataFrame(rows)


def process_nonbgfa(
    records: List[Tuple[str, str, str]],
    calls_dir: Path,
    tmp_dir: Path,
    gfa_bin: Path,
    save_hits: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    hits_all: List[pd.DataFrame] = []

    for group, gid, seq in records:
        work_dir = calls_dir / group / gid
        done     = work_dir / "DONE.flag"
        fa       = tmp_dir / group / f"{gid}.{_md5(gid + group)[:8]}.fa"

        if not done.exists():
            write_single_fasta(fa, gid, seq)
            _run_gfa(gfa_bin, fa, out_prefix=gid, work_dir=work_dir)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text("ok\n")
            try:
                fa.unlink()
            except OSError:
                pass

        hits = _parse_gfa_outputs(work_dir)
        if save_hits and not hits.empty:
            hits["group"]     = group
            hits["genome_id"] = gid
            hits_all.append(hits)

        for motif in _GFA_MOTIFS:
            sub    = hits[hits["motif"] == motif] if not hits.empty else hits.iloc[0:0]
            n_hits = int(len(sub))
            bp_cov = int(sub["hit_len"].sum()) if n_hits else 0
            metrics_rows.append({
                "group": group, "genome_id": gid, "length": len(seq),
                "motif": motif, "n_hits": n_hits, "bp_covered": bp_cov,
            })

    print(f"  Non-B GFA: processed {len(records)} genomes ({len(_GFA_MOTIFS)} motifs each)")
    return (
        pd.DataFrame(metrics_rows),
        pd.concat(hits_all, ignore_index=True) if hits_all else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Length binning
# ---------------------------------------------------------------------------
def _make_bin_edges(lengths: np.ndarray, nbins: int) -> np.ndarray:
    """
    Quantile edges from the pooled length distribution.
    Returns an array of shape (nbins+1,) with finite, non-decreasing values.
    The last edge is extended by 1 to ensure the maximum genome is included.
    """
    lengths = lengths[np.isfinite(lengths)]
    qs    = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(lengths, qs)
    edges = np.unique(edges)
    # Guarantee we have at least 2 edges
    if len(edges) < 2:
        mn, mx = float(lengths.min()), float(lengths.max())
        edges = np.array([mn, mx + 1], dtype=float)
    else:
        edges[-1] += 1  # make last bin inclusive of maximum
    return edges


def _assign_bins(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    cut = pd.cut(df["length"], bins=edges, include_lowest=True, right=True, duplicates="drop")
    df["len_bin"] = cut.astype(str)
    df["L_lo"] = [int(b.left)  if not pd.isna(b) else np.nan for b in cut]
    df["L_hi"] = [int(b.right) if not pd.isna(b) else np.nan for b in cut]
    return df.dropna(subset=["len_bin"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction."""
    m    = len(pvals)
    order = np.argsort(pvals)
    q     = np.empty(m, dtype=float)
    prev  = 1.0
    for i in range(m - 1, -1, -1):
        prev    = min(prev, pvals[order[i]] * m / (i + 1))
        q[i]   = prev
    out          = np.empty(m, dtype=float)
    out[order]   = np.clip(q, 0.0, 1.0)
    return out


def _mw_pvalue(nat: np.ndarray, syn: np.ndarray) -> float:
    if not _HAS_SCIPY or len(nat) == 0 or len(syn) == 0:
        return float("nan")
    try:
        return float(mannwhitneyu(syn, nat, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _permutation_pvalue(nat: np.ndarray, syn: np.ndarray, n_perm: int,
                        rng: np.random.Generator) -> float:
    """
    Two-sided permutation test for the difference of means.
    Labels are shuffled across the pooled sample.
    """
    pooled = np.concatenate([nat, syn])
    n_nat  = len(nat)
    obs    = abs(syn.mean() - nat.mean())
    count  = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(pooled[n_nat:].mean() - pooled[:n_nat].mean())
        if d >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def _bootstrap_delta_mean(
    nat: np.ndarray, syn: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Return (obs_delta, ci_lo, ci_hi) for mean(syn) − mean(nat)."""
    if len(nat) == 0 or len(syn) == 0:
        return float("nan"), float("nan"), float("nan")
    obs    = float(syn.mean() - nat.mean())
    idx_n  = rng.integers(0, len(nat), size=(n_boot, len(nat)))
    idx_s  = rng.integers(0, len(syn), size=(n_boot, len(syn)))
    deltas = nat[idx_n].mean(axis=1)          # shape (n_boot,)
    deltas = syn[idx_s].mean(axis=1) - nat[idx_n].mean(axis=1)
    lo     = float(np.percentile(deltas,  2.5))
    hi     = float(np.percentile(deltas, 97.5))
    return obs, lo, hi


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
    measure: str,
    nbins: int,
    bootstrap: int,
    balance: bool,
    min_per_group: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute per-bin effect sizes, CIs, and p-values for one *measure*.
    Returns a tidy DataFrame with BH-FDR q-values applied across all rows.
    """
    rows: List[dict] = []
    for motif in MOTIF_ORDER:
        d_m = tidy[tidy["motif"] == motif]
        if d_m.empty:
            continue
        for (bin_lbl, L_lo, L_hi), sub in d_m.groupby(["len_bin", "L_lo", "L_hi"], dropna=True):
            nat_r = sub[sub["group"] == "natural"]
            syn_r = sub[sub["group"] == "synthetic"]
            if len(nat_r) < min_per_group or len(syn_r) < min_per_group:
                continue
            used = _balance_bin(sub, rng) if balance else sub
            nat  = used[used["group"] == "natural"][measure].to_numpy(dtype=float)
            syn  = used[used["group"] == "synthetic"][measure].to_numpy(dtype=float)
            dmean, dlo, dhi = _bootstrap_delta_mean(nat, syn, bootstrap, rng)
            p               = _mw_pvalue(nat, syn)
            rows.append({
                "motif": motif, "measure": measure,
                "len_bin": str(bin_lbl), "L_lo": int(L_lo), "L_hi": int(L_hi),
                "n_natural": int(len(nat)), "n_synthetic": int(len(syn)),
                "delta_mean": dmean, "delta_lo": dlo, "delta_hi": dhi, "p": p,
            })

    if not rows:
        return pd.DataFrame()

    out   = pd.DataFrame(rows).sort_values(["motif", "L_lo"])
    pvals = out["p"].to_numpy(dtype=float)
    finite = np.isfinite(pvals)
    out["q"] = np.nan
    if finite.sum() > 0:
        out.loc[finite, "q"] = _bh_fdr(pvals[finite])
    return out


def compute_overall_significance(
    tidy: pd.DataFrame,
    measure: str,
    balance: bool,
    min_per_group: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute motif-level overall significance by pooling balanced bin samples.
    """
    rows: List[dict] = []
    for motif in MOTIF_ORDER:
        d_m   = tidy[tidy["motif"] == motif]
        pieces = []
        for _, sub in d_m.groupby("len_bin", dropna=True):
            if (sub["group"] == "natural").sum() < min_per_group:
                continue
            if (sub["group"] == "synthetic").sum() < min_per_group:
                continue
            pieces.append(_balance_bin(sub, rng) if balance else sub)
        if not pieces:
            continue
        used  = pd.concat(pieces, ignore_index=True)
        nat   = used[used["group"] == "natural"][measure].to_numpy(dtype=float)
        syn   = used[used["group"] == "synthetic"][measure].to_numpy(dtype=float)
        dmean, dlo, dhi = _bootstrap_delta_mean(nat, syn, bootstrap, rng)
        p = _mw_pvalue(nat, syn)
        rows.append({
            "motif": motif, "measure": measure,
            "n_natural": int(len(nat)), "n_synthetic": int(len(syn)),
            "delta_mean": dmean, "delta_lo": dlo, "delta_hi": dhi, "p": p,
        })
    if not rows:
        return pd.DataFrame()
    out   = pd.DataFrame(rows)
    pvals = out["p"].to_numpy(dtype=float)
    finite = np.isfinite(pvals)
    out["q"] = np.nan
    if finite.sum() > 0:
        out.loc[finite, "q"] = _bh_fdr(pvals[finite])
    return out.sort_values(["measure", "motif"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_delta_vs_bin(
    perbin: pd.DataFrame, plots_dir: Path, measure: str
) -> None:
    """Line plot of Δ mean ± 95% CI vs length bin, one figure per motif."""
    _ensure(plots_dir)
    df = perbin[perbin["measure"] == measure]
    if df.empty:
        return

    for motif in MOTIF_ORDER:
        d = df[df["motif"] == motif].sort_values("L_lo")
        if d.empty:
            continue

        x    = np.arange(len(d), dtype=float)
        y    = d["delta_mean"].to_numpy(float)
        lo   = d["delta_lo"].to_numpy(float)
        hi   = d["delta_hi"].to_numpy(float)
        labs = [f"{int(r.L_lo):,}–{int(r.L_hi):,}" for _, r in d.iterrows()]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, y, marker="o")
        ax.fill_between(x, lo, hi, alpha=0.20)
        ax.axhline(0, linewidth=0.8, color="black", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_xlabel("Length bin (bp)")
        ax.set_ylabel(f"Δ mean (synthetic − natural)\n{measure}")
        ax.set_title(f"{motif}: Δ {measure} vs length bin")
        fig.tight_layout()
        stem = plots_dir / f"delta_vs_bin.{motif}.{measure}"
        fig.savefig(str(stem) + ".png", dpi=300)
        fig.savefig(str(stem) + ".pdf")
        plt.close(fig)


def plot_boxplots_by_bin(
    tidy_binned: pd.DataFrame, plots_dir: Path, measure: str
) -> None:
    """
    Side-by-side boxplots (natural vs synthetic) for each length bin,
    one figure per motif.
    """
    _ensure(plots_dir)
    for motif in MOTIF_ORDER:
        d = tidy_binned[tidy_binned["motif"] == motif].sort_values(["L_lo", "L_hi"])
        if d.empty:
            continue

        bins  = d.drop_duplicates(["len_bin", "L_lo"]).sort_values("L_lo")
        n_bins = len(bins)
        if n_bins == 0:
            continue

        nat_data, syn_data, labels = [], [], []
        for _, br in bins.iterrows():
            sub = d[d["len_bin"] == br["len_bin"]]
            nat_data.append(sub[sub["group"] == "natural"][measure].to_numpy(float))
            syn_data.append(sub[sub["group"] == "synthetic"][measure].to_numpy(float))
            labels.append(f"{int(br.L_lo):,}–{int(br.L_hi):,}")

        x     = np.arange(n_bins, dtype=float)
        w     = 0.35
        fig, ax = plt.subplots(figsize=(max(7, n_bins * 1.2), 4))

        bp_nat = ax.boxplot(
            nat_data, positions=x - w / 2, widths=w,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor="#1f77b4", alpha=0.6),
            medianprops=dict(color="black"),
            flierprops=dict(marker=".", markersize=3),
        )
        bp_syn = ax.boxplot(
            syn_data, positions=x + w / 2, widths=w,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor="#ff7f0e", alpha=0.6),
            medianprops=dict(color="black"),
            flierprops=dict(marker=".", markersize=3),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Length bin (bp)")
        ax.set_ylabel(measure)
        ax.set_title(f"{motif}: {measure} by length bin")

        # legend proxy
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor="#1f77b4", alpha=0.6, label="Natural"),
                     Patch(facecolor="#ff7f0e", alpha=0.6, label="Synthetic")],
            frameon=False,
        )
        fig.tight_layout()
        stem = plots_dir / f"boxplots_by_bin.{motif}.{measure}"
        fig.savefig(str(stem) + ".png", dpi=300)
        fig.savefig(str(stem) + ".pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Population-level non-B DNA motif comparison (natural vs synthetic), stratified by genome length."
    )
    ap.add_argument("--natural-fasta",   required=True, help="Multi-record FASTA of natural genomes.")
    ap.add_argument("--synthetic-fasta", required=True, help="Multi-record FASTA of synthetic genomes.")
    ap.add_argument("--outdir",  required=True, help="Output directory.")
    ap.add_argument("--tag",     default="dataset", help="Short label for outputs. (default: dataset)")
    ap.add_argument("--gfa-bin", required=True,
                    help="Path to the Non-B GFA binary (required for DR/IR/MR/STR).")

    tools = ap.add_argument_group("tool selection")
    tools.add_argument("--skip-g4hunter", action="store_true", help="Skip G4Hunter (GQ).")
    tools.add_argument("--skip-zseeker",  action="store_true", help="Skip ZSeeker (Z-DNA).")
    tools.add_argument("--skip-nonbgfa",  action="store_true", help="Skip Non-B GFA (DR/IR/MR/STR).")

    stats = ap.add_argument_group("statistical settings")
    stats.add_argument("--nbins",         type=int, default=10,   help="Number of quantile length bins. (default: 10)")
    stats.add_argument("--bootstrap",     type=int, default=2000, help="Bootstrap replicates per bin. (default: 2000)")
    stats.add_argument("--balance-within-bin", action="store_true",
                       help="Downsample the larger group to match the smaller one within each bin.")
    stats.add_argument("--min-per-bin",   type=int, default=10,
                       help="Minimum genomes per group to analyse a bin. (default: 10)")
    stats.add_argument("--seed",          type=int, default=7,    help="Random seed. (default: 7)")

    misc = ap.add_argument_group("miscellaneous")
    misc.add_argument("--n-jobs",       type=int, default=1,  help="Parallel jobs for ZSeeker. (default: 1)")
    misc.add_argument("--max-genomes",  type=int, default=0,
                      help="If > 0, load only the first N genomes per group. (default: 0 = all)")
    misc.add_argument("--save-hits",    action="store_true",
                      help="Write per-hit parquet files for downstream inspection.")
    args = ap.parse_args()

    gfa_bin = Path(args.gfa_bin).resolve()
    if not args.skip_nonbgfa and not gfa_bin.exists():
        raise SystemExit(f"Non-B GFA binary not found: {gfa_bin}")
    if not _HAS_SCIPY:
        print("[WARN] scipy not available; Mann–Whitney p-values will be NaN. "
              "Install scipy for significance testing.", file=sys.stderr)

    outdir = Path(args.outdir).resolve()
    rng    = np.random.default_rng(args.seed)

    calls_dir   = outdir / "calls"
    metrics_dir = outdir / "metrics"
    plots_dir   = outdir / "plots"
    tmp_dir     = outdir / "tmp"
    for d in (calls_dir, metrics_dir, plots_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load genome sequences
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

    # ------------------------------------------------------------------
    # Run tools
    # ------------------------------------------------------------------
    metrics_parts: List[pd.DataFrame] = []

    if not args.skip_g4hunter:
        print("\nRunning G4Hunter...")
        m, _ = process_g4hunter(records, calls_dir / "g4hunter", tmp_dir / "g4hunter",
                                 save_hits=args.save_hits)
        m.to_csv(metrics_dir / "g4hunter.per_genome.csv", index=False)
        metrics_parts.append(m)

    if not args.skip_zseeker:
        print("\nRunning ZSeeker...")
        m, _ = process_zseeker(records, calls_dir / "zseeker", tmp_dir / "zseeker",
                                n_jobs=args.n_jobs, save_hits=args.save_hits)
        m.to_csv(metrics_dir / "zseeker.per_genome.csv", index=False)
        metrics_parts.append(m)

    if not args.skip_nonbgfa:
        print("\nRunning Non-B GFA...")
        m, _ = process_nonbgfa(records, calls_dir / "nonbgfa", tmp_dir / "nonbgfa",
                                gfa_bin=gfa_bin, save_hits=args.save_hits)
        m.to_csv(metrics_dir / "nonbgfa.per_genome.csv", index=False)
        metrics_parts.append(m)

    if not metrics_parts:
        raise SystemExit("All tools were skipped — nothing to analyse.")

    # ------------------------------------------------------------------
    # Aggregate per-genome metrics
    # ------------------------------------------------------------------
    tidy = pd.concat(metrics_parts, ignore_index=True)
    tidy["group"]  = tidy["group"].str.lower().map(
        lambda x: "synthetic" if x.startswith("syn") else "natural"
    )
    tidy["length"] = tidy["length"].astype(int)
    tidy["n_hits"] = pd.to_numeric(tidy["n_hits"],     errors="coerce").fillna(0)
    tidy["bp_covered"] = pd.to_numeric(tidy["bp_covered"], errors="coerce").fillna(0)
    tidy["bp_per_kb"]   = np.where(tidy["length"] > 0, 1000.0 * tidy["bp_covered"] / tidy["length"], 0.0)
    tidy["hits_per_kb"] = np.where(tidy["length"] > 0, 1000.0 * tidy["n_hits"]     / tidy["length"], 0.0)
    tidy = tidy[tidy["motif"].isin(MOTIF_ORDER)].copy()

    tidy.to_csv(metrics_dir / "per_genome.nonbdna.csv", index=False)
    print(f"\n  Per-genome metrics: {metrics_dir / 'per_genome.nonbdna.csv'}")

    # ------------------------------------------------------------------
    # Length binning
    # ------------------------------------------------------------------
    pooled = tidy.drop_duplicates(subset=["group", "genome_id"])["length"].to_numpy(dtype=float)
    edges  = _make_bin_edges(pooled, args.nbins)
    tidy_b = _assign_bins(tidy, edges)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print("Computing per-bin statistics...")
    stat_parts = []
    for measure in ("bp_per_kb", "hits_per_kb"):
        stat_parts.append(
            compute_per_bin_stats(
                tidy_b, measure=measure,
                nbins=args.nbins, bootstrap=args.bootstrap,
                balance=args.balance_within_bin,
                min_per_group=args.min_per_bin,
                rng=rng,
            )
        )
    perbin = pd.concat(stat_parts, ignore_index=True)
    perbin.to_csv(metrics_dir / "per_bin.nonbdna.tsv", sep="\t", index=False)
    print(f"  Per-bin summary: {metrics_dir / 'per_bin.nonbdna.tsv'}")

    overall_parts = []
    for measure in ("bp_per_kb", "hits_per_kb"):
        overall_parts.append(
            compute_overall_significance(
                tidy_b, measure=measure,
                balance=args.balance_within_bin,
                min_per_group=args.min_per_bin,
                bootstrap=args.bootstrap,
                rng=rng,
            )
        )
    overall = pd.concat(overall_parts, ignore_index=True)
    overall.to_csv(metrics_dir / "significance_summary.csv", index=False)
    print(f"  Significance summary: {metrics_dir / 'significance_summary.csv'}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("Generating plots...")
    for measure in ("bp_per_kb", "hits_per_kb"):
        plot_delta_vs_bin(perbin, plots_dir, measure)
        plot_boxplots_by_bin(tidy_b, plots_dir, measure)
    print(f"  Plots: {plots_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
