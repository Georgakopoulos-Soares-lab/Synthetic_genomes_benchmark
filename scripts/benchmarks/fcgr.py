#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fcgr.py

Frequency Chaos Game Representation (FCGR) benchmark: Original vs Synthetic.

This script reads a manifest CSV containing paired FASTA paths (orig/syn),
computes per-pair FCGR matrices at order k, compares normalized FCGRs using
L1 distance, and produces paper-friendly summary plots and statistics.

FCGR definition
---------------
- Order k => matrix size is 2^k x 2^k (e.g., k=8 -> 256x256).
- Mapping uses a standard 2-bit corner encoding:
    A -> (0,0), C -> (0,1), G -> (1,0), T -> (1,1)
  For each k-mer, x is formed from the first bits, y from the second bits.
- k-mers spanning non-ACGT characters are skipped (Ns reduce total_kmers).

Outputs (written under --outdir)
--------------------------------
metrics/
  fcgr_pair_distances.k<K>.csv      # per-pair L1 distances + lengths + status
  fcgr_summary.k<K>.json            # dataset-level stats (median, CI, p-values, etc.)

plots/
  fcgr_mean_tripanel.k<K>.png       # mean orig, mean syn (shared scale), log2-ratio diff
  fcgr_distance_hist.k<K>.png       # histogram of per-pair distances

Manifest format
---------------
CSV with columns at least: id, orig, syn
- id: any string label
- orig/syn: paths to FASTA (optionally .gz)

Examples
--------
Run from a manifest:
  python fcgr.py --manifest pairs.Human.csv --outdir results/Human/fcgr --k 8

"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Plot style (paper-friendly)
# -------------------------

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 12,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -------------------------
# FASTA reading
# -------------------------

def read_fasta_concat(path: Path) -> str:
    """
    Concatenate all FASTA sequence lines across records (uppercase).
    We intentionally ignore headers and do not preserve contig boundaries
    because FCGR is k-mer counting-based and we skip any k-mers spanning non-ACGT.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    chunks = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    return "".join(chunks).upper()


# -------------------------
# FCGR computation
# -------------------------

BITS: Dict[str, Tuple[int, int]] = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 0),
    "T": (1, 1),
}

def fcgr_counts(seq: str, k: int) -> np.ndarray:
    """
    Compute FCGR count matrix (uint32) of shape (2^k, 2^k).
    Uses a rolling bit build. Skips any k-mers spanning non-ACGT.
    """
    side = 1 << k
    mat = np.zeros((side, side), dtype=np.uint32)
    mask = side - 1

    x = 0
    y = 0
    run = 0

    for ch in seq:
        b = BITS.get(ch)
        if b is None:
            run = 0
            x = 0
            y = 0
            continue

        bx, by = b
        x = ((x << 1) & mask) | bx
        y = ((y << 1) & mask) | by
        run += 1

        if run >= k:
            mat[y, x] += 1

    return mat


def normalize_fcgr(counts: np.ndarray) -> np.ndarray:
    s = float(counts.sum())
    if s <= 0:
        return counts.astype(np.float32)
    return (counts / s).astype(np.float32)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


# -------------------------
# Stats: bootstrap CI + exact sign test fallback
# -------------------------

def bootstrap_median_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(x))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    lo = float(np.percentile(meds, 100 * (alpha / 2)))
    hi = float(np.percentile(meds, 100 * (1 - alpha / 2)))
    return med, lo, hi


def exact_binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """Exact survival function P(X >= k) for Binomial(n,p), log-space stable."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p = math.log(p)
    log_q = math.log(1 - p)

    logs = []
    for i in range(k, n + 1):
        log_coef = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        logs.append(log_coef + i * log_p + (n - i) * log_q)

    m = max(logs)
    s = sum(math.exp(li - m) for li in logs)
    return float(math.exp(m) * s)


def sign_test_greater_than(x: np.ndarray, delta: float = 0.0) -> float:
    """
    One-sided sign test for median(x) > delta.
    Counts how many values exceed delta, ignores ties.
    H0: P(x > delta) = 0.5.
    """
    x = x[np.isfinite(x)]
    gt = int(np.sum(x > delta))
    lt = int(np.sum(x < delta))
    n = gt + lt
    if n == 0:
        return 1.0
    return exact_binom_sf(gt, n, 0.5)


def wilcoxon_one_sample_greater_than(x: np.ndarray, delta: float = 0.0) -> Optional[float]:
    """
    One-sided Wilcoxon signed-rank test on (x - delta), H1: median(x) > delta.
    Returns p-value or None if SciPy unavailable/fails.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None

    d = x - delta
    if np.allclose(d, 0):
        return 1.0

    try:
        res = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return None


def mean_signflip_perm_pvalue(d: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """
    Paired sign-flip permutation test on mean(d) vs 0 (two-sided).
    Included as a robustness check.
    """
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size), replace=True)
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))


# -------------------------
# Visualization
# -------------------------

def _robust_limits(a: np.ndarray, b: np.ndarray, pct_lo: float, pct_hi: float) -> Tuple[float, float]:
    both = np.concatenate([a.ravel(), b.ravel()])
    vmin = float(np.percentile(both, pct_lo))
    vmax = float(np.percentile(both, pct_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(both.min()), float(both.max())
    return vmin, vmax


def save_mean_tripanel(
    mean_orig: np.ndarray,
    mean_syn: np.ndarray,
    out_png: Path,
    label: str,
    k: int,
    eps: float = 1e-10,
    pct_lo: float = 1.0,
    pct_hi: float = 99.9,
    ratio_pct: float = 99.5,
) -> None:
    """
    Tripanel:
      1) log10(mean_orig+eps) with shared scaling
      2) log10(mean_syn+eps)  with shared scaling
      3) log2-ratio log2((orig+eps)/(syn+eps)) with symmetric scaling
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    o = np.log10(mean_orig + eps)
    s = np.log10(mean_syn + eps)
    vmin, vmax = _robust_limits(o, s, pct_lo, pct_hi)

    r = np.log2((mean_orig + eps) / (mean_syn + eps))
    dv = float(np.percentile(np.abs(r.ravel()), ratio_pct))
    dv = max(dv, 1e-6)

    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    im1 = ax1.imshow(o, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax1.set_title(f"Original mean FCGR\nlog10(P+eps), k={k}")
    ax1.axis("off")

    im2 = ax2.imshow(s, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax2.set_title(f"Synthetic mean FCGR\nlog10(P+eps), k={k}")
    ax2.axis("off")

    im3 = ax3.imshow(r, vmin=-dv, vmax=dv, cmap="RdBu_r", interpolation="nearest", aspect="equal")
    ax3.set_title(f"log2 ratio\nlog2((orig+eps)/(syn+eps)), k={k}")
    ax3.axis("off")

    # Colorbars (one per panel; easiest to interpret in papers)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    fig.suptitle(label, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_distance_hist(d: np.ndarray, out_png: Path, label: str, k: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    d = d[np.isfinite(d)]
    plt.figure(figsize=(6, 4))
    plt.hist(d, bins=15)
    plt.xlabel("L1 distance between normalized FCGRs")
    plt.ylabel("Number of pairs")
    plt.title(f"{label}: FCGR distances (k={k})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------
# IO helpers
# -------------------------

def load_manifest(manifest: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    need = {"id", "orig", "syn"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Manifest must include columns: {sorted(need)}")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["orig"] = df["orig"].astype(str)
    df["syn"] = df["syn"].astype(str)
    return df


def infer_label(label: Optional[str], manifest: Optional[Path], tag: Optional[str]) -> str:
    if label:
        return label
    if tag:
        return tag
    if manifest:
        base = re.sub(r"\.csv$", "", manifest.name, flags=re.I)
        m = re.search(r"(?:pairs[._-])(.+)$", base, flags=re.I)
        return (m.group(1) if m else base) or "FCGR"
    return "FCGR"


# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="FCGR benchmark: original vs synthetic.")
    ap.add_argument("--manifest", type=str, default=None,
                    help="CSV with columns: id,orig,syn (recommended).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Compatibility mode: expects <data-root>/<tag>/pairs.<tag>.csv")

    ap.add_argument("--data-root", type=str, default=None,
                    help="Only used with --tag (compat mode).")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output root directory. Writes metrics/ and plots/ subdirs.")
    ap.add_argument("--label", type=str, default=None,
                    help="Nice label for plots/summary (optional).")

    ap.add_argument("--k", type=int, default=8,
                    help="FCGR order k (side=2^k). Default 8 -> 256x256.")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="If >0, process only the first N pairs (debug).")

    # stats
    ap.add_argument("--delta", type=float, default=0.0,
                    help="Threshold for median-distance test: H1 median(d) > delta. Default 0.")
    ap.add_argument("--n-perm", type=int, default=50000,
                    help="Permutations for sign-flip test on the mean.")
    ap.add_argument("--n-boot", type=int, default=20000,
                    help="Bootstrap resamples for median CI.")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed.")
    ap.add_argument("--eps", type=float, default=1e-10,
                    help="Epsilon for log plots/ratios.")
    return ap.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    plots_dir = outdir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Resolve manifest
    manifest = None
    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.exists():
            raise SystemExit(f"Missing manifest: {manifest}")
    else:
        if not args.tag:
            raise SystemExit("Provide either --manifest or --tag (compat mode).")
        if not args.data_root:
            raise SystemExit("Compat mode requires --data-root.")
        data_root = Path(args.data_root)
        manifest = data_root / args.tag / f"pairs.{args.tag}.csv"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest in compat mode: {manifest}")

    label = infer_label(args.label, manifest=manifest, tag=args.tag)

    df = load_manifest(manifest)
    if args.max_pairs and args.max_pairs > 0:
        df = df.head(args.max_pairs).copy()

    rng = np.random.default_rng(args.seed)
    side = 1 << args.k

    mean_orig = np.zeros((side, side), dtype=np.float64)
    mean_syn = np.zeros((side, side), dtype=np.float64)

    rows = []
    n_total = int(len(df))

    for _, row in df.iterrows():
        pid = str(row["id"])
        op = Path(row["orig"])
        sp = Path(row["syn"])

        if not op.exists() or not sp.exists():
            rows.append({"id": pid, "status": "missing_fasta", "l1_fcgr": np.nan})
            continue

        orig_seq = read_fasta_concat(op)
        syn_seq = read_fasta_concat(sp)

        oc = fcgr_counts(orig_seq, args.k)
        sc = fcgr_counts(syn_seq, args.k)

        opmf = normalize_fcgr(oc)
        spmf = normalize_fcgr(sc)

        d = l1_distance(opmf, spmf)

        mean_orig += opmf
        mean_syn += spmf

        rows.append({
            "id": pid,
            "status": "ok",
            "orig_path": str(op),
            "syn_path": str(sp),
            "orig_len": int(len(orig_seq)),
            "syn_len": int(len(syn_seq)),
            "orig_total_kmers": int(oc.sum()),
            "syn_total_kmers": int(sc.sum()),
            "l1_fcgr": float(d),
        })

    res = pd.DataFrame(rows)
    ok = res["status"].eq("ok").to_numpy()
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise SystemExit("No pairs processed successfully.")

    mean_orig /= n_ok
    mean_syn /= n_ok

    d = res.loc[res["status"].eq("ok"), "l1_fcgr"].to_numpy(dtype=float)

    # Significance for median distance > delta (one-sided)
    p_wil = wilcoxon_one_sample_greater_than(d, delta=args.delta)
    if p_wil is None:
        p_wil = sign_test_greater_than(d, delta=args.delta)
        test_used = "sign_test_one_sided_greater"
    else:
        test_used = "wilcoxon_one_sided_greater"

    # Extra robustness check on mean via sign-flip permutations (two-sided)
    p_perm = mean_signflip_perm_pvalue(d, n_perm=args.n_perm, rng=rng)

    med, lo, hi = bootstrap_median_ci(d, n_boot=args.n_boot, rng=rng)

    # Write per-pair metrics
    out_csv = metrics_dir / f"fcgr_pair_distances.k{args.k}.csv"
    res.to_csv(out_csv, index=False)

    summary = {
        "label": label,
        "manifest": str(manifest.resolve()),
        "k": int(args.k),
        "matrix_side": int(side),
        "n_pairs_total": int(n_total),
        "n_pairs_ok": int(n_ok),
        "distance": {
            "name": "L1(normalized_FCGR_orig, normalized_FCGR_syn)",
            "median": float(med),
            "median_ci_95": [float(lo), float(hi)],
            "mean": float(np.mean(d)),
        },
        "significance": {
            "delta_used": float(args.delta),
            "p_value_one_sided_median_gt_delta": float(p_wil),
            "test": test_used,
            "p_value_signflip_mean_two_sided": float(p_perm),
            "n_perm": int(args.n_perm),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        "notes": [
            "FCGR computed at order k; matrix size is 2^k x 2^k.",
            "k-mers spanning non-ACGT characters are skipped.",
            "Mean FCGR plots use log10(P+eps) with shared vmin/vmax across orig/syn.",
            "Difference map uses log2((orig+eps)/(syn+eps)) with symmetric scaling.",
        ],
    }

    out_json = metrics_dir / f"fcgr_summary.k{args.k}.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Plots
    save_mean_tripanel(
        mean_orig=mean_orig,
        mean_syn=mean_syn,
        out_png=plots_dir / f"fcgr_mean_tripanel.k{args.k}.png",
        label=label,
        k=args.k,
        eps=args.eps,
    )
    save_distance_hist(
        d=d,
        out_png=plots_dir / f"fcgr_distance_hist.k{args.k}.png",
        label=label,
        k=args.k,
    )

    print("[OK] FCGR benchmark complete")
    print(f"  manifest: {manifest}")
    print(f"  outdir:   {outdir}")
    print(f"  n_ok:     {n_ok}/{n_total}")
    print(f"  median L1: {med:.6g}  95% CI [{lo:.6g}, {hi:.6g}]")
    print(f"  p(one-sided median>d={args.delta}): {p_wil:.3g} ({test_used})")
    print(f"  p(signflip mean, two-sided):        {p_perm:.3g}")
    print(f"  wrote: {out_csv}")
    print(f"  wrote: {out_json}")


if __name__ == "__main__":
    main()
