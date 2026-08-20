#!/usr/bin/env python3
"""
Decoding-sweep audit: per-window CSV, summary CSV, paired tests, and figure.

Reads:
  revisions/decoding_sweep/generated/{lowtemp,nucleus}/*.syn.fa
  revisions/decoding_sweep/seeds/*.natref.fa
  revisions/decoding_sweep/seed_manifest.csv
  revisions/results/natnat_fcgr_summary_k8.csv   (natural-natural baseline band)

Writes:
  revisions/results/decoding_sweep_per_window.csv
  revisions/results/decoding_sweep_summary.csv
  revisions/results/decoding_sweep_paired_tests.csv
  revisions/figures/decoding_sweep_audit.{png,pdf}

CPU-only.
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import csv
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import bootstrap as scipy_bootstrap

PROJ    = Path(_ROOT)
REV     = PROJ / "revisions"
GEN     = REV / "decoding_sweep" / "generated"
SEEDS   = REV / "decoding_sweep" / "seeds"
MANIFEST = REV / "decoding_sweep" / "seed_manifest.csv"
NATNAT  = REV / "results" / "natnat_fcgr_summary_k8.csv"
RESULTS = REV / "results"
FIGS    = REV / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CONFIGS_ORDERED = ["lowtemp", "nucleus"]
COLORS = {"lowtemp": "#1b9e77", "nucleus": "#7570b3"}

# ── metric hyperparameters (must match run_sweep_metrics.py) ──────────────────
FCGR_K = 8
KMER_K = 6
NULL_K = 9

# ── FCGR helpers ──────────────────────────────────────────────────────────────
BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
_BX = np.full(256, -1, dtype=np.int64)
_BY = np.full(256, -1, dtype=np.int64)
for _ch, (_bx, _by) in BITS.items():
    _BX[ord(_ch)] = _bx
    _BY[ord(_ch)] = _by

_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i


def read_fasta(path: Path) -> str:
    parts = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts).upper()


def fcgr_norm(seq: str, k: int) -> np.ndarray:
    side = 1 << k
    codes = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
    bx = _BX[codes]
    by = _BY[codes]
    if len(bx) < k:
        return np.zeros(side * side, dtype=np.float64)
    wx = np.lib.stride_tricks.sliding_window_view(bx, k)
    wy = np.lib.stride_tricks.sliding_window_view(by, k)
    valid = (wx >= 0).all(axis=1)
    if not valid.any():
        return np.zeros(side * side, dtype=np.float64)
    powers = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    x = (wx[valid] * powers).sum(axis=1)
    y = (wy[valid] * powers).sum(axis=1)
    lin = y * side + x
    mat = np.bincount(lin, minlength=side * side).astype(np.float64)
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def kmer_freq(seq: str, k: int) -> np.ndarray:
    codes = _CODE[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    if len(codes) < k:
        return np.zeros(4 ** k, dtype=np.float64)
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return np.zeros(4 ** k, dtype=np.float64)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    idx = (win[valid] * powers).sum(axis=1)
    cnt = np.bincount(idx, minlength=4 ** k).astype(np.float64)
    s = cnt.sum()
    return cnt / s if s > 0 else cnt


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def canonical_classes(k: int) -> int:
    return (4 ** k + (4 ** (k // 2) if k % 2 == 0 else 0)) // 2


def canonical_nullomer_fraction(seq: str, k: int) -> float:
    """Fraction of canonical k-mer classes absent from seq."""
    codes = _CODE[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = len(codes)
    if n < k:
        return float("nan")
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (win >= 0).all(axis=1)
    win = win[valid]
    if win.size == 0:
        return float("nan")
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    idx  = (win * powers).sum(axis=1)
    comp = 3 - win
    rc   = comp[:, ::-1]
    ridx = (rc * powers).sum(axis=1)
    canon = np.minimum(idx, ridx)
    observed = np.unique(canon).size
    total = canonical_classes(k)
    return 1.0 - observed / total


# ── paired Wilcoxon + Cohen's dz + CI ────────────────────────────────────────

def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Wilcoxon signed-rank test on differences a - b.
    scipy.stats.wilcoxon(a, b) computes d = a - b internally.
    Returns (W, p, direction_string).
    """
    d = a - b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            W, p = wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            return math.nan, math.nan, "n/a"
    direction = "a > b" if np.median(d) > 0 else "a < b"
    return float(W), float(p), direction


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's dz = mean(a-b) / std(a-b, ddof=1)."""
    d = a - b
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return math.nan
    return float(np.mean(d) / sd)


def mean_ci(d: np.ndarray, n_boot: int = 9999, seed: int = 42):
    """Bootstrap 95% CI for the mean of array d."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bh_correct(pvals: list) -> list:
    """BH adjustment; nan inputs → nan outputs."""
    vi = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    adj = [math.nan] * len(pvals)
    if not vi:
        return adj
    vp = [pvals[i] for i in vi]
    m = len(vp)
    order = np.argsort(vp)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    raw = np.array([min(p * m / r, 1.0) for p, r in zip(vp, ranks)])
    for i in range(m - 2, -1, -1):
        raw[order[i]] = min(raw[order[i]], raw[order[i + 1]])
    for j, i in enumerate(vi):
        adj[i] = float(raw[j])
    return adj


# ── parse manifest for coordinates ───────────────────────────────────────────

def parse_manifest():
    """Returns {window_id: {tag, chrom, genomic_start, window_len, seed_len, target_len}}."""
    info = {}
    with open(MANIFEST, newline="") as fh:
        for row in csv.DictReader(fh):
            wid = row["window_id"]
            # src_record: 'orig|orig.{chrom}.{start}.{len}.fa'
            name  = row["src_record"].split("|")[-1].removesuffix(".fa")
            parts = name.split(".")
            win_len = int(parts[-1])
            start   = int(parts[-2])
            chrom   = ".".join(parts[1:-2])
            info[wid] = {
                "tag":         row["tag"],
                "chrom":       chrom,
                "genomic_start": start,
                "window_len":  win_len,
                "seed_len":    int(row["seed_len"]),
                "target_len":  int(row["target_len"]),
            }
    return info


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    manifest = parse_manifest()
    natref_cache: dict = {}

    pw_rows   = []   # one row per (config, window)
    
    for cfg in CONFIGS_ORDERED:
        cdir = GEN / cfg
        if not cdir.is_dir():
            print(f"[warn] {cdir} not found — skipping")
            continue
        for fa in sorted(cdir.glob(f"*.{cfg}.syn.fa")):
            wid = fa.name[: -len(f".{cfg}.syn.fa")]
            info = manifest.get(wid, {})
            natref_p = SEEDS / f"{wid}.natref.fa"
            if not natref_p.exists():
                print(f"[warn] no natref for {wid}")
                continue

            syn = read_fasta(fa)
            if wid not in natref_cache:
                natref_cache[wid] = read_fasta(natref_p)
            nat = natref_cache[wid]

            l1   = float(np.abs(fcgr_norm(syn, FCGR_K) - fcgr_norm(nat, FCGR_K)).sum())
            kj   = jsd(kmer_freq(syn, KMER_K), kmer_freq(nat, KMER_K))
            nsyn = canonical_nullomer_fraction(syn, NULL_K)
            nnat = canonical_nullomer_fraction(nat, NULL_K)
            completed = (len(syn) == info.get("target_len", 300000))

            pw_rows.append({
                "window_id":          wid,
                "species":            info.get("tag", ""),
                "chromosome":         info.get("chrom", ""),
                "start":              info.get("genomic_start", ""),
                "end":                (info.get("genomic_start", 0) +
                                       info.get("window_len", 0)),
                "decoding_config":    cfg,
                "seed_length":        info.get("seed_len", ""),
                "generated_length":   len(syn),
                "kmer_JSD":           f"{kj:.6f}",
                "FCGR_L1":            f"{l1:.6f}",
                "nullomer_fraction":  f"{nsyn:.6f}",
                "tandem_repeat_metric": "not_computed",
                "generation_completed": "yes" if completed else "no",
            })
            print(f"[ok] {cfg}/{wid}: L1={l1:.4f} JSD={kj:.4f} null={nsyn:.3f}", flush=True)

    df = pd.DataFrame(pw_rows)

    # ── A. per-window CSV ──────────────────────────────────────────────────
    pw_path = RESULTS / "decoding_sweep_per_window.csv"
    df.to_csv(pw_path, index=False)
    print(f"\n[done] per-window -> {pw_path}  ({len(df)} rows)")

    # ── B. summary CSV ──────────────────────────────────────────────────────
    METRIC_COLS = {
        "kmer_JSD":           "kmer_JSD",
        "FCGR_L1":            "FCGR_L1",
        "nullomer_fraction":  "nullomer_fraction",
    }

    summ_rows = []
    for cfg in CONFIGS_ORDERED:
        sub = df[df.decoding_config == cfg]
        for col, label in METRIC_COLS.items():
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            ci_lo, ci_hi = mean_ci(vals)
            summ_rows.append({
                "decoding_config": cfg,
                "n_windows":       len(vals),
                "metric":          label,
                "mean":            f"{np.mean(vals):.6f}",
                "median":          f"{np.median(vals):.6f}",
                "standard_deviation": f"{np.std(vals, ddof=1):.6f}",
                "ci_low":          f"{ci_lo:.6f}",
                "ci_high":         f"{ci_hi:.6f}",
            })

    summ_path = RESULTS / "decoding_sweep_summary.csv"
    pd.DataFrame(summ_rows).to_csv(summ_path, index=False)
    print(f"[done] summary -> {summ_path}")

    # ── C. paired tests ────────────────────────────────────────────────────
    # configs: lowtemp vs nucleus (only two configs exist)
    # We pair on window_id (same 15 windows for each config)
    pairs = list(itertools.combinations(CONFIGS_ORDERED, 2))
    metrics = list(METRIC_COLS.keys())

    pt_rows = []
    raw_pvals_per_metric: dict = {m: [] for m in metrics}
    pt_meta: dict = {m: [] for m in metrics}

    for metric in metrics:
        for ca, cb in pairs:
            sub_a = df[df.decoding_config == ca].set_index("window_id")
            sub_b = df[df.decoding_config == cb].set_index("window_id")
            common = sorted(set(sub_a.index) & set(sub_b.index))
            if len(common) < 2:
                raw_pvals_per_metric[metric].append(math.nan)
                pt_meta[metric].append((ca, cb, [], [], math.nan, math.nan, math.nan))
                continue
            a_vals = pd.to_numeric(sub_a.loc[common, metric], errors="coerce").values.astype(float)
            b_vals = pd.to_numeric(sub_b.loc[common, metric], errors="coerce").values.astype(float)
            valid = np.isfinite(a_vals) & np.isfinite(b_vals)
            a_v, b_v = a_vals[valid], b_vals[valid]
            if len(a_v) < 2:
                raw_pvals_per_metric[metric].append(math.nan)
                pt_meta[metric].append((ca, cb, [], [], math.nan, math.nan, math.nan))
                continue
            W, p, direction = paired_wilcoxon(a_v, b_v)
            dz = cohens_dz(a_v, b_v)
            d  = a_v - b_v
            raw_pvals_per_metric[metric].append(p)
            pt_meta[metric].append((ca, cb, a_v, b_v, W, p, dz))

    for metric in metrics:
        adj = bh_correct(raw_pvals_per_metric[metric])
        for i, ((ca, cb, a_v, b_v, W, p, dz), p_adj) in enumerate(
                zip(pt_meta[metric], adj)):
            d = np.array(a_v) - np.array(b_v) if len(a_v) else np.array([])
            pt_rows.append({
                "metric":           metric,
                "comparison":       f"{ca}_vs_{cb}",
                "n_pairs":          len(a_v),
                "mean_difference":  f"{np.mean(d):.6f}"   if d.size else "",
                "median_difference":f"{np.median(d):.6f}" if d.size else "",
                "test_statistic":   f"{W:.1f}"   if not math.isnan(W) else "",
                "p_value":          f"{p:.6f}"   if not math.isnan(p) else "",
                "p_adjusted":       f"{p_adj:.6f}" if not math.isnan(p_adj) else "",
                "significant":      "yes" if (not math.isnan(p_adj) and p_adj < 0.05) else "no",
            })

    pt_path = RESULTS / "decoding_sweep_paired_tests.csv"
    pd.DataFrame(pt_rows).to_csv(pt_path, index=False)
    print(f"[done] paired tests -> {pt_path}")

    # Print results
    print("\n=== Summary (median per config) ===")
    for cfg in CONFIGS_ORDERED:
        sub = df[df.decoding_config == cfg]
        for col in metrics:
            v = pd.to_numeric(sub[col], errors="coerce")
            print(f"  {cfg:10s}  {col:22s}  median={v.median():.4f}  mean={v.mean():.4f}  n={v.notna().sum()}")

    print("\n=== Paired tests (BH-corrected) ===")
    for r in pt_rows:
        print(f"  {r['metric']:22s}  {r['comparison']:25s}  "
              f"diff={r['mean_difference']:>9s}  W={r['test_statistic']:>7s}  "
              f"p={r['p_value']:>8s}  p_adj={r['p_adjusted']:>8s}  sig={r['significant']}")

    # ── D. Figure ────────────────────────────────────────────────────────────
    # Load natnat band for FCGR L1
    natnat_df = pd.read_csv(NATNAT) if NATNAT.exists() else None
    natnat_band = {}
    if natnat_df is not None:
        for tag in ["Publish_Human", "Publish_Mus", "Publish_Drosophila"]:
            row = natnat_df[natnat_df.tag == tag]
            if not row.empty:
                natnat_band[tag] = float(row["median_nat_nat_L1"].values[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Alternative-decoding sweep: per-window synthetic-vs-natural metrics\n"
        "(lowtemp: T=0.7, top_k=4; nucleus: T=1.0, top_p=0.9; n=15 per config)",
        fontsize=10, y=1.01)

    metric_labels = {
        "FCGR_L1":           f"FCGR L1 (k={FCGR_K})\nsynthetic vs natural",
        "kmer_JSD":          f"k-mer JSD (k={KMER_K})\nsynthetic vs natural",
        "nullomer_fraction": f"Canonical nullomer fraction (k={NULL_K})\nsynthetic sequence",
    }

    for ax, metric in zip(axes, ["FCGR_L1", "kmer_JSD", "nullomer_fraction"]):
        for ci, cfg in enumerate(CONFIGS_ORDERED):
            sub = df[df.decoding_config == cfg]
            vals = pd.to_numeric(sub[metric], errors="coerce").values
            color = COLORS[cfg]
            # jitter
            jx = ci + 1 + np.random.default_rng(ci).uniform(-0.12, 0.12, len(vals))
            ax.scatter(jx, vals, color=color, alpha=0.7, s=30, zorder=3,
                       label=cfg if ci == 0 else None)
            # box
            bp = ax.boxplot(vals[np.isfinite(vals)], positions=[ci + 1],
                            widths=0.35, patch_artist=True, showfliers=False,
                            zorder=4)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
            for el in ["medians"]:
                for line in bp[el]:
                    line.set_color("black")

        # natnat band for FCGR
        if metric == "FCGR_L1" and natnat_band:
            overall_band = np.median(list(natnat_band.values()))
            ax.axhline(overall_band, color="#e6550d", linestyle="--", lw=1.3,
                       label=f"nat–nat median L1 ({overall_band:.3f})", zorder=2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(CONFIGS_ORDERED, fontsize=9)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_xlim(0.4, 2.6)

        # annotate paired p-value
        pt_sub = [r for r in pt_rows
                  if r["metric"] == ("kmer_JSD" if metric == "kmer_JSD" else
                                     "FCGR_L1" if metric == "FCGR_L1" else
                                     "nullomer_fraction")]
        if pt_sub:
            r = pt_sub[0]
            pstr = f"Wilcoxon p={r['p_value']}\nn={r['n_pairs']}"
            ax.set_title(pstr, fontsize=8)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if metric == "FCGR_L1" and natnat_band:
            ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"decoding_sweep_audit.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[done] figure -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
