#!/usr/bin/env python3
"""
Metrics for the Evo 2 7B constrained-decoding experiment (Reviewer #1, Major #1,
strategy 2 on the real model). Compares, per window and in aggregate:
  baseline  (Evo 2 default decoding)  vs
  constrained (homopolymer + k-mer over-representation penalties) vs
  natural   (the matched natural reference window)

against the paper's metric families:
  - k-mer Jensen-Shannon divergence (k=6) vs the natural reference
  - homopolymer fraction (runs >= 5 bp) and low-complexity score
  - FCGR L1 distance (k=6) vs the natural reference
  - shallow-classifier AUROC (natural vs generated, k=6 LogReg): lower = more
    natural-like / harder to detect

CPU only. Run with system python3 from /tmp (uses user-site scipy/sklearn).
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(f"{_ROOT}/revisions")
CDEC = REV / "decoding_sweep" / "constrained_decode"
MANIFEST = REV / "decoding_sweep" / "cdec_manifest.csv"
OUTCSV = REV / "results" / "constrained_decode_metrics.csv"
OUTFIG = REV / "figures" / "constrained_decode_metrics.png"
OUTCSV.parent.mkdir(parents=True, exist_ok=True)
OUTFIG.parent.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 50000
KJSD = 6
KFCGR = 6
EVAL_CHUNK = 2048
CONFIGS = ["baseline", "constrained"]
_NT = {"A": 0, "C": 1, "G": 2, "T": 3}


def load_seq(path, limit=None):
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            buf.append(line.strip())
    s = "".join(buf).upper()
    return s[:limit] if limit else s


def encode(seq):
    a = np.array([_NT.get(c, -1) for c in seq], dtype=np.int64)
    return a[a >= 0]


def kmer_freq(idx, k):
    if idx.size < k:
        return np.ones(4 ** k) / (4 ** k)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(idx, k)
    codes = (win * powers).sum(axis=1)
    c = np.bincount(codes, minlength=4 ** k).astype(float)
    return c / c.sum()


def js_div(p, q):
    p = p + 1e-12; q = q + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def fcgr_l1(idx_a, idx_b, k):
    return float(np.abs(kmer_freq(idx_a, k) - kmer_freq(idx_b, k)).sum())


def homopolymer_fraction(idx, min_run=5):
    if idx.size == 0:
        return 0.0
    covered = 0; cur = 1
    for i in range(1, idx.size):
        if idx[i] == idx[i - 1]:
            cur += 1
        else:
            if cur >= min_run:
                covered += cur
            cur = 1
    if cur >= min_run:
        covered += cur
    return covered / idx.size


def lowcomplexity(idx, k=3):
    f = kmer_freq(idx, k)
    f = f[f > 0]
    ent = -(f * np.log2(f)).sum()
    return float(1.0 - ent / (k * 2.0))


def chunks(idx, size):
    return [idx[i:i + size] for i in range(0, idx.size - size + 1, size)]


def auroc(nat_chunks, gen_chunks, k=KJSD):
    if len(nat_chunks) < 5 or len(gen_chunks) < 5:
        return float("nan")
    Xn = np.stack([kmer_freq(c, k) for c in nat_chunks])
    Xg = np.stack([kmer_freq(c, k) for c in gen_chunks])
    X = np.vstack([Xn, Xg]); y = np.r_[np.zeros(len(Xn)), np.ones(len(Xg))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    per_window = []
    nat_all, gen_all = {}, {c: [] for c in CONFIGS}
    nat_chunks_all = []
    gen_chunks_all = {c: [] for c in CONFIGS}
    for r in rows:
        wid = r["window_id"]
        nat = encode(load_seq(r["natref_fasta"], TARGET_LEN))
        nat_kmer = kmer_freq(nat, KJSD)
        nat_chunks_all += chunks(nat, EVAL_CHUNK)
        for cfg in CONFIGS:
            fa = CDEC / cfg / f"{wid}.{cfg}.syn.fa"
            if not fa.exists():
                continue
            g = encode(load_seq(fa, TARGET_LEN))
            gen_chunks_all[cfg] += chunks(g, EVAL_CHUNK)
            per_window.append(dict(
                window=wid, config=cfg,
                kmer_jsd=js_div(nat_kmer.copy(), kmer_freq(g, KJSD)),
                fcgr_l1=fcgr_l1(nat, g, KFCGR),
                homopolymer_frac=homopolymer_fraction(g),
                lowcomplexity=lowcomplexity(g)))
    # natural reference values (per window) for context
    for r in rows:
        wid = r["window_id"]
        nat = encode(load_seq(r["natref_fasta"], TARGET_LEN))
        per_window.append(dict(
            window=wid, config="natural",
            kmer_jsd=0.0, fcgr_l1=0.0,
            homopolymer_frac=homopolymer_fraction(nat),
            lowcomplexity=lowcomplexity(nat)))

    with open(OUTCSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["window", "config", "kmer_jsd",
                                           "fcgr_l1", "homopolymer_frac",
                                           "lowcomplexity"])
        w.writeheader()
        for row in per_window:
            w.writerow(row)

    # aggregate
    def agg(cfg, key):
        vals = [p[key] for p in per_window if p["config"] == cfg]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for cfg in CONFIGS + ["natural"]:
        summary[cfg] = {k: agg(cfg, k) for k in
                        ["kmer_jsd", "fcgr_l1", "homopolymer_frac", "lowcomplexity"]}
        summary[cfg]["auroc"] = (auroc(nat_chunks_all, gen_chunks_all[cfg])
                                 if cfg in CONFIGS else float("nan"))
    print("=== aggregate (mean over 6 windows) ===")
    for cfg in CONFIGS + ["natural"]:
        s = summary[cfg]
        print(f"{cfg:>11}: JSD={s['kmer_jsd']:.4f} FCGR_L1={s['fcgr_l1']:.4f} "
              f"homo={s['homopolymer_frac']:.4f} lowc={s['lowcomplexity']:.4f} "
              f"AUROC={s['auroc']:.3f}")

    # figure
    order = ["baseline", "constrained", "natural"]
    metrics = [("kmer_jsd", "k-mer JSD (k=6)\nvs natural (lower=better)"),
               ("fcgr_l1", "FCGR L1 (k=6)\nvs natural (lower=better)"),
               ("homopolymer_frac", "Homopolymer fraction"),
               ("lowcomplexity", "Low-complexity score"),
               ("auroc", "Classifier AUROC\n(lower=harder to detect)")]
    colors = ["#888888", "#7570b3", "#000000"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[c].get(key, np.nan) for c in order]
        ax.bar(range(len(order)), vals, color=colors)
        if key == "auroc":
            ax.axhline(0.5, ls="--", color="red", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_title(title, fontsize=11)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Constrained decoding on Evo 2 7B (6 eukaryotic windows, 50 kb)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTFIG, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {OUTCSV} and {OUTFIG}")


if __name__ == "__main__":
    main()
