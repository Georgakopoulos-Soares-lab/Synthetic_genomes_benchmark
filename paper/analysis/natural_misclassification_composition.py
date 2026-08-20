#!/usr/bin/env python3
"""
Reviewer #2 (major #2, final sub-part): are NATURAL windows with unusual
composition ever misclassified as synthetic?

The reviewer asks whether the classifier's natural-vs-synthetic decision is
driven by simple compositional properties (GC content, dinucleotide / CpG bias,
low-complexity content, repeat density) rather than deep structural signal.

We reuse the shallow k-mer logistic-regression classifier (k=4, leave-one-tag-out)
from shallow_baseline_classifier.py -- which R1.2 already showed matches the CNN
AUROC -- to obtain an out-of-fold P(synthetic) for every NATURAL window. We then
test whether the most "synthetic-looking" natural windows are compositional
outliers:
  * GC content
  * CpG observed/expected
  * low-complexity fraction (1 - normalised 3-mer Shannon entropy)
  * homopolymer fraction (run-length >= 4)
  * repeat density (1 - zlib compression ratio)

Outputs per-window scores, Spearman correlations, a misclassified-vs-correct
composition contrast, and a diagnostic figure.

CPU-only. Run with system python3 from /tmp.
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(f"{_ROOT}/revisions/scripts")
sys.path.insert(0, str(SCRIPTS))
from shallow_baseline_classifier import (  # noqa: E402
    DOMAINS, load_tag_sequences, chunk_sequence, kmer_features, encode_indices,
)

OUTRES = Path(f"{_ROOT}/revisions/results")
FIG = Path(f"{_ROOT}/revisions/figures")
OUTRES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

K = 4
CHUNK = 1024
MAX_CHUNKS = 40
SEED = 1337


# ---------- composition features (on the full natural window) ----------
def gc_content(idx: np.ndarray) -> float:
    v = idx[idx >= 0]
    if v.size == 0:
        return np.nan
    return float(((v == 1) | (v == 2)).mean())


def cpg_obs_exp(idx: np.ndarray) -> float:
    v = idx[idx >= 0]
    n = v.size
    if n < 2:
        return np.nan
    pC = (v == 1).mean(); pG = (v == 2).mean()
    if pC == 0 or pG == 0:
        return 0.0
    cg = np.sum((v[:-1] == 1) & (v[1:] == 2)) / (n - 1)
    return float(cg / (pC * pG))


def lowcomplexity_frac(idx: np.ndarray) -> float:
    """1 - normalised Shannon entropy of the 3-mer distribution."""
    v = idx[idx >= 0]
    if v.size < 3:
        return np.nan
    w = np.lib.stride_tricks.sliding_window_view(v, 3)
    codes = w[:, 0] * 16 + w[:, 1] * 4 + w[:, 2]
    counts = np.bincount(codes, minlength=64).astype(float)
    p = counts[counts > 0] / counts.sum()
    H = -(p * np.log2(p)).sum()
    return float(1.0 - H / np.log2(64))


def homopolymer_frac(idx: np.ndarray) -> float:
    """Fraction of bases inside homopolymer runs of length >= 4."""
    v = idx[idx >= 0]
    n = v.size
    if n == 0:
        return np.nan
    in_run = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[j + 1] == v[i]:
            j += 1
        runlen = j - i + 1
        if runlen >= 4:
            in_run += runlen
        i = j + 1
    return float(in_run / n)


def repeat_density(seq: str) -> float:
    """1 - zlib compression ratio (higher = more repetitive)."""
    b = seq.encode()
    if not b:
        return np.nan
    comp = len(zlib.compress(b, 6))
    return float(1.0 - comp / len(b))


def window_features(seq: str) -> dict:
    idx = encode_indices(seq).astype(np.int64)
    return {
        "gc": gc_content(idx),
        "cpg_oe": cpg_obs_exp(idx),
        "lowcomplexity": lowcomplexity_frac(idx),
        "homopolymer_frac": homopolymer_frac(idx),
        "repeat_density": repeat_density(seq),
    }


# ---------- build per-window data with classifier P(syn) ----------
def collect(domain: str, tags: list[str]):
    rng = np.random.default_rng(SEED)
    per_tag = {}
    for tag in tags:
        orig, syn = load_tag_sequences(tag)
        if not orig or not syn:
            print(f"[warn] {tag}: missing data", file=sys.stderr)
            continue
        # chunk-level features for the classifier + window bookkeeping
        Xc, yc, win_of_chunk = [], [], []
        windows = []  # (label, seq, comp_features)
        for label, seqs in ((0, orig), (1, syn)):
            for si, seq in enumerate(seqs):
                chunks = chunk_sequence(seq, CHUNK, MAX_CHUNKS, rng)
                if not chunks:
                    continue
                widx = len(windows)
                comp = window_features(seq) if label == 0 else None
                windows.append((label, widx, comp))
                for ch in chunks:
                    Xc.append(kmer_features(ch, K))
                    yc.append(label)
                    win_of_chunk.append(widx)
        if not Xc:
            continue
        per_tag[tag] = (np.vstack(Xc), np.array(yc), np.array(win_of_chunk), windows)
        print(f"[info] {tag}: {len(yc)} chunks, {len(windows)} windows")
    return per_tag


def loto_scores(per_tag: dict):
    """Leave-one-tag-out P(syn) for every window."""
    rows = []
    tags = list(per_tag)
    for held in tags:
        Xtr = np.vstack([per_tag[t][0] for t in tags if t != held])
        ytr = np.concatenate([per_tag[t][1] for t in tags if t != held])
        Xte, yte, wte, windows = per_tag[held]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        p_chunk = clf.predict_proba(sc.transform(Xte))[:, 1]
        # aggregate chunk P(syn) to window level
        for label, widx, comp in windows:
            mask = wte == widx
            if not mask.any():
                continue
            p_syn = float(p_chunk[mask].mean())
            row = {"tag": held, "window_idx": widx, "label": label, "p_syn": p_syn}
            if comp:
                row.update(comp)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    feats = ["gc", "cpg_oe", "lowcomplexity", "homopolymer_frac", "repeat_density"]
    all_nat = []
    summary = []
    for domain in ["euk", "prok", "vir"]:
        per_tag = collect(domain, DOMAINS[domain])
        if len(per_tag) < 2:
            continue
        df = loto_scores(per_tag)
        nat = df[df.label == 0].copy()
        nat["domain"] = domain
        nat["misclassified"] = nat.p_syn > 0.5
        all_nat.append(nat)
        mis_rate = float(nat.misclassified.mean())
        print(f"\n=== {domain}: {len(nat)} natural windows, "
              f"{100*mis_rate:.1f}% misclassified as synthetic ===")
        for f in feats:
            if nat[f].notna().sum() < 5:
                continue
            rho, p = spearmanr(nat["p_syn"], nat[f], nan_policy="omit")
            mis = nat.loc[nat.misclassified, f].dropna()
            cor = nat.loc[~nat.misclassified, f].dropna()
            if len(mis) >= 3 and len(cor) >= 3:
                u, pu = mannwhitneyu(mis, cor, alternative="two-sided")
            else:
                pu = np.nan
            summary.append({
                "domain": domain, "feature": f,
                "spearman_rho_psyn": rho, "spearman_p": p,
                "misclass_median": float(mis.median()) if len(mis) else np.nan,
                "correct_median": float(cor.median()) if len(cor) else np.nan,
                "mwu_p_mis_vs_correct": pu,
                "n_misclassified": int(len(mis)), "n_correct": int(len(cor)),
            })
            print(f"  {f:18s} rho(P_syn)={rho:+.3f} (p={p:.1e})  "
                  f"mis_med={mis.median():.3f} vs correct_med={cor.median():.3f} "
                  f"(MWU p={pu:.1e})")

    natdf = pd.concat(all_nat, ignore_index=True)
    natdf.to_csv(OUTRES / "natural_misclassification_per_window.csv", index=False)
    sumdf = pd.DataFrame(summary)
    sumdf.to_csv(OUTRES / "natural_misclassification_summary.csv", index=False)

    # ---- figure: euk natural windows, P(syn) vs GC and vs low-complexity ----
    euk = natdf[natdf.domain == "euk"]
    if not euk.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, f, lab in zip(
            axes, ["gc", "lowcomplexity", "repeat_density"],
            ["GC content", "Low-complexity (1-norm 3-mer entropy)", "Repeat density (1-zlib ratio)"]):
            ok = euk[euk[f].notna()]
            colors = np.where(ok.misclassified, "#d95f02", "#1b9e77")
            ax.scatter(ok[f], ok.p_syn, c=colors, s=18, alpha=0.7)
            ax.axhline(0.5, color="grey", ls="--", lw=0.8)
            rho, _ = spearmanr(ok.p_syn, ok[f], nan_policy="omit")
            ax.set_xlabel(lab); ax.set_ylabel("P(synthetic) | natural window")
            ax.set_title(f"rho={rho:+.2f}", fontsize=10)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color="#1b9e77", label="correctly natural"),
                            Patch(color="#d95f02", label="misclassified as synthetic")],
                   loc="lower center", ncol=2, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.03))
        fig.suptitle("Do compositional outliers among natural windows look synthetic? "
                     "(eukaryotes)", fontsize=12, weight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        out = FIG / "natural_misclassification_composition.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\n[done] wrote {out}")

    print(f"[wrote] {OUTRES/'natural_misclassification_per_window.csv'}")
    print(f"[wrote] {OUTRES/'natural_misclassification_summary.csv'}")


if __name__ == "__main__":
    main()
