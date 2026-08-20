#!/usr/bin/env python3
"""
natnat_fcgr_permutation.py
==========================
Matched-label permutation test for FCGR L1 natural–natural baseline.

AUDIT FINDINGS ADDRESSED
  The original natnat_fcgr_baseline.py used one-sided Mann-Whitney U on
  sampled pairwise L1 distances.  Because every synthetic window is the
  matched counterpart of a specific natural window (confirmed: the i-th
  entry in <TAG>.orig.concat.fa and <TAG>.syn.concat.fa share the same
  genomic locus, equal counts in all 25 species), the MWU treats multiple
  distances arising from the same window as independent observations, which
  inflates the effective sample size and makes p-values anticonservatively
  biased.

  Additional issue: the old script capped windows at max_windows=40 and
  pairs at max_pairs=400.  The new script uses all available windows and
  the full 2n×2n precomputed distance matrix.

PARAMETERS (fixed for reproducibility)
  k              = 8          FCGR order (normalized, flattened 2^k × 2^k)
  B              = 10 000     Monte Carlo permutations per species
  seed           = 1337
  FDR method     = Benjamini-Hochberg across all species

ALGORITHM (per species, n matched pairs)
  1. Compute FCGR(k=8) vector for each of the 2n windows (nat first, then syn).
  2. Stack into vecs (2n, 4^k) and compute the full 2n×2n L1 distance matrix
     D using scipy.spatial.distance.cdist(vecs, vecs, 'cityblock').
  3. Observed indices: nat_idx = arange(n), syn_idx = arange(n, 2n).
     delta_obs = median(D[syn_idx, :][:, nat_idx].ravel())
               - median(D[nat_idx, :][:, nat_idx][upper-tri])
  4. For each of B Monte Carlo permutations (rng.integers(0,2,n) → b):
       perm_nat[k] = k    if b[k]==0 else k+n
       perm_syn[k] = k+n  if b[k]==0 else k
       Δ_perm = median(D[perm_syn, :][:, perm_nat].ravel())
              - median(D[perm_nat, :][:, perm_nat][upper-tri])
  5. p = (1 + #{Δ_perm ≥ δ_obs}) / (1 + B).
  BH-FDR across all species.

VALIDATION (embedded, runs at start-up)
  V1: swap-idempotence — applying b=ones twice restores delta_obs.
  V2: toy 4-pair exact check — compares exact Δ distribution (2^4=16
      permutations) against Monte Carlo on the same toy data.

SPECIES INCLUDED
  euk (20): Publish_Aedes, Apis, Arabidopsis, Bos, Branchiostoma,
            Caenorhabditis, Canis, Danio, Drosophila, Gallus, Gossypium,
            Mus, Nematostella, Oryza, Saccharina, Takifugu, Triticum,
            Xenopus, Zea, Human
  vir  (5): Kitrinoviricota, Nucleocytoviricota, Peploviricota,
            Preplasmiviricota, Uroviricota
  NOTE: Uroviricota has 48 matched pairs (old script capped at 40).

OUTPUT
  revisions/results/natnat_fcgr_permutation_results.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
PROJ = Path(_ROOT)
HARM = PROJ / "results" / "harmonized"
OUT  = PROJ / "revisions" / "results"

K    = 8        # FCGR order
B    = 10_000   # Monte Carlo permutations
SEED = 1337
Q_THRESH = 0.05

DOMAIN_TAGS: dict[str, list[str]] = {
    "euk": [
        "Publish_Aedes", "Publish_Apis", "Publish_Arabidopsis", "Publish_Bos",
        "Publish_Branchiostoma", "Publish_Caenorhabditis", "Publish_Canis",
        "Publish_Danio", "Publish_Drosophila", "Publish_Gallus",
        "Publish_Gossypium", "Publish_Mus", "Publish_Nematostella",
        "Publish_Oryza", "Publish_Saccharina", "Publish_Takifugu",
        "Publish_Triticum", "Publish_Xenopus", "Publish_Zea", "Publish_Human",
    ],
    "vir": [
        "Kitrinoviricota", "Nucleocytoviricota", "Peploviricota",
        "Preplasmiviricota", "Uroviricota",
    ],
}

# Per-base CGR bit lookup (index by ASCII byte value; -1 = non-ACGT)
_BX = np.full(256, -1, np.int64)
_BY = np.full(256, -1, np.int64)
for _ch, (_bx, _by) in {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}.items():
    _BX[ord(_ch)] = _bx
    _BY[ord(_ch)] = _by


# ──────────────────────────────────────────────────────────────────────────────
# FCGR COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def iter_fasta(path: Path):
    """Yield upper-cased sequence strings from a FASTA file."""
    seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq).upper()
                    seq = []
            else:
                seq.append(line.strip())
    if seq:
        yield "".join(seq).upper()


def fcgr_norm(seq: str, k: int) -> np.ndarray:
    """Return a normalized, flattened FCGR vector (length 4^k)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_delta(D: np.ndarray, nat_idx: np.ndarray,
                  syn_idx: np.ndarray, triu_r: np.ndarray,
                  triu_c: np.ndarray) -> float:
    """Δ = median(syn-nat distances) – median(nat-nat upper-tri distances)."""
    d_nn = D[nat_idx[triu_r], nat_idx[triu_c]]
    d_sn = D[np.ix_(syn_idx, nat_idx)].ravel()
    return float(np.median(d_sn) - np.median(d_nn))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; handles NaN by passing them through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    out[valid] = q
    return out


def permutation_pvalue(D: np.ndarray, n: int, B: int, seed: int,
                       delta_obs: float) -> tuple[float, str, int]:
    """
    Monte Carlo permutation p-value.

    Returns (p_value, permutation_type, n_permutations_used).
    Always uses Monte Carlo (exact enumeration is not feasible for n up to 48,
    since 2^48 >> 10000).
    """
    rng = np.random.default_rng(seed)
    triu_r, triu_c = np.triu_indices(n, k=1)
    base_nat = np.arange(n, dtype=np.int64)
    base_syn = np.arange(n, 2 * n, dtype=np.int64)

    count_ge = 0
    for _ in range(B):
        b = rng.integers(0, 2, size=n)
        p_nat = np.where(b == 0, base_nat, base_syn)
        p_syn = np.where(b == 0, base_syn, base_nat)
        d = compute_delta(D, p_nat, p_syn, triu_r, triu_c)
        if d >= delta_obs:
            count_ge += 1

    p = (1 + count_ge) / (1 + B)
    return float(p), "monte_carlo", B


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
def _toy_exact_deltas(nat: np.ndarray, syn: np.ndarray) -> np.ndarray:
    """Enumerate all 2^n matched-pair label permutations exactly (toy/small n)."""
    n = len(nat)
    all_vals = np.concatenate([nat, syn])
    D_toy = np.abs(all_vals[:, None] - all_vals[None, :])
    triu_r, triu_c = np.triu_indices(n, k=1)
    base_nat = np.arange(n)
    base_syn = np.arange(n, 2 * n)
    deltas = []
    for mask in range(2 ** n):
        b = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int64)
        p_nat = np.where(b == 0, base_nat, base_syn)
        p_syn = np.where(b == 0, base_syn, base_nat)
        d = compute_delta(D_toy, p_nat, p_syn, triu_r, triu_c)
        deltas.append(d)
    return np.array(deltas)


def run_validation() -> None:
    """Run embedded validation checks; raise AssertionError on failure."""
    rng_val = np.random.default_rng(42)

    # ── V1: swap-idempotence ─────────────────────────────────────────────────
    nat_v = rng_val.uniform(0, 1, (6, 16))
    syn_v = rng_val.uniform(0, 1, (6, 16))
    for v in (nat_v, syn_v):
        v /= v.sum(axis=1, keepdims=True)
    vecs = np.vstack([nat_v, syn_v])
    D_v  = cdist(vecs, vecs, "cityblock")
    n_v  = 6
    triu_r, triu_c = np.triu_indices(n_v, k=1)
    base_nat = np.arange(n_v)
    base_syn = np.arange(n_v, 2 * n_v)

    delta_orig = compute_delta(D_v, base_nat, base_syn, triu_r, triu_c)

    # Swap all labels once
    b1 = np.ones(n_v, dtype=np.int64)
    p_nat1 = np.where(b1 == 0, base_nat, base_syn)
    p_syn1 = np.where(b1 == 0, base_syn, base_nat)
    delta_swap1 = compute_delta(D_v, p_nat1, p_syn1, triu_r, triu_c)

    # Swap all labels again (restores original)
    p_nat2 = np.where(b1 == 0, p_nat1, p_syn1)
    p_syn2 = np.where(b1 == 0, p_syn1, p_nat1)
    delta_swap2 = compute_delta(D_v, p_nat2, p_syn2, triu_r, triu_c)

    assert abs(delta_swap2 - delta_orig) < 1e-12, (
        f"V1 FAIL: double-swap delta {delta_swap2} != original {delta_orig}"
    )
    print(f"[validation] V1 swap-idempotence PASS "
          f"(delta_orig={delta_orig:.6f}, delta_swap2={delta_swap2:.6f})")

    # ── V2: toy exact vs Monte Carlo agreement ───────────────────────────────
    toy_nat = np.array([0.0, 0.4, 0.7, 0.2])
    toy_syn = np.array([0.5, 0.9, 0.1, 0.8])
    n_toy = len(toy_nat)
    exact_deltas = _toy_exact_deltas(toy_nat, toy_syn)
    delta_obs_toy = exact_deltas[0]  # mask=0 = original labels

    # Compute exact p-value
    p_exact = float((exact_deltas >= delta_obs_toy).sum()) / len(exact_deltas)

    # Monte Carlo estimate
    all_toy = np.concatenate([toy_nat, toy_syn])
    D_toy   = np.abs(all_toy[:, None] - all_toy[None, :])
    p_mc, _, _ = permutation_pvalue(D_toy, n_toy, B=50_000, seed=0,
                                    delta_obs=delta_obs_toy)
    # Allow 0.01 tolerance (Monte Carlo noise)
    assert abs(p_mc - p_exact) < 0.01, (
        f"V2 FAIL: exact p={p_exact:.4f} vs MC p={p_mc:.4f} (diff={abs(p_mc-p_exact):.4f})"
    )
    print(f"[validation] V2 toy exact vs MC PASS "
          f"(exact={p_exact:.4f}, MC={p_mc:.4f}, |diff|={abs(p_mc-p_exact):.4f})")

    print("[validation] All checks passed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    run_validation()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for domain, tags in DOMAIN_TAGS.items():
        for tag in tags:
            of = HARM / tag / f"{tag}.orig.concat.fa"
            sf = HARM / tag / f"{tag}.syn.concat.fa"
            if not (of.exists() and sf.exists()):
                print(f"[skip] {tag}: concat FASTAs missing", file=sys.stderr)
                continue

            nat_seqs = [s for s in iter_fasta(of) if len(s) >= (1 << K)]
            syn_seqs = [s for s in iter_fasta(sf) if len(s) >= (1 << K)]

            n = min(len(nat_seqs), len(syn_seqs))   # positional pairs
            if n < 2:
                print(f"[skip] {tag}: too few matched windows (n={n})",
                      file=sys.stderr)
                continue

            # Use all n matched pairs (no cap)
            nat_seqs = nat_seqs[:n]
            syn_seqs = syn_seqs[:n]

            # ── Compute FCGR vectors ────────────────────────────────────────
            print(f"[{domain}] {tag}: computing FCGR for {n} matched pairs ...",
                  flush=True)
            nat_vecs = np.array([fcgr_norm(s, K) for s in nat_seqs])
            syn_vecs = np.array([fcgr_norm(s, K) for s in syn_seqs])
            vecs = np.vstack([nat_vecs, syn_vecs])  # (2n, 4^K)

            # ── Full 2n×2n L1 distance matrix ──────────────────────────────
            D = cdist(vecs, vecs, "cityblock")

            # ── Observed delta ──────────────────────────────────────────────
            nat_idx = np.arange(n, dtype=np.int64)
            syn_idx = np.arange(n, 2 * n, dtype=np.int64)
            triu_r, triu_c = np.triu_indices(n, k=1)
            delta_obs = compute_delta(D, nat_idx, syn_idx, triu_r, triu_c)

            n_nn = len(triu_r)      # C(n, 2)
            n_sn = n * n            # n²

            med_nn = float(np.median(D[np.ix_(nat_idx, nat_idx)][triu_r, triu_c]))
            med_sn = float(np.median(D[np.ix_(syn_idx, nat_idx)].ravel()))
            ratio  = med_sn / med_nn if med_nn > 0 else float("nan")

            # ── Permutation p-value ─────────────────────────────────────────
            p_val, perm_type, n_perms = permutation_pvalue(
                D, n, B, SEED, delta_obs
            )

            print(f"  nat-nat={med_nn:.4f}  syn-nat={med_sn:.4f}  "
                  f"ratio={ratio:.3f}  Δ={delta_obs:.4f}  p={p_val:.4e}")

            rows.append({
                "species":             tag,
                "domain":              domain,
                "n_matched_windows":   n,
                "n_nat_nat_distances": n_nn,
                "n_syn_nat_distances": n_sn,
                "median_nat_nat":      round(med_nn, 8),
                "median_syn_nat":      round(med_sn, 8),
                "median_ratio":        round(ratio, 8),
                "delta_median":        round(delta_obs, 8),
                "permutation_type":    perm_type,
                "n_permutations":      n_perms,
                "p_value":             p_val,
                "q_value":             np.nan,    # filled after BH below
                "significant":         False,
            })

    if not rows:
        print("[error] no results produced", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    # ── BH-FDR across all species ───────────────────────────────────────────
    df["q_value"] = bh_fdr(df["p_value"].values)
    df["significant"] = df["q_value"] < Q_THRESH

    # ── Write output ────────────────────────────────────────────────────────
    out_path = OUT / "natnat_fcgr_permutation_results.csv"
    df[[
        "species", "domain", "n_matched_windows",
        "n_nat_nat_distances", "n_syn_nat_distances",
        "median_nat_nat", "median_syn_nat", "median_ratio",
        "delta_median", "permutation_type", "n_permutations",
        "p_value", "q_value", "significant",
    ]].to_csv(out_path, index=False)

    # ── Console summary ─────────────────────────────────────────────────────
    n_sig   = int(df["significant"].sum())
    n_total = len(df)
    print(f"\n[FCGR permutation summary]")
    print(f"  Species tested:       {n_total}")
    print(f"  Significant (q<0.05): {n_sig} / {n_total} "
          f"({100 * n_sig / n_total:.0f}%)")
    print(f"  Median ratio (all):   "
          f"{df['median_ratio'].median():.3f}")
    print(f"\n[ok] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
