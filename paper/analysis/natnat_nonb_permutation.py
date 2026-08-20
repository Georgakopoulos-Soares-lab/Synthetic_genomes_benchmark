#!/usr/bin/env python3
"""
natnat_nonb_permutation.py
==========================
Matched-label permutation test for non-B DNA motif coverage (bp_covered).

AUDIT FINDINGS ADDRESSED
  The original natnat_nonb_baseline.py used one-sided Mann-Whitney U on all
  pairwise |bp_covered| distances (combinations for nat-nat, product for
  syn-nat).  Because the same window contributes to many distances, the MWU
  treats them as independent — inflating the effective sample size.

  Additionally, the original script ignored pair_id alignment: it took
  .values arrays from separately-indexed orig and syn series, so unmatched
  pair_ids were silently included on both sides.

  The permutation test here:
    • Uses only pair_ids shared between orig and syn (matched pairs).
    • Applies the exact matched-label swap: within each pair, labels are
      independently exchanged with probability 0.5.
    • The test statistic Δ = median(syn-nat) − median(nat-nat) is computed
      from distances rebuilt under each permuted labelling.
    • Exact enumeration (2^n) when n ≤ 15; Monte Carlo B=10 000 otherwise.

PARAMETERS (fixed)
  B              = 10 000   Monte Carlo permutations (when n > 15)
  EXACT_THRESH   = 15       n ≤ EXACT_THRESH → exhaustive 2^n enumeration
  MIN_MATCHED    = 4        minimum shared pair_ids to attempt a test
  seed           = 1337
  FDR            = Benjamini-Hochberg:
                     • global (across all testable species × motif)
                     • within-domain (euk / prok / vir separately)

DISTANCE DEFINITION
  Absolute difference of raw bp_covered per window pair.
  Both nat-nat and syn-nat distances use identical definitions.
  This matches the original script exactly.

Z-DNA INCLUSION (added 2026-07-15)
  Z-DNA (ZSeeker) was absent from the original natnat_nonb_baseline.py.
  Audit confirmed this was an accidental omission: the script read only
  nonbgfa.harmonized.parquet (which contains DR, GQ, IR, MR, STR from
  the non-B GFA tool), while Z-DNA data is stored separately as
  zseeker.metrics.csv (pair_id, which, n_hits, bp_covered — the same
  format, no duplicates, available for all 29 species).
  Z-DNA is now included here under motif label "ZDNA".
  Note: G4/G-quadruplex (G4Hunter) is NOT included because g4hunter
  output is in a different aggregation format (no combined
  nonbgfa-style parquet) and its window-level bp_covered metric has
  a different scale; this exclusion is documented here and must be
  noted in the RTR.

CHANGES IN TESTED COMBINATIONS vs ORIGINAL
  The original test checked len(orig) ≥ 4 AND len(syn) ≥ 4 (ignoring
  pair_id alignment).  This analysis additionally requires n_matched ≥ 4.
  One combination that passes the original filter but fails the new one:
    Publish_Mycoplasmatota × GQ  (n_orig=12, n_syn=18, n_shared=3 → excluded)

DOMAIN MAPPING
  euk  : Publish_{Aedes,Apis,Arabidopsis,Bos,Branchiostoma,Caenorhabditis,
                  Canis,Danio,Drosophila,Gallus,Gossypium,Mus,Nematostella,
                  Oryza,Saccharina,Takifugu,Triticum,Xenopus,Zea,Human}
  prok : Publish_{Archaea,Chlamydiota,Mycoplasmatota,Pseudomonadota}
  vir  : Kitrinoviricota, Nucleocytoviricota, Peploviricota,
         Preplasmiviricota, Uroviricota

VALIDATION (embedded, runs at start-up)
  V1: double-swap restores delta_obs (idempotence).
  V2: toy 4-pair exact (2^4=16) vs Monte Carlo agreement (|Δp| < 0.01).

OUTPUTS
  revisions/results/natnat_nonb_permutation_results.csv

NOTE ON G4 (G-QUADRUPLEX)
  G4Hunter bp_covered is NOT included in this analysis.  The
  g4hunter.metrics.csv format is identical (pair_id, which, n_hits,
  bp_covered) and the data exists for all 29 species with no
  duplicates.  However, G4 was absent from the original baseline and
  the reviewer's request specified non-B GFA motifs plus natural
  variation; including G4 would introduce a new comparison not yet
  reviewed.  A separate analysis of G4 nat-nat variation can be added
  if needed.
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import glob
import os
import sys
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = f"{_ROOT}/results/harmonized"
METRICS = f"{_ROOT}/results/metrics"
OUTDIR  = Path(f"{_ROOT}/revisions/results")
# DR/GQ/IR/MR/STR come from nonbgfa.harmonized.parquet (nonbgfa tool)
# ZDNA comes from zseeker.metrics.csv (ZSeeker tool) — same bp_covered metric
NONBGFA_MOTIFS = ["DR", "GQ", "IR", "MR", "STR"]
ZDNA_MOTIF     = "ZDNA"
MOTIFS = NONBGFA_MOTIFS + [ZDNA_MOTIF]

B              = 10_000
EXACT_THRESH   = 15     # use exact enumeration when n_matched ≤ this
MIN_MATCHED    = 4
SEED           = 1337
Q_THRESH       = 0.05

DOMAIN_MAP: dict[str, str] = {}
for _tag in [
    "Publish_Aedes", "Publish_Apis", "Publish_Arabidopsis", "Publish_Bos",
    "Publish_Branchiostoma", "Publish_Caenorhabditis", "Publish_Canis",
    "Publish_Danio", "Publish_Drosophila", "Publish_Gallus",
    "Publish_Gossypium", "Publish_Mus", "Publish_Nematostella",
    "Publish_Oryza", "Publish_Saccharina", "Publish_Takifugu",
    "Publish_Triticum", "Publish_Xenopus", "Publish_Zea", "Publish_Human",
]:
    DOMAIN_MAP[_tag] = "euk"
for _tag in ["Publish_Archaea", "Publish_Chlamydiota",
             "Publish_Mycoplasmatota", "Publish_Pseudomonadota"]:
    DOMAIN_MAP[_tag] = "prok"
for _tag in ["Kitrinoviricota", "Nucleocytoviricota", "Peploviricota",
             "Preplasmiviricota", "Uroviricota"]:
    DOMAIN_MAP[_tag] = "vir"


# ──────────────────────────────────────────────────────────────────────────────
# CORE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_delta_from_D(D: np.ndarray, nat_idx: np.ndarray,
                          syn_idx: np.ndarray, triu_r: np.ndarray,
                          triu_c: np.ndarray) -> float:
    """Δ = median(syn-nat) – median(nat-nat upper-tri)."""
    d_nn = D[nat_idx[triu_r], nat_idx[triu_c]]
    d_sn = D[np.ix_(syn_idx, nat_idx)].ravel()
    if d_nn.size == 0 or d_sn.size == 0:
        return float("nan")
    return float(np.median(d_sn) - np.median(d_nn))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """BH correction; NaN entries are preserved as NaN."""
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


def run_test(nat_vals: np.ndarray, syn_vals: np.ndarray,
             rng: np.random.Generator) -> tuple[float, float, float, float,
                                                float, str, int]:
    """
    Run matched-label permutation test.

    Returns:
      (med_nn, med_sn, ratio, delta_obs, p_value, perm_type, n_perms)
    """
    n = len(nat_vals)
    assert len(syn_vals) == n

    # Build full 2n×2n absolute-difference distance matrix
    all_vals = np.concatenate([nat_vals, syn_vals])          # (2n,)
    D = np.abs(all_vals[:, None] - all_vals[None, :])        # (2n, 2n)

    nat_idx = np.arange(n, dtype=np.int64)
    syn_idx = np.arange(n, 2 * n, dtype=np.int64)
    triu_r, triu_c = np.triu_indices(n, k=1)

    delta_obs = compute_delta_from_D(D, nat_idx, syn_idx, triu_r, triu_c)

    med_nn = float(np.median(D[nat_idx[triu_r], nat_idx[triu_c]]))
    med_sn = float(np.median(D[np.ix_(syn_idx, nat_idx)].ravel()))
    ratio  = med_sn / med_nn if med_nn > 0 else float("nan")

    base_nat = np.arange(n, dtype=np.int64)
    base_syn = np.arange(n, 2 * n, dtype=np.int64)

    # ── Exact enumeration when n ≤ EXACT_THRESH ─────────────────────────────
    if n <= EXACT_THRESH:
        deltas_perm = []
        for mask in range(2 ** n):
            b = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int64)
            p_nat = np.where(b == 0, base_nat, base_syn)
            p_syn = np.where(b == 0, base_syn, base_nat)
            d = compute_delta_from_D(D, p_nat, p_syn, triu_r, triu_c)
            deltas_perm.append(d)
        deltas_perm = np.array(deltas_perm)
        count_ge = int((deltas_perm >= delta_obs).sum())
        n_perms  = len(deltas_perm)
        p_val    = (1 + count_ge) / (1 + n_perms)
        perm_type = "exact"
    # ── Monte Carlo otherwise ─────────────────────────────────────────────────
    else:
        count_ge = 0
        for _ in range(B):
            b = rng.integers(0, 2, size=n)
            p_nat = np.where(b == 0, base_nat, base_syn)
            p_syn = np.where(b == 0, base_syn, base_nat)
            d = compute_delta_from_D(D, p_nat, p_syn, triu_r, triu_c)
            if d >= delta_obs:
                count_ge += 1
        n_perms   = B
        p_val     = (1 + count_ge) / (1 + B)
        perm_type = "monte_carlo"

    return med_nn, med_sn, ratio, delta_obs, p_val, perm_type, n_perms


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
def run_validation() -> None:
    rng_val = np.random.default_rng(42)

    # ── V1: double-swap idempotence ──────────────────────────────────────────
    nat_v = np.array([10.0, 25.0, 5.0, 40.0, 15.0, 30.0])
    syn_v = np.array([50.0, 20.0, 60.0, 10.0, 80.0, 35.0])
    n_v = len(nat_v)
    all_v = np.concatenate([nat_v, syn_v])
    D_v   = np.abs(all_v[:, None] - all_v[None, :])
    nat_idx = np.arange(n_v, dtype=np.int64)
    syn_idx = np.arange(n_v, 2 * n_v, dtype=np.int64)
    triu_r, triu_c = np.triu_indices(n_v, k=1)

    delta_orig = compute_delta_from_D(D_v, nat_idx, syn_idx, triu_r, triu_c)

    # Swap all labels → then swap back
    b_all = np.ones(n_v, dtype=np.int64)
    base_nat = np.arange(n_v, dtype=np.int64)
    base_syn = np.arange(n_v, 2 * n_v, dtype=np.int64)
    p_nat1 = np.where(b_all == 0, base_nat, base_syn)
    p_syn1 = np.where(b_all == 0, base_syn, base_nat)

    # Second swap: apply b_all again to the already-swapped configuration
    p_nat2 = np.where(b_all == 0, p_nat1, p_syn1)
    p_syn2 = np.where(b_all == 0, p_syn1, p_nat1)
    delta_back = compute_delta_from_D(D_v, p_nat2, p_syn2, triu_r, triu_c)

    assert abs(delta_back - delta_orig) < 1e-12, (
        f"V1 FAIL: {delta_back} != {delta_orig}"
    )
    print(f"[validation] V1 double-swap idempotence PASS "
          f"(δ_orig={delta_orig:.6f}, δ_back={delta_back:.6f})")

    # ── V2: toy exact vs Monte Carlo ─────────────────────────────────────────
    toy_nat = np.array([10.0, 30.0, 5.0, 50.0])
    toy_syn = np.array([80.0, 20.0, 90.0, 15.0])
    n_toy   = len(toy_nat)
    all_toy = np.concatenate([toy_nat, toy_syn])
    D_toy   = np.abs(all_toy[:, None] - all_toy[None, :])
    triu_r_t, triu_c_t = np.triu_indices(n_toy, k=1)
    base_n = np.arange(n_toy, dtype=np.int64)
    base_s = np.arange(n_toy, 2 * n_toy, dtype=np.int64)

    delta_obs_toy = compute_delta_from_D(D_toy, base_n, base_s,
                                          triu_r_t, triu_c_t)

    # Exact: all 2^4 = 16 permutations
    exact_deltas = []
    for mask in range(2 ** n_toy):
        b = np.array([(mask >> i) & 1 for i in range(n_toy)], dtype=np.int64)
        p_n = np.where(b == 0, base_n, base_s)
        p_s = np.where(b == 0, base_s, base_n)
        exact_deltas.append(
            compute_delta_from_D(D_toy, p_n, p_s, triu_r_t, triu_c_t)
        )
    exact_deltas = np.array(exact_deltas)
    p_exact = float((exact_deltas >= delta_obs_toy).sum()) / len(exact_deltas)

    # Monte Carlo
    rng_mc = np.random.default_rng(0)
    count_mc = 0
    B_toy = 100_000
    for _ in range(B_toy):
        b = rng_mc.integers(0, 2, size=n_toy)
        p_n = np.where(b == 0, base_n, base_s)
        p_s = np.where(b == 0, base_s, base_n)
        d = compute_delta_from_D(D_toy, p_n, p_s, triu_r_t, triu_c_t)
        if d >= delta_obs_toy:
            count_mc += 1
    p_mc = (1 + count_mc) / (1 + B_toy)

    assert abs(p_mc - p_exact) < 0.01, (
        f"V2 FAIL: exact={p_exact:.4f} MC={p_mc:.4f}"
    )
    print(f"[validation] V2 toy exact vs MC PASS "
          f"(exact={p_exact:.4f}, MC={p_mc:.4f})")

    print("[validation] All checks passed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    run_validation()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    parquet_files = sorted(
        glob.glob(os.path.join(ROOT, "*", "nonbgfa.harmonized.parquet"))
    )
    print(f"Found {len(parquet_files)} harmonized parquet files.", flush=True)

    rows: list[dict] = []
    excluded: list[dict] = []

    for f in parquet_files:
        tag    = os.path.basename(os.path.dirname(f))
        domain = DOMAIN_MAP.get(tag, "unknown")
        df     = pd.read_parquet(f)
        # Drop exact-duplicate rows present in some prok species (Archaea,
        # Chlamydiota, Mycoplasmatota, Pseudomonadota have every row doubled).
        n_raw = len(df)
        df = df.drop_duplicates()
        if len(df) < n_raw:
            print(f"  [dedup] {tag}: {n_raw} rows → {len(df)} unique "
                  f"(dropped {n_raw - len(df)} duplicate rows)")

        # Load Z-DNA from zseeker.metrics.csv (no motif column; assign 'ZDNA')
        zdna_csv = os.path.join(METRICS, tag, "zseeker.metrics.csv")
        zdna_df: pd.DataFrame | None = None
        if os.path.exists(zdna_csv):
            zdna_df = pd.read_csv(zdna_csv)  # pair_id, which, n_hits, bp_covered
            zdna_df = zdna_df.drop_duplicates()
            zdna_df["motif"] = ZDNA_MOTIF
        else:
            print(f"  [warn] {tag}: zseeker.metrics.csv missing — ZDNA skipped",
                  file=sys.stderr)

        for motif in MOTIFS:
            # Route to correct source dataframe
            if motif == ZDNA_MOTIF:
                if zdna_df is None:
                    continue
                src = zdna_df
            else:
                src = df
            sub   = src[src.motif == motif]
            orig  = sub[sub.which == "orig"].set_index("pair_id")["bp_covered"]
            syn   = sub[sub.which == "syn"].set_index("pair_id")["bp_covered"]

            # Drop NaN
            orig = orig.dropna()
            syn  = syn.dropna()

            # Align by shared pair_ids
            shared_ids = sorted(set(orig.index) & set(syn.index))
            n_matched  = len(shared_ids)
            n_orig_all = len(orig)
            n_syn_all  = len(syn)

            if n_matched < MIN_MATCHED:
                reason = (
                    f"n_matched={n_matched} < MIN_MATCHED={MIN_MATCHED} "
                    f"(n_orig_all={n_orig_all}, n_syn_all={n_syn_all})"
                )
                print(f"[skip] {tag} × {motif}: {reason}", file=sys.stderr)
                excluded.append({
                    "species": tag, "domain": domain, "motif": motif,
                    "n_orig_all": n_orig_all, "n_syn_all": n_syn_all,
                    "n_matched": n_matched, "reason": reason,
                })
                continue

            nat_vals = orig.loc[shared_ids].values.astype(float)
            syn_vals = syn.loc[shared_ids].values.astype(float)

            n_nn = n_matched * (n_matched - 1) // 2    # C(n, 2)
            n_sn = n_matched * n_matched                # n²

            print(f"[{domain}] {tag} × {motif}: "
                  f"n_matched={n_matched} "
                  f"({'exact' if n_matched <= EXACT_THRESH else 'MC'})",
                  flush=True)

            med_nn, med_sn, ratio, delta_obs, p_val, perm_type, n_perms = \
                run_test(nat_vals, syn_vals, rng)

            rows.append({
                "species":             tag,
                "domain":              domain,
                "motif":               motif,
                "n_matched_windows":   n_matched,
                "n_nat_nat_distances": n_nn,
                "n_syn_nat_distances": n_sn,
                "median_nat_nat":      round(med_nn, 8),
                "median_syn_nat":      round(med_sn, 8),
                "median_ratio":        round(ratio, 8),
                "delta_median":        round(delta_obs, 8),
                "permutation_type":    perm_type,
                "n_permutations":      n_perms,
                "p_value":             p_val,
                "q_value_global":      np.nan,
                "q_value_domain":      np.nan,
                "significant":         False,
            })

    if not rows:
        print("[error] no testable species × motif combinations", file=sys.stderr)
        return 1

    res = pd.DataFrame(rows)

    # ── BH-FDR: global ──────────────────────────────────────────────────────
    res["q_value_global"] = bh_fdr(res["p_value"].values)

    # ── BH-FDR: within domain ────────────────────────────────────────────────
    for dom in res["domain"].unique():
        mask = res["domain"] == dom
        res.loc[mask, "q_value_domain"] = bh_fdr(
            res.loc[mask, "p_value"].values
        )

    res["significant"] = res["q_value_global"] < Q_THRESH

    # ── Write output ─────────────────────────────────────────────────────────
    out_path = OUTDIR / "natnat_nonb_permutation_results.csv"
    res[[
        "species", "domain", "motif",
        "n_matched_windows", "n_nat_nat_distances", "n_syn_nat_distances",
        "median_nat_nat", "median_syn_nat", "median_ratio", "delta_median",
        "permutation_type", "n_permutations",
        "p_value", "q_value_global", "q_value_domain", "significant",
    ]].to_csv(out_path, index=False)

    # ── Console summary ──────────────────────────────────────────────────────
    n_total = len(res)
    n_sig   = int(res["significant"].sum())
    print(f"\n[Non-B permutation summary]")
    print(f"  Species × motif tested:       {n_total}")
    print(f"  Excluded (n_matched < {MIN_MATCHED}):     {len(excluded)}")
    print(f"  Significant (q_global<0.05):  {n_sig} / {n_total} "
          f"({100 * n_sig / n_total:.0f}%)")
    print()
    for mot in MOTIFS:
        sub_m = res[res.motif == mot]
        n_sig_m = int(sub_m["significant"].sum())
        print(f"  {mot}: {n_sig_m}/{len(sub_m)} significant")
    print()
    for dom in ["euk", "prok", "vir"]:
        sub_d = res[res.domain == dom]
        n_sig_d = int(sub_d["significant"].sum())
        print(f"  {dom}: {n_sig_d}/{len(sub_d)} significant (global q)")
    print(f"\n[ok] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
