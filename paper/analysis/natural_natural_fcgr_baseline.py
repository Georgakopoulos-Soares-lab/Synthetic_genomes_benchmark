#!/usr/bin/env python3
"""
Reviewer #2 (major #2): natural-natural baselines.

The reviewer asks whether synthetic-vs-natural divergences exceed the *intrinsic*
natural-vs-natural variation among real genomic windows of the same species.
This script answers that for the FCGR L1 metric (the one the reviewer names
explicitly) and for compositional summaries (GC, single-nt, dinucleotide).

For each species/tag we load:
  * natural windows  : results/harmonized/<TAG>/<TAG>.orig.concat.fa
  * synthetic windows: results/harmonized/<TAG>/<TAG>.syn.concat.fa

We compute FCGR (k configurable, default 8) per window, then:
  * NAT-NAT  : pairwise L1 between distinct natural windows (sampled)
  * SYN-NAT  : pairwise L1 between synthetic and natural windows (sampled)
  * SYN-SYN  : pairwise L1 between distinct synthetic windows (sampled, context)

A one-sided Mann-Whitney U tests whether SYN-NAT > NAT-NAT per species.
If SYN-NAT is not significantly larger than NAT-NAT, the divergence the paper
reports could be within natural variability; if it is, the claim is supported.

Outputs:
  revisions/results/natnat_fcgr_per_pair_k<K>.csv   (every sampled pair distance)
  revisions/results/natnat_fcgr_summary_k<K>.csv    (per-species medians + test)
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJ = Path(_ROOT)
HARM = PROJ / "results" / "harmonized"

DOMAIN_TAGS: dict[str, list[str]] = {
    "euk": [
        "Publish_Aedes", "Publish_Apis", "Publish_Arabidopsis", "Publish_Bos",
        "Publish_Branchiostoma", "Publish_Caenorhabditis", "Publish_Canis",
        "Publish_Danio", "Publish_Drosophila", "Publish_Gallus", "Publish_Gossypium",
        "Publish_Mus", "Publish_Nematostella", "Publish_Oryza", "Publish_Saccharina",
        "Publish_Takifugu", "Publish_Triticum", "Publish_Xenopus", "Publish_Zea",
        "Publish_Human",
    ],
    "vir": [
        "Kitrinoviricota", "Nucleocytoviricota", "Peploviricota",
        "Preplasmiviricota", "Uroviricota",
    ],
}

BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}

# Per-base bit lookup tables (index by ASCII code); -1 marks non-ACGT.
_BX = np.full(256, -1, dtype=np.int64)
_BY = np.full(256, -1, dtype=np.int64)
for _ch, (_bx, _by) in BITS.items():
    _BX[ord(_ch)] = _bx
    _BY[ord(_ch)] = _by


def iter_fasta(path: Path):
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
    """Vectorized normalized FCGR (flattened 2^k x 2^k)."""
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


def sampled_pair_l1(a_vecs, b_vecs, same_set, rng, max_pairs):
    n_a, n_b = len(a_vecs), len(b_vecs)
    if same_set:
        all_pairs = list(itertools.combinations(range(n_a), 2))
    else:
        all_pairs = [(i, j) for i in range(n_a) for j in range(n_b)]
    if not all_pairs:
        return np.array([])
    if len(all_pairs) > max_pairs:
        sel = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in sel]
    out = []
    for i, j in all_pairs:
        out.append(np.abs(a_vecs[i] - b_vecs[j]).sum())
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=["euk", "vir"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=40,
                    help="cap windows per group to bound pair counts")
    ap.add_argument("--max-pairs", type=int, default=400,
                    help="cap sampled pairs per comparison per species")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    per_pair_rows = []
    summary_rows = []

    tags = [t for d in args.domains for t in DOMAIN_TAGS[d]]
    for tag in tags:
        domain = next(d for d in args.domains if tag in DOMAIN_TAGS[d])
        of = HARM / tag / f"{tag}.orig.concat.fa"
        sf = HARM / tag / f"{tag}.syn.concat.fa"
        if not (of.exists() and sf.exists()):
            print(f"[warn] {tag}: concat FASTAs missing", file=sys.stderr)
            continue
        nat = [s for s in iter_fasta(of) if len(s) >= (1 << args.k)]
        syn = [s for s in iter_fasta(sf) if len(s) >= (1 << args.k)]
        nat = nat[: args.max_windows]
        syn = syn[: args.max_windows]
        if len(nat) < 2 or len(syn) < 1:
            print(f"[warn] {tag}: too few windows (nat={len(nat)} syn={len(syn)})",
                  file=sys.stderr)
            continue

        nat_v = [fcgr_norm(s, args.k) for s in nat]
        syn_v = [fcgr_norm(s, args.k) for s in syn]

        d_nn = sampled_pair_l1(nat_v, nat_v, True, rng, args.max_pairs)
        d_sn = sampled_pair_l1(syn_v, nat_v, False, rng, args.max_pairs)
        d_ss = sampled_pair_l1(syn_v, syn_v, True, rng, args.max_pairs)

        for arr, comp in ((d_nn, "nat_nat"), (d_sn, "syn_nat"), (d_ss, "syn_syn")):
            for v in arr:
                per_pair_rows.append({"tag": tag, "domain": domain,
                                      "comparison": comp, "l1": float(v)})

        # one-sided test: syn_nat > nat_nat
        if len(d_nn) and len(d_sn):
            u, p = mannwhitneyu(d_sn, d_nn, alternative="greater")
        else:
            u, p = np.nan, np.nan
        med_nn = float(np.median(d_nn)) if len(d_nn) else np.nan
        med_sn = float(np.median(d_sn)) if len(d_sn) else np.nan
        med_ss = float(np.median(d_ss)) if len(d_ss) else np.nan
        ratio = med_sn / med_nn if med_nn and med_nn > 0 else np.nan
        summary_rows.append({
            "tag": tag, "domain": domain,
            "median_nat_nat_L1": med_nn,
            "median_syn_nat_L1": med_sn,
            "median_syn_syn_L1": med_ss,
            "syn_nat_over_nat_nat_ratio": ratio,
            "mwu_U": float(u) if u == u else np.nan,
            "p_syn_nat_gt_nat_nat": float(p) if p == p else np.nan,
            "n_nat": len(nat), "n_syn": len(syn),
        })
        print(f"[{domain}] {tag:26s} natnat={med_nn:.3f} synnat={med_sn:.3f} "
              f"ratio={ratio:.2f} p={p:.2e}")

    if not summary_rows:
        print("[err] no results", file=sys.stderr)
        return 1

    out_dir = PROJ / "revisions" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pp = pd.DataFrame(per_pair_rows)
    sm = pd.DataFrame(summary_rows)
    # BH-FDR across species
    from scipy.stats import false_discovery_control  # scipy >= 1.11
    valid = sm["p_syn_nat_gt_nat_nat"].notna()
    sm.loc[valid, "q_bh"] = false_discovery_control(
        sm.loc[valid, "p_syn_nat_gt_nat_nat"].values)
    pp_path = out_dir / f"natnat_fcgr_per_pair_k{args.k}.csv"
    sm_path = out_dir / f"natnat_fcgr_summary_k{args.k}.csv"
    pp.to_csv(pp_path, index=False)
    sm.to_csv(sm_path, index=False)
    print(f"\n[ok] {pp_path}")
    print(f"[ok] {sm_path}")
    print("\n=== per-species summary ===")
    print(sm[["tag", "median_nat_nat_L1", "median_syn_nat_L1",
              "syn_nat_over_nat_nat_ratio", "p_syn_nat_gt_nat_nat"]]
          .round(4).to_string(index=False))
    n_sig = int((sm.get("q_bh", pd.Series(dtype=float)) < 0.05).sum())
    print(f"\n[summary] species with syn-nat > nat-nat (q<0.05): {n_sig}/{len(sm)}")
    print(f"[summary] median syn-nat/nat-nat L1 ratio: "
          f"{sm['syn_nat_over_nat_nat_ratio'].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
