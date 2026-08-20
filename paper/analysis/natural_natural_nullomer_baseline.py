#!/usr/bin/env python3
"""
Reviewer #2 (major #2), part 2: natural-natural baseline for NULLOMERS.

Mirrors natural_natural_fcgr_baseline.py but for canonical nullomer fractions.
For each species/tag we compute the per-window canonical nullomer fraction
(k=9 by default, denominator = number of canonical k-mer classes -- the R2.3
fix) for natural ("orig") and synthetic ("syn") harmonized windows, then test
whether the synthetic per-window distribution is shifted relative to the
natural window-to-window distribution (the intrinsic natural variability).

A two-sided Mann-Whitney U per species, BH-FDR across species. We report the
natural median and IQR so reviewers can see whether the synthetic median falls
outside the natural variability band.

Outputs:
  revisions/results/natnat_nullomer_per_window_k<K>.csv
  revisions/results/natnat_nullomer_summary_k<K>.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJ = Path(_ROOT)
HARM = PROJ / "results" / "harmonized"
RESULTS = PROJ / "revisions" / "results"

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

_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i


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


def canonical_classes(k: int) -> int:
    return (4 ** k + (4 ** (k // 2) if k % 2 == 0 else 0)) // 2


def canonical_nullomer_fraction(seq: str, k: int) -> float:
    codes = _CODE[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    if len(codes) < k:
        return float("nan")
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    win = win[(win >= 0).all(axis=1)]
    if win.size == 0:
        return float("nan")
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    idx = (win * powers).sum(axis=1)
    rc = (3 - win)[:, ::-1]
    ridx = (rc * powers).sum(axis=1)
    canon = np.minimum(idx, ridx)
    observed = np.unique(canon).size
    return 1.0 - observed / canonical_classes(k)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=["euk", "vir"])
    ap.add_argument("--k", type=int, default=9)
    args = ap.parse_args()

    per_rows = []
    summ_rows = []
    for dom in args.domains:
        for tag in DOMAIN_TAGS.get(dom, []):
            o = HARM / tag / f"{tag}.orig.concat.fa"
            s = HARM / tag / f"{tag}.syn.concat.fa"
            if not (o.exists() and s.exists()):
                print(f"[skip] {tag}: missing concat", flush=True)
                continue
            nat = [canonical_nullomer_fraction(x, args.k) for x in iter_fasta(o)]
            syn = [canonical_nullomer_fraction(x, args.k) for x in iter_fasta(s)]
            nat = np.array([v for v in nat if v == v])
            syn = np.array([v for v in syn if v == v])
            if len(nat) < 3 or len(syn) < 3:
                print(f"[skip] {tag}: too few windows", flush=True)
                continue
            for v in nat:
                per_rows.append({"domain": dom, "tag": tag, "group": "natural", "nullfrac": v})
            for v in syn:
                per_rows.append({"domain": dom, "tag": tag, "group": "synthetic", "nullfrac": v})
            try:
                u, p = mannwhitneyu(syn, nat, alternative="two-sided")
            except ValueError:
                u, p = float("nan"), float("nan")
            q1, q3 = np.percentile(nat, [25, 75])
            syn_med = float(np.median(syn))
            outside = bool(syn_med < q1 or syn_med > q3)
            summ_rows.append({
                "domain": dom, "tag": tag,
                "n_nat": len(nat), "n_syn": len(syn),
                "nat_median": float(np.median(nat)),
                "nat_iqr_lo": float(q1), "nat_iqr_hi": float(q3),
                "syn_median": syn_med,
                "delta_syn_minus_nat": syn_med - float(np.median(nat)),
                "syn_median_outside_nat_iqr": outside,
                "mwu_p": float(p),
            })
            print(f"[ok] {tag}: nat_med={np.median(nat):.3f} syn_med={syn_med:.3f} "
                  f"p={p:.2e} outside_IQR={outside}", flush=True)

    if not summ_rows:
        print("[error] nothing computed")
        return 1
    summ = pd.DataFrame(summ_rows)
    summ["mwu_q"] = bh_fdr(summ["mwu_p"].values)
    per = pd.DataFrame(per_rows)
    per_p = RESULTS / f"natnat_nullomer_per_window_k{args.k}.csv"
    sum_p = RESULTS / f"natnat_nullomer_summary_k{args.k}.csv"
    per.to_csv(per_p, index=False)
    summ.to_csv(sum_p, index=False)
    print(f"\n[done] per-window -> {per_p}")
    print(f"[done] summary    -> {sum_p}")
    n_sig = int((summ["mwu_q"] < 0.05).sum())
    n_out = int(summ["syn_median_outside_nat_iqr"].sum())
    print(f"[result] {n_sig}/{len(summ)} species: syn nullomer distribution "
          f"differs from natural (q<0.05); {n_out}/{len(summ)} have syn median "
          f"outside the natural IQR.")
    print("\n" + summ[["tag", "nat_median", "nat_iqr_lo", "nat_iqr_hi",
                        "syn_median", "syn_median_outside_nat_iqr", "mwu_q"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
