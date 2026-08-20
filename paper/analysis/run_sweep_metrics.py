#!/usr/bin/env python3
"""
Metrics for the alternative-decoding sweep (R3.7).

For every generated window under
    revisions/decoding_sweep/generated/<config>/<window_id>.<config>.syn.fa
we compare the synthetic sequence against its natural reference
    revisions/decoding_sweep/seeds/<window_id>.natref.fa
using the same compositional metrics the paper relies on:

  * FCGR L1 (k=8)            : syn-vs-natref chaos-game distance
  * k-mer JSD (k=6)          : Jensen-Shannon divergence of k-mer spectra
  * canonical nullomer frac  : k=9 canonical nullomer fraction for syn and natref

The question the reviewer poses (R3.7) is whether the failures the paper reports
PERSIST under lower-temperature and nucleus (top-p) decoding. We therefore also
load the intrinsic natural-natural FCGR L1 band (revisions/results/
natnat_fcgr_summary_k8.csv) so each config's syn-nat distance can be read
against natural variability.

Outputs:
  revisions/results/sweep_metrics_per_window.csv
  revisions/results/sweep_metrics_summary.csv

CPU-only; run with system python3 from /tmp.
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(_ROOT)
SWEEP = PROJ / "revisions" / "decoding_sweep"
RESULTS = PROJ / "revisions" / "results"

BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
_BX = np.full(256, -1, dtype=np.int64)
_BY = np.full(256, -1, dtype=np.int64)
for _ch, (_bx, _by) in BITS.items():
    _BX[ord(_ch)] = _bx
    _BY[ord(_ch)] = _by

# base -> 2-bit code for k-mer indexing
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i


def read_fasta_seq(path: Path) -> str:
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
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def canonical_classes(k: int) -> int:
    return (4 ** k + (4 ** (k // 2) if k % 2 == 0 else 0)) // 2


def canonical_nullomer_fraction(seq: str, k: int) -> float:
    """Fraction of canonical k-mer classes absent from the sequence."""
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
    idx = (win * powers).sum(axis=1)
    # reverse complement index: complement = 3 - code, then reverse order
    comp = 3 - win
    rc = comp[:, ::-1]
    ridx = (rc * powers).sum(axis=1)
    canon = np.minimum(idx, ridx)
    observed = np.unique(canon).size
    total = canonical_classes(k)
    return 1.0 - observed / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fcgr-k", type=int, default=8)
    ap.add_argument("--kmer-k", type=int, default=6)
    ap.add_argument("--null-k", type=int, default=9)
    ap.add_argument("--generated", default=str(SWEEP / "generated"))
    ap.add_argument("--seeds", default=str(SWEEP / "seeds"))
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Limit to these config subdirs (default: all found).")
    args = ap.parse_args()

    gen_root = Path(args.generated)
    seeds = Path(args.seeds)
    if not gen_root.is_dir():
        print(f"[error] no generated dir yet: {gen_root}")
        return 1

    configs = args.configs or sorted(p.name for p in gen_root.iterdir() if p.is_dir())
    rows = []
    natref_cache: dict[str, str] = {}
    for cfg in configs:
        cdir = gen_root / cfg
        for fa in sorted(cdir.glob(f"*.{cfg}.syn.fa")):
            wid = fa.name[: -len(f".{cfg}.syn.fa")]
            natref_p = seeds / f"{wid}.natref.fa"
            if not natref_p.exists():
                print(f"[warn] no natref for {wid}; skipping")
                continue
            syn = read_fasta_seq(fa)
            if wid not in natref_cache:
                natref_cache[wid] = read_fasta_seq(natref_p)
            nat = natref_cache[wid]
            l1 = float(np.abs(fcgr_norm(syn, args.fcgr_k) -
                              fcgr_norm(nat, args.fcgr_k)).sum())
            kj = jsd(kmer_freq(syn, args.kmer_k), kmer_freq(nat, args.kmer_k))
            null_syn = canonical_nullomer_fraction(syn, args.null_k)
            null_nat = canonical_nullomer_fraction(nat, args.null_k)
            tag = wid.rsplit(".w", 1)[0]
            rows.append({
                "config": cfg, "tag": tag, "window_id": wid,
                "syn_len": len(syn), "nat_len": len(nat),
                f"fcgr_l1_k{args.fcgr_k}": l1,
                f"kmer_jsd_k{args.kmer_k}": kj,
                f"nullfrac_syn_k{args.null_k}": null_syn,
                f"nullfrac_nat_k{args.null_k}": null_nat,
                f"nullfrac_delta_k{args.null_k}": null_syn - null_nat,
            })
            print(f"[ok] {cfg}/{wid}: FCGR_L1={l1:.4f} kmerJSD={kj:.4f} "
                  f"null_syn={null_syn:.3f} null_nat={null_nat:.3f}", flush=True)

    if not rows:
        print("[error] no generated sequences found to score.")
        return 1
    df = pd.DataFrame(rows)
    per_window = RESULTS / "sweep_metrics_per_window.csv"
    df.to_csv(per_window, index=False)

    num_cols = [c for c in df.columns if c.startswith(("fcgr", "kmer", "null"))]
    summ = (df.groupby(["config", "tag"])[num_cols]
              .median().reset_index())
    summary = RESULTS / "sweep_metrics_summary.csv"
    summ.to_csv(summary, index=False)

    # attach natural-natural FCGR band for context if available
    natnat = RESULTS / f"natnat_fcgr_summary_k{args.fcgr_k}.csv"
    band = ""
    if natnat.exists():
        band = f" (natural-natural band: {natnat})"
    print(f"\n[done] per-window -> {per_window}")
    print(f"[done] summary    -> {summary}{band}")
    print("\n=== median FCGR L1 by config x tag ===")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
