#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
species_significance.py

Single-species significance from per-window ks_stat values.

Given a metrics CSV produced by kmer_spectra.py (pair mode), compute:
- n_windows
- mean_ks
- median_ks
- one-sided p-value for median(ks_stat) > DELTA
  (Wilcoxon signed-rank on ks_stat - DELTA; fallback: exact binomial sign test)

Outputs:
- JSON summary
- optional CSV summary row

Example:
  python species_significance.py \
    --metrics results/Human/pair/metrics.kauto.csv \
    --tag Human \
    --delta 0.01 \
    --out-json results/Human/significance/Human.significance.json \
    --out-csv  results/Human/significance/Human.significance.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd


def try_wilcoxon_one_sided_greater(x: np.ndarray, delta: float) -> Optional[float]:
    """
    One-sided Wilcoxon signed-rank test on (x - delta), testing median(x) > delta.
    Returns p-value, or None if SciPy unavailable / fails.
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


def exact_binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """Exact survival function P(X >= k) for Binomial(n, p), computed in log-space."""
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


def sign_test_one_sided_greater(x: np.ndarray, delta: float) -> float:
    """
    One-sided sign test for median(x) > delta:
    Count observations > delta, ignore ties.
    Under H0, P(>delta) = 0.5.
    """
    gt = int(np.sum(x > delta))
    lt = int(np.sum(x < delta))
    n = gt + lt
    if n == 0:
        return 1.0
    return exact_binom_sf(gt, n, 0.5)


def species_p_value(x: np.ndarray, delta: float) -> tuple[float, str]:
    """Prefer Wilcoxon; fallback to exact sign test."""
    p = try_wilcoxon_one_sided_greater(x, delta)
    if p is not None:
        return p, "wilcoxon_one_sided_greater"
    return sign_test_one_sided_greater(x, delta), "sign_test_one_sided_greater"


def parse_args():
    ap = argparse.ArgumentParser(description="Single-species significance from ks_stat replicates.")
    ap.add_argument("--metrics", required=True, help="CSV containing column ks_stat (from kmer_spectra.py pair).")
    ap.add_argument("--tag", default=None, help="Species tag/name for output metadata (e.g., Human).")
    ap.add_argument("--delta", type=float, default=0.01, help="Practical-zero threshold for ks_stat.")
    ap.add_argument("--out-json", required=True, help="Output JSON path.")
    ap.add_argument("--out-csv", default=None, help="Optional output CSV path (one-row summary).")
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.metrics)
    if "ks_stat" not in df.columns:
        raise SystemExit(f"Missing 'ks_stat' in {args.metrics}")

    ks = pd.to_numeric(df["ks_stat"], errors="coerce").dropna().to_numpy(dtype=float)
    if ks.size == 0:
        raise SystemExit(f"No valid ks_stat values in {args.metrics}")

    p, method = species_p_value(ks, args.delta)

    summary = {
        "tag": args.tag,
        "metrics_csv": os.path.abspath(args.metrics),
        "n_windows": int(ks.size),
        "mean_ks": float(np.mean(ks)),
        "median_ks": float(np.median(ks)),
        "delta_used": float(args.delta),
        "p_value": float(p),
        "test": method,
        "notes": "One-sided test of median(ks_stat) > delta. ks_stat is a KS-like distance on Chor-normalized PMF-CDF.",
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote JSON: {args.out_json}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out = pd.DataFrame([{
            "tag": summary["tag"],
            "n_windows": summary["n_windows"],
            "mean_ks": summary["mean_ks"],
            "median_ks": summary["median_ks"],
            "delta_used": summary["delta_used"],
            "p_value": summary["p_value"],
            "test": summary["test"],
            "metrics_csv": summary["metrics_csv"],
        }])
        out.to_csv(args.out_csv, index=False)
        print(f"[OK] wrote CSV:  {args.out_csv}")

    print(f"n_windows={summary['n_windows']} mean_ks={summary['mean_ks']:.6g} "
          f"median_ks={summary['median_ks']:.6g} p={summary['p_value']:.6g} ({summary['test']})")


if __name__ == "__main__":
    main()
