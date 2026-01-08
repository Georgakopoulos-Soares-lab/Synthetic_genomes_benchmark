#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact


def read_fimo_hits(fimo_tsv: Path, site_p: float, site_q: float) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Return (n_hits, motif_id, motif_alt_id).
    Accepts fimo.tsv or fimo.tsv.gz.
    """
    if not fimo_tsv.exists():
        return 0, None, None

    try:
        df = pd.read_csv(fimo_tsv, sep="\t", comment="#", compression="infer")
    except pd.errors.EmptyDataError:
        return 0, None, None

    if df.empty:
        return 0, None, None

    need = {"p-value", "q-value", "motif_id", "motif_alt_id"}
    if not need.issubset(df.columns):
        return 0, None, None

    df_f = df[(df["p-value"] < site_p) & (df["q-value"] < site_q)]
    n = int(len(df_f))

    if n > 0:
        return n, str(df_f["motif_id"].iloc[0]), str(df_f["motif_alt_id"].iloc[0])
    return 0, str(df["motif_id"].iloc[0]), str(df["motif_alt_id"].iloc[0])


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]

    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev

    qvals = np.empty(n, dtype=float)
    qvals[order] = np.clip(q, 0.0, 1.0)
    return qvals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate FIMO outputs into per-motif counts + Fisher test + volcano.")
    p.add_argument("--fimo-root", required=True, help="Directory containing pair_*/orig/<motif>/fimo.tsv(.gz) and syn/...")
    p.add_argument("--out-csv", required=True, help="Output CSV path.")
    p.add_argument("--out-png", required=True, help="Output volcano PNG path.")
    p.add_argument("--site-p", type=float, default=1e-4, help="Site-level p-value threshold (default: 1e-4).")
    p.add_argument("--site-q", type=float, default=1.0, help="Site-level q-value threshold (default: 1).")
    p.add_argument("--volcano-p", type=float, default=0.05, help="Volcano significance line (default: 0.05).")
    p.add_argument("--volcano-log2fc", type=float, default=1.0, help="Volcano |log2FC| line (default: 1.0).")
    p.add_argument("--top-n", type=int, default=10, help="Top N motifs in legend (by p-value).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    base = Path(args.fimo_root)

    pair_dirs = sorted([p for p in base.glob("pair_*") if p.is_dir()])
    if not pair_dirs:
        raise SystemExit(f"No pair_* directories under {base.resolve()}")

    motif_stats = {}
    total_orig = 0
    total_syn = 0

    for pair_dir in pair_dirs:
        orig_dir = pair_dir / "orig"
        syn_dir = pair_dir / "syn"
        if not orig_dir.is_dir() or not syn_dir.is_dir():
            continue

        for motif_dir in sorted([p for p in orig_dir.iterdir() if p.is_dir()]):
            motif_name = motif_dir.name

            fimo_o = motif_dir / "fimo.tsv"
            if not fimo_o.exists():
                fimo_o = motif_dir / "fimo.tsv.gz"

            motif_dir_s = syn_dir / motif_name
            fimo_s = motif_dir_s / "fimo.tsv"
            if not fimo_s.exists():
                fimo_s = motif_dir_s / "fimo.tsv.gz"

            h_o, mid_o, malt_o = read_fimo_hits(fimo_o, args.site_p, args.site_q)
            h_s, mid_s, malt_s = read_fimo_hits(fimo_s, args.site_p, args.site_q)

            motif_id = mid_o or mid_s
            motif_alt = malt_o or malt_s
            motif_key = motif_alt or motif_id or motif_name

            if motif_key not in motif_stats:
                motif_stats[motif_key] = {"motif_id": motif_id, "motif_alt_id": motif_alt, "orig_hits": 0, "syn_hits": 0}

            motif_stats[motif_key]["orig_hits"] += h_o
            motif_stats[motif_key]["syn_hits"] += h_s
            total_orig += h_o
            total_syn += h_s

    rows = []
    for motif_key, d in motif_stats.items():
        h_o = int(d["orig_hits"])
        h_s = int(d["syn_hits"])
        if h_o + h_s == 0:
            continue

        other_o = max(total_orig - h_o, 0)
        other_s = max(total_syn - h_s, 0)

        table = [[h_o, other_o], [h_s, other_s]]
        _, p = fisher_exact(table)

        p = float(p) if p and p > 0 else 1e-300
        log2fc = math.log((h_o + 0.5) / (h_s + 0.5), 2)
        neglog10p = -math.log10(p)

        rows.append({
            "motif_key": motif_key,
            "motif_id": d["motif_id"],
            "motif_alt_id": d["motif_alt_id"],
            "hits_orig": h_o,
            "hits_syn": h_s,
            "total_hits_orig_all_motifs": total_orig,
            "total_hits_syn_all_motifs": total_syn,
            "log2FC_orig_over_syn": log2fc,
            "pvalue_fisher": p,
            "neglog10_pvalue": neglog10p,
        })

    if not rows:
        raise SystemExit("No motif hits survived filtering (check thresholds or inputs).")

    df = pd.DataFrame(rows).sort_values("pvalue_fisher").reset_index(drop=True)
    df["p_fdr"] = benjamini_hochberg(df["pvalue_fisher"].to_numpy(dtype=float))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    x = df["log2FC_orig_over_syn"].to_numpy()
    y = df["neglog10_pvalue"].to_numpy()
    pvals = df["pvalue_fisher"].to_numpy()

    colors = []
    for xi, pv in zip(x, pvals):
        if pv < args.volcano_p and abs(xi) > args.volcano_log2fc:
            colors.append("red" if xi > 0 else "blue")
        else:
            colors.append("grey")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, c=colors, alpha=0.7, linewidths=0)
    plt.axvline(args.volcano_log2fc, linestyle="--")
    plt.axvline(-args.volcano_log2fc, linestyle="--")
    plt.axhline(-math.log10(args.volcano_p), linestyle="--")
    plt.xlabel("log2(Original / Synthetic)")
    plt.ylabel("-log10(Fisher p-value)")

    top = df.nsmallest(args.top_n, "pvalue_fisher")
    cmap = plt.get_cmap("tab10")
    handles, labels = [], []
    for i, (_, r) in enumerate(top.iterrows()):
        sc = plt.scatter([r["log2FC_orig_over_syn"]], [r["neglog10_pvalue"]],
                         s=40, color=cmap(i % 10), edgecolor="black", linewidths=0.5, zorder=3)
        handles.append(sc)
        labels.append(str(r["motif_key"]))

    if handles:
        plt.legend(handles, labels, title=f"Top {len(handles)} motifs", loc="lower left",
                   fontsize=6, title_fontsize=7, frameon=True)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")


if __name__ == "__main__":
    main()
