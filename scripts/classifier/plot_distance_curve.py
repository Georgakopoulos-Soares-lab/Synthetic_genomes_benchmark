#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_distance_curve.py

Plot metric vs distance from start.

Inputs:
  - *_summary.csv produced by deep_detector_distance_eval.py (distance_bp, mean_<metric>, std_<metric>), OR
  - *_per_tag_distance.csv (distance_bp, <metric>, ...), which will be aggregated across tags.

Std visualization (optional):
  - --show-std : error bars (±1 std)
  - --shade-std: shaded band (±1 std)

Examples:
  python plot_distance_curve.py --in distance_run_summary.csv --shade-std
  python plot_distance_curve.py --in distance_run_per_tag_distance.csv --metric auc --shade-std
  python plot_distance_curve.py --in distance_run_summary.csv --out auc_vs_distance.png --show-std
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV: *_summary.csv or *_per_tag_distance.csv")
    ap.add_argument(
        "--metric",
        default="auc",
        help="Metric for per-tag input (default: auc). Summary input uses mean_<metric> (or mean_auc).",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output image path (png). Default: <input_stem>_metric_vs_distance.png",
    )

    ap.add_argument("--xlabel", default="Distance from start (bp)")
    ap.add_argument("--ylabel", default="AUROC")
    ap.add_argument(
        "--xlog",
        action="store_true",
        help="Use log scale for x-axis. If any distance is <= 0, log scaling is skipped.",
    )
    ap.add_argument("--ylim", nargs=2, type=float, default=None, help="Set y-limits, e.g. --ylim 0.4 1.0")

    ap.add_argument("--show-std", action="store_true", help="Show ±1 std as error bars (if available).")
    ap.add_argument("--shade-std", action="store_true", help="Shade ±1 std band (if available).")

    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--tick-size", type=int, default=14)
    ap.add_argument("--marker-size", type=float, default=6.0)
    ap.add_argument("--line-width", type=float, default=1.5)
    ap.add_argument("--cap-size", type=float, default=3.0)
    ap.add_argument("--shade-alpha", type=float, default=0.18)

    ap.add_argument("--save-pdf", action="store_true")
    return ap.parse_args()


def _is_summary_df(df: pd.DataFrame) -> bool:
    if "distance_bp" not in df.columns:
        return False
    if "mean_auc" in df.columns:
        return True
    return any(c.startswith("mean_") for c in df.columns)


def _aggregate_from_per_tag(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "distance_bp" not in df.columns:
        raise ValueError("per-tag input must contain a distance_bp column")
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in columns: {sorted(df.columns)}")

    dfx = df.loc[pd.notna(df[metric]), ["distance_bp", metric]].copy()
    agg = (
        dfx.groupby("distance_bp", as_index=False)
        .agg(mean_metric=(metric, "mean"), std_metric=(metric, "std"))
        .sort_values("distance_bp")
        .reset_index(drop=True)
    )
    return agg


def _prepare_xy(df: pd.DataFrame, metric: str, want_std: bool):
    """
    Returns x, y, ystd (ystd may be None)
    """
    ystd = None

    if _is_summary_df(df):
        mean_col = f"mean_{metric}" if f"mean_{metric}" in df.columns else "mean_auc"
        std_col = f"std_{metric}" if f"std_{metric}" in df.columns else ("std_auc" if "std_auc" in df.columns else None)

        if mean_col not in df.columns:
            raise ValueError(f"Summary input missing '{mean_col}'")

        x = df["distance_bp"].to_numpy(dtype=float)
        y = df[mean_col].to_numpy(dtype=float)

        if want_std and std_col is not None and std_col in df.columns:
            ystd = df[std_col].to_numpy(dtype=float)

    else:
        m = metric
        if m == "mean_auc":
            m = "auc"

        agg = _aggregate_from_per_tag(df, metric=m)
        x = agg["distance_bp"].to_numpy(dtype=float)
        y = agg["mean_metric"].to_numpy(dtype=float)
        if want_std:
            ystd = agg["std_metric"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if ystd is not None:
        ystd = ystd[order]

    return x, y, ystd


def main():
    args = parse_args()
    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    df = pd.read_csv(inp)

    want_std = bool(args.show_std or args.shade_std)
    x, y, ystd = _prepare_xy(df, metric=args.metric, want_std=want_std)

    out_png = Path(args.out) if args.out else inp.with_name(inp.stem + "_metric_vs_distance.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=220)

    (line,) = plt.plot(
        x,
        y,
        marker="o",
        markersize=args.marker_size,
        linewidth=args.line_width,
    )

    if ystd is not None and args.shade_std:
        c = line.get_color()
        plt.fill_between(x, y - ystd, y + ystd, color=c, alpha=args.shade_alpha, linewidth=0)

    if ystd is not None and args.show_std:
        plt.errorbar(
            x,
            y,
            yerr=ystd,
            fmt="none",
            capsize=args.cap_size,
            linewidth=args.line_width,
        )

    plt.xlabel(args.xlabel, fontsize=args.font_size)
    plt.ylabel(args.ylabel, fontsize=args.font_size)
    plt.xticks(fontsize=args.tick_size)
    plt.yticks(fontsize=args.tick_size)

    if args.xlog:
        if np.all(x > 0):
            plt.xscale("log")

    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(out_png)
    print(f"[ok] wrote: {out_png}")

    if args.save_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        plt.savefig(out_pdf)
        print(f"[ok] wrote: {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())