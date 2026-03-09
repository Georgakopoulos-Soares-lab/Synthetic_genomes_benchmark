#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_distance_curve_broken_overlay.py

Overlay 2+ distance summary curves on a broken x-axis (two panels).

Inputs:
  - Multiple *_summary.csv files produced by deep_detector_distance_eval.py
    Each must contain:
      - distance_bp
      - mean_<metric> (or mean_auc)
      - optional std_<metric> (or std_auc) if --show-std

Output:
  - PNG (and optional PDF)

Example:
  python plot_distance_curve_broken_overlay.py \
    --ins euk_distance_summary.csv prok_distance_summary.csv \
    --labels euk prok \
    --metric auc \
    --xbreak 20000 \
    --out distance_euk_prok_auc.png \
    --show-std --save-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", nargs="+", required=True, help="Input *_summary.csv files (2+).")
    ap.add_argument("--labels", nargs="+", default=[], help="Labels for each input (same count as --ins).")
    ap.add_argument("--metric", default="auc", help="Metric: auc/f1/acc/... (expects mean_<metric>, std_<metric>).")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--show-std", action="store_true")
    ap.add_argument("--xlabel", default="Distance from start (bp)")
    ap.add_argument("--ylabel", default="AUROC")
    ap.add_argument("--xbreak", type=float, default=20000.0)
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--font-size", type=int, default=18)
    ap.add_argument("--tick-size", type=int, default=16)
    ap.add_argument("--marker-size", type=float, default=7.0)
    ap.add_argument("--line-width", type=float, default=2.0)
    ap.add_argument("--cap-size", type=float, default=3.0)
    ap.add_argument("--gap", type=float, default=0.02)
    return ap.parse_args()


def add_break_marks(ax_left, ax_right):
    d = 0.015
    kw = dict(transform=ax_left.transAxes, color="k", clip_on=False, linewidth=1.2)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kw)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kw)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kw)


def read_summary(path: Path, metric: str):
    df = pd.read_csv(path)
    if "distance_bp" not in df.columns:
        raise ValueError(f"{path} missing distance_bp")

    mean_col = f"mean_{metric}" if f"mean_{metric}" in df.columns else "mean_auc"
    std_col = f"std_{metric}" if f"std_{metric}" in df.columns else ("std_auc" if "std_auc" in df.columns else None)

    if mean_col not in df.columns:
        raise ValueError(f"{path} missing {mean_col}")

    x = df["distance_bp"].to_numpy(dtype=float)
    y = df[mean_col].to_numpy(dtype=float)
    s = df[std_col].to_numpy(dtype=float) if (std_col is not None and std_col in df.columns) else None

    o = np.argsort(x)
    return x[o], y[o], (s[o] if s is not None else None)


def main():
    args = parse_args()
    ins = [Path(p) for p in args.ins]
    for p in ins:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    labels = args.labels
    if labels and len(labels) != len(ins):
        raise SystemExit("--labels must match number of --ins")
    if not labels:
        labels = [p.stem.replace("_summary", "") for p in ins]

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": float(args.gap)},
        figsize=(15, 6),
    )

    for path, lab in zip(ins, labels):
        x, y, s = read_summary(path, args.metric)

        for ax in (ax1, ax2):
            if args.show_std and s is not None and np.isfinite(s).any():
                ax.errorbar(
                    x, y, yerr=s,
                    fmt="-o",
                    capsize=args.cap_size,
                    markersize=args.marker_size,
                    linewidth=args.line_width,
                    label=lab if ax is ax1 else None,
                )
            else:
                ax.plot(
                    x, y, "-o",
                    markersize=args.marker_size,
                    linewidth=args.line_width,
                    label=lab if ax is ax1 else None,
                )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    xbreak = float(args.xbreak)
    xmax = max(float(np.nanmax(read_summary(p, args.metric)[0])) for p in ins)
    ax1.set_xlim(0, xbreak)
    ax2.set_xlim(xbreak, xmax)

    ax1.set_ylabel(args.ylabel, fontsize=args.font_size)
    fig.supxlabel(args.xlabel, fontsize=args.font_size)
    ax1.tick_params(axis="both", labelsize=args.tick_size)
    ax2.tick_params(axis="both", labelsize=args.tick_size)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=False)

    add_break_marks(ax1, ax2)
    ax1.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[ok] wrote: {out_png}")

    if args.save_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        plt.savefig(out_pdf)
        print(f"[ok] wrote: {out_pdf}")


if __name__ == "__main__":
    main()