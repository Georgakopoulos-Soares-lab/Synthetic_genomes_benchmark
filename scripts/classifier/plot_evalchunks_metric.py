#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_evalchunks_metric_3dom_tagstd.py

Plot mean metric vs n_eval_chunks for euk/prok/vir with shaded ±1 std across holdout tags,
from an aggregated per-tag CSV (e.g. all_domains_all_runs_per_tag.csv).

Expected columns:
  domain, run_id, holdout_tag, <metric>
where run_id contains 'nev<INT>' (e.g., euk_nev16_r5).

Outputs:
  - PNG (and optional PDF)
  - CSV of aggregated mean/std used for plotting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


YLABELS = {
    "auc": "AUROC",
    "f1": "F1 score",
    "acc": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to all_domains_all_runs_per_tag.csv")
    ap.add_argument("--metric", required=True, choices=list(YLABELS.keys()))
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--ylabel", default="", help="Override y-axis label")
    ap.add_argument("--save-pdf", action="store_true")

    ap.add_argument("--font-scale", type=float, default=1.25)
    ap.add_argument("--shade-alpha", type=float, default=0.18)

    ap.add_argument("--domains", default="euk,prok,vir", help="Comma-separated domain order")
    ap.add_argument("--domain-map", default="",
                    help="Optional mapping for domain values in CSV, e.g. 'virus:vir,prokaryotes:prok'")

    ap.add_argument("--tag-list", default="",
                    help="Optional CSV/TXT with columns: domain,tag (only keep these tags)")
    ap.add_argument("--no-tag-filter", action="store_true",
                    help="Do not filter by tag list even if --tag-list is provided")

    return ap.parse_args()


def parse_domain_map(s: str) -> dict:
    out = {}
    if not s.strip():
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out[a.strip()] = b.strip()
    return out


def load_tag_allowlist(path: Path) -> dict:
    df = pd.read_csv(path)
    need = {"domain", "tag"}
    if not need.issubset(df.columns):
        raise ValueError(f"--tag-list must include columns {sorted(need)}")
    allow = {}
    for dom, sub in df.groupby("domain"):
        allow[str(dom)] = set(sub["tag"].astype(str).tolist())
    return allow


def main() -> int:
    args = parse_args()
    inp = Path(args.csv)
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"[error] missing input: {inp}", file=sys.stderr)
        return 2

    df = pd.read_csv(inp)
    needed = {"domain", "run_id", "holdout_tag", args.metric}
    missing = needed - set(df.columns)
    if missing:
        print(f"[error] input missing columns: {sorted(missing)}", file=sys.stderr)
        print(f"[error] columns: {list(df.columns)}", file=sys.stderr)
        return 2

    df["domain"] = df["domain"].astype(str).str.strip()
    df["run_id"] = df["run_id"].astype(str)
    df["holdout_tag"] = df["holdout_tag"].astype(str)

    dmap = parse_domain_map(args.domain_map)
    if dmap:
        df["domain"] = df["domain"].map(lambda x: dmap.get(x, x))

    nev = df["run_id"].str.extract(r"nev(\d+)")[0]
    df["n_eval_chunks"] = pd.to_numeric(nev, errors="coerce")
    df = df[pd.notna(df["n_eval_chunks"])].copy()
    df["n_eval_chunks"] = df["n_eval_chunks"].astype(int)

    allow = None
    if args.tag_list:
        allow = load_tag_allowlist(Path(args.tag_list))

    if allow is not None and not args.no_tag_filter:
        keep_parts = []
        for dom, sub in df.groupby("domain"):
            dom = str(dom)
            if dom in allow:
                keep_parts.append(sub[sub["holdout_tag"].isin(allow[dom])])
            else:
                keep_parts.append(sub)
        df = pd.concat(keep_parts, ignore_index=True)

    agg = (
        df.groupby(["domain", "n_eval_chunks"], as_index=False)
          .agg(
              mean_metric=(args.metric, "mean"),
              std_metric=(args.metric, "std"),
              n_tags=("holdout_tag", "nunique"),
          )
          .sort_values(["domain", "n_eval_chunks"])
          .reset_index(drop=True)
    )

    base = float(plt.rcParams.get("font.size", 10.0))
    plt.rcParams.update(
        {
            "font.size": base * args.font_scale,
            "axes.labelsize": base * args.font_scale,
            "xtick.labelsize": base * args.font_scale * 0.95,
            "ytick.labelsize": base * args.font_scale * 0.95,
            "legend.fontsize": base * args.font_scale * 0.95,
        }
    )

    domain_order = [d.strip() for d in args.domains.split(",") if d.strip()]
    present = [d for d in domain_order if d in set(agg["domain"])]
    if not present:
        present = sorted(agg["domain"].unique())

    plt.figure(figsize=(10, 6), dpi=200)

    for domain in present:
        d = agg[agg["domain"] == domain].sort_values("n_eval_chunks")
        if d.empty:
            continue
        x = d["n_eval_chunks"].to_numpy()
        y = d["mean_metric"].to_numpy()
        s = d["std_metric"].to_numpy()

        (line,) = plt.plot(x, y, marker="o", label=domain)

        ok = np.isfinite(s)
        if ok.any():
            c = line.get_color()
            plt.fill_between(x[ok], (y - s)[ok], (y + s)[ok], color=c, alpha=args.shade_alpha, linewidth=0)

    plt.xlabel("Evaluation chunks per sequence")
    ylabel = args.ylabel.strip() if args.ylabel.strip() else YLABELS.get(args.metric, args.metric)
    plt.ylabel(ylabel)

    xt = sorted(agg["n_eval_chunks"].unique().tolist())
    plt.xticks(xt)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title=None)
    plt.tight_layout()

    plt.savefig(out_png)
    print(f"[ok] wrote: {out_png}")

    if args.save_pdf:
        out_pdf = out_png.with_suffix(".pdf")
        plt.savefig(out_pdf)
        print(f"[ok] wrote: {out_pdf}")

    out_csv = out_png.with_suffix(".data.csv")
    agg_out = agg.rename(columns={"mean_metric": f"mean_{args.metric}", "std_metric": f"std_{args.metric}"})
    agg_out.to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())