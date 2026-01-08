#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


MOTIF_ORDER = ["GQ", "Z-DNA", "DR", "IR", "MR", "STR"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="results/<TAG>/nonbdna")
    p.add_argument("--eps", type=float, default=0.5, help="Pseudocount for log2 ratios (default 0.5)")
    p.add_argument("--ymax-bp", type=float, default=None, help="Optional y-max for bp_covered plot")
    p.add_argument("--ymax-hits", type=float, default=None, help="Optional y-max for n_hits plot")
    p.add_argument("--q01", type=float, default=0.01)
    p.add_argument("--q05", type=float, default=0.05)
    p.add_argument("--q10", type=float, default=0.10)
    p.add_argument("--sym01", type=str, default="**")
    p.add_argument("--sym05", type=str, default="*")
    p.add_argument("--sym10", type=str, default="•")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _require_cols(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}. Found: {list(df.columns)}")


def load_tidy(metrics_dir: Path) -> pd.DataFrame:
    g4f = metrics_dir / "g4hunter.metrics.csv"
    zf = metrics_dir / "zseeker.metrics.csv"
    nf = metrics_dir / "nonbgfa.metrics.csv"

    for f in (g4f, zf, nf):
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")

    g4 = pd.read_csv(g4f)
    _require_cols(g4, ["pair_id", "which", "n_hits", "bp_covered"], g4f)
    g4 = g4[["pair_id", "which", "n_hits", "bp_covered"]].copy()
    g4["motif"] = "GQ"

    z = pd.read_csv(zf)
    _require_cols(z, ["pair_id", "which", "n_hits", "bp_covered"], zf)
    z = z[["pair_id", "which", "n_hits", "bp_covered"]].copy()
    z["motif"] = "Z-DNA"

    nb = pd.read_csv(nf)
    _require_cols(nb, ["pair_id", "motif", "which", "n_hits", "bp_covered"], nf)
    nb = nb[["pair_id", "motif", "which", "n_hits", "bp_covered"]].copy()
    nb = nb[nb["motif"].isin(["DR", "IR", "MR", "STR"])].copy()

    tidy = pd.concat([g4, z, nb], ignore_index=True)

    tidy["which"] = tidy["which"].astype(str).str.lower().map(
        lambda x: "syn" if x.startswith("syn") else "orig"
    )
    for c in ["n_hits", "bp_covered"]:
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce").fillna(0.0)

    tidy = tidy[tidy["motif"].isin(MOTIF_ORDER)].copy()
    tidy["motif"] = pd.Categorical(tidy["motif"], categories=MOTIF_ORDER, ordered=True)
    return tidy


def paired_table(tidy: pd.DataFrame, motif: str, measure: str) -> pd.DataFrame:
    sub = tidy[tidy["motif"] == motif][["pair_id", "which", measure]].copy()
    piv = sub.pivot_table(index="pair_id", columns="which", values=measure, aggfunc="sum")
    if "orig" not in piv.columns:
        piv["orig"] = np.nan
    if "syn" not in piv.columns:
        piv["syn"] = np.nan
    piv = piv.dropna(subset=["orig", "syn"])
    piv = piv.reset_index()
    return piv


def wilcoxon_p(orig: np.ndarray, syn: np.ndarray) -> float:
    if len(orig) == 0:
        return float("nan")
    if wilcoxon is None:
        return float("nan")
    d = orig - syn
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def summarize_one(tidy: pd.DataFrame, motif: str, measure: str, eps: float) -> dict:
    piv = paired_table(tidy, motif, measure)
    orig = piv["orig"].to_numpy(dtype=float)
    syn = piv["syn"].to_numpy(dtype=float)

    n_pairs = int(len(piv))

    mean_o = float(np.mean(orig)) if n_pairs else float("nan")
    mean_s = float(np.mean(syn)) if n_pairs else float("nan")
    med_o = float(np.median(orig)) if n_pairs else float("nan")
    med_s = float(np.median(syn)) if n_pairs else float("nan")

    log2fc_mean = float(np.log2((mean_o + eps) / (mean_s + eps))) if n_pairs else float("nan")
    log2fc_median = float(np.log2((med_o + eps) / (med_s + eps))) if n_pairs else float("nan")

    # per-pair log2 ratio (more stable than ratio of medians)
    pair_log2 = np.log2((orig + eps) / (syn + eps)) if n_pairs else np.array([], dtype=float)
    log2fc_pair_median = float(np.median(pair_log2)) if n_pairs else float("nan")
    log2fc_pair_mean = float(np.mean(pair_log2)) if n_pairs else float("nan")

    p = wilcoxon_p(orig, syn)

    return {
        "motif": motif,
        "measure": measure,
        "n_pairs": n_pairs,
        "mean_orig": mean_o,
        "mean_syn": mean_s,
        "median_orig": med_o,
        "median_syn": med_s,
        "log2FC_mean": log2fc_mean,
        "log2FC_median": log2fc_median,
        "log2FC_pair_mean": log2fc_pair_mean,
        "log2FC_pair_median": log2fc_pair_median,
        "p": p,
    }


def q_to_mark(q: float, args: argparse.Namespace) -> str | None:
    if q is None or not np.isfinite(q):
        return None
    if q < args.q01:
        return args.sym01
    if q < args.q05:
        return args.sym05
    if q < args.q10:
        return args.sym10
    return None


def plot_boxplot(tidy: pd.DataFrame, out_png: Path, measure: str, q_by_motif: dict, args: argparse.Namespace, ymax: float | None):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(MOTIF_ORDER), dtype=float)

    data_orig, data_syn = [], []
    for m in MOTIF_ORDER:
        d_m = tidy[tidy["motif"] == m]
        data_orig.append(d_m[d_m["which"] == "orig"][measure].to_numpy(dtype=float))
        data_syn.append(d_m[d_m["which"] == "syn"][measure].to_numpy(dtype=float))

    width = 0.30
    pos_orig = x - width / 2.0
    pos_syn = x + width / 2.0

    bp1 = ax.boxplot(data_orig, positions=pos_orig, widths=width, patch_artist=True, manage_ticks=False)
    bp2 = ax.boxplot(data_syn, positions=pos_syn, widths=width, patch_artist=True, manage_ticks=False)

    c_orig = "#1f77b4"
    c_syn = "#ff7f0e"

    for patch in bp1["boxes"]:
        patch.set_facecolor(c_orig)
        patch.set_edgecolor("black")
        patch.set_alpha(0.9)
    for patch in bp2["boxes"]:
        patch.set_facecolor(c_syn)
        patch.set_edgecolor("black")
        patch.set_alpha(0.9)

    for median in bp1["medians"] + bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(MOTIF_ORDER, fontsize=11)
    ax.set_xlabel("Motif", fontsize=12)
    ax.set_ylabel(measure, fontsize=12)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=c_orig, edgecolor="black", label="Original"),
            mpatches.Patch(facecolor=c_syn, edgecolor="black", label="Synthetic"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=10,
    )

    if ymax is not None:
        ax.set_ylim(0, float(ymax))

    # significance marks
    y_top = ax.get_ylim()[1]
    y_pad = 0.03 * y_top
    for i, m in enumerate(MOTIF_ORDER):
        q = q_by_motif.get(m, np.nan)
        mark = q_to_mark(q, args)
        if not mark:
            continue

        local_max = 0.0
        if len(data_orig[i]) > 0:
            local_max = max(local_max, float(np.nanmax(data_orig[i])))
        if len(data_syn[i]) > 0:
            local_max = max(local_max, float(np.nanmax(data_syn[i])))

        y = min(local_max + y_pad, y_top * 0.98)
        ax.text(x[i], y, mark, ha="center", va="bottom", fontsize=12, weight="bold", color="black")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    metrics_dir = outdir / "metrics"
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)

    tidy = load_tidy(metrics_dir)

    rows = []
    for measure in ["bp_covered", "n_hits"]:
        for motif in MOTIF_ORDER:
            rows.append(summarize_one(tidy, motif, measure, eps=args.eps))

    res = pd.DataFrame(rows)

    # BH-FDR per measure across the 6 motifs
    res["q"] = np.nan
    for measure in ["bp_covered", "n_hits"]:
        msk = (res["measure"] == measure) & np.isfinite(res["p"].to_numpy(dtype=float))
        pvals = res.loc[msk, "p"].to_numpy(dtype=float)
        if len(pvals):
            res.loc[msk, "q"] = bh_fdr(pvals)

    out_csv = outdir / "significance.summary.csv"
    res.to_csv(out_csv, index=False)

    # plots use q for that measure
    for measure, ymax, outname in [
        ("bp_covered", args.ymax_bp, "nonbdna_bp_covered_boxplot.png"),
        ("n_hits", args.ymax_hits, "nonbdna_n_hits_boxplot.png"),
    ]:
        q_by_motif = {
            r["motif"]: float(r["q"]) if np.isfinite(r["q"]) else np.nan
            for _, r in res[res["measure"] == measure].iterrows()
        }
        plot_boxplot(
            tidy=tidy,
            out_png=plots_dir / outname,
            measure=measure,
            q_by_motif=q_by_motif,
            args=args,
            ymax=ymax,
        )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] plots: {plots_dir}")


if __name__ == "__main__":
    main()
