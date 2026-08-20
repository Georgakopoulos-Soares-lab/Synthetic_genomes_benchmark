#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
composition.py

Fast per-window composition and divergence summary for a set of matched pairs.

This is the cheapest benchmark in the suite and the one to run first on a new
generator: no external tools, no training, seconds per window. It reports the
actual numbers (not just significance) for the properties that generative
models most often get wrong, plus the paired divergence of each synthetic
window from its own natural counterpart.

Per-window properties (computed for natural and synthetic separately)
---------------------------------------------------------------------
    gc                  GC fraction
    cpg_oe              CpG observed/expected -- collapses toward 1.0 when a
                        model fails to reproduce vertebrate CpG depletion
    homopolymer_frac    fraction of bases inside a run of >= --min-run
    low_complexity      1 - H/H_max over 3-mers; rises with repetitive output
    entropy_k3          Shannon entropy of the 3-mer distribution (bits)
    nullomer_frac       fraction of canonical k-mer classes absent from the
                        window, using the canonical-class denominator

Paired divergence (synthetic window vs its own natural counterpart)
-------------------------------------------------------------------
    kmer_jsd            Jensen-Shannon divergence of k-mer spectra (bits)
    fcgr_l1             L1 distance between normalised FCGR vectors

Statistics
----------
Per-window properties are compared with a paired Wilcoxon signed-rank test and
an exact paired sign-flip permutation test on the mean difference, with
Benjamini-Hochberg correction across properties. The permutation test is exact
for up to 20 pairs and needs no distributional assumption, which matters at the
sample sizes typical of window-level benchmarks.

For a divergence that is *calibrated against natural variation* rather than
merely significant, follow this with ``natural_baseline.py``.

Outputs
-------
    <outdir>/composition.per_pair.csv    one row per pair, all metrics
    <outdir>/composition.summary.csv     paired tests and effect sizes
    <outdir>/composition.png             paired dot plots (unless --no-plot)

Example
-------
    python scripts/benchmarks/composition.py \\
        --manifest manifests/pairs.Homo_sapiens.csv \\
        --outdir results/Homo_sapiens/composition
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seqio as S  # noqa: E402

WINDOW_METRICS = [
    "gc", "cpg_oe", "homopolymer_frac", "low_complexity",
    "entropy_k3", "nullomer_frac",
]
PAIR_METRICS = ["kmer_jsd", "fcgr_l1"]



def window_metrics(seq: str, min_run: int, nullomer_k: int) -> dict[str, float]:
    return {
        "gc": S.gc_content(seq),
        "cpg_oe": S.cpg_observed_expected(seq),
        "homopolymer_frac": S.homopolymer_fraction(seq, min_run),
        "low_complexity": S.low_complexity(seq, 3),
        "entropy_k3": S.shannon_entropy(seq, 3),
        "nullomer_frac": nullomer_fraction(seq, nullomer_k),
    }


def nullomer_fraction(seq: str, k: int) -> float:
    """Fraction of canonical k-mer classes absent from this window.

    The denominator is the number of canonical (strand-collapsed) classes, not
    ``4**k``. Using ``4**k`` with canonically-counted k-mers -- the mistake that
    is easy to make because KMC counts canonically by default -- floors this
    value near 0.5 for odd k regardless of the sequence.
    """
    counts = S.kmer_counts(seq, k, canonical=True)
    classes = S.canonical_kmer_classes(k)
    return float((classes - int((counts > 0).sum())) / classes)


def paired_tests(orig: np.ndarray, syn: np.ndarray, seed: int,
                 n_perm: int) -> dict[str, float]:
    """Paired Wilcoxon + exact sign-flip permutation on syn - orig."""
    from scipy.stats import wilcoxon

    mask = np.isfinite(orig) & np.isfinite(syn)
    o, s = orig[mask], syn[mask]
    n = o.size
    if n < 3:
        return {"n_pairs": n, "p_wilcoxon": np.nan, "p_permutation": np.nan,
                "perm_method": "insufficient", "n_perm_used": 0}

    diff = s - o
    try:
        if np.allclose(diff, 0):
            p_w = 1.0
        else:
            p_w = float(wilcoxon(s, o, alternative="two-sided",
                                 zero_method="wilcox").pvalue)
    except ValueError:
        p_w = np.nan

    p_perm, method, n_used = S.signflip_pvalue(
        diff, n_perm=n_perm, seed=seed, alternative="two-sided"
    )
    return {"n_pairs": n, "p_wilcoxon": p_w, "p_permutation": p_perm,
            "perm_method": method, "n_perm_used": n_used}


def cohens_dz(orig: np.ndarray, syn: np.ndarray) -> float:
    """Paired effect size: mean(diff) / sd(diff)."""
    mask = np.isfinite(orig) & np.isfinite(syn)
    diff = syn[mask] - orig[mask]
    if diff.size < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd > 0 else np.nan


def make_plot(per_pair: pd.DataFrame, out_png: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in WINDOW_METRICS
               if f"orig_{m}" in per_pair.columns
               and per_pair[f"orig_{m}"].notna().any()]
    if not metrics:
        return
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, metrics):
        o = per_pair[f"orig_{m}"].to_numpy(dtype=float)
        s = per_pair[f"syn_{m}"].to_numpy(dtype=float)
        keep = np.isfinite(o) & np.isfinite(s)
        o, s = o[keep], s[keep]
        for oi, si in zip(o, s):
            ax.plot([0, 1], [oi, si], color="0.75", lw=0.9, zorder=1)
        ax.scatter(np.zeros_like(o), o, s=26, color="#2c3e50", zorder=3,
                   label="natural")
        ax.scatter(np.ones_like(s), s, s=26, color="#c0392b", zorder=3,
                   label="synthetic")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["natural", "synthetic"], fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(m, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle(f"{label}: per-window composition", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, help="Pairs CSV (id,orig,syn).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--kmer-k", type=int, default=6,
                    help="k for the k-mer JSD between matched windows.")
    ap.add_argument("--fcgr-k", type=int, default=6, help="FCGR order.")
    ap.add_argument("--nullomer-k", type=int, default=11,
                    help="k for the per-window canonical nullomer fraction.")
    ap.add_argument("--min-run", type=int, default=5,
                    help="Minimum homopolymer run length counted as a run.")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="Monte Carlo permutations when >20 pairs.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else None
    manifest = Path(args.manifest)
    label = args.label or re.sub(r"^pairs\.", "", manifest.stem)

    df = S.load_manifest(manifest, data_root, args.max_pairs)
    print(f"[info] {label}: {len(df)} matched pairs")

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        nat = S.read_fasta_concat(r["orig"]).upper()
        syn = S.read_fasta_concat(r["syn"]).upper()
        row: dict[str, object] = {
            "id": r["id"], "orig_len": len(nat), "syn_len": len(syn),
        }
        for name, val in window_metrics(nat, args.min_run, args.nullomer_k).items():
            row[f"orig_{name}"] = val
        for name, val in window_metrics(syn, args.min_run, args.nullomer_k).items():
            row[f"syn_{name}"] = val
        row["kmer_jsd"] = S.js_divergence(
            S.kmer_freqs(nat, args.kmer_k), S.kmer_freqs(syn, args.kmer_k)
        )
        row["fcgr_l1"] = S.l1_distance(
            S.fcgr_vector(nat, args.fcgr_k), S.fcgr_vector(syn, args.fcgr_k)
        )
        rows.append(row)
        if i % 10 == 0 or i == len(df):
            print(f"[info] {i}/{len(df)} pairs done")

    per_pair = pd.DataFrame(rows)
    pp_path = outdir / "composition.per_pair.csv"
    per_pair.to_csv(pp_path, index=False)
    print(f"[ok] wrote {pp_path}")

    summary_rows = []
    for m in WINDOW_METRICS:
        o = per_pair[f"orig_{m}"].to_numpy(dtype=float)
        s = per_pair[f"syn_{m}"].to_numpy(dtype=float)
        stats = paired_tests(o, s, args.seed, args.n_perm)
        summary_rows.append({
            "metric": m,
            "natural_mean": float(np.nanmean(o)),
            "natural_median": float(np.nanmedian(o)),
            "synthetic_mean": float(np.nanmean(s)),
            "synthetic_median": float(np.nanmedian(s)),
            "mean_difference": float(np.nanmean(s - o)),
            "cohens_dz": cohens_dz(o, s),
            **stats,
        })
    for m in PAIR_METRICS:
        v = per_pair[m].to_numpy(dtype=float)
        summary_rows.append({
            "metric": m,
            "natural_mean": np.nan, "natural_median": np.nan,
            "synthetic_mean": float(np.nanmean(v)),
            "synthetic_median": float(np.nanmedian(v)),
            "mean_difference": np.nan, "cohens_dz": np.nan,
            "n_pairs": int(np.isfinite(v).sum()),
            "p_wilcoxon": np.nan, "p_permutation": np.nan,
            "perm_method": "not_applicable", "n_perm_used": 0,
        })

    summary = pd.DataFrame(summary_rows)
    summary["q_permutation"] = S.bh_fdr(summary["p_permutation"].to_numpy(float))
    summary["q_wilcoxon"] = S.bh_fdr(summary["p_wilcoxon"].to_numpy(float))
    sum_path = outdir / "composition.summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[ok] wrote {sum_path}")

    print(f"\n=== {label}: per-window composition (n={len(per_pair)}) ===")
    print(f"{'metric':18s} {'natural':>11s} {'synthetic':>11s} "
          f"{'diff':>10s} {'dz':>7s} {'q':>10s}")
    for _, r in summary.iterrows():
        if r["metric"] in PAIR_METRICS:
            continue
        flag = ""
        if np.isfinite(r["q_permutation"]) and r["q_permutation"] < 0.05:
            flag = "  <- shifted " + ("up" if r["mean_difference"] > 0 else "down")
        print(f"{r['metric']:18s} {r['natural_mean']:11.4f} "
              f"{r['synthetic_mean']:11.4f} {r['mean_difference']:+10.4f} "
              f"{r['cohens_dz']:7.2f} {r['q_permutation']:10.3g}{flag}")
    print("\n=== paired divergence from own natural window ===")
    for _, r in summary.iterrows():
        if r["metric"] in PAIR_METRICS:
            print(f"{r['metric']:18s} median={r['synthetic_median']:.4f}  "
                  f"mean={r['synthetic_mean']:.4f}")

    if not args.no_plot:
        png = outdir / "composition.png"
        make_plot(per_pair, png, label)
        print(f"[ok] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
