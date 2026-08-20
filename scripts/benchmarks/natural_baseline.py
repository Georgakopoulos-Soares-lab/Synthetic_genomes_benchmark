#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
natural_baseline.py

Calibrate synthetic-vs-natural divergence against natural-vs-natural variation.

Every other benchmark in this suite answers "how far is synthetic from
natural?". On its own that number is not interpretable, because real genomic
windows from the same genome already differ from each other. This benchmark
supplies the missing reference: it asks whether synthetic windows sit *outside*
the spread that natural windows already show among themselves.

Design (per metric)
-------------------
Given n matched pairs from the manifest, each window is reduced to a feature
(a vector for ``fcgr``/``kmer``, a scalar otherwise) and all pairwise distances
are formed:

    nat-nat : distances between distinct natural windows        (n choose 2)
    syn-nat : distances between synthetic and natural windows   (n * n)

The statistic is

    delta = median(syn-nat) - median(nat-nat)

delta > 0 means synthetic windows are further from natural windows than
natural windows are from each other.

Why a permutation test rather than Mann-Whitney
-----------------------------------------------
Each window contributes to many pairwise distances, so the distances are not
independent observations. A rank test over them treats n(n-1)/2 correlated
values as independent, inflating the effective sample size and producing
anticonservative p-values. This benchmark instead permutes the thing that is
actually exchangeable under the null: the natural/synthetic *label within each
matched pair*. Each pair's labels are swapped with probability 0.5, the whole
distance structure is rebuilt, and delta is recomputed. Exhaustive enumeration
of all 2^n labellings is used when n is small enough, Monte Carlo otherwise.

Effect size
-----------
``ratio = median(syn-nat) / median(nat-nat)`` is reported alongside delta. A
ratio of 1.0 means the synthetic windows are indistinguishable from natural
variation by that metric; 2.0 means they are twice as far apart as real windows
are from each other.

Outputs
-------
    <outdir>/natural_baseline.per_metric.csv   one row per metric
    <outdir>/natural_baseline.distances.csv    the underlying distance summaries
    <outdir>/natural_baseline.png              forest plot of ratios (unless --no-plot)

Example
-------
    python scripts/benchmarks/natural_baseline.py \\
        --manifest manifests/pairs.Homo_sapiens.csv \\
        --outdir results/Homo_sapiens/natural_baseline \\
        --metrics fcgr kmer gc homopolymer low_complexity cpg_oe
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seqio as S  # noqa: E402

VECTOR_METRICS = {"fcgr", "kmer"}
SCALAR_METRICS = {
    "gc": S.gc_content,
    "homopolymer": S.homopolymer_fraction,
    "low_complexity": S.low_complexity,
    "cpg_oe": S.cpg_observed_expected,
    "nullomer_fraction": None,  # handled specially (needs k)
}
DEFAULT_METRICS = ["fcgr", "kmer", "gc", "homopolymer", "low_complexity", "cpg_oe"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def nullomer_fraction(seq: str, k: int) -> float:
    """Fraction of canonical k-mer classes absent from a single window.

    Uses the canonical-class denominator (see ``_seqio.canonical_kmer_classes``)
    so the value is not floored at ~0.5 for odd k.
    """
    counts = S.kmer_counts(seq, k, canonical=True)
    observed = int((counts > 0).sum())
    classes = S.canonical_kmer_classes(k)
    return float((classes - observed) / classes)


def window_features(seq: str, metric: str, k_fcgr: int, k_kmer: int,
                    k_null: int) -> np.ndarray:
    """Reduce one window to its feature vector (length 1 for scalar metrics)."""
    if metric == "fcgr":
        return S.fcgr_vector(seq, k_fcgr)
    if metric == "kmer":
        return S.kmer_freqs(seq, k_kmer)
    if metric == "nullomer_fraction":
        return np.array([nullomer_fraction(seq, k_null)])
    fn = SCALAR_METRICS[metric]
    return np.array([fn(seq)])


# ---------------------------------------------------------------------------
# Statistic and permutation test
# ---------------------------------------------------------------------------

def pairwise_l1(vectors: np.ndarray) -> np.ndarray:
    """Full L1 distance matrix for a stack of feature vectors.

    Computed in blocks so a 2n x 2n matrix over 4^k-dimensional FCGR vectors
    does not need an (2n, 2n, 4^k) temporary.
    """
    m = vectors.shape[0]
    d = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        d[i] = np.abs(vectors[i][None, :] - vectors).sum(axis=1)
    np.fill_diagonal(d, 0.0)
    return d


def compute_delta(dist: np.ndarray, nat_idx: np.ndarray, syn_idx: np.ndarray,
                  triu_r: np.ndarray, triu_c: np.ndarray) -> tuple[float, float, float]:
    """Return (delta, median_nat_nat, median_syn_nat) for one labelling."""
    d_nn = dist[nat_idx[triu_r], nat_idx[triu_c]]
    d_sn = dist[np.ix_(syn_idx, nat_idx)].ravel()
    m_nn = float(np.median(d_nn))
    m_sn = float(np.median(d_sn))
    return m_sn - m_nn, m_nn, m_sn


def matched_label_permutation(
    dist: np.ndarray,
    n: int,
    delta_obs: float,
    n_perm: int,
    seed: int,
    exact_threshold: int,
) -> tuple[float, str, int]:
    """One-sided p-value for delta_obs under matched-pair label exchange.

    Windows are indexed so that row i is the natural member and row n+i the
    synthetic member of pair i. A labelling is a length-n bit vector saying
    which pairs have their two members swapped.
    """
    triu_r, triu_c = np.triu_indices(n, k=1)
    base_nat = np.arange(n, dtype=np.int64)
    base_syn = np.arange(n, 2 * n, dtype=np.int64)

    def delta_for(bits: np.ndarray) -> float:
        p_nat = np.where(bits == 0, base_nat, base_syn)
        p_syn = np.where(bits == 0, base_syn, base_nat)
        return compute_delta(dist, p_nat, p_syn, triu_r, triu_c)[0]

    if n <= exact_threshold:
        count = 0
        total = 0
        for combo in itertools.product((0, 1), repeat=n):
            if delta_for(np.array(combo, dtype=np.int64)) >= delta_obs - 1e-15:
                count += 1
            total += 1
        return float(count / total), "exact", total

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        if delta_for(rng.integers(0, 2, size=n)) >= delta_obs - 1e-15:
            count += 1
    return float((1 + count) / (1 + n_perm)), "monte_carlo", n_perm


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(df: pd.DataFrame, out_png: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["ratio"]).reset_index(drop=True)
    if d.empty:
        return
    y = np.arange(len(d))
    sig = d["q_value"] < 0.05

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(d) + 2.4),
                           layout="constrained")
    ax.axvline(1.0, color="0.35", lw=1.4, ls="--", zorder=1,
               label="natural-natural variation")
    ax.scatter(d["ratio"][sig], y[sig], s=95, color="#c0392b", zorder=3,
               label="beyond natural variation (q < 0.05)")
    ax.scatter(d["ratio"][~sig], y[~sig], s=95, facecolors="none",
               edgecolors="#2c3e50", linewidths=1.6, zorder=3,
               label="within natural variation")
    for yi, r in zip(y, d["ratio"]):
        ax.plot([1.0, r], [yi, yi], color="0.6", lw=1.1, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(d["metric"])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("median(syn-nat distance) / median(nat-nat distance)")
    ax.set_title(f"{label}: divergence relative to natural variation", pad=12)
    ax.grid(axis="x", alpha=0.25)
    # Outside the axes: the largest ratio is often the bottom row, where an
    # inset legend would sit on top of the most important marker.
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, help="Pairs CSV (id,orig,syn).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--data-root", default=None,
                    help="Prefix for relative FASTA paths in the manifest.")
    ap.add_argument("--label", default=None, help="Label used in plot titles.")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help=f"Any of: {sorted(VECTOR_METRICS | set(SCALAR_METRICS))}")
    ap.add_argument("--scalar-csv", default=None,
                    help="Optional CSV of extra per-window scalars with columns "
                         "id,metric,orig,syn (e.g. non-B DNA bp coverage).")
    ap.add_argument("--fcgr-k", type=int, default=8, help="FCGR order.")
    ap.add_argument("--kmer-k", type=int, default=6, help="k for k-mer frequencies.")
    ap.add_argument("--nullomer-k", type=int, default=11,
                    help="k for the per-window canonical nullomer fraction.")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Use only the first N pairs (0 = all).")
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="Monte Carlo permutations when exhaustive enumeration "
                         "is not used.")
    ap.add_argument("--exact-threshold", type=int, default=15,
                    help="Enumerate all 2^n labellings when n <= this.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--null-check", action="store_true",
                    help="Also run the identical test with the natural windows "
                         "split in half and pitted against each other. This is a "
                         "negative control: it must come out with ratio ~1 and a "
                         "non-significant p, otherwise the test is miscalibrated "
                         "for your data.")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def load_scalar_csv(path: Path, ids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read extra per-window scalars keyed by metric name.

    Expected columns: ``id,metric,orig,syn``. Rows whose ``id`` is not in the
    manifest are ignored; metrics missing any pair are dropped.
    """
    df = pd.read_csv(path)
    need = {"id", "metric", "orig", "syn"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path}: --scalar-csv needs columns {sorted(need)}")
    df["id"] = df["id"].astype(str)
    index = {pid: i for i, pid in enumerate(ids)}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric, grp in df.groupby("metric"):
        nat = np.full(len(ids), np.nan)
        syn = np.full(len(ids), np.nan)
        for _, r in grp.iterrows():
            i = index.get(str(r["id"]))
            if i is not None:
                nat[i] = float(r["orig"])
                syn[i] = float(r["syn"])
        if np.isfinite(nat).all() and np.isfinite(syn).all():
            out[str(metric)] = (nat, syn)
        else:
            print(f"[warn] --scalar-csv metric '{metric}' incomplete; skipped")
    return out


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root) if args.data_root else None
    df = S.load_manifest(Path(args.manifest), data_root, args.max_pairs)
    n = len(df)
    label = args.label or Path(args.manifest).stem.replace("pairs.", "")

    if n < 4:
        raise SystemExit(f"need at least 4 matched pairs, got {n}")

    unknown = set(args.metrics) - VECTOR_METRICS - set(SCALAR_METRICS)
    if unknown:
        raise SystemExit(f"unknown metric(s): {sorted(unknown)}")

    print(f"[info] {label}: {n} matched pairs, metrics={args.metrics}")

    # Read every window once; the per-metric loop reuses the sequences.
    print("[info] reading sequences ...")
    nat_seqs = [S.read_fasta_concat(p).upper() for p in df["orig"]]
    syn_seqs = [S.read_fasta_concat(p).upper() for p in df["syn"]]

    extra = {}
    if args.scalar_csv:
        extra = load_scalar_csv(Path(args.scalar_csv), df["id"].tolist())
        print(f"[info] loaded {len(extra)} extra scalar metric(s) from "
              f"{args.scalar_csv}")

    # ---- main comparison: synthetic vs natural --------------------------
    metric_features = build_metric_features(
        nat_seqs, syn_seqs, args.metrics, args, extra
    )
    rows, dist_rows = run_comparison(metric_features, n, args, "syn_vs_nat")

    # ---- negative control: natural vs natural ---------------------------
    if args.null_check:
        half = n // 2
        if half < 4:
            print("[warn] --null-check needs at least 8 pairs; skipped")
        else:
            print(f"[info] null check: {half} natural-vs-natural pairs")
            ctrl_features = build_metric_features(
                nat_seqs[:half], nat_seqs[half:2 * half], args.metrics, args,
                {k: (v[0][:half], v[0][half:2 * half]) for k, v in extra.items()},
            )
            ctrl_rows, ctrl_dist = run_comparison(
                ctrl_features, half, args, "nat_vs_nat"
            )
            rows += ctrl_rows
            dist_rows += ctrl_dist

    res = pd.DataFrame(rows)
    if "p_value" in res.columns:
        for comp, grp in res.groupby("comparison"):
            res.loc[grp.index, "q_value"] = S.bh_fdr(
                grp["p_value"].to_numpy(dtype=float)
            )
        res["beyond_natural_variation"] = (res["q_value"] < 0.05) & (res["delta"] > 0)

    per_metric = outdir / "natural_baseline.per_metric.csv"
    res.to_csv(per_metric, index=False)
    print(f"[ok] wrote {per_metric}")

    if dist_rows:
        dpath = outdir / "natural_baseline.distances.csv"
        pd.DataFrame(dist_rows).to_csv(dpath, index=False)
        print(f"[ok] wrote {dpath}")

    if args.null_check and "beyond_natural_variation" in res.columns:
        bad = res[(res["comparison"] == "nat_vs_nat")
                  & res["beyond_natural_variation"].fillna(False)]
        if len(bad):
            print(f"[warn] null check FAILED for {list(bad['metric'])}: natural "
                  f"windows were flagged as beyond natural variation. Treat the "
                  f"syn_vs_nat calls for those metrics with caution — the natural "
                  f"windows are probably not exchangeable (e.g. mixed "
                  f"chromosomes, lengths or GC regimes).")
        else:
            print("[ok] null check passed: no natural-vs-natural metric was "
                  "flagged as significant")

    if not args.no_plot and "ratio" in res.columns:
        png = outdir / "natural_baseline.png"
        make_plot(res[res["comparison"] == "syn_vs_nat"], png, label)
        print(f"[ok] wrote {png}")

    return 0


def build_metric_features(
    nat_seqs: list[str],
    syn_seqs: list[str],
    metrics: list[str],
    args: argparse.Namespace,
    extra: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    """Stack [natural windows; synthetic windows] features for every metric."""
    out: list[tuple[str, np.ndarray]] = []
    for metric in metrics:
        print(f"[info] featurising {metric} ...")
        nat_v = np.vstack([window_features(s, metric, args.fcgr_k, args.kmer_k,
                                           args.nullomer_k) for s in nat_seqs])
        syn_v = np.vstack([window_features(s, metric, args.fcgr_k, args.kmer_k,
                                           args.nullomer_k) for s in syn_seqs])
        out.append((metric, np.vstack([nat_v, syn_v])))
    for metric, (nat_s, syn_s) in extra.items():
        out.append((metric, np.concatenate([nat_s, syn_s])[:, None]))
    return out


def run_comparison(
    metric_features: list[tuple[str, np.ndarray]],
    n: int,
    args: argparse.Namespace,
    comparison: str,
) -> tuple[list[dict], list[dict]]:
    """Run the matched-label permutation test for every metric."""
    rows: list[dict] = []
    dist_rows: list[dict] = []
    triu_r, triu_c = np.triu_indices(n, k=1)

    for metric, vectors in metric_features:
        if not np.isfinite(vectors).all():
            print(f"[warn] {metric}: non-finite features; skipped")
            rows.append({"comparison": comparison, "metric": metric,
                         "n_pairs": n, "status": "non_finite"})
            continue

        dist = pairwise_l1(vectors)
        delta, med_nn, med_sn = compute_delta(
            dist, np.arange(n), np.arange(n, 2 * n), triu_r, triu_c
        )
        p, method, n_used = matched_label_permutation(
            dist, n, delta, args.n_perm, args.seed, args.exact_threshold
        )
        ratio = med_sn / med_nn if med_nn > 0 else np.nan

        rows.append({
            "comparison": comparison,
            "metric": metric,
            "n_pairs": n,
            "median_nat_nat": med_nn,
            "median_syn_nat": med_sn,
            "delta": delta,
            "ratio": ratio,
            "p_value": p,
            "perm_method": method,
            "n_perm_used": n_used,
            "status": "ok",
        })
        d_nn = dist[triu_r, triu_c]
        d_sn = dist[np.ix_(np.arange(n, 2 * n), np.arange(n))].ravel()
        dist_rows.append({
            "comparison": comparison,
            "metric": metric,
            "nat_nat_q25": float(np.percentile(d_nn, 25)),
            "nat_nat_median": med_nn,
            "nat_nat_q75": float(np.percentile(d_nn, 75)),
            "syn_nat_q25": float(np.percentile(d_sn, 25)),
            "syn_nat_median": med_sn,
            "syn_nat_q75": float(np.percentile(d_sn, 75)),
        })
        print(f"  [{comparison}] {metric:18s} nat-nat={med_nn:.4g}  "
              f"syn-nat={med_sn:.4g}  ratio={ratio:.3f}  p={p:.4g} ({method})")

    return rows, dist_rows


if __name__ == "__main__":
    raise SystemExit(main())
