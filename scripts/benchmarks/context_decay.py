#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
context_decay.py

Does generated sequence get worse the further it runs from its prompt?

A genomic language model conditioned on a natural seed produces sequence that
is initially close to the reference and drifts as generation continues. This
benchmark measures that drift directly: it trains a small classifier to
separate natural from synthetic *within each distance bin* from the end of the
conditioning seed, and reports AUROC as a function of that distance.

    AUROC ~ 0.5 near the seed, rising with distance
        -> the model tracks its context at first and loses it further out
    AUROC flat and high everywhere
        -> the output never resembled the reference; conditioning is not
           the limiting factor
    AUROC flat and low everywhere
        -> no measurable decay at this window length

Running it at several seed lengths answers the practical question that follows:
does a longer prompt delay the onset of decay, or only shift the curve?

Cross-validation
----------------
AUROC is estimated leave-one-window-out: for each distance bin, the classifier
trains on chunks from all windows except one and is evaluated on the held-out
window. Pooling chunks from the same window across train and test would let the
model recognise the locus instead of the natural/synthetic distinction, which
inflates AUROC -- badly so in near-seed bins where synthetic and natural still
share sequence.

Region comparison
-----------------
With ``--compare-regions`` the per-window AUROCs are averaged over a near-seed
region and a long-range region and compared with an exact paired sign-flip
permutation test, giving a single "did it decay?" p-value per manifest.

Outputs
-------
    <outdir>/context_decay.per_window.csv   one row per (bin, held-out window)
    <outdir>/context_decay.by_bin.csv       mean/sd/median AUROC per bin
    <outdir>/context_decay.regions.csv      near vs long-range test (optional)
    <outdir>/context_decay.png              decay curve (unless --no-plot)

Example
-------
    python scripts/benchmarks/context_decay.py \\
        --manifest manifests/pairs.Homo_sapiens.csv \\
        --outdir results/Homo_sapiens/context_decay \\
        --seed-len 3000 --bin-size 20000 --compare-regions
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seqio as S  # noqa: E402


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def bin_chunks(
    seq: str,
    start: int,
    end: int,
    chunk: int,
    max_chunks: int,
    rng: np.random.Generator,
) -> list[str]:
    """Non-overlapping all-ACGT chunks from ``seq[start:end]``."""
    if start >= len(seq):
        return []
    return S.chunk_sequence(seq[start:min(end, len(seq))], chunk, max_chunks, rng)


def collect(
    df: pd.DataFrame,
    seed_len_col: str | None,
    seed_len: int,
    bin_size: int,
    max_distance: int,
    chunk: int,
    max_chunks: int,
    kmer_k: int,
    rng: np.random.Generator,
) -> dict[float, dict[str, dict[str, np.ndarray]]]:
    """Featurise every window into ``{bin_start_bp: {window: {syn, nat}}}``."""
    per_bin: dict[float, dict[str, dict[str, np.ndarray]]] = {}

    for _, row in df.iterrows():
        wid = str(row["id"])
        nat = S.read_fasta_concat(row["orig"]).upper()
        syn = S.read_fasta_concat(row["syn"]).upper()
        sl = int(row[seed_len_col]) if seed_len_col else seed_len

        usable = min(len(nat), len(syn)) - sl
        if usable < chunk:
            print(f"[warn] {wid}: only {usable} bp past the seed; skipped")
            continue
        limit = usable if max_distance <= 0 else min(usable, max_distance)

        for b0 in range(0, limit, bin_size):
            b1 = min(b0 + bin_size, limit)
            if b1 - b0 < chunk:
                continue
            nat_c = bin_chunks(nat, sl + b0, sl + b1, chunk, max_chunks, rng)
            syn_c = bin_chunks(syn, sl + b0, sl + b1, chunk, max_chunks, rng)
            if not nat_c or not syn_c:
                continue
            slot = per_bin.setdefault(float(b0), {})
            slot[wid] = {
                "nat": np.vstack([S.kmer_freqs(c, kmer_k) for c in nat_c]),
                "syn": np.vstack([S.kmer_freqs(c, kmer_k) for c in syn_c]),
            }
        print(f"[info] {wid}: seed={sl} bp, {limit} bp analysed")
    return per_bin


# ---------------------------------------------------------------------------
# Leave-one-window-out AUROC
# ---------------------------------------------------------------------------

def lowo_auroc(window_feats: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    """AUROC for each held-out window, training on all the others."""
    wids = sorted(window_feats)
    out: dict[str, float] = {}
    for held in wids:
        x_tr, y_tr = [], []
        for wid in wids:
            if wid == held:
                continue
            d = window_feats[wid]
            for key, lab in (("syn", 1.0), ("nat", 0.0)):
                if d[key].shape[0]:
                    x_tr.append(d[key])
                    y_tr.append(np.full(d[key].shape[0], lab))
        d_te = window_feats[held]
        if not x_tr or not d_te["syn"].shape[0] or not d_te["nat"].shape[0]:
            out[held] = np.nan
            continue

        x_tr_m = np.vstack(x_tr)
        y_tr_m = np.concatenate(y_tr)
        if len(np.unique(y_tr_m)) < 2:
            out[held] = np.nan
            continue

        x_te = np.vstack([d_te["syn"], d_te["nat"]])
        y_te = np.concatenate([
            np.ones(d_te["syn"].shape[0]), np.zeros(d_te["nat"].shape[0])
        ])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(x_tr_m, y_tr_m)
        out[held] = float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(by_bin: pd.DataFrame, out_png: Path, label: str,
              seed_desc: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    d = by_bin.sort_values("bin_start_bp")
    x = d["bin_center_bp"] / 1000.0
    ax.plot(x, d["auroc_mean"], marker="o", lw=2.0, color="#c0392b",
            label="AUROC (leave-one-window-out)")
    ax.fill_between(x, d["auroc_mean"] - d["auroc_sd"],
                    d["auroc_mean"] + d["auroc_sd"], alpha=0.18,
                    color="#c0392b", lw=0)
    ax.axhline(0.5, color="0.4", ls="--", lw=1.2)
    ax.text(x.min(), 0.507, " indistinguishable from natural", color="0.35",
            fontsize=8, va="bottom")
    ax.set_xlabel("distance from end of conditioning seed (kb)")
    ax.set_ylabel("AUROC (natural vs synthetic)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{label}: context decay ({seed_desc})", pad=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
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
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed-len", type=int, default=0,
                    help="Length in bp of the natural prompt at the start of "
                         "each synthetic sequence. Ignored if the manifest has "
                         "a seed_len column.")
    ap.add_argument("--seed-len-column", default="seed_len",
                    help="Manifest column holding a per-window seed length.")
    ap.add_argument("--bin-size", type=int, default=20000,
                    help="Distance bin width in bp.")
    ap.add_argument("--max-distance", type=int, default=0,
                    help="Stop this many bp past the seed (0 = use whole window).")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Max chunks per window per bin (0 = all).")
    ap.add_argument("--kmer-k", type=int, default=6)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--compare-regions", action="store_true",
                    help="Test near-seed vs long-range AUROC with a paired "
                         "sign-flip permutation test.")
    ap.add_argument("--near-max-kb", type=float, default=20.0,
                    help="Bins starting below this are the near-seed region.")
    ap.add_argument("--long-min-kb", type=float, default=40.0,
                    help="Bins starting at or above this are the long-range "
                         "region.")
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
    seed_col = args.seed_len_column if args.seed_len_column in df.columns else None
    if seed_col is None and args.seed_len <= 0:
        raise SystemExit(
            "No seed length available. Pass --seed-len <bp>, or add a "
            f"'{args.seed_len_column}' column to the manifest. If the synthetic "
            "sequences were not generated from a natural prompt, this benchmark "
            "does not apply -- use detectability.py instead."
        )
    seed_desc = (f"per-window {seed_col}" if seed_col
                 else f"{args.seed_len / 1000:g} kb seed")

    if len(df) < 3:
        raise SystemExit("leave-one-window-out needs at least 3 windows")

    rng = np.random.default_rng(args.seed)
    per_bin = collect(df, seed_col, args.seed_len, args.bin_size,
                      args.max_distance, args.chunk, args.max_chunks,
                      args.kmer_k, rng)
    if not per_bin:
        raise SystemExit("no usable distance bins; check --seed-len and --bin-size")

    rows = []
    for b0 in sorted(per_bin):
        feats = per_bin[b0]
        if len(feats) < 3:
            continue
        for wid, auroc in lowo_auroc(feats).items():
            rows.append({
                "bin_start_bp": b0,
                "bin_end_bp": b0 + args.bin_size,
                "bin_center_bp": b0 + args.bin_size / 2.0,
                "held_out_window": wid,
                "auroc": auroc,
                "n_syn_chunks": int(feats[wid]["syn"].shape[0]),
                "n_nat_chunks": int(feats[wid]["nat"].shape[0]),
                "n_train_windows": len(feats) - 1,
            })
        done = [r["auroc"] for r in rows if r["bin_start_bp"] == b0]
        print(f"  bin {b0/1000:6.1f}-{(b0+args.bin_size)/1000:.1f} kb: "
              f"AUROC={np.nanmean(done):.3f} (n={len(done)} windows)")

    if not rows:
        raise SystemExit("no bin had at least 3 windows")

    per_window = pd.DataFrame(rows)
    pw_path = outdir / "context_decay.per_window.csv"
    per_window.to_csv(pw_path, index=False)
    print(f"[ok] wrote {pw_path}")

    by_bin = (
        per_window.groupby(["bin_start_bp", "bin_end_bp", "bin_center_bp"])
        .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
             auroc_median=("auroc", "median"), n_windows=("auroc", "count"))
        .reset_index()
        .fillna({"auroc_sd": 0.0})
    )
    bb_path = outdir / "context_decay.by_bin.csv"
    by_bin.to_csv(bb_path, index=False)
    print(f"[ok] wrote {bb_path}")

    if args.compare_regions:
        near = per_window[per_window["bin_start_bp"] < args.near_max_kb * 1000]
        far = per_window[per_window["bin_start_bp"] >= args.long_min_kb * 1000]
        if near.empty or far.empty:
            print("[warn] --compare-regions: one region is empty; "
                  "adjust --near-max-kb / --long-min-kb")
        else:
            near_w = near.groupby("held_out_window")["auroc"].mean()
            far_w = far.groupby("held_out_window")["auroc"].mean()
            shared = sorted(set(near_w.index) & set(far_w.index))
            diffs = np.array([far_w[w] - near_w[w] for w in shared])
            p, method, n_used = S.signflip_pvalue(
                diffs, n_perm=10000, seed=args.seed, alternative="greater"
            )
            reg = pd.DataFrame([{
                "n_windows": len(shared),
                "near_region_kb": f"0-{args.near_max_kb:g}",
                "long_region_kb": f"{args.long_min_kb:g}+",
                "near_auroc_mean": float(np.mean([near_w[w] for w in shared])),
                "long_auroc_mean": float(np.mean([far_w[w] for w in shared])),
                "auroc_increase": float(np.mean(diffs)),
                "p_value": p,
                "perm_method": method,
                "n_perm_used": n_used,
            }])
            rpath = outdir / "context_decay.regions.csv"
            reg.to_csv(rpath, index=False)
            r = reg.iloc[0]
            print(f"\n=== region comparison ({len(shared)} windows) ===")
            print(f"  near-seed  (0-{args.near_max_kb:g} kb): "
                  f"AUROC={r['near_auroc_mean']:.3f}")
            print(f"  long-range ({args.long_min_kb:g}+ kb): "
                  f"AUROC={r['long_auroc_mean']:.3f}")
            print(f"  increase={r['auroc_increase']:+.3f}  p={p:.4g} ({method})")
            print(f"[ok] wrote {rpath}")

    if not args.no_plot:
        png = outdir / "context_decay.png"
        make_plot(by_bin, png, label, seed_desc)
        print(f"[ok] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
