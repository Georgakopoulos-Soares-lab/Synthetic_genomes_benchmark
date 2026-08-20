#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detectability.py

How easily can a *shallow* model tell synthetic sequence from natural?

The deep detector in ``scripts/classifier/`` answers "are these sequences
distinguishable at all". This benchmark answers a sharper question: how much of
that separability is available to a linear model reading nothing but k-mer
frequencies or low-order Markov transition probabilities? If a logistic
regression on 4-mer frequencies already matches the CNN, the divergence being
detected is elementary compositional drift, not a failure of long-range
structure -- and that changes what a generator has to fix.

It runs on CPU in minutes and needs no training infrastructure, so it doubles as
the cheap first-pass detectability score for a new generator.

Protocol
--------
* Sequences are cut into non-overlapping chunks of ``--chunk`` bp (default
  1024). Chunks containing any non-ACGT base are dropped.
* Features are computed per chunk: ``kmer<K>`` (canonical or forward k-mer
  frequencies), ``markov1`` (16 transition probabilities + 4 mononucleotide
  frequencies) or ``gc`` (GC + mononucleotide frequencies -- a deliberate
  floor: anything above this is more than GC drift).
* Cross-validation is **grouped by manifest pair**, so chunks from a natural
  window and from its matched synthetic counterpart are always in the same
  fold. Without this, a model can memorise a locus rather than learn the
  natural/synthetic distinction and AUROC is inflated.
* A sequence-level score is the mean of its chunk probabilities, matching how
  the deep detector pools chunk logits. AUROC is reported as a function of the
  number of chunks averaged (``--n-eval-list``), which shows how much sequence
  a detector needs before it becomes confident.

With several manifests passed at once, cross-validation switches to
leave-one-manifest-out: train on all other tags/species, test on the held-out
one. That measures whether the signal generalises across genomes rather than
being species-specific.

Interpretation
--------------
    AUROC ~ 0.5   synthetic is indistinguishable by shallow composition
    AUROC ~ 0.7   clear compositional drift
    AUROC > 0.9   trivially separable; a linear k-mer model suffices

Outputs
-------
    <outdir>/detectability.folds.csv     one row per fold x model x feature x n_eval
    <outdir>/detectability.summary.csv   mean/std across folds
    <outdir>/detectability.png           AUROC vs chunks averaged (unless --no-plot)

Example
-------
    python scripts/benchmarks/detectability.py \\
        --manifest manifests/pairs.Homo_sapiens.csv \\
        --outdir results/Homo_sapiens/detectability \\
        --features kmer4 markov1 gc
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seqio as S  # noqa: E402

# Below this many held-out control sequences the natural-vs-natural AUROC is
# dominated by fold-level noise and says nothing about leakage.
MIN_CONTROL_SEQUENCES = 20


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def featurize(chunk: str, feature: str, canonical: bool) -> np.ndarray:
    """Feature vector for one chunk. ``feature`` is kmer<K>, markov1 or gc."""
    m = re.fullmatch(r"kmer(\d+)", feature)
    if m:
        return S.kmer_freqs(chunk, int(m.group(1)), canonical=canonical)
    if feature == "markov1":
        idx = S.encode_acgt(chunk)
        valid = idx[idx >= 0].astype(np.int64)
        mono = np.bincount(valid, minlength=4).astype(np.float64)
        if mono.sum() > 0:
            mono = mono / mono.sum()
        # Transition counts over adjacent all-ACGT positions.
        pairs = S.kmer_counts(chunk, 2)
        total = pairs.sum()
        trans = pairs / total if total > 0 else pairs
        return np.concatenate([trans, mono])
    if feature == "gc":
        idx = S.encode_acgt(chunk)
        valid = idx[idx >= 0].astype(np.int64)
        mono = np.bincount(valid, minlength=4).astype(np.float64)
        if mono.sum() > 0:
            mono = mono / mono.sum()
        return np.concatenate([[mono[1] + mono[2]], mono])
    raise ValueError(f"unknown feature: {feature}")


def feature_dim(feature: str) -> int:
    m = re.fullmatch(r"kmer(\d+)", feature)
    if m:
        return 4 ** int(m.group(1))
    return {"markov1": 20, "gc": 5}[feature]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_chunks(
    manifests: list[Path],
    data_root: Path | None,
    chunk: int,
    max_chunks: int,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Return a long table of chunks: tag, pair, label, sequence.

    ``label`` is 0 for natural and 1 for synthetic. ``pair`` is the grouping
    key for cross-validation (both members of a matched pair share it).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mpath in manifests:
        tag = re.sub(r"^pairs\.", "", mpath.stem)
        df = S.load_manifest(mpath, data_root, max_pairs)
        for _, r in df.iterrows():
            for label, col in ((0, "orig"), (1, "syn")):
                seq = S.read_fasta_concat(r[col]).upper()
                for j, piece in enumerate(
                    S.chunk_sequence(seq, chunk, max_chunks, rng)
                ):
                    rows.append({
                        "tag": tag,
                        "pair": f"{tag}::{r['id']}",
                        "label": label,
                        "seq_id": f"{tag}::{r['id']}::{label}",
                        "chunk_index": j,
                        "chunk": piece,
                    })
        print(f"[info] {tag}: {sum(1 for x in rows if x['tag'] == tag)} chunks")
    if not rows:
        raise SystemExit("no usable chunks; check --chunk against window length")
    return pd.DataFrame(rows)


def make_folds(table: pd.DataFrame, n_folds: int, seed: int) -> list[np.ndarray]:
    """Fold assignment as boolean test masks, grouped so a pair never splits.

    A precomputed ``fold`` column is honoured if present (used by the
    natural-vs-natural control, which must control label balance per fold).
    With more than one tag present, folds are leave-one-tag-out. Otherwise
    pairs are shuffled into ``n_folds`` groups.
    """
    if "fold" in table.columns:
        names = sorted(table["fold"].unique())
        return [(table["fold"] == f).to_numpy() for f in names], list(names)

    tags = table["tag"].unique()
    if len(tags) > 1:
        return [(table["tag"] == t).to_numpy() for t in sorted(tags)], sorted(tags)

    pairs = np.array(sorted(table["pair"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_folds = int(min(n_folds, len(pairs)))
    if n_folds < 2:
        raise SystemExit("need at least 2 pairs for cross-validation")
    groups = np.array_split(pairs, n_folds)
    masks = [table["pair"].isin(set(g)).to_numpy() for g in groups]
    return masks, [f"fold{i + 1}" for i in range(n_folds)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def sequence_scores(
    chunk_scores: np.ndarray,
    seq_ids: np.ndarray,
    labels: np.ndarray,
    chunk_index: np.ndarray,
    n_eval: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Average up to ``n_eval`` chunk scores per sequence.

    Chunks are sampled without replacement so the estimate does not depend on
    position within the window; sequences with fewer than ``n_eval`` chunks use
    all of the ones they have.
    """
    order = np.argsort(seq_ids, kind="stable")
    sids = seq_ids[order]
    scores = chunk_scores[order]
    labs = labels[order]
    del chunk_index

    out_scores, out_labels = [], []
    start = 0
    for i in range(1, len(sids) + 1):
        if i == len(sids) or sids[i] != sids[start]:
            block = scores[start:i]
            if block.size > n_eval:
                pick = rng.choice(block.size, size=n_eval, replace=False)
                block = block[pick]
            out_scores.append(float(np.mean(block)))
            out_labels.append(int(labs[start]))
            start = i
    return np.asarray(out_scores), np.asarray(out_labels)


def run(
    table: pd.DataFrame,
    features: list[str],
    models: list[str],
    canonical: bool,
    n_eval_list: list[int],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    masks, fold_names = make_folds(table, n_folds, seed)
    labels = table["label"].to_numpy()
    seq_ids = table["seq_id"].to_numpy()
    chunk_index = table["chunk_index"].to_numpy()
    chunks = table["chunk"].tolist()

    rows = []
    for feature in features:
        print(f"[info] featurising {feature} "
              f"({len(chunks)} chunks x {feature_dim(feature)} dims) ...")
        X = np.vstack([featurize(c, feature, canonical) for c in chunks])

        for model_name in models:
            for mask, fold_name in zip(masks, fold_names):
                train, test = ~mask, mask
                if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
                    continue

                scaler = StandardScaler().fit(X[train])
                x_tr, x_te = scaler.transform(X[train]), scaler.transform(X[test])

                if model_name == "logreg":
                    clf = LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")
                    clf.fit(x_tr, labels[train])
                    scores = clf.predict_proba(x_te)[:, 1]
                elif model_name == "svm_lin":
                    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
                    clf.fit(x_tr, labels[train])
                    scores = clf.decision_function(x_te)
                else:
                    raise SystemExit(f"unknown model: {model_name}")

                chunk_auroc = roc_auc_score(labels[test], scores)
                rng = np.random.default_rng(seed)
                for n_eval in n_eval_list:
                    s, y = sequence_scores(scores, seq_ids[test], labels[test],
                                           chunk_index[test], n_eval, rng)
                    if len(np.unique(y)) < 2:
                        continue
                    auroc = roc_auc_score(y, s)
                    f1 = f1_score(y, (s >= np.median(s)).astype(int))
                    rows.append({
                        "fold": fold_name,
                        "model": model_name,
                        "feature": feature,
                        "n_eval_chunks": n_eval,
                        "auroc": auroc,
                        "f1": f1,
                        "chunk_auroc": chunk_auroc,
                        "n_seq": int(len(y)),
                        "n_train_chunks": int(train.sum()),
                        "n_test_chunks": int(test.sum()),
                    })
                print(f"  [{feature}/{model_name}] {fold_name}: "
                      f"chunk AUROC={chunk_auroc:.3f}")
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, out_png: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for (feature, model), grp in summary.groupby(["feature", "model"]):
        grp = grp.sort_values("n_eval_chunks")
        ax.errorbar(grp["n_eval_chunks"], grp["auroc_mean"],
                    yerr=grp["auroc_std"], marker="o", capsize=3,
                    lw=1.8, label=f"{feature} / {model}")
    ax.axhline(0.5, color="0.4", ls="--", lw=1.2)
    # Axes coordinates: the x limits are not settled until the log scale and
    # the data limits have both been applied.
    ax.text(0.01, 0.5, "chance", color="0.35", fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xlabel("chunks averaged per sequence")
    ax.set_ylabel("AUROC (natural vs synthetic)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{label}: shallow detectability", pad=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, nargs="+",
                    help="One or more pairs CSVs. Several switch the "
                         "cross-validation to leave-one-manifest-out.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--features", nargs="+", default=["kmer4", "markov1", "gc"],
                    help="kmer<K> (e.g. kmer4, kmer6), markov1, gc")
    ap.add_argument("--models", nargs="+", default=["logreg"],
                    choices=["logreg", "svm_lin"])
    ap.add_argument("--canonical", action="store_true",
                    help="Collapse k-mers with their reverse complement.")
    ap.add_argument("--chunk", type=int, default=1024, help="Chunk length (bp).")
    ap.add_argument("--max-chunks", type=int, default=32,
                    help="Max chunks sampled per window (0 = all).")
    ap.add_argument("--n-eval-list", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32],
                    help="Chunk counts to average when scoring a sequence.")
    ap.add_argument("--folds", type=int, default=5,
                    help="CV folds when a single manifest is given.")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--null-check", action="store_true",
                    help="Also score natural windows against other natural "
                         "windows. This negative control should land near "
                         "AUROC 0.5; a high value means the pipeline is leaking "
                         "(e.g. windows differ systematically by locus) and the "
                         "real score is inflated.")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else None
    manifests = [Path(m) for m in args.manifest]
    label = args.label or re.sub(r"^pairs\.", "", manifests[0].stem)

    table = build_chunks(manifests, data_root, args.chunk, args.max_chunks,
                         args.max_pairs, args.seed)
    print(f"[info] {len(table)} chunks from {table['pair'].nunique()} pairs "
          f"across {table['tag'].nunique()} tag(s)")

    folds = run(table, args.features, args.models, args.canonical,
                sorted(args.n_eval_list), args.folds, args.seed)
    if folds.empty:
        raise SystemExit("no evaluable folds; try fewer --folds or more pairs")
    folds.insert(0, "comparison", "syn_vs_nat")

    if args.null_check:
        ctrl = natural_control_table(table, args.folds, args.seed)
        if ctrl is None:
            print("[warn] --null-check needs at least 4 pairs; skipped")
        else:
            print("[info] null check: natural vs natural")
            ctrl_folds = run(ctrl, args.features, args.models, args.canonical,
                             sorted(args.n_eval_list), args.folds, args.seed)
            if not ctrl_folds.empty:
                ctrl_folds.insert(0, "comparison", "nat_vs_nat")
                folds = pd.concat([folds, ctrl_folds], ignore_index=True)

    fold_path = outdir / "detectability.folds.csv"
    folds.to_csv(fold_path, index=False)
    print(f"[ok] wrote {fold_path}")

    summary = (
        folds.groupby(["comparison", "model", "feature", "n_eval_chunks"])
        .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
             f1_mean=("f1", "mean"), f1_std=("f1", "std"),
             chunk_auroc_mean=("chunk_auroc", "mean"),
             n_folds=("auroc", "size"))
        .reset_index()
        .fillna({"auroc_std": 0.0, "f1_std": 0.0})
    )
    sum_path = outdir / "detectability.summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[ok] wrote {sum_path}")

    main_sum = summary[summary["comparison"] == "syn_vs_nat"]
    print("\n=== detectability (AUROC, mean +/- sd across folds) ===")
    top = main_sum[main_sum["n_eval_chunks"] == main_sum["n_eval_chunks"].max()]
    for _, r in top.iterrows():
        print(f"  {r['feature']:10s} {r['model']:8s} "
              f"n_eval={int(r['n_eval_chunks']):3d}  "
              f"AUROC={r['auroc_mean']:.3f} +/- {r['auroc_std']:.3f}  "
              f"F1={r['f1_mean']:.3f}")

    ctrl_sum = summary[summary["comparison"] == "nat_vs_nat"]
    if len(ctrl_sum):
        ctrl_top = ctrl_sum[
            ctrl_sum["n_eval_chunks"] == ctrl_sum["n_eval_chunks"].max()
        ]
        worst = float(ctrl_top["auroc_mean"].max())
        n_ctrl_seq = int(folds.loc[folds["comparison"] == "nat_vs_nat",
                                   "n_seq"].sum())
        print("\n=== null check: natural vs natural (should be ~0.5) ===")
        for _, r in ctrl_top.iterrows():
            print(f"  {r['feature']:10s} {r['model']:8s} "
                  f"AUROC={r['auroc_mean']:.3f} +/- {r['auroc_std']:.3f}")
        if n_ctrl_seq < MIN_CONTROL_SEQUENCES:
            print(f"[warn] null check is underpowered: only {n_ctrl_seq} held-out "
                  f"control sequences, so a fold's AUROC is 0 or 1 and the "
                  f"spread is meaningless. Interpret it only with roughly "
                  f"{MIN_CONTROL_SEQUENCES // 2}+ pairs.")
        elif worst > 0.75:
            print(f"[warn] null check FAILED (max control AUROC {worst:.3f} > "
                  f"0.75): natural windows are separable from each other, so "
                  f"the syn_vs_nat AUROC above is inflated. Use windows that "
                  f"are more homogeneous in length, chromosome and GC regime.")
        else:
            print(f"[ok] null check passed (max control AUROC {worst:.3f})")

    if not args.no_plot:
        png = outdir / "detectability.png"
        make_plot(main_sum, png, label)
        print(f"[ok] wrote {png}")
    return 0


def natural_control_table(
    table: pd.DataFrame, n_folds: int, seed: int
) -> pd.DataFrame | None:
    """Relabel natural windows into a fake natural-vs-natural comparison.

    Half the natural windows keep label 0 and the other half are relabelled 1,
    preserving the chunk structure so the control runs through exactly the same
    code path as the real comparison. Folds are assigned here rather than in
    :func:`make_folds` because each fold must contain both fake labels -- with
    random assignment most folds come out single-label and are silently
    skipped, leaving the control resting on one or two folds.
    """
    nat = table[table["label"] == 0].copy()
    pairs = np.array(sorted(nat["pair"].unique()))
    if len(pairs) < 4:
        return None
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    # Each control fold must hold several sequences of each label or its AUROC
    # collapses to 0/1 and the control reports noise as a leak.
    n_folds = int(min(n_folds, len(pairs) // 4))
    if n_folds < 2:
        return None

    # Deal alternating fake labels, then deal each label group round-robin
    # across folds, so every fold holds both labels.
    label_of, fold_of = {}, {}
    for group_label in (0, 1):
        members = pairs[group_label::2]
        for i, p in enumerate(members):
            label_of[p] = group_label
            fold_of[p] = f"fold{i % n_folds + 1}"

    nat["label"] = [label_of[p] for p in nat["pair"]]
    nat["fold"] = [fold_of[p] for p in nat["pair"]]
    nat["seq_id"] = [f"{p}::ctrl" for p in nat["pair"]]
    return nat.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
