#!/usr/bin/env python3
"""
Reviewer #1 (major #2) and Reviewer #2: shallow baseline classifiers.

Question: does a simple, non-deep model trained on raw k-mer-frequency vectors (or
low-order Markov transition features) distinguish synthetic from natural sequences
nearly as well as the dilated CNN (AUROC up to ~0.97)? If yes, the divergence the
CNN exploits is largely elementary compositional/statistical, not evidence of deep
hierarchical structure learning.

This mirrors the CNN protocol as closely as possible so the comparison is fair:
  * Same per-domain tag sets (euk / prok / vir).
  * Same data source: <PROJ>/data/generated/<TAG>/pairs.<TAG>.csv  (columns id,orig,syn).
  * Same chunking: fixed-length L=1024 bp chunks, sequence-level score = mean of
    chunk-level predicted probabilities (the CNN averages chunk logits).
  * Same evaluation: leave-one-tag-out CV; AUROC/F1 reported as a function of the
    number of evaluation chunks per sequence (1,2,4,8,16,32).

Models (shallow, no deep learning):
  * logreg  : L2 logistic regression on k-mer frequency vector.
  * svm_lin : linear SVM (LinearSVC) on k-mer frequency vector.

Feature sets:
  * kmer<K> : canonical (strand-collapsed) K-mer relative frequencies (default K=4).
  * markov<O>: order-O Markov conditional transition probabilities
               (markov1 = dinucleotide-conditional, 16 dims; markov2 = 64 dims).

Outputs (under --outdir):
  * baseline_evalchunks_<domain>.csv  : per holdout tag x n_eval x model x feature.
  * baseline_summary.csv              : mean/std AUROC,F1 across holdout tags.
  * baseline_vs_cnn_<domain>.png      : AUROC vs n_eval, shallow models (+ CNN ref if provided).
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import gzip
import itertools
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ----------------------------- domain tag sets (match the CNN sbatch) -----------------------------
TAGS = {
    "euk": [
        "Publish_Mus", "Publish_Gallus", "Publish_Xenopus", "Publish_Takifugu",
        "Publish_Zea", "Publish_Oryza", "Publish_Triticum", "Publish_Arabidopsis",
        "Publish_Gossypium", "Publish_Apis", "Publish_Aedes", "Publish_Branchiostoma",
        "Publish_Danio", "Publish_Bos", "Publish_Nematostella", "Publish_Drosophila",
        "Publish_Canis", "Publish_Caenorhabditis", "Publish_Saccharina", "Homo_sapiens",
    ],
    "prok": ["Publish_Mycoplasmatota", "Publish_Chlamydiota", "Publish_Pseudomonadota", "Publish_Archaea"],
    "vir": ["Nucleocytoviricota", "Peploviricota", "Uroviricota", "Preplasmiviricota", "megadna"],
}

ENC = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGT", "TGCA")


# ----------------------------- IO -----------------------------
def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def read_fasta(path: Path) -> str:
    out: List[str] = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            out.append(line.strip())
    return "".join(out).upper()


def load_sequences(proj: Path, tag: str) -> List[Tuple[str, int]]:
    """Return list of (sequence, label) for a tag. label 0=orig(natural), 1=syn."""
    pairs = proj / "data" / "generated" / tag / f"pairs.{tag}.csv"
    df = pd.read_csv(pairs)
    seqs: List[Tuple[str, int]] = []
    for _, r in df.iterrows():
        for col, label in (("orig", 0), ("syn", 1)):
            p = Path(str(r[col]))
            if not p.is_absolute():
                p = (proj / p).resolve()
            if not p.exists():
                continue
            s = read_fasta(p)
            if len(s) >= 256:
                seqs.append((s, label))
    return seqs


# ----------------------------- chunking + features -----------------------------
def sample_chunks(seq: str, L: int, n: int, rng: random.Random) -> List[str]:
    if len(seq) <= L:
        return [seq]
    starts = [rng.randint(0, len(seq) - L) for _ in range(n)]
    return [seq[s:s + L] for s in starts]


def _canonical_index_map(k: int) -> Tuple[np.ndarray, int]:
    """Map each of 4**k k-mer indices to a canonical class index. Returns (map, n_classes)."""
    bases = "ACGT"
    canon: Dict[str, int] = {}
    idx_map = np.zeros(4 ** k, dtype=np.int64)
    for i, mer in enumerate(itertools.product(bases, repeat=k)):
        s = "".join(mer)
        rc = s.translate(COMP)[::-1]
        key = min(s, rc)
        if key not in canon:
            canon[key] = len(canon)
        idx_map[i] = canon[key]
    return idx_map, len(canon)


def _seq_to_kmer_indices(seq: str, k: int) -> np.ndarray:
    codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    lut = np.full(256, -1, dtype=np.int64)
    for b, v in ENC.items():
        lut[ord(b)] = v
    c = lut[codes]
    n = len(c)
    if n < k:
        return np.empty(0, dtype=np.int64)
    valid = c >= 0
    # rolling base-4 index; invalidate windows containing non-ACGT
    idx = np.zeros(n - k + 1, dtype=np.int64)
    ok = np.ones(n - k + 1, dtype=bool)
    pw = 1
    for j in range(k):
        cj = c[k - 1 - j: n - j] if j < k else None
        col = c[(k - 1 - j):(n - j)]
        col_valid = valid[(k - 1 - j):(n - j)]
        col_safe = np.where(col_valid, col, 0)
        idx += col_safe * pw
        ok &= col_valid
        pw *= 4
    return idx[ok]


class KmerFeaturizer:
    def __init__(self, k: int, canonical: bool = True):
        self.k = k
        self.canonical = canonical
        if canonical:
            self.idx_map, self.dim = _canonical_index_map(k)
        else:
            self.idx_map, self.dim = None, 4 ** k

    def __call__(self, chunk: str) -> np.ndarray:
        idx = _seq_to_kmer_indices(chunk, self.k)
        vec = np.zeros(self.dim, dtype=np.float64)
        if idx.size == 0:
            return vec
        if self.canonical:
            idx = self.idx_map[idx]
        np.add.at(vec, idx, 1.0)
        tot = vec.sum()
        if tot > 0:
            vec /= tot
        return vec


class MarkovFeaturizer:
    """Order-O Markov conditional transition probabilities P(next | prev O-mer)."""
    def __init__(self, order: int):
        self.order = order
        self.kf = KmerFeaturizer(order + 1, canonical=False)
        self.dim = 4 ** (order + 1)
        self.n_contexts = 4 ** order

    def __call__(self, chunk: str) -> np.ndarray:
        idx = _seq_to_kmer_indices(chunk, self.order + 1)
        counts = np.zeros(self.dim, dtype=np.float64)
        if idx.size:
            np.add.at(counts, idx, 1.0)
        counts = counts.reshape(self.n_contexts, 4)
        rowsum = counts.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        probs = counts / rowsum
        return probs.reshape(-1)


def make_featurizer(name: str):
    if name.startswith("kmer"):
        return KmerFeaturizer(int(name[4:]), canonical=True)
    if name.startswith("markov"):
        return MarkovFeaturizer(int(name[6:]))
    raise ValueError(f"unknown feature set: {name}")


# ----------------------------- build chunk matrix per tag -----------------------------
def build_tag_features(
    proj: Path, tag: str, featurizer, L: int, n_chunks: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[n_seq*n_chunks, dim], y_chunk, seq_id_chunk)."""
    rng = random.Random(seed)
    seqs = load_sequences(proj, tag)
    X, y, sid = [], [], []
    for si, (seq, label) in enumerate(seqs):
        for ch in sample_chunks(seq, L, n_chunks, rng):
            X.append(featurizer(ch))
            y.append(label)
            sid.append(si)
    if not X:
        return np.empty((0, featurizer.dim)), np.empty(0), np.empty(0)
    return np.vstack(X), np.asarray(y), np.asarray(sid)


# ----------------------------- model wrappers -----------------------------
def fit_model(name: str, X: np.ndarray, y: np.ndarray):
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif name == "svm_lin":
        clf = LinearSVC(C=1.0)
    else:
        raise ValueError(name)
    clf.fit(X, y)
    return clf


def chunk_scores(name: str, clf, X: np.ndarray) -> np.ndarray:
    if name == "logreg":
        return clf.predict_proba(X)[:, 1]
    # LinearSVC: use decision_function as a monotone score
    return clf.decision_function(X)


# ----------------------------- main eval -----------------------------
def evaluate_domain(
    proj: Path, domain: str, featurizer_name: str, model_names: Sequence[str],
    L: int, n_chunks: int, n_eval_list: Sequence[int], seed: int,
) -> pd.DataFrame:
    featurizer = make_featurizer(featurizer_name)
    tags = TAGS[domain]

    print(f"[{domain}/{featurizer_name}] caching features for {len(tags)} tags...")
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for t in tags:
        cache[t] = build_tag_features(proj, t, featurizer, L, n_chunks, seed)
        print(f"  {t}: {cache[t][0].shape[0]} chunks")

    rows = []
    for holdout in tags:
        Xtr_parts, ytr_parts = [], []
        for t in tags:
            if t == holdout:
                continue
            Xt, yt, _ = cache[t]
            if Xt.shape[0]:
                Xtr_parts.append(Xt)
                ytr_parts.append(yt)
        if not Xtr_parts:
            continue
        Xtr = np.vstack(Xtr_parts)
        ytr = np.concatenate(ytr_parts)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)

        Xte, yte, sid = cache[holdout]
        if Xte.shape[0] == 0 or len(np.unique(yte)) < 2:
            continue
        Xte_s = scaler.transform(Xte)

        for model_name in model_names:
            clf = fit_model(model_name, Xtr_s, ytr)
            sc = chunk_scores(model_name, clf, Xte_s)

            # aggregate chunk scores to sequence level for each n_eval
            seq_ids = np.unique(sid)
            seq_label = {s: yte[sid == s][0] for s in seq_ids}
            rng = np.random.default_rng(seed)
            for n_eval in n_eval_list:
                seq_score, seq_y = [], []
                for s in seq_ids:
                    mask = np.where(sid == s)[0]
                    take = mask if len(mask) <= n_eval else rng.choice(mask, n_eval, replace=False)
                    seq_score.append(float(np.mean(sc[take])))
                    seq_y.append(seq_label[s])
                seq_y = np.asarray(seq_y)
                seq_score = np.asarray(seq_score)
                if len(np.unique(seq_y)) < 2:
                    continue
                auc = roc_auc_score(seq_y, seq_score)
                pred = (seq_score >= np.median(seq_score)).astype(int)
                f1 = f1_score(seq_y, pred)
                rows.append({
                    "domain": domain, "feature": featurizer_name, "model": model_name,
                    "holdout_tag": holdout, "n_eval": n_eval, "auc": auc, "f1": f1,
                    "n_test_seq": len(seq_y), "feat_dim": featurizer.dim,
                })
        print(f"  holdout={holdout} done")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proj", default=_ROOT)
    ap.add_argument("--outdir", default=f"{_ROOT}/revisions/results/baseline_classifier")
    ap.add_argument("--domains", nargs="+", default=["euk", "prok", "vir"])
    ap.add_argument("--features", nargs="+", default=["kmer4", "markov1", "kmer6"])
    ap.add_argument("--models", nargs="+", default=["logreg", "svm_lin"])
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--n-chunks", type=int, default=32, help="chunks cached per sequence (max n_eval)")
    ap.add_argument("--n-eval-list", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    proj = Path(args.proj)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for domain in args.domains:
        for feat in args.features:
            df = evaluate_domain(
                proj, domain, feat, args.models, args.L, args.n_chunks,
                args.n_eval_list, args.seed,
            )
            if not df.empty:
                df.to_csv(outdir / f"baseline_evalchunks_{domain}_{feat}.csv", index=False)
                all_rows.append(df)

    if not all_rows:
        print("[err] no results produced")
        return 1
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(outdir / "baseline_evalchunks_all.csv", index=False)

    summary = (
        combined.groupby(["domain", "feature", "model", "n_eval"])
        .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
             f1_mean=("f1", "mean"), f1_std=("f1", "std"),
             n_tags=("holdout_tag", "nunique"))
        .reset_index()
    )
    summary.to_csv(outdir / "baseline_summary.csv", index=False)
    print("\n[summary] peak AUROC per domain/feature/model (max over n_eval):")
    peak = summary.loc[summary.groupby(["domain", "feature", "model"])["auc_mean"].idxmax()]
    print(peak[["domain", "feature", "model", "n_eval", "auc_mean", "auc_std"]].to_string(index=False))
    print(f"\n[ok] wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
