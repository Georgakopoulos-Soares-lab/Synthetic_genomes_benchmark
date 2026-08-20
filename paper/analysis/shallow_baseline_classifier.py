#!/usr/bin/env python3
"""
Reviewer #1 (major #2) and Reviewer #2 control: shallow baseline classifier.

Question: does the dilated-CNN's high AUROC (up to 0.97) reflect deep structural
learning, or can a shallow model on raw k-mer frequency vectors / Markov
transition probabilities reach comparable performance? If a LogReg/SVM on simple
compositional features already matches the CNN, the separability is an elementary
statistical artifact (k-mer / GC drift), not long-range structural collapse.

Design mirrors the CNN protocol:
  * per domain (eukaryotes, prokaryotes, viruses)
  * leave-one-tag-out cross-validation (species for euk, phylum for prok/viral)
  * sequences split into non-overlapping 1024 bp chunks
  * chunk-level features -> chunk logits averaged to a sequence-level score
  * report AUROC and F1 (mean +/- std across held-out tags)

Features (selectable):
  * kmer      : normalized k-mer frequency vector (default k=4 -> 256 dims)
  * markov1   : order-1 Markov transition probabilities (16 dims) + monont freq
  * gc        : GC content + single-nucleotide frequencies (sanity floor, 5 dims)

Models: logistic regression and linear SVM (both L2).

Data sources (auto-resolved per tag):
  * <PROJ>/results/harmonized/<TAG>/<TAG>.{orig,syn}.concat.fa            (euk, viral)
  * <PROJ>/results/harmonized/<TAG>/nullomers/pair_*/<TAG>.pair_*.{orig,syn}.fa (prok)
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJ = Path(_ROOT)
HARM = PROJ / "results" / "harmonized"

# Domain -> tag lists. Tags must have either concat FASTAs or per-pair fa files.
DOMAINS: dict[str, list[str]] = {
    "euk": [
        "Publish_Aedes", "Publish_Apis", "Publish_Arabidopsis", "Publish_Bos",
        "Publish_Branchiostoma", "Publish_Caenorhabditis", "Publish_Canis",
        "Publish_Danio", "Publish_Drosophila", "Publish_Gallus", "Publish_Gossypium",
        "Publish_Mus", "Publish_Nematostella", "Publish_Oryza", "Publish_Saccharina",
        "Publish_Takifugu", "Publish_Triticum", "Publish_Xenopus", "Publish_Zea",
        "Publish_Human",
    ],
    "prok": [
        "Publish_Archaea", "Publish_Chlamydiota",
        "Publish_Mycoplasmatota", "Publish_Pseudomonadota",
    ],
    "vir": [
        "Kitrinoviricota", "Nucleocytoviricota", "Peploviricota",
        "Preplasmiviricota", "Uroviricota",
    ],
}

_B2I = {"A": 0, "C": 1, "G": 2, "T": 3}


def iter_fasta(path: Path):
    """Yield uppercase sequences (one per record) from a FASTA file."""
    seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq).upper()
                    seq = []
            else:
                seq.append(line.strip())
    if seq:
        yield "".join(seq).upper()


def load_tag_sequences(tag: str) -> tuple[list[str], list[str]]:
    """Return (orig_seqs, syn_seqs) for a tag from concat or per-pair FASTAs."""
    d = HARM / tag
    orig_concat = d / f"{tag}.orig.concat.fa"
    syn_concat = d / f"{tag}.syn.concat.fa"
    if orig_concat.exists() and syn_concat.exists():
        return list(iter_fasta(orig_concat)), list(iter_fasta(syn_concat))

    # per-pair layout (prokaryotes)
    pair_dir = d / "nullomers"
    if pair_dir.exists():
        orig_seqs, syn_seqs = [], []
        for pd_ in sorted(pair_dir.glob("pair_*")):
            of = next(pd_.glob("*.orig.fa"), None)
            sf = next(pd_.glob("*.syn.fa"), None)
            if of is not None:
                orig_seqs.extend(iter_fasta(of))
            if sf is not None:
                syn_seqs.extend(iter_fasta(sf))
        return orig_seqs, syn_seqs

    return [], []


def chunk_sequence(seq: str, chunk: int, max_chunks: int, rng: np.random.Generator):
    """Split into non-overlapping chunks; randomly sample up to max_chunks."""
    n = len(seq) // chunk
    if n == 0:
        return []
    starts = np.arange(n) * chunk
    if n > max_chunks:
        starts = rng.choice(starts, size=max_chunks, replace=False)
    return [seq[s : s + chunk] for s in starts]


def encode_indices(s: str) -> np.ndarray:
    arr = np.full(len(s), -1, dtype=np.int8)
    for b, i in _B2I.items():
        arr[np.frombuffer(s.encode(), dtype=np.uint8) == ord(b)] = i
    return arr


def kmer_features(s: str, k: int) -> np.ndarray:
    """Normalized k-mer frequency vector over the 4**k space (vectorized)."""
    idx = encode_indices(s).astype(np.int64)
    dim = 4 ** k
    vec = np.zeros(dim, dtype=np.float64)
    if len(idx) < k:
        return vec
    windows = np.lib.stride_tricks.sliding_window_view(idx, k)  # (L-k+1, k)
    valid = (windows >= 0).all(axis=1)
    if not valid.any():
        return vec
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    codes = (windows[valid] * powers).sum(axis=1)
    counts = np.bincount(codes, minlength=dim).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def markov1_features(s: str) -> np.ndarray:
    """Order-1 transition frequencies (16) + mononucleotide freq (4) = 20 dims."""
    idx = encode_indices(s)
    trans = np.zeros(16, dtype=np.float64)
    mono = np.zeros(4, dtype=np.float64)
    prev = -1
    tt = 0
    for b in idx:
        if b >= 0:
            mono[b] += 1
            if prev >= 0:
                trans[prev * 4 + b] += 1
                tt += 1
        prev = b
    if tt > 0:
        trans /= tt
    if mono.sum() > 0:
        mono /= mono.sum()
    return np.concatenate([trans, mono])


def gc_features(s: str) -> np.ndarray:
    idx = encode_indices(s)
    mono = np.zeros(4, dtype=np.float64)
    for b in idx:
        if b >= 0:
            mono[b] += 1
    tot = mono.sum()
    if tot > 0:
        mono /= tot
    gc = mono[1] + mono[2]
    return np.concatenate([[gc], mono])


def featurize(chunk: str, feature: str, k: int) -> np.ndarray:
    if feature == "kmer":
        return kmer_features(chunk, k)
    if feature == "markov1":
        return markov1_features(chunk)
    if feature == "gc":
        return gc_features(chunk)
    raise ValueError(feature)


def build_domain_matrix(tags, feature, k, chunk, max_chunks, seed):
    """Return dict tag -> (X chunks, y labels, seq_ids) for leave-one-tag-out."""
    rng = np.random.default_rng(seed)
    per_tag = {}
    for tag in tags:
        orig, syn = load_tag_sequences(tag)
        if not orig or not syn:
            print(f"[warn] {tag}: missing data (orig={len(orig)} syn={len(syn)})",
                  file=sys.stderr)
            continue
        X, y, sid = [], [], []
        for label, seqs in ((0, orig), (1, syn)):
            for si, seq in enumerate(seqs):
                for ch in chunk_sequence(seq, chunk, max_chunks, rng):
                    X.append(featurize(ch, feature, k))
                    y.append(label)
                    sid.append(f"{label}_{si}")
        if not X:
            continue
        per_tag[tag] = (np.vstack(X), np.array(y), np.array(sid))
        print(f"[info] {tag}: {len(y)} chunks "
              f"({int((np.array(y)==0).sum())} orig / {int((np.array(y)==1).sum())} syn)")
    return per_tag


def seq_level_scores(chunk_scores, seq_ids, y_chunks):
    """Average chunk scores within a sequence -> sequence-level score+label."""
    order = {}
    for cs, sid, yc in zip(chunk_scores, seq_ids, y_chunks):
        order.setdefault(sid, [[], None])
        order[sid][0].append(cs)
        order[sid][1] = yc
    s_scores, s_labels = [], []
    for sid, (scores, lab) in order.items():
        s_scores.append(float(np.mean(scores)))
        s_labels.append(int(lab))
    return np.array(s_scores), np.array(s_labels)


def run_domain(domain, tags, feature, k, chunk, max_chunks, model_name, seed):
    per_tag = build_domain_matrix(tags, feature, k, chunk, max_chunks, seed)
    usable = list(per_tag)
    if len(usable) < 2:
        print(f"[skip] {domain}: <2 usable tags", file=sys.stderr)
        return None

    fold_rows = []
    for holdout in usable:
        Xtr = np.vstack([per_tag[t][0] for t in usable if t != holdout])
        ytr = np.concatenate([per_tag[t][1] for t in usable if t != holdout])
        Xte, yte, sid_te = per_tag[holdout]
        if len(np.unique(yte)) < 2:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        if model_name == "logreg":
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            chunk_scores = clf.predict_proba(Xte_s)[:, 1]
        else:  # linsvm
            clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
            clf.fit(Xtr_s, ytr)
            chunk_scores = clf.decision_function(Xte_s)

        ss, sl = seq_level_scores(chunk_scores, sid_te, yte)
        if len(np.unique(sl)) < 2:
            continue
        auroc = roc_auc_score(sl, ss)
        thr = np.median(ss)
        f1 = f1_score(sl, (ss > thr).astype(int))
        fold_rows.append({"holdout": holdout, "auroc": auroc, "f1": f1,
                          "n_seq": int(len(sl))})
        print(f"[{domain}/{model_name}/{feature}] holdout={holdout:28s} "
              f"AUROC={auroc:.3f} F1={f1:.3f}")

    if not fold_rows:
        return None
    aur = np.array([r["auroc"] for r in fold_rows])
    f1s = np.array([r["f1"] for r in fold_rows])
    summary = {
        "domain": domain, "model": model_name, "feature": feature, "k": k,
        "n_folds": len(fold_rows),
        "auroc_mean": float(aur.mean()), "auroc_std": float(aur.std()),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        "folds": fold_rows,
    }
    print(f"==> {domain}/{model_name}/{feature}: "
          f"AUROC={aur.mean():.3f}+/-{aur.std():.3f}  "
          f"F1={f1s.mean():.3f}+/-{f1s.std():.3f}  (n={len(fold_rows)})")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=["euk", "prok", "vir"])
    ap.add_argument("--features", nargs="+", default=["kmer", "markov1", "gc"])
    ap.add_argument("--models", nargs="+", default=["logreg", "linsvm"])
    ap.add_argument("--k", type=int, default=4, help="k for kmer features")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-chunks", type=int, default=32,
                    help="max chunks sampled per sequence")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(PROJ / "revisions" / "results" /
                                         "shallow_baseline_results.json"))
    args = ap.parse_args()

    all_summaries = []
    for domain in args.domains:
        tags = DOMAINS[domain]
        for feature in args.features:
            for model_name in args.models:
                s = run_domain(domain, tags, feature, args.k, args.chunk,
                               args.max_chunks, model_name, args.seed)
                if s is not None:
                    all_summaries.append(s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"\n[ok] wrote {len(all_summaries)} summaries -> {out}")

    # compact table
    print("\n=== SUMMARY (AUROC mean +/- std) ===")
    print(f"{'domain':6s} {'model':8s} {'feature':8s} {'AUROC':>14s} {'F1':>14s}")
    for s in all_summaries:
        print(f"{s['domain']:6s} {s['model']:8s} {s['feature']:8s} "
              f"{s['auroc_mean']:.3f}+/-{s['auroc_std']:.3f}   "
              f"{s['f1_mean']:.3f}+/-{s['f1_std']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
