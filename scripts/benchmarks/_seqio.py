#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
_seqio.py

Shared sequence I/O, encoding and statistics helpers for the benchmark suite.

Imported by the benchmarks that were added alongside the paired-window
benchmarks (composition, detectability, natural_baseline, context_decay). Kept
dependency-light: numpy only, plus pandas for manifest loading.

Everything here operates on the same manifest contract as the rest of the
suite: a CSV with at least the columns ``id,orig,syn`` where ``orig`` and
``syn`` are paths to FASTA files holding a matched pair of windows.
"""

from __future__ import annotations

import gzip
import itertools
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

__all__ = [
    "read_fasta_concat",
    "iter_fasta",
    "load_manifest",
    "resolve_path",
    "encode_acgt",
    "kmer_counts",
    "kmer_freqs",
    "canonical_kmer_classes",
    "fcgr_vector",
    "js_divergence",
    "l1_distance",
    "gc_content",
    "homopolymer_fraction",
    "low_complexity",
    "shannon_entropy",
    "cpg_observed_expected",
    "chunk_sequence",
    "bh_fdr",
    "signflip_pvalue",
]

_BASES = "ACGT"
_B2I = {b: i for i, b in enumerate(_BASES)}

# ASCII lookup: base index for A/C/G/T (upper and lower case), -1 otherwise.
_CODE = np.full(256, -1, dtype=np.int8)
for _b, _i in _B2I.items():
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i

# CGR quadrant bits, indexed the same way.
_BX = np.full(256, -1, dtype=np.int64)
_BY = np.full(256, -1, dtype=np.int64)
for _ch, (_x, _y) in {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}.items():
    _BX[ord(_ch)] = _BX[ord(_ch.lower())] = _x
    _BY[ord(_ch)] = _BY[ord(_ch.lower())] = _y


# ---------------------------------------------------------------------------
# FASTA / manifest I/O
# ---------------------------------------------------------------------------

def _opener(path: Path):
    return gzip.open if str(path).endswith(".gz") else open


def read_fasta_concat(path: Path) -> str:
    """Concatenate every sequence line in a FASTA file into one string.

    Record boundaries are dropped. Callers that must not count k-mers across
    contigs should use :func:`iter_fasta` instead. Case is preserved so that
    soft-masked input can still be detected; use ``.upper()`` downstream when
    case is irrelevant.
    """
    path = Path(path)
    chunks: list[str] = []
    with _opener(path)(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    return "".join(chunks)


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` for each record in a FASTA file."""
    path = Path(path)
    header: Optional[str] = None
    buf: list[str] = []
    with _opener(path)(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
    if header is not None:
        yield header, "".join(buf)


def resolve_path(p: str, data_root: Optional[Path]) -> Path:
    """Resolve a manifest path, optionally prefixed by ``data_root``.

    Absolute paths are returned unchanged. Relative paths are tried first as
    given (i.e. relative to the working directory, which is how the rest of the
    suite resolves them) and then under ``data_root``.
    """
    path = Path(p)
    if path.is_absolute() or data_root is None:
        return path
    if path.exists():
        return path
    return Path(data_root) / path


def load_manifest(
    manifest: Path,
    data_root: Optional[Path] = None,
    max_pairs: int = 0,
    require_existing: bool = True,
) -> pd.DataFrame:
    """Load a ``id,orig,syn`` manifest and resolve the FASTA paths.

    Returns the dataframe with ``orig``/``syn`` replaced by resolved ``Path``
    objects. Rows whose FASTA files are missing are dropped (with a warning)
    when ``require_existing`` is set.
    """
    manifest = Path(manifest)
    df = pd.read_csv(manifest)
    need = {"id", "orig", "syn"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(
            f"{manifest}: manifest must include columns {sorted(need)}; "
            f"missing {sorted(missing)}"
        )
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["orig"] = [resolve_path(str(p), data_root) for p in df["orig"]]
    df["syn"] = [resolve_path(str(p), data_root) for p in df["syn"]]

    if require_existing:
        ok = [o.exists() and s.exists() for o, s in zip(df["orig"], df["syn"])]
        n_bad = len(ok) - int(np.sum(ok))
        if n_bad:
            bad = df.loc[[not b for b in ok], "id"].tolist()
            print(
                f"[warn] {manifest.name}: dropping {n_bad} pair(s) with missing "
                f"FASTA files: {', '.join(bad[:5])}"
                + (" ..." if n_bad > 5 else "")
            )
        df = df.loc[ok].reset_index(drop=True)

    if max_pairs and max_pairs > 0:
        df = df.head(max_pairs).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"{manifest}: no usable pairs")
    return df


# ---------------------------------------------------------------------------
# Encoding and k-mer statistics
# ---------------------------------------------------------------------------

def encode_acgt(seq: str) -> np.ndarray:
    """Encode a DNA string as int8 indices (A=0, C=1, G=2, T=3; -1 otherwise)."""
    if not seq:
        return np.empty(0, dtype=np.int8)
    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    return _CODE[raw]


def _kmer_codes(idx: np.ndarray, k: int) -> np.ndarray:
    """Integer codes of every all-ACGT k-mer window in an encoded sequence."""
    if idx.size < k:
        return np.empty(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(idx.astype(np.int64), k)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return (win[valid] * powers).sum(axis=1)


_RC_CACHE: dict[int, np.ndarray] = {}


def _revcomp_code_table(k: int) -> np.ndarray:
    """Map each k-mer code to the code of its reverse complement.

    Cached: at k=11 this is a 4M-element table that would otherwise be rebuilt
    for every window.
    """
    if k in _RC_CACHE:
        return _RC_CACHE[k]
    codes = np.arange(4 ** k, dtype=np.int64)
    rc = np.zeros_like(codes)
    tmp = codes.copy()
    for _ in range(k):
        rc = rc * 4 + (3 - (tmp % 4))
        tmp //= 4
    _RC_CACHE[k] = rc
    return rc


def kmer_counts(seq: str, k: int, canonical: bool = False) -> np.ndarray:
    """Raw k-mer counts over the ``4**k`` space.

    With ``canonical=True`` each k-mer and its reverse complement are collapsed
    onto the lexicographically smaller of the two codes; the counts of the
    larger code are then zero. See :func:`canonical_kmer_classes` for the
    matching denominator.
    """
    codes = _kmer_codes(encode_acgt(seq), k)
    counts = np.bincount(codes, minlength=4 ** k).astype(np.float64)
    if canonical:
        rc = _revcomp_code_table(k)
        collapsed = np.zeros_like(counts)
        lo = np.minimum(np.arange(4 ** k), rc)
        np.add.at(collapsed, lo, counts)
        counts = collapsed
    return counts


def kmer_freqs(seq: str, k: int, canonical: bool = False) -> np.ndarray:
    """Normalised k-mer frequency vector (sums to 1, or all-zero if no k-mers)."""
    counts = kmer_counts(seq, k, canonical=canonical)
    total = counts.sum()
    return counts / total if total > 0 else counts


def canonical_kmer_classes(k: int) -> int:
    """Number of distinct canonical (strand-collapsed) k-mer classes.

    A k-mer equals its own reverse complement only when k is even, and there
    are exactly ``4**(k/2)`` such palindromes, so

        classes(k) = (4**k + P(k)) / 2,   P(k) = 0 (odd k) or 4**(k/2) (even k)

    This is the correct denominator whenever k-mers were counted canonically
    (which is KMC's default). Dividing a canonical count by the full ``4**k``
    space instead floors any "fraction absent" statistic near 0.5 for odd k.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    palindromes = 4 ** (k // 2) if k % 2 == 0 else 0
    return (4 ** k + palindromes) // 2


def fcgr_vector(seq: str, k: int) -> np.ndarray:
    """Normalised, flattened Frequency Chaos Game Representation (length 4**k).

    k-mers containing any non-ACGT base are skipped, matching
    ``scripts/benchmarks/fcgr.py``.
    """
    side = 1 << k
    if not seq:
        return np.zeros(side * side, dtype=np.float64)
    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    bx, by = _BX[raw], _BY[raw]
    if bx.size < k:
        return np.zeros(side * side, dtype=np.float64)
    wx = np.lib.stride_tricks.sliding_window_view(bx, k)
    wy = np.lib.stride_tricks.sliding_window_view(by, k)
    valid = (wx >= 0).all(axis=1)
    if not valid.any():
        return np.zeros(side * side, dtype=np.float64)
    powers = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    x = (wx[valid] * powers).sum(axis=1)
    y = (wy[valid] * powers).sum(axis=1)
    mat = np.bincount(y * side + x, minlength=side * side).astype(np.float64)
    total = mat.sum()
    return mat / total if total > 0 else mat


# ---------------------------------------------------------------------------
# Divergences and per-window composition metrics
# ---------------------------------------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float((p * np.log2(p / m)).sum())
    kl_qm = float((q * np.log2(q / m)).sum())
    return 0.5 * (kl_pm + kl_qm)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """L1 (cityblock) distance between two vectors."""
    return float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def gc_content(seq: str) -> float:
    """GC fraction over ACGT bases only (NaN when there are none)."""
    idx = encode_acgt(seq)
    valid = idx[idx >= 0]
    if valid.size == 0:
        return float("nan")
    return float(((valid == 1) | (valid == 2)).sum() / valid.size)


def homopolymer_fraction(seq: str, min_run: int = 5) -> float:
    """Fraction of ACGT bases inside a homopolymer run of >= ``min_run``.

    Runs are broken by any non-ACGT base. Generative models tend to inflate
    this relative to natural sequence, so it is a sensitive, cheap degradation
    signal.
    """
    idx = encode_acgt(seq)
    valid = idx[idx >= 0]
    n = valid.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    # Run-length encode via change points.
    change = np.flatnonzero(valid[1:] != valid[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    lengths = ends - starts
    return float(lengths[lengths >= min_run].sum() / n)


def shannon_entropy(seq: str, k: int = 3) -> float:
    """Shannon entropy (bits) of the k-mer distribution."""
    f = kmer_freqs(seq, k)
    f = f[f > 0]
    if f.size == 0:
        return float("nan")
    return float(-(f * np.log2(f)).sum())


def low_complexity(seq: str, k: int = 3) -> float:
    """Normalised low-complexity score ``1 - H/H_max`` over k-mers.

    0 = maximally diverse k-mer usage, 1 = a single repeated k-mer.
    """
    h = shannon_entropy(seq, k)
    if not np.isfinite(h):
        return float("nan")
    return float(1.0 - h / (2.0 * k))


def cpg_observed_expected(seq: str) -> float:
    """CpG observed/expected ratio: ``f(CG) / (f(C) * f(G))``."""
    idx = encode_acgt(seq)
    valid = idx[idx >= 0]
    if valid.size < 2:
        return float("nan")
    mono = np.bincount(valid, minlength=4).astype(np.float64)
    mono /= mono.sum()
    di = kmer_freqs(seq, 2)
    expected = mono[1] * mono[2]
    if expected <= 0:
        return float("nan")
    return float(di[1 * 4 + 2] / expected)


def chunk_sequence(
    seq: str,
    chunk: int,
    max_chunks: int = 0,
    rng: Optional[np.random.Generator] = None,
    require_acgt: bool = True,
) -> list[str]:
    """Split a sequence into non-overlapping chunks of fixed length.

    Chunks containing any non-ACGT base are dropped when ``require_acgt`` is
    set (the whole chunk is inspected, not a sample of it). When ``max_chunks``
    is positive and more chunks survive, a random subset of that size is
    returned using ``rng``.
    """
    n = len(seq) // chunk
    if n == 0:
        return []
    out = []
    for i in range(n):
        piece = seq[i * chunk:(i + 1) * chunk]
        if require_acgt:
            idx = encode_acgt(piece)
            if (idx < 0).any():
                continue
        out.append(piece)
    if max_chunks and len(out) > max_chunks:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(out), size=max_chunks, replace=False)
        out = [out[i] for i in sorted(pick)]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR. NaN p-values pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    out[valid] = q
    return out


def signflip_pvalue(
    diffs: np.ndarray,
    n_perm: int = 10000,
    seed: int = 1337,
    alternative: str = "two-sided",
    exact_threshold: int = 20,
) -> tuple[float, str, int]:
    """Paired sign-flip permutation test on the mean of ``diffs``.

    Under the null that the pairing carries no information, each difference is
    equally likely to have either sign, so flipping signs at random generates
    the null distribution of the mean. Enumerates all ``2**n`` sign patterns
    exactly when ``n <= exact_threshold``, otherwise draws ``n_perm`` Monte
    Carlo samples.

    Returns ``(p_value, method, n_used)``.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan"), "empty", 0
    obs = float(np.mean(d))

    if n <= exact_threshold:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        means = signs @ d / n
        total = means.size
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([1.0, -1.0], size=(n_perm, n))
        means = signs @ d / n
        total = n_perm
        method = "monte_carlo"

    if alternative == "two-sided":
        count = int((np.abs(means) >= abs(obs) - 1e-15).sum())
    elif alternative == "greater":
        count = int((means >= obs - 1e-15).sum())
    elif alternative == "less":
        count = int((means <= obs + 1e-15).sum())
    else:
        raise ValueError(f"unknown alternative: {alternative}")

    if method == "exact":
        return float(count / total), method, total
    return float((1 + count) / (1 + total)), method, total
