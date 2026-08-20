#!/usr/bin/env python3
"""
Reviewer #3 (minor #5): replot the k-mer spectra of Figure 1B-G with a
log-scaled x-axis so the reported differences are visible (the original linear
axes run to ~1000 while nearly all density sits below ~200-300).

Recomputes Chor-et-al.-normalised k=7 spectra (fraction of all 4^k possible
k-mer types at each abundance) from the harmonised natural vs synthetic concat
FASTAs, contig-aware (k-mers never cross record boundaries), strand-specific
(matching the manuscript default).

Panels: B Homo sapiens, C Mus musculus, D Canis lupus familiaris, E Bos taurus,
F Gallus gallus, G Xenopus tropicalis.

CPU-only. Run with system python3 from /tmp.
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARM = Path(f"{_ROOT}/results/harmonized")
FIG = Path(f"{_ROOT}/revisions/figures")
FIG.mkdir(parents=True, exist_ok=True)

K = 7
PANELS = [
    ("B", "Homo sapiens", "Publish_Human"),
    ("C", "Mus musculus", "Publish_Mus"),
    ("D", "Canis lupus familiaris", "Publish_Canis"),
    ("E", "Bos taurus", "Publish_Bos"),
    ("F", "Gallus gallus", "Publish_Gallus"),
    ("G", "Xenopus tropicalis", "Publish_Xenopus"),
]
_B2I = {65: 0, 67: 1, 71: 2, 84: 3}  # A C G T


def iter_fasta(path: Path):
    seq = []
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


def encode(s: str) -> np.ndarray:
    a = np.frombuffer(s.encode(), dtype=np.uint8)
    out = np.full(a.shape, -1, dtype=np.int64)
    for code, i in _B2I.items():
        out[a == code] = i
    return out


def kmer_counts(path: Path, k: int) -> np.ndarray:
    """Return abundance array over the 4**k k-mer space, contig-aware."""
    dim = 4 ** k
    counts = np.zeros(dim, dtype=np.int64)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    for rec in iter_fasta(path):
        idx = encode(rec)
        if idx.size < k:
            continue
        win = np.lib.stride_tricks.sliding_window_view(idx, k)
        valid = (win >= 0).all(axis=1)
        if not valid.any():
            continue
        codes = (win[valid] * powers).sum(axis=1)
        counts += np.bincount(codes, minlength=dim)
    return counts


def spectrum(counts: np.ndarray, dim: int):
    """Chor normalisation: x = abundance, y = fraction of all k-mer types."""
    max_ab = int(counts.max()) if counts.size else 0
    hist = np.bincount(counts, minlength=max_ab + 1).astype(float)
    # drop abundance 0 (k-mer types never observed) from the curve body but keep
    # for completeness; manuscript plots observed abundances
    x = np.arange(1, max_ab + 1)
    y = hist[1:max_ab + 1] / dim
    return x, y


def smooth(y, w=5):
    if len(y) < w:
        return y
    kern = np.ones(w) / w
    return np.convolve(y, kern, mode="same")


def main():
    dim = 4 ** K
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 8))
    axes = axes.ravel()
    for ax, (letter, name, tag) in zip(axes, PANELS):
        d = HARM / tag
        of = d / f"{tag}.orig.concat.fa"
        sf = d / f"{tag}.syn.concat.fa"
        if not of.exists() or not sf.exists():
            _iname = name.replace(' ', r'\ ')
            ax.set_title(f"{letter}. $\\mathit{{{_iname}}}$ (missing)")
            print(f"[warn] missing {tag}", file=sys.stderr)
            continue
        co = kmer_counts(of, K)
        cs = kmer_counts(sf, K)
        xo, yo = spectrum(co, dim)
        xs, ys = spectrum(cs, dim)
        ax.fill_between(xo, smooth(yo), color="#1b9e77", alpha=0.45, label="Natural")
        ax.fill_between(xs, smooth(ys), color="#d95f02", alpha=0.45, label="Synthetic")
        ax.plot(xo, smooth(yo), color="#1b9e77", lw=1.0)
        ax.plot(xs, smooth(ys), color="#d95f02", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel(f"k-mer abundance (k={K}, log scale)")
        ax.set_ylabel("Fraction of k-mer types")
        _iname = name.replace(' ', r'\ ')
        ax.set_title(f"{letter}. $\\mathit{{{_iname}}}$", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        print(f"[info] {tag}: max_ab nat={int(co.max())} syn={int(cs.max())}")
    fig.tight_layout()
    out = FIG / "fig1BG_kmer_spectra_logx.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
