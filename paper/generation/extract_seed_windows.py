#!/usr/bin/env python3
"""
Extract natural seed windows + reference windows for the alternative-decoding
sweep (Reviewer #3 comment #7; Reviewer #1 minor #2).

For each selected benchmark tag we read the harmonized natural ("orig") FASTA,
pick the first N records deterministically, and emit:

  - a seed FASTA  (first --seed-len bp)  -> prompt for Evo2 generation
  - a natural reference FASTA (first --target-len bp) -> for metric comparison

We also write a manifest CSV consumed by the generation launcher with one row
per (tag, window) carrying the phylotag, seed path, reference path and target
length.

CPU-only. Run with system python3 from /tmp.
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

NONBDNA = Path(_ROOT)
HARM = NONBDNA / "results" / "harmonized"
PHYLO_CSV = NONBDNA / "revisions" / "results" / "supplementary_phylotags.csv"

# Eukaryote / viral tags resolve to a single harmonized concat FASTA.
# Prokaryote tags are stored per-pair; resolved separately below.
ORIG_CONCAT = "{tag}/{tag}.orig.concat.fa"


def iter_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    hdr, buf = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if hdr is not None:
            yield hdr, "".join(buf)


def load_phylotag_map():
    """tag -> first phylotag string found in the supplementary table."""
    m = {}
    if not PHYLO_CSV.exists():
        return m
    with open(PHYLO_CSV, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            tag = (row.get("benchmark_tag") or "").strip()
            pt = (row.get("phylotag") or "").strip()
            if tag and pt and pt.lower() != "nan" and tag not in m:
                m[tag] = pt
    return m


def resolve_orig_fasta(tag):
    """Return path to the natural FASTA for a tag, or None."""
    p = HARM / ORIG_CONCAT.format(tag=tag)
    if p.exists():
        return p
    # prokaryote per-pair fallback: concatenate pair_*/<TAG>.pair_*.orig.fa
    pdir = HARM / tag / "nullomers"
    if pdir.is_dir():
        pairs = sorted(pdir.glob("pair_*"))
        if pairs:
            return ("PAIRS", pdir, pairs)
    return None


def iter_records_for_tag(tag):
    """Yield (record_id, sequence) for a tag from whichever source exists."""
    src = resolve_orig_fasta(tag)
    if src is None:
        return
    if isinstance(src, Path):
        for hdr, seq in iter_fasta(src):
            yield hdr.split()[0], seq
    else:
        _, pdir, pairs = src
        for pd in pairs:
            cands = list(pd.glob(f"{tag}.*.orig.fa")) + list(pd.glob("*.orig.fa"))
            for c in cands:
                for hdr, seq in iter_fasta(c):
                    yield f"{pd.name}.{hdr.split()[0]}", seq


def clean_dna(s):
    return "".join(ch for ch in s.upper() if ch in "ACGT")


def wrap80(seq):
    return "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="Benchmark tags (harmonized dir names).")
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--seed-len", type=int, default=3000)
    ap.add_argument("--target-len", type=int, default=300000)
    ap.add_argument("--min-len", type=int, default=None,
                    help="Skip records shorter than this (default: target-len).")
    ap.add_argument("--outdir", default=str(NONBDNA / "revisions" / "decoding_sweep" / "seeds"))
    ap.add_argument("--manifest", default=str(NONBDNA / "revisions" / "decoding_sweep" / "seed_manifest.csv"))
    args = ap.parse_args()

    min_len = args.min_len if args.min_len is not None else args.target_len
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    phylo = load_phylotag_map()
    rows = []
    for tag in args.tags:
        pt = phylo.get(tag, "")
        if not pt:
            print(f"[warn] no phylotag for {tag}; generating without one", file=sys.stderr)
        taken = 0
        for rid, seq in iter_records_for_tag(tag):
            seq = clean_dna(seq)
            if len(seq) < min_len:
                continue
            wid = f"{tag}.w{taken:02d}"
            seed_seq = seq[:args.seed_len]
            ref_seq = seq[:args.target_len]
            seed_p = outdir / f"{wid}.seed.fa"
            ref_p = outdir / f"{wid}.natref.fa"
            with open(seed_p, "w") as fh:
                fh.write(f">{wid}.seed src={rid}\n{wrap80(seed_seq)}\n")
            with open(ref_p, "w") as fh:
                fh.write(f">{wid}.natref src={rid}\n{wrap80(ref_seq)}\n")
            rows.append({
                "tag": tag,
                "window_id": wid,
                "src_record": rid,
                "phylotag": pt,
                "seed_fasta": str(seed_p),
                "natref_fasta": str(ref_p),
                "seed_len": args.seed_len,
                "target_len": args.target_len,
            })
            taken += 1
            if taken >= args.n_windows:
                break
        print(f"[info] {tag}: extracted {taken} windows", file=sys.stderr)

    with open(args.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["tag", "window_id", "src_record", "phylotag",
                            "seed_fasta", "natref_fasta", "seed_len", "target_len"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {len(rows)} windows -> {args.manifest}")


if __name__ == "__main__":
    main()
