#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_manifest.py

Build the ``pairs.<TAG>.csv`` manifest that every benchmark in this repository
takes as input.

The manifest is a CSV with the columns ``id,orig,syn``, one row per matched
pair of windows, where ``orig`` points at a natural FASTA and ``syn`` at the
synthetic sequence generated for that same locus. Extra columns are preserved
by the benchmarks and ignored unless they are asked for (``context_decay.py``
reads ``seed_len`` if present).

Three ways to build one
-----------------------
``--mode regex`` (default)
    Pair files by a shared key extracted with regular expressions. This handles
    the common case where natural and synthetic files are named differently but
    both encode the same locus, e.g.

        orig.chr1.165826768.300000.fa
        syn_chr1:165826769-166126768.fasta

    Give a pattern for each side whose capture groups produce the same key. The
    default patterns understand ``<contig>.<start>.<length>`` on the natural
    side and ``<contig>:<start+1>-<end>`` on the synthetic side, i.e. the
    layout produced by ``generation/``.

``--mode sorted``
    Pair the i-th natural file with the i-th synthetic file after sorting both
    lists. Use only when the two directories are known to correspond
    positionally; the script refuses if the counts differ.

``--mode records``
    Both inputs are single multi-record FASTA files and records are paired by
    order within the file. Each record is written out as its own FASTA under
    ``--split-dir`` so the per-window benchmarks can address them individually.

Validation
----------
Whatever the mode, the resulting pairs are checked before writing: both files
must exist, be non-empty, and contain sequence. Length differences beyond
``--max-length-ratio`` are reported, since a badly mismatched pair usually
means the pairing rule is wrong rather than the generator being unusual.

Examples
--------
    # Natural and synthetic in one directory, named as the generator writes them
    python scripts/make_manifest.py --tag Homo_sapiens \\
        --orig-dir data/Homo_sapiens --syn-dir data/Homo_sapiens \\
        --out manifests/pairs.Homo_sapiens.csv

    # Two directories, positional pairing
    python scripts/make_manifest.py --tag MyRun --mode sorted \\
        --orig-dir natural/ --syn-dir generated/ --out manifests/pairs.MyRun.csv

    # Two multi-record FASTAs
    python scripts/make_manifest.py --tag MyRun --mode records \\
        --orig-fasta natural.fa --syn-fasta generated.fa \\
        --split-dir data/MyRun --out manifests/pairs.MyRun.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))
import _seqio as S  # noqa: E402

# Natural: orig.<contig>.<start>.<length>.fa  -> key (contig, start)
DEFAULT_ORIG_PATTERN = r"^orig\.(?P<contig>.+)\.(?P<start>\d+)\.(?P<length>\d+)\.(?:fa|fasta)$"
# Synthetic: syn_<contig>:<start+1>-<end>.fasta -> key (contig, start)
DEFAULT_SYN_PATTERN = r"^syn_(?P<contig>.+):(?P<start1>\d+)-(?P<end>\d+)\.(?:fa|fasta)$"

FASTA_SUFFIXES = ("*.fa", "*.fasta", "*.fna", "*.fa.gz", "*.fasta.gz")


def list_fastas(directory: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in FASTA_SUFFIXES:
        out.extend(directory.glob(pattern))
    return sorted(set(out))


def key_from_match(m: re.Match) -> tuple:
    """Build a locus key from a regex match, normalising coordinates.

    ``start1`` (1-based, as in ``chr1:165826769-166126768``) is converted to
    the 0-based ``start`` used by the natural filenames so the two sides agree.
    """
    d = m.groupdict()
    contig = d.get("contig", "")
    if "start" in d and d["start"] is not None:
        return (contig, int(d["start"]))
    if "start1" in d and d["start1"] is not None:
        return (contig, int(d["start1"]) - 1)
    # No coordinates captured: fall back to the whole match's groups.
    return (contig,) + tuple(v for k, v in sorted(d.items()) if v is not None)


def pair_by_regex(orig_files, syn_files, orig_pat, syn_pat):
    """Pair files whose names yield the same locus key under the two patterns.

    ``orig_files`` and ``syn_files`` may be the same list -- natural and
    synthetic windows often live in one directory. A file that matches the
    other side's pattern is therefore never reported as unpaired.
    """
    o_re, s_re = re.compile(orig_pat), re.compile(syn_pat)
    o_keys, o_unmatched = {}, []
    for p in orig_files:
        m = o_re.match(p.name)
        if m:
            o_keys.setdefault(key_from_match(m), p)
        elif not s_re.match(p.name):
            o_unmatched.append(p)

    pairs, s_unmatched = [], []
    used = set()
    for p in sorted(syn_files):
        m = s_re.match(p.name)
        if not m:
            if not o_re.match(p.name):
                s_unmatched.append(p)
            continue
        key = key_from_match(m)
        if key in o_keys:
            pairs.append((key, o_keys[key], p))
            used.add(key)
        else:
            s_unmatched.append(p)

    leftover_orig = [p for k, p in o_keys.items() if k not in used] + o_unmatched
    return sorted(pairs), sorted(leftover_orig), sorted(s_unmatched)


def pair_by_order(orig_files, syn_files):
    if len(orig_files) != len(syn_files):
        raise SystemExit(
            f"--mode sorted needs equal counts, got {len(orig_files)} natural "
            f"and {len(syn_files)} synthetic files. Use --mode regex, or check "
            f"for stray files in the directories."
        )
    return [((i,), o, s) for i, (o, s) in
            enumerate(zip(sorted(orig_files), sorted(syn_files)))], [], []


def split_records(fasta: Path, outdir: Path, prefix: str) -> list[Path]:
    """Write each record of a multi-record FASTA to its own file."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (header, seq) in enumerate(S.iter_fasta(fasta)):
        path = outdir / f"{prefix}.{i:04d}.fa"
        with open(path, "w") as fh:
            fh.write(f">{header}\n")
            for j in range(0, len(seq), 80):
                fh.write(seq[j:j + 80] + "\n")
        written.append(path)
    if not written:
        raise SystemExit(f"{fasta}: no FASTA records found")
    return written


def make_id(key, index: int) -> str:
    if len(key) == 2 and isinstance(key[1], int):
        return f"{key[0]}_{key[1]}"
    if len(key) == 1 and isinstance(key[0], int):
        return f"pair_{index:04d}"
    return "_".join(str(k) for k in key)


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tag", required=True, help="Dataset tag, e.g. Homo_sapiens.")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default manifests/pairs.<TAG>.csv).")
    ap.add_argument("--mode", choices=["regex", "sorted", "records"],
                    default="regex")
    ap.add_argument("--orig-dir", default=None, help="Directory of natural FASTAs.")
    ap.add_argument("--syn-dir", default=None,
                    help="Directory of synthetic FASTAs (default: --orig-dir).")
    ap.add_argument("--orig-fasta", default=None,
                    help="Multi-record natural FASTA (--mode records).")
    ap.add_argument("--syn-fasta", default=None,
                    help="Multi-record synthetic FASTA (--mode records).")
    ap.add_argument("--split-dir", default=None,
                    help="Where to write per-record FASTAs (--mode records).")
    ap.add_argument("--orig-pattern", default=DEFAULT_ORIG_PATTERN,
                    help="Regex matching natural filenames (--mode regex).")
    ap.add_argument("--syn-pattern", default=DEFAULT_SYN_PATTERN,
                    help="Regex matching synthetic filenames (--mode regex).")
    ap.add_argument("--seed-len", type=int, default=0,
                    help="If >0, add a seed_len column so context_decay.py "
                         "knows how much of each synthetic sequence was the "
                         "natural prompt.")
    ap.add_argument("--relative-to", default=None,
                    help="Write paths relative to this directory (e.g. the "
                         "repository root, which is how the benchmarks resolve "
                         "them by default).")
    ap.add_argument("--max-length-ratio", type=float, default=1.5,
                    help="Warn when the longer window of a pair exceeds the "
                         "shorter by more than this factor.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out) if args.out else Path("manifests") / f"pairs.{args.tag}.csv"

    if args.mode == "records":
        if not args.orig_fasta or not args.syn_fasta:
            raise SystemExit("--mode records needs --orig-fasta and --syn-fasta")
        split_dir = Path(args.split_dir or f"data/{args.tag}")
        orig_files = split_records(Path(args.orig_fasta), split_dir, "orig")
        syn_files = split_records(Path(args.syn_fasta), split_dir, "syn")
        print(f"[info] split {len(orig_files)} natural and {len(syn_files)} "
              f"synthetic records into {split_dir}")
        if len(orig_files) != len(syn_files):
            raise SystemExit(
                f"record counts differ ({len(orig_files)} vs {len(syn_files)}); "
                f"records are paired by order, so they must match"
            )
        pairs, extra_o, extra_s = pair_by_order(orig_files, syn_files)
    else:
        if not args.orig_dir:
            raise SystemExit("--orig-dir is required")
        orig_dir = Path(args.orig_dir)
        syn_dir = Path(args.syn_dir) if args.syn_dir else orig_dir
        orig_files = list_fastas(orig_dir)
        syn_files = list_fastas(syn_dir)
        if not orig_files or not syn_files:
            raise SystemExit(
                f"no FASTA files found ({len(orig_files)} in {orig_dir}, "
                f"{len(syn_files)} in {syn_dir})"
            )
        if args.mode == "regex":
            pairs, extra_o, extra_s = pair_by_regex(
                orig_files, syn_files, args.orig_pattern, args.syn_pattern
            )
        else:
            pairs, extra_o, extra_s = pair_by_order(orig_files, syn_files)

    if not pairs:
        raise SystemExit(
            "no pairs matched. Check --orig-pattern / --syn-pattern against "
            "your filenames, or use --mode sorted if the two directories "
            "correspond positionally."
        )

    # Validate before writing.
    rows, problems = [], []
    for i, (key, o, s) in enumerate(pairs):
        lo, ls = len(S.read_fasta_concat(o)), len(S.read_fasta_concat(s))
        if lo == 0 or ls == 0:
            problems.append(f"{o.name} / {s.name}: empty sequence "
                            f"({lo} vs {ls} bp)")
            continue
        ratio = max(lo, ls) / min(lo, ls)
        if ratio > args.max_length_ratio:
            problems.append(f"{o.name} / {s.name}: lengths differ {ratio:.1f}x "
                            f"({lo} vs {ls} bp)")
        row = {
            "id": make_id(key, i),
            "orig": str(rel(o, args.relative_to)),
            "syn": str(rel(s, args.relative_to)),
            "orig_len": lo,
            "syn_len": ls,
        }
        if args.seed_len > 0:
            row["seed_len"] = args.seed_len
        rows.append(row)

    if not rows:
        raise SystemExit("every candidate pair failed validation; see above")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {out} with {len(rows)} pairs")
    if extra_o:
        print(f"[warn] {len(extra_o)} natural file(s) had no synthetic partner, "
              f"e.g. {extra_o[0].name}")
    if extra_s:
        print(f"[warn] {len(extra_s)} synthetic file(s) had no natural partner, "
              f"e.g. {extra_s[0].name}")
    for p in problems[:10]:
        print(f"[warn] {p}")
    if len(problems) > 10:
        print(f"[warn] ... and {len(problems) - 10} more")
    return 0


def rel(path: Path, base: str | None) -> Path:
    if not base:
        return path
    try:
        return path.resolve().relative_to(Path(base).resolve())
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
