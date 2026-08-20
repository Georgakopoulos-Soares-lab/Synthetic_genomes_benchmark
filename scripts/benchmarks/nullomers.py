#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nullomers.py

Nullomer-style comparison using KMC set operations on k-mers.

For each pair (orig, syn) in a manifest CSV (id,orig,syn), compute:
- orig_unique: distinct k-mers in original
- syn_unique:  distinct k-mers in synthetic
- shared_unique: intersection size
- missing_in_syn: (orig \\ syn)
- novel_in_syn:   (syn \\ orig)
- jaccard: shared / (orig + syn - shared)
- missing_rate: missing_in_syn / orig_unique
- novel_rate:   novel_in_syn / syn_unique

and, per sequence, the absolute nullomer content:
- kmer_space: number of distinct k-mer classes that *could* be observed
- orig_nullomers / syn_nullomers: classes absent from that sequence
- orig_nullomer_fraction / syn_nullomer_fraction: absent / kmer_space

Denominator (important)
-----------------------
KMC counts canonical k-mers by default: a k-mer and its reverse complement are
collapsed into one class, so the number of observable classes is NOT 4**k but

    canonical_classes(k) = (4**k + P(k)) / 2,  P(k) = 0 for odd k, 4**(k/2) even

Dividing a canonical distinct-count by 4**k floors any "fraction absent"
statistic near 0.5 for odd k no matter what the sequence looks like, because
the count can never exceed 4**k / 2. This script picks the denominator that
matches how the k-mers were actually counted -- see --count-mode. Pass
--count-mode both-strands to make KMC count each strand separately (-b), in
which case the denominator is the full 4**k space.

Outputs:
- <outdir>/nullomers.k<K>.pair_metrics.csv

Dependencies:
- kmc
- kmc_tools

Notes:
- Uses explicit FASTA mode (-fm). Works for .fa/.fasta/.fna and gzipped FASTA as well.
- KMC output "total k-mers" in `kmc_tools info` is treated as the number of distinct k-mers in the DB.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise SystemExit(f"Missing dependency '{exe}'. Install it and ensure it is on PATH.")
    return p


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def kmer_space(k: int, canonical: bool) -> int:
    """Number of distinct k-mer classes observable under the counting mode.

    With canonical counting (KMC's default) a k-mer and its reverse complement
    are one class. A k-mer can equal its own reverse complement only when k is
    even, and there are exactly 4**(k/2) such palindromes, so

        canonical_classes(k) = (4**k + P(k)) / 2

    with P(k) = 0 for odd k and 4**(k/2) for even k. With both-strand counting
    every k-mer is its own class and the space is the full 4**k.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not canonical:
        return 4 ** k
    palindromes = 4 ** (k // 2) if k % 2 == 0 else 0
    return (4 ** k + palindromes) // 2


def _kmc_build_db(
    fasta: Path,
    db_prefix: Path,
    tmp_dir: Path,
    k: int,
    threads: int,
    max_count: int,
    canonical: bool = True,
) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _which_or_die("kmc"),
        f"-k{k}",
        "-fm",                 # IMPORTANT: multi-FASTA
        "-ci1",
        f"-cs{max_count}",
        f"-t{threads}",
    ]
    if not canonical:
        cmd.append("-b")       # count both strands separately
    cmd += [
        str(fasta),
        str(db_prefix),
        str(tmp_dir),
    ]
    _run(cmd)


def _kmc_db_size(db_prefix: Path) -> int:
    """
    Return number of distinct k-mers in a KMC database.

    Different KMC versions label this differently; accept:
      - 'total k-mers      :  <int>'  (common in KMC 3.x output)
      - 'No. of unique k-mers : <int>' (older variants)
    """
    cmd = [_which_or_die("kmc_tools"), "info", str(db_prefix)]
    p = _run(cmd)
    out = p.stdout

    # KMC 3.x often prints: "total k-mers      :  83405"
    m = re.search(r"^\s*total\s+k-mers\s*:\s*([0-9]+)\s*$", out, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return int(m.group(1))

    # Some versions: "No. of unique k-mers : <int>"
    m = re.search(r"unique\s+k-mers\s*:\s*([0-9]+)\s*$", out, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return int(m.group(1))

    raise RuntimeError(f"Could not parse DB size from `kmc_tools info` output:\n{out}")


def _kmc_simple(op: str, a: Path, b: Path, out: Path) -> None:
    cmd = [_which_or_die("kmc_tools"), "simple", str(a), str(b), op, str(out)]
    _run(cmd)


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        need = {"id", "orig", "syn"}
        if not need.issubset(set(rdr.fieldnames or [])):
            raise SystemExit(f"Manifest must have columns {sorted(need)} (got {rdr.fieldnames})")

        for i, row in enumerate(rdr, start=1):
            if not row.get("id"):
                row["id"] = f"pair_{i:03d}"
            rows.append(row)
    if not rows:
        raise SystemExit("Manifest is empty.")
    return rows


def run_nullomers(
    manifest: Path,
    outdir: Path,
    k: int,
    threads: int,
    max_pairs: int,
    max_count: int,
    keep_tmp: bool,
    canonical: bool = True,
) -> Path:
    _which_or_die("kmc")
    _which_or_die("kmc_tools")

    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root = outdir / ".tmp_kmc"
    db_root = outdir / ".kmc_db"
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest)

    out_csv = outdir / f"nullomers.k{k}.pair_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        space = kmer_space(k, canonical)
        mode = "canonical" if canonical else "both_strands"
        print(f"[info] k={k} count-mode={mode} kmer_space={space}"
              f" (4**k = {4 ** k})")
        w.writerow([
            "pair_id", "k",
            "orig_unique", "syn_unique", "shared_unique",
            "missing_in_syn", "novel_in_syn",
            "jaccard", "missing_rate", "novel_rate",
            "count_mode", "kmer_space",
            "orig_nullomers", "syn_nullomers",
            "orig_nullomer_fraction", "syn_nullomer_fraction",
            "status"
        ])

        for i, r in enumerate(rows, start=1):
            if max_pairs and i > max_pairs:
                break

            pid = str(r["id"])
            orig = Path(r["orig"])
            syn = Path(r["syn"])

            if not orig.exists() or not syn.exists():
                w.writerow([pid, k, 0, 0, 0, 0, 0, 0, 0, 0, mode, space,
                            "", "", "", "", "missing_fasta"])
                continue

            pair_tmp = tmp_root / pid
            pair_db = db_root / pid
            pair_db.mkdir(parents=True, exist_ok=True)

            db_o = pair_db / "orig"
            db_s = pair_db / "syn"
            db_i = pair_db / "intersect"
            db_m = pair_db / "missing_in_syn"
            db_n = pair_db / "novel_in_syn"

            status = "ok"
            try:
                _kmc_build_db(orig, db_o, pair_tmp / "orig", k=k, threads=threads,
                              max_count=max_count, canonical=canonical)
                _kmc_build_db(syn,  db_s, pair_tmp / "syn",  k=k, threads=threads,
                              max_count=max_count, canonical=canonical)

                orig_unique = _kmc_db_size(db_o)
                syn_unique = _kmc_db_size(db_s)

                if orig_unique == 0 or syn_unique == 0:
                    status = "empty_kmc_db"
                    w.writerow([pid, k, orig_unique, syn_unique, 0, 0, 0, 0, 0, 0,
                                mode, space, "", "", "", "", status])
                    continue

                _kmc_simple("intersect", db_o, db_s, db_i)
                _kmc_simple("kmers_subtract", db_o, db_s, db_m)  # orig \ syn
                _kmc_simple("kmers_subtract", db_s, db_o, db_n)  # syn \ orig

                shared_unique = _kmc_db_size(db_i)
                missing_in_syn = _kmc_db_size(db_m)
                novel_in_syn = _kmc_db_size(db_n)

                denom = orig_unique + syn_unique - shared_unique
                jacc = (shared_unique / denom) if denom > 0 else 0.0
                missing_rate = (missing_in_syn / orig_unique) if orig_unique > 0 else 0.0
                novel_rate = (novel_in_syn / syn_unique) if syn_unique > 0 else 0.0

                orig_nullomers = space - orig_unique
                syn_nullomers = space - syn_unique
                if orig_nullomers < 0 or syn_nullomers < 0:
                    # More distinct k-mers than the space allows: the counting
                    # mode assumed here does not match how KMC actually ran.
                    status = "denominator_mismatch"
                    print(f"[warn] {pid}: distinct k-mers exceed the {mode} "
                          f"k-mer space ({max(orig_unique, syn_unique)} > "
                          f"{space}); check --count-mode")

                w.writerow([
                    pid, k,
                    orig_unique, syn_unique, shared_unique,
                    missing_in_syn, novel_in_syn,
                    f"{jacc:.8g}", f"{missing_rate:.8g}", f"{novel_rate:.8g}",
                    mode, space,
                    orig_nullomers, syn_nullomers,
                    f"{orig_nullomers / space:.8g}", f"{syn_nullomers / space:.8g}",
                    status
                ])

            except subprocess.CalledProcessError as e:
                w.writerow([pid, k, 0, 0, 0, 0, 0, 0, 0, 0, mode, space,
                            "", "", "", "", "kmc_failed"])
            finally:
                if not keep_tmp:
                    shutil.rmtree(pair_tmp, ignore_errors=True)

    return out_csv


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Nullomer benchmark using KMC on a manifest (id,orig,syn).")
    ap.add_argument("--manifest", required=True, help="CSV with columns: id, orig, syn")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=9, help="k-mer length (default: 9)")
    ap.add_argument("--threads", type=int, default=0, help="Threads (0 = auto)")
    ap.add_argument("--max-pairs", type=int, default=0, help="If >0, process only first N pairs")
    ap.add_argument("--max-count", type=int, default=65535, help="Max counter value for KMC (-cs)")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep temporary KMC files")
    ap.add_argument("--count-mode", choices=["canonical", "both-strands"],
                    default="canonical",
                    help="How KMC counts k-mers, which sets the nullomer "
                         "denominator. 'canonical' (KMC default) collapses each "
                         "k-mer with its reverse complement and divides by "
                         "(4**k + P(k))/2; 'both-strands' passes -b and divides "
                         "by 4**k. Mixing the two floors the nullomer fraction "
                         "near 0.5 for odd k.")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    threads = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)

    out_csv = run_nullomers(
        manifest=Path(args.manifest),
        outdir=Path(args.outdir),
        k=args.k,
        threads=threads,
        max_pairs=args.max_pairs,
        max_count=args.max_count,
        keep_tmp=args.keep_tmp,
        canonical=(args.count_mode == "canonical"),
    )
    print(f"[ok] wrote: {out_csv}")


if __name__ == "__main__":
    main()
