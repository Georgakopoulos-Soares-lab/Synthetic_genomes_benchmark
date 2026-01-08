#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import gzip
import subprocess
from pathlib import Path
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "orig", "syn"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Manifest must contain columns: {sorted(need)}")
    return df


def _cache_key(p: Path) -> str:
    return hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()


def ensure_plain_fasta(in_path: Path, cache_dir: Path) -> Path:
    if in_path.suffix != ".gz":
        return in_path
    ensure_dir(cache_dir)
    stem = in_path.name[:-3]
    out = cache_dir / f"{_cache_key(in_path)}.{stem}"
    if out.exists() and out.stat().st_size > 0:
        return out
    with gzip.open(in_path, "rt") as fin, out.open("w") as fout:
        fout.write(fin.read())
    return out


def have_outputs(pair_dir: Path, pid: str) -> bool:
    o = list(pair_dir.glob(f"{pid}_orig.*"))
    s = list(pair_dir.glob(f"{pid}_syn.*"))
    return bool(o) and bool(s)


def run_one(gfa_bin: Path, fasta: Path, out_prefix: str, work_dir: Path) -> None:
    ensure_dir(work_dir)
    cmd = [str(gfa_bin), "-skipWGET", "-skipAPR", "-skipZ", "-seq", str(fasta), "-out", out_prefix]
    subprocess.run(cmd, cwd=str(work_dir), check=False)  # gfa can be noisy; don't hard-fail


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True, help="results/<TAG>/nonbdna")
    ap.add_argument("--gfa-bin", required=True, help="Path to gfa binary")
    ap.add_argument("--max-pairs", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()
    gfa_bin = Path(args.gfa_bin).resolve()

    if not gfa_bin.exists():
        raise SystemExit(f"gfa binary not found: {gfa_bin}")

    calls_root = outdir / "calls" / "nonbgfa"
    tmp_cache = outdir / "tmp" / "fasta_cache"
    ensure_dir(calls_root); ensure_dir(tmp_cache)

    df = read_manifest(manifest)
    if args.max_pairs and args.max_pairs > 0:
        df = df.head(args.max_pairs).copy()

    for _, r in df.iterrows():
        pid = str(r["id"])
        orig_in = Path(str(r["orig"]))
        syn_in = Path(str(r["syn"]))
        if (not orig_in.exists()) or (not syn_in.exists()):
            continue

        pair_dir = calls_root / pid
        ensure_dir(pair_dir)

        if have_outputs(pair_dir, pid):
            continue

        orig = ensure_plain_fasta(orig_in, tmp_cache)
        syn = ensure_plain_fasta(syn_in, tmp_cache)

        run_one(gfa_bin, orig, f"{pid}_orig", pair_dir)
        run_one(gfa_bin, syn,  f"{pid}_syn",  pair_dir)


if __name__ == "__main__":
    main()
