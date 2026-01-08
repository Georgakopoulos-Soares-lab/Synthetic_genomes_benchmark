#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
import pandas as pd
import gzip


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


def run_zseeker_on_fasta(fasta: Path, work_dir: Path, n_jobs: int) -> Path:
    ensure_dir(work_dir)
    out_dir = work_dir / "zdna_extractions"
    ensure_dir(out_dir)

    cmd = ["ZSeeker", "--fasta", str(fasta), "--n_jobs", str(n_jobs)]
    subprocess.run(cmd, cwd=str(work_dir), check=True)

    cands = sorted(out_dir.glob("*zdna_score.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No ZSeeker output found under {out_dir}")
    return cands[0]


def _read_zseeker_table(path: Path) -> pd.DataFrame:
    with path.open("r", errors="ignore") as fh:
        first = fh.readline()
    sep = "," if first.count(",") > first.count("\t") else "\t"
    return pd.read_csv(path, sep=sep)


def harmonize_zseeker(tab_path: Path, pair_id: str, which: str, genome_id: str) -> pd.DataFrame:
    df = _read_zseeker_table(tab_path).dropna(subset=["Start", "End"])
    out = pd.DataFrame({
        "pair_id": pair_id,
        "which": which,
        "genome_id": genome_id,
        "source": "zseeker",
        "class": "Z_DNA",
        "contig": df["Chromosome"].astype(str),
        "start": pd.to_numeric(df["Start"], errors="coerce").astype("Int64"),
        "end": pd.to_numeric(df["End"], errors="coerce").astype("Int64"),
        "strand": ".",
        "score": pd.to_numeric(df.get("Z-DNA Score", pd.NA), errors="coerce"),
    }).dropna(subset=["start", "end"])
    out["length"] = (out["end"] - out["start"]).astype("Int64")
    return out


def genome_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suf in [".fna", ".fa", ".fasta"]:
        if name.endswith(suf):
            name = name[:-len(suf)]
            break
    return name


def summarize_metrics(h: pd.DataFrame) -> pd.DataFrame:
    if h.empty:
        return pd.DataFrame(columns=["pair_id", "which", "n_hits", "bp_covered"])
    grp = h.groupby(["pair_id", "which"], dropna=False).agg(
        n_hits=("contig", "count"),
        bp_covered=("length", "sum"),
    ).reset_index()
    return grp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True, help="results/<TAG>/nonbdna")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()

    harmon_dir = outdir / "harmonized"
    metrics_dir = outdir / "metrics"
    tmp_dir = outdir / "tmp" / "zseeker"
    cache_dir = outdir / "tmp" / "fasta_cache"
    ensure_dir(harmon_dir); ensure_dir(metrics_dir); ensure_dir(tmp_dir); ensure_dir(cache_dir)

    df = read_manifest(manifest)
    if args.max_pairs and args.max_pairs > 0:
        df = df.head(args.max_pairs).copy()

    rows = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        orig_in = Path(str(r["orig"]))
        syn_in = Path(str(r["syn"]))

        out_parq = harmon_dir / f"{pid}.zseeker.parquet"
        if out_parq.exists() and out_parq.stat().st_size > 0:
            continue

        if (not orig_in.exists()) or (not syn_in.exists()):
            continue

        orig = ensure_plain_fasta(orig_in, cache_dir)
        syn = ensure_plain_fasta(syn_in, cache_dir)

        wdir = tmp_dir / pid
        ensure_dir(wdir)

        o_tab = run_zseeker_on_fasta(orig, wdir / "orig", args.n_jobs)
        s_tab = run_zseeker_on_fasta(syn,  wdir / "syn",  args.n_jobs)

        ho = harmonize_zseeker(o_tab, pid, "orig", genome_id_from_path(orig_in))
        hs = harmonize_zseeker(s_tab, pid, "syn",  genome_id_from_path(syn_in))
        h = pd.concat([ho, hs], ignore_index=True)

        h.to_parquet(out_parq, index=False)
        rows.append(h)

    if rows:
        all_h = pd.concat(rows, ignore_index=True)
    else:
        # still produce empty metrics file for consistency
        all_h = pd.DataFrame(columns=["pair_id","which","genome_id","source","class","contig","start","end","strand","score","length"])

    metrics = summarize_metrics(all_h)
    metrics.to_csv(metrics_dir / "zseeker.metrics.csv", index=False)

    # Also write a tag-level harmonized parquet for downstream plotting/significance later
    all_h.to_parquet(harmon_dir / "zseeker.harmonized.parquet", index=False)


if __name__ == "__main__":
    main()
