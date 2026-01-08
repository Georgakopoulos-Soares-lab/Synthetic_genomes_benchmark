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


def genome_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suf in [".fna", ".fa", ".fasta"]:
        if name.endswith(suf):
            name = name[:-len(suf)]
            break
    return name


def run_g4hunter(fasta: Path, outdir: Path) -> Path:
    ensure_dir(outdir)
    cmd = ["g4hunter", str(fasta), "--outdir", str(outdir)]
    subprocess.run(cmd, check=True)

    cands = list(outdir.glob("*_pG4s.g4_hunter.tsv"))
    if not cands:
        raise FileNotFoundError(f"No G4Hunter TSV in {outdir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def harmonize_g4hunter(tsv: Path, pair_id: str, which: str, genome_id: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if not {"seqID", "start", "end"}.issubset(df.columns):
        raise ValueError(f"G4Hunter TSV missing required columns: {tsv}")

    out = pd.DataFrame({
        "pair_id": pair_id,
        "which": which,
        "genome_id": genome_id,
        "source": "g4hunter",
        "class": df["type"] if "type" in df.columns else "pG4",
        "contig": df["seqID"].astype(str),
        "start": pd.to_numeric(df["start"], errors="coerce").astype("Int64"),
        "end": pd.to_numeric(df["end"], errors="coerce").astype("Int64"),
        "strand": df["strand"] if "strand" in df.columns else ".",
        "score": pd.to_numeric(df["score"], errors="coerce") if "score" in df.columns else pd.NA,
    }).dropna(subset=["start", "end"])
    out["length"] = (
        pd.to_numeric(df["length"], errors="coerce").astype("Int64")
        if "length" in df.columns
        else (out["end"] - out["start"]).astype("Int64")
    )
    return out


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
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-pairs", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()

    harmon_dir = outdir / "harmonized"
    metrics_dir = outdir / "metrics"
    tmp_dir = outdir / "tmp" / "g4hunter"
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

        out_parq = harmon_dir / f"{pid}.g4hunter.parquet"
        if out_parq.exists() and out_parq.stat().st_size > 0:
            continue

        if (not orig_in.exists()) or (not syn_in.exists()):
            continue

        orig = ensure_plain_fasta(orig_in, cache_dir)
        syn = ensure_plain_fasta(syn_in, cache_dir)

        pair_tmp = tmp_dir / pid
        o_tsv = run_g4hunter(orig, pair_tmp / "orig")
        s_tsv = run_g4hunter(syn,  pair_tmp / "syn")

        ho = harmonize_g4hunter(o_tsv, pid, "orig", genome_id_from_path(orig_in))
        hs = harmonize_g4hunter(s_tsv, pid, "syn",  genome_id_from_path(syn_in))
        h = pd.concat([ho, hs], ignore_index=True)

        h.to_parquet(out_parq, index=False)
        rows.append(h)

    all_h = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["pair_id","which","genome_id","source","class","contig","start","end","strand","score","length"]
    )

    summarize_metrics(all_h).to_csv(metrics_dir / "g4hunter.metrics.csv", index=False)
    all_h.to_parquet(harmon_dir / "g4hunter.harmonized.parquet", index=False)


if __name__ == "__main__":
    main()
