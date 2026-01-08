#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_pair_parquets(harmon_dir: Path, suffix: str) -> pd.DataFrame:
    paths = sorted(harmon_dir.glob(f"*.{suffix}.parquet"))
    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def zlike_metrics(h: pd.DataFrame) -> pd.DataFrame:
    if h.empty:
        return pd.DataFrame(columns=["pair_id","which","n_hits","bp_covered"])
    grp = h.groupby(["pair_id","which"], dropna=False).agg(
        n_hits=("contig","count"),
        bp_covered=("length","sum"),
    ).reset_index()
    return grp


def infer_motif_from_name(fname: str) -> str:
    b = Path(fname).name
    # matches _DR.tsv, _DR.gff, .DR.tsv, etc.
    m = re.search(r"(?:^|[_.])(DR|IR|MR|STR|GQ)(?:[_.]|$)", b, flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNK"


def infer_which(fname: str) -> str:
    b = Path(fname).name.lower()
    # your files look like: pair_001_orig_DR.tsv
    if "_orig_" in b or b.endswith("_orig.tsv") or b.endswith("_orig.gff"):
        return "orig"
    if "_syn_" in b or b.endswith("_syn.tsv") or b.endswith("_syn.gff"):
        return "syn"
    if "_orig" in b:
        return "orig"
    if "_syn" in b:
        return "syn"
    return "unk"


def read_intervals_generic(path: Path) -> pd.DataFrame:
    """
    Return DataFrame with columns: chrom,start,end
    Supports:
      - nonbgfa *.gff (GFF3-ish): chrom, source, type, start, end, ...
      - nonbgfa *.tsv: typically has header with Start/End or chrom/start/end columns
      - fallback: headerless tab/whitespace with >=3 columns
    """
    suf = path.suffix.lower()

    # --- GFF: start/end are columns 4/5 (1-based) -> indices 3/4 (0-based)
    if suf == ".gff":
        try:
            df = pd.read_csv(path, sep="\t", comment="#", header=None, engine="python")
            if df.shape[1] >= 5:
                out = pd.DataFrame({
                    "chrom": df.iloc[:, 0].astype(str),
                    "start": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
                    "end": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
                }).dropna(subset=["start", "end"])
                return out
        except Exception:
            return pd.DataFrame(columns=["chrom", "start", "end"])

    # --- TSV: try with header first
    if suf == ".tsv":
        try:
            df = pd.read_csv(path, sep="\t", comment="#", header=0, engine="python")
            cols = {c.lower(): c for c in df.columns}

            def pick(*names):
                for n in names:
                    if n in df.columns:
                        return n
                    if n.lower() in cols:
                        return cols[n.lower()]
                return None

            c_chrom = pick("chrom", "Chrom", "chr", "Chromosome", "contig", "seqid", "seqID")
            c_start = pick("start", "Start")
            c_end = pick("end", "End", "stop", "Stop")

            if c_start and c_end:
                out = pd.DataFrame({
                    "chrom": df[c_chrom].astype(str) if c_chrom else "chr",
                    "start": pd.to_numeric(df[c_start], errors="coerce"),
                    "end": pd.to_numeric(df[c_end], errors="coerce"),
                }).dropna(subset=["start", "end"])
                return out
        except Exception:
            pass

    # --- Fallback: headerless tab then whitespace, first 3 columns
    for sep in ["\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python")
            if df.shape[1] >= 3:
                out = df.iloc[:, :3].copy()
                out.columns = ["chrom", "start", "end"]
                out["start"] = pd.to_numeric(out["start"], errors="coerce")
                out["end"] = pd.to_numeric(out["end"], errors="coerce")
                out = out.dropna(subset=["start", "end"])
                return out
        except Exception:
            continue

    return pd.DataFrame(columns=["chrom", "start", "end"])


def aggregate_nonbgfa(calls_root: Path) -> pd.DataFrame:
    rows = []
    if not calls_root.exists():
        return pd.DataFrame(columns=["pair_id","which","motif","n_hits","bp_covered"])

    for pair_dir in sorted([p for p in calls_root.iterdir() if p.is_dir()]):
        pid = pair_dir.name
        for f in sorted(pair_dir.glob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".bed", ".tsv", ".txt", ".gff"}:
                continue

            which = infer_which(f.name)
            motif = infer_motif_from_name(f.name)
            if which == "unk" or motif == "UNK":
                continue

            df = read_intervals_generic(f)
            if df.empty:
                continue

            n_hits = int(df.shape[0])
            bp_cov = int((df["end"] - df["start"]).clip(lower=0).sum())

            rows.append({
                "pair_id": pid,
                "which": which,
                "motif": motif,
                "n_hits": n_hits,
                "bp_covered": bp_cov,
            })

    return pd.DataFrame(rows, columns=["pair_id","which","motif","n_hits","bp_covered"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="results/<TAG>/nonbdna")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    harmon_dir = outdir / "harmonized"
    metrics_dir = outdir / "metrics"
    ensure_dir(harmon_dir); ensure_dir(metrics_dir)

    # zseeker / g4hunter: tag-level concat + metrics
    z = load_pair_parquets(harmon_dir, "zseeker")
    z.to_parquet(harmon_dir / "zseeker.harmonized.parquet", index=False)
    zlike_metrics(z).to_csv(metrics_dir / "zseeker.metrics.csv", index=False)

    g4 = load_pair_parquets(harmon_dir, "g4hunter")
    g4.to_parquet(harmon_dir / "g4hunter.harmonized.parquet", index=False)
    zlike_metrics(g4).to_csv(metrics_dir / "g4hunter.metrics.csv", index=False)

    # nonbgfa: parse raw calls -> tidy + metrics
    calls_root = outdir / "calls" / "nonbgfa"
    gfa = aggregate_nonbgfa(calls_root)
    gfa.to_parquet(harmon_dir / "nonbgfa.harmonized.parquet", index=False)
    gfa.to_csv(metrics_dir / "nonbgfa.metrics.csv", index=False)


if __name__ == "__main__":
    main()
