#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_species_benchmark.py

One-shot runner for a single species benchmark.

By default, expects these scripts in the SAME directory as this file:
  - kmer_spectra.py
  - kmer_spectra_significance.py

Runs:
  1) kmer_spectra.py pair       -> per-window plots + metrics CSV
  2) kmer_spectra.py aggregate  -> mean CDF±CI, ΔCDF±CI, ΔPMF + CSV/JSON
  3) kmer_spectra_significance.py    -> per-species p-value on ks_stat replicates

Example:
  python run_species_benchmark.py \
    --manifest pairs.Human.csv \
    --outdir results/Human \
    --k auto \
    --delta 0.01 \
    --canonical \
    --softmask-to-N \
    --xmax 1200 \
    --rare-max 2 \
    --high-min 200
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def infer_tag(tag: str | None, manifest_path: str) -> str:
    """Optional: tag only used for nicer filenames/metadata."""
    if tag:
        return tag
    base = os.path.basename(manifest_path)
    base = re.sub(r"\.csv$", "", base, flags=re.I)
    m = re.search(r"(?:pairs[._-])(.+)$", base, flags=re.I)
    return (m.group(1) if m else base) or "run"


def parse_args():
    ap = argparse.ArgumentParser(description="Run pair+aggregate+significance for one species.")

    # Optional overrides; default is same dir as this script
    ap.add_argument("--kmer-spectra", default=None, help="Override path to kmer_spectra.py")
    ap.add_argument("--species-significance", default=None, help="Override path to kmer_spectra_significance.py")

    ap.add_argument("--manifest", required=True, help="pairs CSV with columns id,orig,syn")
    ap.add_argument("--outdir", required=True, help="Output directory root for this species.")
    ap.add_argument("--tag", default=None, help="Optional tag for filenames/metadata (auto-inferred if omitted).")

    ap.add_argument("--k", default="auto", help="k or 'auto'")
    ap.add_argument("--delta", type=float, default=0.01, help="Practical-zero for median ks_stat test.")

    ap.add_argument("--canonical", action="store_true", help="Use canonical k-mers")
    ap.add_argument("--softmask-to-N", action="store_true", help="Treat lowercase as N (break k-mers)")
    ap.add_argument("--xmax", type=int, default=None, help="Max abundance plotted/cutoff")

    ap.add_argument("--rare-max", type=int, default=2, help="Rare bin upper bound (inclusive)")
    ap.add_argument("--high-min", type=int, default=200, help="High bin threshold (exclusive)")

    # pair-mode knobs
    ap.add_argument("--no-hist", action="store_true", help="Pair mode: step plots instead of histogram bars")
    ap.add_argument("--smooth-win", type=int, default=5, help="Pair mode: moving average window for histogram")
    ap.add_argument("--no-pdf", action="store_true", help="Pair mode: do not write PDFs (PNG only)")

    return ap.parse_args()


def main():
    args = parse_args()

    here = Path(__file__).resolve().parent

    kmer_py = Path(args.kmer_spectra).resolve() if args.kmer_spectra else (here / "kmer_spectra.py")
    sig_py  = Path(args.species_significance).resolve() if args.species_significance else (here / "kmer_spectra_significance.py")

    if not kmer_py.is_file():
        raise SystemExit(f"Could not find kmer_spectra.py at: {kmer_py}")
    if not sig_py.is_file():
        raise SystemExit(f"Could not find kmer_spectra_significance.py at: {sig_py}")

    tag = infer_tag(args.tag, args.manifest)

    out_pair = Path(args.outdir) / "pair"
    out_agg  = Path(args.outdir) / "aggregate"
    out_sig  = Path(args.outdir) / "significance"
    out_pair.mkdir(parents=True, exist_ok=True)
    out_agg.mkdir(parents=True, exist_ok=True)
    out_sig.mkdir(parents=True, exist_ok=True)

    # ---- 1) pair ----
    cmd_pair = [
        sys.executable, str(kmer_py), "pair",
        "--k", str(args.k),
        "--manifest", args.manifest,
        "--outdir", str(out_pair),
        "--smooth-win", str(args.smooth_win),
    ]
    if args.canonical:
        cmd_pair.append("--canonical")
    if args.softmask_to_N:
        cmd_pair.append("--softmask-to-N")
    if args.no_hist:
        cmd_pair.append("--no-hist")
    if args.no_pdf:
        cmd_pair.append("--no-pdf")
    if args.xmax is not None:
        cmd_pair += ["--xmax", str(args.xmax)]
    run(cmd_pair)

    # metrics CSV path (auto vs fixed k)
    metrics_csv = out_pair / f"metrics.k{args.k}.csv"
    if not metrics_csv.exists():
        metrics_csv = out_pair / "metrics.kauto.csv"
    if not metrics_csv.exists():
        raise SystemExit(f"Could not find metrics CSV in {out_pair}")

    # ---- 2) aggregate ----
    cmd_agg = [
        sys.executable, str(kmer_py), "aggregate",
        "--k", str(args.k),
        "--manifest", args.manifest,
        "--outdir", str(out_agg),
        "--tag", tag,
        "--rare-max", str(args.rare_max),
        "--high-min", str(args.high_min),
    ]
    if args.canonical:
        cmd_agg.append("--canonical")
    if args.softmask_to_N:
        cmd_agg.append("--softmask-to-N")
    if args.xmax is not None:
        cmd_agg += ["--xmax", str(args.xmax)]
    run(cmd_agg)

    # ---- 3) significance ----
    out_json = out_sig / f"{tag}.significance.json"
    out_csv  = out_sig / f"{tag}.significance.csv"

    cmd_sig = [
        sys.executable, str(sig_py),
        "--metrics", str(metrics_csv),
        "--tag", tag,
        "--delta", str(args.delta),
        "--out-json", str(out_json),
        "--out-csv", str(out_csv),
    ]
    run(cmd_sig)

    print("\n[OK] Done.")
    print(f"  Pair results:       {out_pair}")
    print(f"  Aggregate results:  {out_agg}")
    print(f"  Significance:       {out_json}")


if __name__ == "__main__":
    main()
