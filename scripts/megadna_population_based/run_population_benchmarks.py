#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_population_benchmarks.py  —  Master runner for all population-based benchmarks

Calls the four individual benchmark scripts in sequence:

  1. kmer_spectra.py     — k-mer abundance spectra (Chor-style)
  2. fcgr_lenbins.py     — Frequency Chaos Game Representation
  3. nonbdna_lenbins.py  — Non-B DNA structural motifs (G4Hunter / ZSeeker / nonbgfa)
  4. nullomers_lenbins.py — Nullomer fraction via KMC

Each sub-script is invoked as a subprocess so it runs in its own environment.
The runner creates a dedicated sub-directory under --outdir for each benchmark
and passes all shared parameters (--tag, --nbins, --bootstrap, etc.) through.
Benchmark-specific options are forwarded via pass-through argument groups.

Quick start
-----------
python run_population_benchmarks.py \\
    --natural-fasta   natural.fasta \\
    --synthetic-fasta synthetic.fasta \\
    --tag  my_phages \\
    --outdir  results/population_benchmarks \\
    --gfa-bin /path/to/gfa \\
    --balance-within-bin

Skip individual benchmarks with --skip-kmer-spectra, --skip-fcgr,
--skip-nonbdna, --skip-nullomers.

The scripts directory
---------------------
By default the runner looks for the four scripts in the same directory as
this file.  Override with --scripts-dir if you place them elsewhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _python() -> str:
    """Return the path of the currently running Python interpreter."""
    return sys.executable


def _run(
    label: str,
    cmd: List[str],
    log_path: Optional[Path] = None,
) -> int:
    """
    Run *cmd* as a subprocess, streaming stdout/stderr to the terminal and,
    optionally, to *log_path* simultaneously.
    Returns the exit code.
    """
    print(f"\n{'='*72}")
    print(f"[BENCHMARK] {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*72}")
    t0 = time.monotonic()

    log_fh = log_path.open("w") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
        proc.wait()
    finally:
        if log_fh:
            log_fh.close()

    elapsed = time.monotonic() - t0
    status  = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[{label}] {status}  ({elapsed:.1f}s)")
    return proc.returncode


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run all population-based synthetic genome benchmarks in sequence: "
            "k-mer spectra, FCGR, non-B DNA motifs, and nullomers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- required inputs --------------------------------------------------
    inp = ap.add_argument_group("required inputs")
    inp.add_argument("--natural-fasta",   required=True,
                     help="Multi-record FASTA of natural genomes.")
    inp.add_argument("--synthetic-fasta", required=True,
                     help="Multi-record FASTA of synthetic genomes.")
    inp.add_argument("--outdir",  required=True,
                     help="Root output directory. Sub-directories are created per benchmark.")
    inp.add_argument("--tag",     default="dataset",
                     help="Short label used in output file names.")

    # ---- benchmark selection ---------------------------------------------
    sel = ap.add_argument_group("benchmark selection")
    sel.add_argument("--skip-kmer-spectra", action="store_true",
                     help="Skip the k-mer spectra benchmark.")
    sel.add_argument("--skip-fcgr",         action="store_true",
                     help="Skip the FCGR benchmark.")
    sel.add_argument("--skip-nonbdna",      action="store_true",
                     help="Skip the non-B DNA motif benchmark.")
    sel.add_argument("--skip-nullomers",    action="store_true",
                     help="Skip the nullomers benchmark.")

    # ---- shared statistical settings -------------------------------------
    shr = ap.add_argument_group("shared statistical settings")
    shr.add_argument("--nbins",     type=int, default=10,
                     help="Number of pooled quantile length bins for all benchmarks.")
    shr.add_argument("--bootstrap", type=int, default=2000,
                     help="Bootstrap replicates per bin for all benchmarks.")
    shr.add_argument("--balance-within-bin", action="store_true",
                     help="Downsample the larger group within each bin (recommended).")
    shr.add_argument("--min-per-bin", type=int, default=10,
                     help="Minimum genomes per group to include a bin.")
    shr.add_argument("--seed",       type=int, default=7,
                     help="Random seed propagated to all sub-scripts.")

    # ---- kmer_spectra-specific -------------------------------------------
    ks = ap.add_argument_group("k-mer spectra options (kmer_spectra.py)")
    ks.add_argument("--kmer-k",    default="auto",
                    help="k-mer length; 'auto' selects based on genome size.")
    ks.add_argument("--kmer-xmax", type=int, default=None,
                    help="Truncate abundance axis at this value.")

    # ---- fcgr-specific ---------------------------------------------------
    fcgr = ap.add_argument_group("FCGR options (fcgr_lenbins.py)")
    fcgr.add_argument("--fcgr-k", type=int, default=8,
                      help="FCGR order k (matrix size = 2^k × 2^k).")

    # ---- nonbdna-specific ------------------------------------------------
    nbd = ap.add_argument_group("non-B DNA options (nonbdna_lenbins.py)")
    nbd.add_argument("--gfa-bin",   default=None,
                     help="Path to the Non-B GFA binary. Required unless --skip-nonbdna.")
    nbd.add_argument("--nonbdna-n-jobs", type=int, default=1,
                     help="Parallel jobs for ZSeeker.")
    nbd.add_argument("--skip-g4hunter", action="store_true",
                     help="Skip G4Hunter within the non-B DNA benchmark.")
    nbd.add_argument("--skip-zseeker",  action="store_true",
                     help="Skip ZSeeker within the non-B DNA benchmark.")
    nbd.add_argument("--skip-gfa",      action="store_true",
                     help="Skip Non-B GFA within the non-B DNA benchmark.")

    # ---- nullomers-specific ----------------------------------------------
    nul = ap.add_argument_group("nullomers options (nullomers_lenbins.py)")
    nul.add_argument("--nullomers-k-min", type=int, default=7,
                     help="Smallest k-mer order for nullomer analysis.")
    nul.add_argument("--nullomers-k-max", type=int, default=9,
                     help="Largest k-mer order for nullomer analysis.")
    nul.add_argument("--nullomers-threads", type=int, default=4,
                     help="CPU threads passed to KMC.")

    # ---- misc ------------------------------------------------------------
    misc = ap.add_argument_group("miscellaneous")
    misc.add_argument("--scripts-dir", default=None,
                      help="Directory containing the four benchmark scripts. "
                           "Defaults to the directory of this file.")
    misc.add_argument("--keep-going", action="store_true",
                      help="Continue with remaining benchmarks even if one fails.")
    misc.add_argument("--max-genomes", type=int, default=0,
                      help="If > 0, load only the first N genomes per group (debug mode).")

    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap   = build_parser()
    args = ap.parse_args()

    # Resolve paths
    outdir   = Path(args.outdir).resolve()
    nat_fa   = Path(args.natural_fasta).resolve()
    syn_fa   = Path(args.synthetic_fasta).resolve()
    py       = _python()

    scripts_dir = (
        Path(args.scripts_dir).resolve()
        if args.scripts_dir
        else Path(__file__).resolve().parent
    )

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate non-B DNA requirement
    if not args.skip_nonbdna and not args.gfa_bin:
        ap.error(
            "--gfa-bin is required unless --skip-nonbdna is set. "
            "Provide the path to the Non-B GFA binary or add --skip-nonbdna."
        )

    # ------------------------------------------------------------------
    # Shared arguments passed to every sub-script
    # ------------------------------------------------------------------
    shared = [
        "--natural-fasta",   str(nat_fa),
        "--synthetic-fasta", str(syn_fa),
        "--tag",             args.tag,
        "--nbins",           str(args.nbins),
        "--bootstrap",       str(args.bootstrap),
        "--min-per-bin",     str(args.min_per_bin),
        "--seed",            str(args.seed),
    ]
    if args.balance_within_bin:
        shared.append("--balance-within-bin")
    if args.max_genomes:
        shared += ["--max-genomes", str(args.max_genomes)]

    results: dict[str, str] = {}
    failures = 0

    # ================================================================
    # 1. k-mer spectra
    # ================================================================
    if not args.skip_kmer_spectra:
        sub_outdir = outdir / "kmer_spectra"
        sub_outdir.mkdir(parents=True, exist_ok=True)
        cmd = (
            [py, str(scripts_dir / "kmer_spectra.py")]
            + shared
            + ["--outdir", str(sub_outdir), "--k", args.kmer_k]
        )
        if args.kmer_xmax is not None:
            cmd += ["--xmax", str(args.kmer_xmax)]
        rc = _run("k-mer spectra", cmd, logs_dir / "kmer_spectra.log")
        results["kmer_spectra"] = "OK" if rc == 0 else f"FAILED ({rc})"
        if rc != 0:
            failures += 1
            if not args.keep_going:
                _print_summary(results)
                sys.exit(rc)
    else:
        results["kmer_spectra"] = "skipped"

    # ================================================================
    # 2. FCGR
    # ================================================================
    if not args.skip_fcgr:
        sub_outdir = outdir / "fcgr"
        sub_outdir.mkdir(parents=True, exist_ok=True)
        cmd = (
            [py, str(scripts_dir / "fcgr_lenbins.py")]
            + shared
            + ["--outdir", str(sub_outdir), "--k", str(args.fcgr_k)]
        )
        rc = _run("FCGR", cmd, logs_dir / "fcgr.log")
        results["fcgr"] = "OK" if rc == 0 else f"FAILED ({rc})"
        if rc != 0:
            failures += 1
            if not args.keep_going:
                _print_summary(results)
                sys.exit(rc)
    else:
        results["fcgr"] = "skipped"

    # ================================================================
    # 3. Non-B DNA
    # ================================================================
    if not args.skip_nonbdna:
        sub_outdir = outdir / "nonbdna"
        sub_outdir.mkdir(parents=True, exist_ok=True)
        cmd = (
            [py, str(scripts_dir / "nonbdna_lenbins.py")]
            + shared
            + [
                "--outdir",  str(sub_outdir),
                "--gfa-bin", str(Path(args.gfa_bin).resolve()),
                "--n-jobs",  str(args.nonbdna_n_jobs),
            ]
        )
        if args.skip_g4hunter:
            cmd.append("--skip-g4hunter")
        if args.skip_zseeker:
            cmd.append("--skip-zseeker")
        if args.skip_gfa:
            cmd.append("--skip-nonbgfa")
        rc = _run("non-B DNA", cmd, logs_dir / "nonbdna.log")
        results["nonbdna"] = "OK" if rc == 0 else f"FAILED ({rc})"
        if rc != 0:
            failures += 1
            if not args.keep_going:
                _print_summary(results)
                sys.exit(rc)
    else:
        results["nonbdna"] = "skipped"

    # ================================================================
    # 4. Nullomers
    # ================================================================
    if not args.skip_nullomers:
        sub_outdir = outdir / "nullomers"
        sub_outdir.mkdir(parents=True, exist_ok=True)
        cmd = (
            [py, str(scripts_dir / "nullomers_lenbins.py")]
            + shared
            + [
                "--outdir",   str(sub_outdir),
                "--k-min",    str(args.nullomers_k_min),
                "--k-max",    str(args.nullomers_k_max),
                "--threads",  str(args.nullomers_threads),
            ]
        )
        rc = _run("nullomers", cmd, logs_dir / "nullomers.log")
        results["nullomers"] = "OK" if rc == 0 else f"FAILED ({rc})"
        if rc != 0:
            failures += 1
            if not args.keep_going:
                _print_summary(results)
                sys.exit(rc)
    else:
        results["nullomers"] = "skipped"

    # ================================================================
    # Summary
    # ================================================================
    _print_summary(results)
    if failures:
        sys.exit(1)


def _print_summary(results: dict) -> None:
    labels = {
        "kmer_spectra": "k-mer spectra  (kmer_spectra.py)",
        "fcgr":         "FCGR           (fcgr_lenbins.py)",
        "nonbdna":      "non-B DNA      (nonbdna_lenbins.py)",
        "nullomers":    "Nullomers      (nullomers_lenbins.py)",
    }
    order = ["kmer_spectra", "fcgr", "nonbdna", "nullomers"]
    print(f"\n{'='*72}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*72}")
    for k in order:
        if k in results:
            status = results[k]
            icon   = "✓" if status == "OK" else ("—" if status == "skipped" else "✗")
            print(f"  {icon}  {labels[k]:<45}  {status}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
