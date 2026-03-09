#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


def python_exe() -> str:
    return sys.executable


def repo_root() -> Path:
    return Path.cwd().resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def script_path(root: Path, rel: str) -> Path:
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing script: {p}")
    return p


def find_manifest(root: Path, tag: str) -> Optional[Path]:
    for c in [
        root / f"pairs.{tag}.csv",
        root / "manifests" / f"pairs.{tag}.csv",
        root / "manifests" / f"{tag}.csv",
        root / "data" / "manifests" / f"pairs.{tag}.csv",
    ]:
        if c.exists():
            return c
    return None


def run_kmer_suite(
    root: Path,
    tag: str,
    manifest: Path,
    outdir: Path,
    k: str,
    canonical: bool,
    softmask_to_N: bool,
    xmax: Optional[int],
    smooth_win: int,
    rare_max: int,
    high_min: int,
    write_pdf: bool,
    sig_delta: float,
) -> None:
    kmer = script_path(root, "scripts/benchmarks/kmer_spectra.py")
    sig = script_path(root, "scripts/benchmarks/kmer_spectra_significance.py")

    # --- spectra: pair ---
    pair_cmd = [
        python_exe(), str(kmer),
        "pair",
        "--k", k,
        "--manifest", str(manifest),
        "--outdir", str(outdir),
        "--smooth-win", str(smooth_win),
    ]
    if canonical:
        pair_cmd.append("--canonical")
    if softmask_to_N:
        pair_cmd.append("--softmask-to-N")
    if xmax is not None:
        pair_cmd += ["--xmax", str(xmax)]
    if not write_pdf:
        pair_cmd.append("--no-pdf")

    run_cmd(pair_cmd, cwd=root)

    # --- significance (single species) ---
    metrics_csv = outdir / f"metrics.k{k}.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Expected metrics CSV not found: {metrics_csv}")

    sig_out = outdir / "significance"
    ensure_dir(sig_out)

    sig_cmd = [
        python_exe(), str(sig),
        "--metrics", str(metrics_csv),
        "--tag", tag,
        "--delta", str(sig_delta),
        "--out-json", str(sig_out / f"{tag}.significance.k{k}.json"),
        "--out-csv", str(sig_out / f"{tag}.significance.k{k}.csv"),
    ]
    run_cmd(sig_cmd, cwd=root)

    # --- spectra: aggregate ---
    agg_cmd = [
        python_exe(), str(kmer),
        "aggregate",
        "--k", k,
        "--manifest", str(manifest),
        "--outdir", str(outdir),
        "--tag", tag,
        "--rare-max", str(rare_max),
        "--high-min", str(high_min),
    ]
    if canonical:
        agg_cmd.append("--canonical")
    if softmask_to_N:
        agg_cmd.append("--softmask-to-N")
    if xmax is not None:
        agg_cmd += ["--xmax", str(xmax)]

    run_cmd(agg_cmd, cwd=root)


def run_fcgr(
    root: Path,
    tag: str,
    manifest: Path,
    outdir: Path,
    k: int,
    max_pairs: int,
    delta: float,
    n_perm: int,
    n_boot: int,
    seed: int,
    eps: float,
) -> None:
    fcgr = script_path(root, "scripts/benchmarks/fcgr.py")

    cmd = [
        python_exe(), str(fcgr),
        "--manifest", str(manifest),
        "--outdir", str(outdir),
        "--label", tag,
        "--k", str(k),
        "--delta", str(delta),
        "--n-perm", str(n_perm),
        "--n-boot", str(n_boot),
        "--seed", str(seed),
        "--eps", str(eps),
    ]
    if max_pairs and max_pairs > 0:
        cmd += ["--max-pairs", str(max_pairs)]

    run_cmd(cmd, cwd=root)


def run_nullomers(
    root: Path,
    manifest: Path,
    outdir: Path,
    k: int,
    threads: int,
    max_pairs: int,
) -> None:
    nullomers = script_path(root, "scripts/benchmarks/nullomers.py")

    cmd = [
        python_exe(), str(nullomers),
        "--manifest", str(manifest),
        "--outdir", str(outdir),
        "--k", str(k),
        "--threads", str(threads),
    ]
    if max_pairs and max_pairs > 0:
        cmd += ["--max-pairs", str(max_pairs)]

    run_cmd(cmd, cwd=root)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--tag", required=True, help="e.g., Human")
    p.add_argument("--manifest", default=None, help="Pairs CSV (id,orig,syn). If omitted, auto-search common locations.")
    p.add_argument("--results-root", default="results", help="Default: ./results")

    p.add_argument("--spectra", action="store_true", help="Run k-mer spectra (pair + aggregate + significance).")
    p.add_argument("--fcgr", action="store_true", help="Run FCGR benchmark.")
    p.add_argument("--nullomers", action="store_true", help="Run nullomer benchmark (KMC).")
    p.add_argument("--all", action="store_true", help="Run everything currently wired (spectra + fcgr).")

    p.add_argument("--max-pairs", type=int, default=0, help="If >0, process only first N pairs (debug).")

    # spectra knobs
    p.add_argument("--kmer-k", default="auto", help="k for spectra (int or 'auto').")
    p.add_argument("--canonical", action="store_true", help="Use canonical k-mers (revcomp collapsed).")
    p.add_argument("--softmask-to-N", action="store_true", help="Treat lowercase as N.")
    p.add_argument("--xmax", type=int, default=500, help="Plot cutoff (abundance).")
    p.add_argument("--smooth-win", type=int, default=5, help="Histogram smoothing window (pair spectra).")
    p.add_argument("--rare-max", type=int, default=2, help="Rare bin upper bound (aggregate).")
    p.add_argument("--high-min", type=int, default=200, help="High bin threshold (aggregate).")
    p.add_argument("--pdf", action="store_true", help="Also write PDFs (pair mode).")
    p.add_argument("--sig-delta", type=float, default=0.01, help="Practical-zero threshold for ks_stat significance.")

    # fcgr knobs
    p.add_argument("--fcgr-k", type=int, default=8, help="FCGR order k (side=2^k).")
    p.add_argument("--fcgr-delta", type=float, default=0.0, help="Threshold for median(distance) > delta (FCGR).")
    p.add_argument("--n-perm", type=int, default=50000, help="Sign-flip permutations (FCGR).")
    p.add_argument("--n-boot", type=int, default=20000, help="Bootstrap resamples (FCGR).")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (FCGR).")
    p.add_argument("--eps", type=float, default=1e-10, help="Epsilon for log plots/ratios (FCGR).")

    # nullomers

    p.add_argument("--nullomers-k", type=int, default=7, help="k for nullomers (KMC).")
    p.add_argument("--threads", type=int, default=0, help="Threads for external tools (0=auto).")

    # tfbs (FIMO)
    p.add_argument("--tfbs", action="store_true", help="Run TFBS benchmark (FIMO + aggregation).")
    p.add_argument("--tfbs-meme-dir", default="ref/jaspar/vertebrates/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme",
                   help="Directory with *.meme files (relative to repo root by default).")
    p.add_argument("--tfbs-meme-file", default=None, help="Single .meme file (overrides --tfbs-meme-dir).")
    p.add_argument("--tfbs-workers", type=int, default=0, help="Parallel sequences for FIMO (0=CPU count).")
    p.add_argument("--tfbs-extra-fimo-args", default="--thresh 1e-4 --verbosity 1",
                   help="Extra args passed to fimo.")
    p.add_argument("--tfbs-site-p", type=float, default=1e-4, help="Site-level p-value filter for aggregation.")
    p.add_argument("--tfbs-site-q", type=float, default=1.0, help="Site-level q-value filter for aggregation.")
    p.add_argument("--tfbs-gzip-tsv", action="store_true", help="Compress fimo.tsv to fimo.tsv.gz.")

    p.add_argument("--nonbdna", action="store_true", help="Run non-B DNA benchmarks (ZSeeker + G4Hunter + non-B GFA + aggregate).")
    p.add_argument("--gfa-bin", default="non-B_gfa/gfa", help="Path to gfa binary (default repo-relative).")
    p.add_argument("--zseeker-jobs", type=int, default=1, help="Threads for ZSeeker.")
    p.add_argument("--nonbdna-ymax-bp", type=float, default=None, help="y-max for nonbdna bp_covered plot")

    return p


def main() -> None:
    args = build_parser().parse_args()
    root = repo_root()

    # --all means "everything wired so far"
    if args.all:
        args.spectra = True
        args.fcgr = True
        args.nullomers = True
        args.tfbs = True
        args.nonbdna = True



    if not args.spectra and not args.fcgr and not args.nullomers and not args.tfbs and not args.nonbdna:
        raise SystemExit("Nothing to run. Use --spectra, --fcgr, --nullomers, --tfbs, --nonbdna or --all.")

    manifest = Path(args.manifest).resolve() if args.manifest else find_manifest(root, args.tag)
    if manifest is None or not manifest.exists():
        raise SystemExit(
            "Manifest not found. Provide --manifest or place it in one of:\n"
            f"  {root / f'pairs.{args.tag}.csv'}\n"
            f"  {root / 'manifests' / f'pairs.{args.tag}.csv'}\n"
            f"  {root / 'manifests' / f'{args.tag}.csv'}\n"
            f"  {root / 'data' / 'manifests' / f'pairs.{args.tag}.csv'}\n"
        )

    results_root = (root / args.results_root).resolve()

    spectra_out = results_root / args.tag / "spectra"
    fcgr_out = results_root / args.tag / "fcgr"
    null_out = results_root / args.tag / "nullomers"
    tfbs_out = results_root / args.tag / "tfbs"
    tfbs_fimo_out = tfbs_out / "fimo"

    ensure_dir(spectra_out)
    ensure_dir(fcgr_out)
    ensure_dir(null_out)
    ensure_dir(tfbs_out)
    ensure_dir(tfbs_fimo_out)


    if args.spectra:
        run_kmer_suite(
            root=root,
            tag=args.tag,
            manifest=manifest,
            outdir=spectra_out,
            k=args.kmer_k,
            canonical=args.canonical,
            softmask_to_N=args.softmask_to_N,
            xmax=args.xmax,
            smooth_win=args.smooth_win,
            rare_max=args.rare_max,
            high_min=args.high_min,
            write_pdf=args.pdf,
            sig_delta=args.sig_delta,
        )

    if args.fcgr:
        run_fcgr(
            root=root,
            tag=args.tag,
            manifest=manifest,
            outdir=fcgr_out,
            k=args.fcgr_k,
            max_pairs=args.max_pairs,
            delta=args.fcgr_delta,
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            seed=args.seed,
            eps=args.eps,
        )

    if args.nullomers:
        threads = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)
        run_nullomers(
            root=root,
            manifest=manifest,
            outdir=null_out,
            k=args.nullomers_k,
            threads=threads,
            max_pairs=args.max_pairs,
        )


    if args.tfbs:
        tfbs_fimo = script_path(root, "scripts/benchmarks/tfbs_fimo.py")
        tfbs_agg  = script_path(root, "scripts/benchmarks/tfbs_aggregate.py")

        meme_dir = (root / args.tfbs_meme_dir).resolve()
        meme_file = (root / args.tfbs_meme_file).resolve() if args.tfbs_meme_file else None

        fimo_cmd = [
            python_exe(), str(tfbs_fimo),
            "--tag", args.tag,
            "--manifest", str(manifest),
            "--outdir", str(tfbs_fimo_out),
            "--extra-fimo-args", args.tfbs_extra_fimo_args,
            "--max-pairs", str(args.max_pairs),
        ]
        if args.tfbs_workers and args.tfbs_workers > 0:
            fimo_cmd += ["--workers", str(args.tfbs_workers)]
        if args.tfbs_gzip_tsv:
            fimo_cmd.append("--gzip-tsv")

        if meme_file:
            fimo_cmd += ["--meme-file", str(meme_file)]
        else:
            fimo_cmd += ["--meme-dir", str(meme_dir)]

        run_cmd(fimo_cmd, cwd=root)

        out_csv = tfbs_out / "tfbs_aggregate_stats.csv"
        out_png = tfbs_out / "tfbs_volcano_agg.png"

        agg_cmd = [
            python_exe(), str(tfbs_agg),
            "--fimo-root", str(tfbs_fimo_out),
            "--out-csv", str(out_csv),
            "--out-png", str(out_png),
            "--site-p", str(args.tfbs_site_p),
            "--site-q", str(args.tfbs_site_q),
        ]
        run_cmd(agg_cmd, cwd=root)

        print(f"[ok] tfbs: wrote {out_csv}")
        print(f"[ok] tfbs: wrote {out_png}")

    if args.nonbdna:
        nonbdna_out = results_root / args.tag / "nonbdna"
        ensure_dir(nonbdna_out)

        z_py   = script_path(root, "scripts/benchmarks/nonbdna_zseeker.py")
        g4_py  = script_path(root, "scripts/benchmarks/nonbdna_g4hunter.py")
        gfa_py = script_path(root, "scripts/benchmarks/nonbdna_nonbgfa.py")
        agg_py = script_path(root, "scripts/benchmarks/nonbdna_aggregate.py")

        # ZSeeker
        run_cmd([
            python_exe(), str(z_py),
            "--manifest", str(manifest),
            "--outdir", str(nonbdna_out),
            "--n-jobs", str(args.zseeker_jobs),
            "--max-pairs", str(args.max_pairs),
        ], cwd=root)

        # G4Hunter
        run_cmd([
            python_exe(), str(g4_py),
            "--manifest", str(manifest),
            "--outdir", str(nonbdna_out),
            "--max-pairs", str(args.max_pairs),
        ], cwd=root)

        # non-B_gfa calls
        gfa_bin = (root / args.gfa_bin).resolve()
        run_cmd([
            python_exe(), str(gfa_py),
            "--manifest", str(manifest),
            "--outdir", str(nonbdna_out),
            "--gfa-bin", str(gfa_bin),
            "--max-pairs", str(args.max_pairs),
        ], cwd=root)

        # aggregate to tag-level metrics/parquets
        run_cmd([
            python_exe(), str(agg_py),
            "--outdir", str(nonbdna_out),
        ], cwd=root)

        print(f"[ok] nonbdna: wrote metrics under {nonbdna_out / 'metrics'}")

        cmd = [
            python_exe(),
            str(script_path(root, "scripts/benchmarks/nonbdna_significance_plot.py")),
            "--outdir", str(nonbdna_out),
        ]
        if args.nonbdna_ymax_bp is not None:
            cmd += ["--ymax-bp", str(args.nonbdna_ymax_bp)]

        run_cmd(cmd, cwd=root)

                                                                                                                                                                                                                                                                                                                                                                                                                            
if __name__ == "__main__":
    main()
