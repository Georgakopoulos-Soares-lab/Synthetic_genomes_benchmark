#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise SystemExit(f"Missing dependency '{exe}'. Ensure it is installed and on PATH.")
    return p


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        need = {"id", "orig", "syn"}
        if not need.issubset(set(rdr.fieldnames or [])):
            raise SystemExit(f"Manifest must have columns {sorted(need)} (got {rdr.fieldnames})")
        rows = []
        for i, row in enumerate(rdr, start=1):
            if not row.get("id"):
                row["id"] = f"pair_{i:03d}"
            rows.append(row)
        if not rows:
            raise SystemExit("Manifest is empty.")
        return rows


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def acgt_background_from_fasta(path: Path) -> Dict[str, float]:
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    total = 0
    with open_text_auto(path) as f:
        for line in f:
            if not line or line.startswith(">"):
                continue
            for ch in line.strip().upper():
                if ch in counts:
                    counts[ch] += 1
                    total += 1
    if total == 0:
        return {b: 0.25 for b in "ACGT"}
    return {b: counts[b] / total for b in "ACGT"}


def write_background(bg: Dict[str, float], out: Path) -> None:
    out.write_text(
        f"A {bg['A']:.8f}\nC {bg['C']:.8f}\nG {bg['G']:.8f}\nT {bg['T']:.8f}\n",
        encoding="utf-8",
    )


def list_memes(meme_file: Optional[str], meme_dir: Optional[str]) -> List[Path]:
    if meme_file:
        p = Path(meme_file)
        if not p.is_file():
            raise FileNotFoundError(p)
        return [p]
    if not meme_dir:
        raise SystemExit("Provide --meme-file or --meme-dir")
    d = Path(meme_dir)
    if not d.is_dir():
        raise NotADirectoryError(d)
    memes = sorted(d.glob("*.meme"))
    if not memes:
        raise SystemExit(f"No *.meme files found under: {d}")
    return memes


def safe_meme_name(p: Path) -> str:
    return p.name[:-5] if p.name.endswith(".meme") else p.stem


def fimo_done(outdir: Path) -> bool:
    return (outdir / "fimo.tsv").is_file() or (outdir / "fimo.tsv.gz").is_file()


def prune_fimo_outdir(outdir: Path, keep_gff: bool, gzip_tsv: bool) -> None:
    tsv = outdir / "fimo.tsv"
    gff = outdir / "fimo.gff"

    if gzip_tsv and tsv.is_file():
        gz = outdir / "fimo.tsv.gz"
        with tsv.open("rb") as f_in, gzip.open(gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        tsv.unlink(missing_ok=True)

    for p in list(outdir.iterdir()):
        name = p.name
        keep = False
        if gzip_tsv:
            keep = (name == "fimo.tsv.gz") or (keep_gff and name == "fimo.gff")
        else:
            keep = (name == "fimo.tsv") or (keep_gff and name == "fimo.gff")

        if keep:
            continue

        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def run_one_sequence(job: Tuple[str, str, Path, Path, List[Path], str, str, bool, bool, bool]):
    """
    (pair_id, which, fasta, out_root, memes, fimo_bin, extra_args, keep_gff, gzip_tsv, dry_run)
    """
    pair_id, which, fasta, out_root, memes, fimo_bin, extra_args, keep_gff, gzip_tsv, dry_run = job
    out_root.mkdir(parents=True, exist_ok=True)

    bg_path = out_root / "background.txt"
    if not bg_path.exists():
        bg = acgt_background_from_fasta(fasta)
        write_background(bg, bg_path)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    n_ok = n_skip = n_fail = 0

    for meme in memes:
        mname = safe_meme_name(meme)
        outdir = out_root / mname
        outdir.mkdir(parents=True, exist_ok=True)

        if fimo_done(outdir):
            n_skip += 1
            continue

        cmd = [fimo_bin, "-oc", str(outdir), "--bfile", str(bg_path)]
        if extra_args.strip():
            cmd.extend(extra_args.strip().split())
        cmd.extend([str(meme), str(fasta)])

        if dry_run:
            n_ok += 1
            continue

        with open(os.devnull, "wb") as devnull:
            rc = subprocess.run(cmd, env=env, stdout=devnull, stderr=devnull).returncode

        if rc == 0:
            prune_fimo_outdir(outdir, keep_gff=keep_gff, gzip_tsv=gzip_tsv)
            n_ok += 1
        else:
            n_fail += 1

    return (pair_id, which, n_ok, n_skip, n_fail, str(out_root))


def run_streamed(jobs: List[Tuple], workers: int):
    total = len(jobs)
    if total == 0:
        raise SystemExit("No jobs prepared (check manifest paths).")

    interrupted = {"flag": False}

    def _sigint(_s, _f):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    done_count = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = iter(jobs)
        live = set()

        for _ in range(min(workers * 2, total)):
            try:
                live.add(ex.submit(run_one_sequence, next(it)))
            except StopIteration:
                break

        while live:
            finished, live = wait(live, return_when=FIRST_COMPLETED)
            for fut in finished:
                pair_id, which, n_ok, n_skip, n_fail, out_root = fut.result()
                done_count += 1
                status = "OK" if n_fail == 0 else "FAIL"
                print(f"[{done_count}/{total}] {pair_id} {which}: {status}  motifs_ok={n_ok} skip={n_skip} fail={n_fail} -> {out_root}")

                if not interrupted["flag"]:
                    try:
                        live.add(ex.submit(run_one_sequence, next(it)))
                    except StopIteration:
                        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run FIMO over (orig,syn) sequences in a manifest (id,orig,syn).")
    p.add_argument("--tag", required=True, help="Tag used for default output layout (e.g., Publish_Human).")
    p.add_argument("--manifest", required=True, help="Pairs CSV with columns: id, orig, syn.")
    p.add_argument("--outdir", required=True, help="Output directory (e.g., results/<tag>/tfbs/fimo).")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--meme-file", default=None, help="Single .meme file.")
    g.add_argument("--meme-dir", default=None, help="Directory with *.meme files.")

    p.add_argument("--extra-fimo-args", default="--thresh 1e-4 --verbosity 1", help="Extra args passed to FIMO.")
    p.add_argument("--workers", type=int, default=0, help="Parallel sequences (0=CPU count).")
    p.add_argument("--keep-gff", action="store_true", help="Keep fimo.gff outputs.")
    p.add_argument("--gzip-tsv", action="store_true", help="Compress fimo.tsv to fimo.tsv.gz.")
    p.add_argument("--max-pairs", type=int, default=0, help="If >0, process only first N pairs.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    which_or_die("fimo")

    args = build_parser().parse_args()
    manifest = Path(args.manifest)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest)
    if args.max_pairs and args.max_pairs > 0:
        rows = rows[: args.max_pairs]

    memes = list_memes(args.meme_file, args.meme_dir)

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    fimo_bin = which_or_die("fimo")

    jobs: List[Tuple] = []
    for r in rows:
        pid = str(r["id"])
        orig = Path(r["orig"])
        syn = Path(r["syn"])
        if not orig.is_file() or not syn.is_file():
            continue

        pair_root = outdir / pid
        jobs.append((pid, "orig", orig, pair_root / "orig", memes, fimo_bin, args.extra_fimo_args, args.keep_gff, args.gzip_tsv, args.dry_run))
        jobs.append((pid, "syn",  syn,  pair_root / "syn",  memes, fimo_bin, args.extra_fimo_args, args.keep_gff, args.gzip_tsv, args.dry_run))

    print(f"jobs={len(jobs)}  memes={len(memes)}  workers={workers}  outdir={outdir}")
    run_streamed(jobs, workers=workers)


if __name__ == "__main__":
    main()
