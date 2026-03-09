#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path

import pandas as pd

TAGS_EUK = [
    "Aedes_aegypti","Apis_mellifera","Arabidopsis_thaliana","Bos_taurus","Branchiostoma_floridae",
    "Caenorhabditis_elegans","Canis_familiaris_lupus","Danio_rerio","Drosophila_melanogaster",
    "Gallus_gallus","Gossypium_barbadense","Homo_sapiens","Mus_musculus","Nematostella_vectensis",
    "Oryza_sativa","Saccharina_japonica","Takifugu_rubripes","Triticum_aestivum",
    "Xenopus_tropicalis","Zea_mays",
]
TAGS_PROK = ["Archaea","Chlamydiota","Mycoplasmatota","Pseudomonadota"]
TAGS_VIR = ["Kitrinoviricota","Nucleocytoviricota","Peploviricota","Preplasmiviricota","Uroviricota","megaDNA_bacteriophages"]

DEFAULT_NEV = [1, 2, 4, 8, 16, 32]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--n-eval-list", nargs="+", type=int, default=DEFAULT_NEV)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-train-chunks", type=int, default=4)
    ap.add_argument("--calib-frac", type=float, default=0.10)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--stem-kernel", type=int, default=15)
    ap.add_argument("--block-kernel", type=int, default=7)
    ap.add_argument("--dilations", nargs=4, type=int, default=[1, 2, 2, 4])

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-pdf", action="store_true")
    return ap.parse_args()


def run_cmd(cmd):
    print("[cmd]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def run_domain(domain: str, tags: list[str], args, outdir: Path) -> list[Path]:
    script = Path(__file__).resolve().parent / "deep_detector_dilated_resnet1d.py"
    domdir = outdir / domain
    domdir.mkdir(parents=True, exist_ok=True)

    per_tag_paths: list[Path] = []
    for nev in args.n_eval_list:
        run_id = f"{domain}_nev{nev}_r1"
        out_csv = domdir / f"{run_id}.csv"

        cmd = [
            "python3", str(script),
            "--dataset-root", args.dataset_root,
            "--tags", *tags,
            "--lengths", str(args.L),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--n-train-chunks", str(args.n_train_chunks),
            "--n-eval-chunks", str(nev),
            "--calib-frac", str(args.calib_frac),
            "--dropout", str(args.dropout),
            "--base-ch", str(args.base_ch),
            "--stem-kernel", str(args.stem_kernel),
            "--block-kernel", str(args.block_kernel),
            "--dilations", *map(str, args.dilations),
            "--eval", "leave_one_tag_out",
            "--run-id", run_id,
            "--out", str(out_csv),
            "--device", args.device,
        ]
        run_cmd(cmd)

        per_tag = out_csv.with_name(out_csv.stem + "_per_tag.csv")
        if not per_tag.exists():
            raise FileNotFoundError(f"Missing per-tag output: {per_tag}")
        per_tag_paths.append(per_tag)

    return per_tag_paths


def merge_per_tag(per_tag_paths: list[Path], out_csv: Path):
    dfs = []
    for p in per_tag_paths:
        df = pd.read_csv(p)
        if "domain" not in df.columns:
            rid = str(df["run_id"].iloc[0]) if "run_id" in df.columns else p.stem
            df["domain"] = rid.split("_")[0]
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print("[ok] wrote:", out_csv, "rows=", len(out))


def plot_evalchunks(outdir: Path, merged_csv: Path, save_pdf: bool):
    plot_script = Path(__file__).resolve().parent / "plot_evalchunks_metric.py"
    for metric in ["auc", "f1"]:
        out_png = outdir / f"evalchunks_{metric}_tagstd.png"
        cmd = ["python3", str(plot_script), "--csv", str(merged_csv), "--metric", metric, "--out", str(out_png)]
        if save_pdf:
            cmd.append("--save-pdf")
        run_cmd(cmd)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_tag_paths = []
    per_tag_paths += run_domain("euk", TAGS_EUK, args, outdir)
    per_tag_paths += run_domain("prok", TAGS_PROK, args, outdir)
    per_tag_paths += run_domain("vir", TAGS_VIR, args, outdir)

    merged_csv = outdir / "all_domains_all_runs_per_tag.csv"
    merge_per_tag(per_tag_paths, merged_csv)
    plot_evalchunks(outdir, merged_csv, save_pdf=args.save_pdf)


if __name__ == "__main__":
    main()