#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


TAGS_EUK = [
    "Aedes_aegypti","Apis_mellifera","Arabidopsis_thaliana","Bos_taurus","Branchiostoma_floridae",
    "Caenorhabditis_elegans","Canis_familiaris_lupus","Danio_rerio","Drosophila_melanogaster",
    "Gallus_gallus","Gossypium_barbadense","Homo_sapiens","Mus_musculus","Nematostella_vectensis",
    "Oryza_sativa","Saccharina_japonica","Takifugu_rubripes","Triticum_aestivum",
    "Xenopus_tropicalis","Zea_mays",
]
TAGS_PROK = ["Archaea","Chlamydiota","Mycoplasmatota","Pseudomonadota"]
TAGS_VIR = ["Kitrinoviricota","Nucleocytoviricota","Peploviricota","Preplasmiviricota","Uroviricota","megaDNA_bacteriophages"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Root of dataset (contains data/<tag>/pairs.<tag>.csv)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--L", type=int, default=1024)

    ap.add_argument("--distances-eukprok", nargs="+", type=int, required=True)
    ap.add_argument("--distances-vir", nargs="+", type=int, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-train-chunks", type=int, default=4)
    ap.add_argument("--n-calib-chunks", type=int, default=32)
    ap.add_argument("--calib-frac", type=float, default=0.10)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--stem-kernel", type=int, default=15)
    ap.add_argument("--block-kernel", type=int, default=7)
    ap.add_argument("--dilations", nargs=4, type=int, default=[1, 2, 2, 4])

    ap.add_argument("--distance-chunks", type=int, default=1)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--metric", default="auc", choices=["auc", "f1", "acc", "precision", "recall"])
    ap.add_argument("--ylabel", default="AUROC")

    ap.add_argument("--xbreak", type=float, default=20000.0, help="Broken-axis split for euk/prok plot.")

    ap.add_argument("--show-std", action="store_true")
    ap.add_argument("--shade-std", action="store_true")
    ap.add_argument("--save-pdf", action="store_true")

    return ap.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_distance(domain: str, tags: list[str], distances: list[int], args, outdir: Path) -> Path:
    script = Path(__file__).resolve().parent / "deep_detector_distance_eval.py"
    prefix = outdir / f"{domain}_distance_L{args.L}"

    cmd = [
        "python3", str(script),
        "--dataset-root", args.dataset_root,
        "--tags", *tags,
        "--L", str(args.L),
        "--distances", *map(str, distances),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--n-train-chunks", str(args.n_train_chunks),
        "--n-calib-chunks", str(args.n_calib_chunks),
        "--calib-frac", str(args.calib_frac),
        "--dropout", str(args.dropout),
        "--base-ch", str(args.base_ch),
        "--stem-kernel", str(args.stem_kernel),
        "--block-kernel", str(args.block_kernel),
        "--dilations", *map(str, args.dilations),
        "--distance-chunks", str(args.distance_chunks),
        "--device", args.device,
        "--run-id", f"{domain}_distance_L{args.L}",
        "--out-prefix", str(prefix),
    ]
    run_cmd(cmd)
    return prefix


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # run experiments
    prefix_euk = run_distance("euk", TAGS_EUK, list(map(int, args.distances_eukprok)), args, outdir)
    prefix_prok = run_distance("prok", TAGS_PROK, list(map(int, args.distances_eukprok)), args, outdir)
    prefix_vir = run_distance("vir", TAGS_VIR, list(map(int, args.distances_vir)), args, outdir)

    euk_sum = prefix_euk.with_name(prefix_euk.name + "_summary.csv")
    prok_sum = prefix_prok.with_name(prefix_prok.name + "_summary.csv")
    vir_sum = prefix_vir.with_name(prefix_vir.name + "_summary.csv")

    # plot euk+prok with broken axis overlay
    plot_broken = Path(__file__).resolve().parent / "plot_distance_curve_broken_overlay.py"
    out_eukprok = outdir / f"distance_euk_prok_{args.metric}.png"
    cmd_ep = [
        "python3", str(plot_broken),
        "--ins", str(euk_sum), str(prok_sum),
        "--labels", "euk", "prok",
        "--metric", args.metric,
        "--ylabel", args.ylabel,
        "--xbreak", str(args.xbreak),
        "--out", str(out_eukprok),
    ]
    if args.show_std:
        cmd_ep.append("--show-std")
    if args.save_pdf:
        cmd_ep.append("--save-pdf")
    run_cmd(cmd_ep)

    # plot viral with single-axis plotter (your plot_distance_curve.py)
    plot_viral = Path(__file__).resolve().parent / "plot_distance_curve.py"
    out_vir = outdir / f"distance_vir_{args.metric}.png"
    cmd_v = [
        "python3", str(plot_viral),
        "--in", str(vir_sum),
        "--metric", args.metric,
        "--ylabel", args.ylabel,
        "--out", str(out_vir),
    ]
    if args.shade_std:
        cmd_v.append("--shade-std")
    if args.show_std:
        cmd_v.append("--show-std")
    if args.save_pdf:
        cmd_v.append("--save-pdf")
    run_cmd(cmd_v)


if __name__ == "__main__":
    main()