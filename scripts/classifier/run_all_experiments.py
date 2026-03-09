#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--L", type=int, default=1024)

    ap.add_argument("--distances-eukprok", nargs="+", type=int, required=True)
    ap.add_argument("--distances-vir", nargs="+", type=int, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-train-chunks", type=int, default=4)
    ap.add_argument("--calib-frac", type=float, default=0.10)

    ap.add_argument("--n-eval-list", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-calib-chunks", type=int, default=32)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--stem-kernel", type=int, default=15)
    ap.add_argument("--block-kernel", type=int, default=7)
    ap.add_argument("--dilations", nargs=4, type=int, default=[1, 2, 2, 4])

    ap.add_argument("--distance-chunks", type=int, default=1)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--metric", default="auc", choices=["auc", "f1", "acc", "precision", "recall"])
    ap.add_argument("--ylabel", default="AUROC")
    ap.add_argument("--xbreak", type=float, default=20000.0)

    ap.add_argument("--show-std", action="store_true")
    ap.add_argument("--shade-std", action="store_true")
    ap.add_argument("--save-pdf", action="store_true")

    return ap.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent

    out_eval = outdir / "evalchunks"
    out_dist = outdir / "distance"
    out_eval.mkdir(parents=True, exist_ok=True)
    out_dist.mkdir(parents=True, exist_ok=True)

    # 1) evalchunks
    run_eval = here / "run_evalchunks.py"
    cmd_eval = [
        "python3", str(run_eval),
        "--dataset-root", args.dataset_root,
        "--outdir", str(out_eval),
        "--L", str(args.L),
        "--n-eval-list", *map(str, args.n_eval_list),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--n-train-chunks", str(args.n_train_chunks),
        "--calib-frac", str(args.calib_frac),
        "--dropout", str(args.dropout),
        "--base-ch", str(args.base_ch),
        "--stem-kernel", str(args.stem_kernel),
        "--block-kernel", str(args.block_kernel),
        "--dilations", *map(str, args.dilations),
        "--device", args.device,
    ]
    if args.save_pdf:
        cmd_eval.append("--save-pdf")
    run_cmd(cmd_eval)

    # 2) distance
    run_dist = here / "run_distance.py"
    cmd_dist = [
        "python3", str(run_dist),
        "--dataset-root", args.dataset_root,
        "--outdir", str(out_dist),
        "--L", str(args.L),
        "--distances-eukprok", *map(str, args.distances_eukprok),
        "--distances-vir", *map(str, args.distances_vir),
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
        "--metric", args.metric,
        "--ylabel", args.ylabel,
        "--xbreak", str(args.xbreak),
    ]
    if args.show_std:
        cmd_dist.append("--show-std")
    if args.shade_std:
        cmd_dist.append("--shade-std")
    if args.save_pdf:
        cmd_dist.append("--save-pdf")
    run_cmd(cmd_dist)

    print("[done] outputs:")
    print(" evalchunks:", out_eval)
    print(" distance:  ", out_dist)
    print("[done] key plots:")
    print(" ", out_eval / f"evalchunks_{args.metric}_tagstd.png")
    print(" ", out_dist / f"distance_euk_prok_{args.metric}.png")
    print(" ", out_dist / f"distance_vir_{args.metric}.png")


if __name__ == "__main__":
    main()