#!/usr/bin/env python3
import argparse
from pathlib import Path
from .pipeline import run_pipeline

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nonbdna-gen",
        description="Generate synthetic genome windows via Evo2 (Apptainer).",
    )
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--workdir", required=True, help="Run work directory (created if missing)")
    args = ap.parse_args()

    run_pipeline(config_path=Path(args.config), workdir=Path(args.workdir))

if __name__ == "__main__":
    main()
