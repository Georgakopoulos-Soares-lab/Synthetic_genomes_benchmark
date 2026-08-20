#!/usr/bin/env python3
"""
Reviewer #1 (minor #1): supplementary phylotag table.

Collates every species-specific Evo 2 phylogenetic tag (phylotag) used to
condition generation, from two authoritative sources:
  1. scripts/fill_phylotag_manifest.py  (STATIC_TAG_TO_PHYLO; eukaryotes + a few)
  2. data/external/<TAG>/bonus_species.tsv  (evo2_prompt; prokaryotes + viruses)

Each phylotag has the form:
  |D__<domain>;P__<phylum>;C__<class>;O__<order>;F__<family>;G__<genus>;S__<species>|

The table parses these ranks into columns for the supplement and is grouped by
benchmark tag / domain.

Output: revisions/results/supplementary_phylotags.csv
"""
from __future__ import annotations

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import ast
import re
from pathlib import Path

import pandas as pd

PROJ = Path(_ROOT)
FILL = PROJ / "scripts" / "fill_phylotag_manifest.py"
EXTERNAL = PROJ / "data" / "external"

RANK_KEYS = [("D__", "domain"), ("P__", "phylum"), ("C__", "class"),
             ("O__", "order"), ("F__", "family"), ("G__", "genus"),
             ("S__", "species")]


def parse_phylotag(tag: str) -> dict:
    """Parse |D__..;P__..;..| into a rank dict (titlecased)."""
    out = {name: "" for _, name in RANK_KEYS}
    if not isinstance(tag, str):
        return out
    inner = tag.strip().strip("|")
    for part in inner.split(";"):
        part = part.strip()
        for prefix, name in RANK_KEYS:
            if part.startswith(prefix):
                out[name] = part[len(prefix):].strip().title()
    return out


def load_static_map() -> dict:
    """Extract STATIC_TAG_TO_PHYLO dict literal from fill_phylotag_manifest.py."""
    text = FILL.read_text()
    m = re.search(r"STATIC_TAG_TO_PHYLO\s*=\s*(\{.*?\n\})", text, re.DOTALL)
    if not m:
        return {}
    return ast.literal_eval(m.group(1))


def main() -> int:
    rows = []

    # 1) Static eukaryote (+misc) species-level tags
    static_map = load_static_map()
    for tag, phylo in static_map.items():
        ranks = parse_phylotag(phylo)
        rows.append({"benchmark_tag": tag, "organism_name": ranks["species"].title(),
                     "phylotag": phylo.strip(), "source": "static_map", **ranks})

    # 2) Per-genome bonus_species.tsv (prokaryotes + viruses)
    for tsv in sorted(EXTERNAL.glob("*/bonus_species.tsv")):
        tag = tsv.parent.name
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "evo2_prompt" not in df.columns:
            continue
        for _, r in df.iterrows():
            phylo = str(r["evo2_prompt"]).strip()
            ranks = parse_phylotag(phylo)
            rows.append({"benchmark_tag": tag,
                         "organism_name": str(r.get("organism_name", "")).strip(),
                         "phylotag": phylo, "source": "bonus_species", **ranks})

    df = pd.DataFrame(rows)
    # dedupe identical (tag, phylotag) entries
    df = df.drop_duplicates(subset=["benchmark_tag", "phylotag"]).reset_index(drop=True)
    # order columns
    cols = ["benchmark_tag", "organism_name", "domain", "phylum", "class",
            "order", "family", "genus", "species", "phylotag", "source"]
    df = df[cols].sort_values(["domain", "benchmark_tag", "species"])

    out = PROJ / "revisions" / "results" / "supplementary_phylotags.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[ok] {out}  ({len(df)} unique phylotags)")
    print("\n=== counts by domain ===")
    print(df["domain"].value_counts().to_string())
    print("\n=== counts by benchmark_tag ===")
    print(df["benchmark_tag"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
