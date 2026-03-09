#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_io.py

Utilities for loading the Zenodo synthetic_dataset.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLS = {"id", "orig", "syn"}


def pairs_csv_for_tag(dataset_root: Path, tag: str) -> Path:
    """
    Default Zenodo layout: <root>/data/<tag>/pairs.<tag>.csv
    """
    p = dataset_root / "data" / tag / f"pairs.{tag}.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"pairs CSV not found for tag='{tag}': {p}")


def resolve_root_relative(dataset_root: Path, relpath: str) -> Path:
    """
    All paths in the CSV are relative to dataset_root.
    """
    p = Path(str(relpath))
    if p.is_absolute():
        # allow absolute just in case; but your dataset uses root-relative
        return p
    return (dataset_root / p).resolve()


def load_pairs_csv(dataset_root: Path, pairs_csv: Path, tag: str) -> List[Dict]:
    """
    Returns a list of items (orig+syn) with metadata preserved.
    Each item has keys:
      tag, id, which, label, path, meta (dict of extra columns)
    """
    df = pd.read_csv(pairs_csv)
    if not REQUIRED_COLS.issubset(df.columns):
        missing = sorted(REQUIRED_COLS - set(df.columns))
        raise ValueError(f"{pairs_csv} missing required columns: {missing}")

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLS]

    items: List[Dict] = []
    for _, r in df.iterrows():
        pid = str(r["id"])

        meta = {c: r[c] for c in extra_cols}

        for which, label, col in (("orig", 0, "orig"), ("syn", 1, "syn")):
            p = resolve_root_relative(dataset_root, str(r[col]))
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing file referenced by pairs CSV.\n"
                    f"  tag={tag}\n  id={pid}\n  which={which}\n  raw={r[col]}\n  resolved={p}\n  pairs_csv={pairs_csv}"
                )
            items.append(
                {
                    "tag": tag,
                    "id": pid,
                    "which": which,
                    "label": int(label),
                    "path": str(p),
                    "meta": meta,
                }
            )
    return items