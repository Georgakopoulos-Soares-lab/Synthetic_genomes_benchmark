import csv
from pathlib import Path
from typing import Dict, Iterable

HEADER = ["id", "orig", "syn", "contig", "start", "len", "seed"]

def write_manifest_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})
