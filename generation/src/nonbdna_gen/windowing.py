import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

CHR_PAT = re.compile(r"^chr([0-9]{1,2}|1[0-9]|2[0-2]|X|Y|M|MT)$", re.I)

def fasta_iter(path: Path):
    hdr = None
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if hdr is not None:
            yield hdr, "".join(buf)

def is_primary(name: str) -> bool:
    if name.startswith("NC_"):
        return True
    if CHR_PAT.fullmatch(name):
        return True
    return False

def n_ratio(s: str) -> float:
    if not s:
        return 1.0
    n = sum(1 for ch in s if ch in "Nn")
    return n / len(s)

def sample_windows_and_write(
    *,
    ref_fa: Path,
    n_windows: int,
    window_len: int,
    min_gap: int,
    n_threshold: float,
    seed: int,
    primary_only: bool,
    win_tsv: Path,
    meta_json: Path,
    win_dir: Path,
) -> List[Tuple[str, int, int]]:
    random.seed(seed)
    os.makedirs(win_dir, exist_ok=True)

    contigs = []
    for hdr, seq in fasta_iter(ref_fa):
        name = hdr.split()[0]
        ln = len(seq)
        if ln <= 0:
            continue
        if (not primary_only) or is_primary(name):
            contigs.append((name, ln))

    if not contigs:
        for hdr, seq in fasta_iter(ref_fa):
            name = hdr.split()[0]
            ln = len(seq)
            if ln > 0:
                contigs.append((name, ln))

    total_len = sum(l for _, l in contigs)
    if total_len == 0:
        raise ValueError("Empty reference?")

    raw_share = {n: (n_windows * (l / total_len)) for n, l in contigs}
    alloc = {n: int(raw_share[n]) for n, _ in contigs}
    cur = sum(alloc.values())

    if cur < n_windows:
        remainder = n_windows - cur
        fracs = sorted(contigs, key=lambda x: (raw_share[x[0]] - int(raw_share[x[0]])), reverse=True)
        idx = 0
        while remainder > 0 and fracs:
            n, _ = fracs[idx]
            alloc[n] += 1
            remainder -= 1
            idx = (idx + 1) % len(fracs)
    elif cur > n_windows:
        excess = cur - n_windows
        fracs = sorted(contigs, key=lambda x: (raw_share[x[0]] - int(raw_share[x[0]])))
        idx = 0
        while excess > 0 and fracs:
            n, _ = fracs[idx]
            if alloc[n] > 0:
                alloc[n] -= 1
                excess -= 1
            idx = (idx + 1) % len(fracs)
            if idx == 0 and all(alloc[n] == 0 for n, _ in fracs):
                break

    taken = defaultdict(list)

    def ok_place(chr_name: str, start: int, length: int) -> bool:
        end = start + length - 1
        for (a, b) in taken[chr_name]:
            if not (end + min_gap < a or b + min_gap < start):
                return False
        return True

    windows: List[Tuple[str, int, int]] = []
    lens = {n: l for n, l in contigs}

    for hdr, seq in fasta_iter(ref_fa):
        name = hdr.split()[0]
        if name not in lens:
            continue
        Lc = lens[name]
        need = alloc.get(name, 0)
        if need <= 0:
            continue

        wlen = window_len if Lc >= window_len else Lc
        if wlen <= 0:
            continue

        got = 0
        tries = 0
        MAX_TRIES = 20000

        while got < need and tries < MAX_TRIES:
            tries += 1
            start = 0 if Lc == wlen else random.randint(0, Lc - wlen)
            if not ok_place(name, start, wlen):
                continue
            subseq = seq[start:start + wlen]
            if n_ratio(subseq) > n_threshold:
                continue

            taken[name].append((start, start + wlen - 1))
            windows.append((name, start, wlen))

            outp = win_dir / f"orig.{name}.{start}.{wlen}.fa"
            with open(outp, "w") as w:
                w.write(f">{name}:{start+1}-{start+wlen}\n")
                for i in range(0, len(subseq), 80):
                    w.write(subseq[i:i+80] + "\n")

            got += 1

        if got < need:
            tries_relax = 0
            MAX_RELAX_TRIES = 50000
            while got < need and tries_relax < MAX_RELAX_TRIES:
                tries_relax += 1
                start = 0 if Lc == wlen else random.randint(0, Lc - wlen)
                if not ok_place(name, start, wlen):
                    continue

                subseq = seq[start:start + wlen]
                taken[name].append((start, start + wlen - 1))
                windows.append((name, start, wlen))

                outp = win_dir / f"orig.{name}.{start}.{wlen}.fa"
                with open(outp, "w") as w:
                    w.write(f">{name}:{start+1}-{start+wlen}\n")
                    for i in range(0, len(subseq), 80):
                        w.write(subseq[i:i+80] + "\n")

                got += 1

    with open(win_tsv, "w") as w:
        for chr_name, st, ln in windows:
            w.write(f"{chr_name}\t{st}\t{ln}\n")

    meta = {
        "seed": seed,
        "ref_fa": str(ref_fa),
        "n_windows": n_windows,
        "window_len_requested": window_len,
        "min_gap": min_gap,
        "n_threshold": n_threshold,
        "chromosomes_primary_filtered": [{"name": n, "length": l} for n, l in contigs],
        "allocation": alloc,
        "windows_count": len(windows),
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return windows
