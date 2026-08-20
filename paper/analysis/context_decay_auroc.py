#!/usr/bin/env python3
"""
Distance-from-seed AUROC — leave-one-window-out (LOWO) cross-validation.

Replaces the original chunk-pooled StratifiedKFold with LOWO-CV: for each
(config, distance bin), a 6-mer logistic regression is trained on chunks from
four source windows and evaluated on the held-out fifth window, so every AUROC
estimate is genuinely out-of-window.

Chunk filtering now checks the full 1,024-bp chunk for ambiguous bases (not a
sampled subset).

Outputs (written to --outdir):
  context_decay_per_window.csv    — one row per (config, bin, held-out window)
                                    with coordinates, AUROC, and train/test counts
  context_decay_auroc.csv         — summary: mean / SD / median AUROC across the
                                    five held-out windows per (config, bin);
                                    backward-compatible with plot_context_decay.py
  context_decay_region_window.csv — per-window mean AUROC for the near-seed
                                    (0–20 kb) and long-range (40–100 kb) regions
  context_decay_permtest.csv      — exact paired sign-flip permutation tests
                                    between all three seed-length pairs for both
                                    regions, BH-corrected over 6 comparisons

CPU-only.  Run after run_seed_length_decay.sh / seed_length_decay.sbatch.

Usage:
    python3 context_decay_auroc.py \\
        --winlen-dir /path/to/winlen_generated \\
        --manifest   /path/to/seed_manifest_long.csv \\
        --outdir     /path/to/results/context_decay
"""

import argparse
import csv
import itertools
import math
import warnings
from itertools import groupby
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED_LEN_FOR_CONFIG = {"seed3k": 3000, "seed10k": 10000, "seed20k": 20000}
CONFIGS_ORDERED     = ["seed3k", "seed10k", "seed20k"]

# Distance bins (in kb) that define each analysis region.
# near_seed : 0–10 kb and 10–20 kb  (the first two bins after the seed end)
# long_range: 40–90 kb              (six bins well past the decay onset)
NEAR_SEED_BINS_KB  = frozenset({0.0, 10.0})
LONG_RANGE_BINS_KB = frozenset({40.0, 50.0, 60.0, 70.0, 80.0, 90.0})

_ACGT: frozenset = frozenset("ACGT")
_VOCAB_CACHE: dict = {}


# ── k-mer features ─────────────────────────────────────────────────────────────

def _get_vocab(k: int) -> dict:
    if k not in _VOCAB_CACHE:
        _VOCAB_CACHE[k] = {
            "".join(p): i
            for i, p in enumerate(itertools.product("ACGT", repeat=k))
        }
    return _VOCAB_CACHE[k]


def kmer_vec(seq: str, k: int) -> np.ndarray:
    """Normalised k-mer frequency vector (forward strand, 4^k features)."""
    vocab  = _get_vocab(k)
    size   = 4 ** k
    counts = np.zeros(size, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        j = vocab.get(seq[i : i + k])
        if j is not None:
            counts[j] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def featurise(chunks: list, k: int) -> np.ndarray:
    """Convert a list of DNA strings to a (n, 4^k) float32 feature matrix."""
    if not chunks:
        return np.empty((0, 4 ** k), dtype=np.float32)
    return np.array([kmer_vec(c, k) for c in chunks], dtype=np.float32)


# ── FASTA loading ──────────────────────────────────────────────────────────────

def load_seq(fasta_path: Path) -> str:
    parts = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts).upper()


# ── Chunk extraction ───────────────────────────────────────────────────────────

def extract_chunks(seq: str, start: int, end: int, chunk_size: int) -> list:
    """Non-overlapping chunks of length chunk_size from seq[start:end].

    A chunk is discarded if it contains *any* character outside {A, C, G, T}
    (the entire chunk is inspected, not a sampled subset).
    """
    chunks = []
    for cs in range(start, end - chunk_size + 1, chunk_size):
        chunk = seq[cs : cs + chunk_size]
        if len(chunk) == chunk_size and frozenset(chunk).issubset(_ACGT):
            chunks.append(chunk)
    return chunks


# ── Coordinate parsing ─────────────────────────────────────────────────────────

def parse_coords(src_record: str) -> tuple:
    """Return (chrom, genomic_start, window_length_bp) from manifest src_record.

    src_record format: 'orig|orig.{chrom}.{start}.{length}.fa'
    chrom may contain internal dots (e.g. NC_000067.7); the last two dot-fields
    are always the integer start and length, so we parse from the right.
    """
    name   = src_record.split("|")[-1].removesuffix(".fa")  # drop scheme + .fa
    parts  = name.split(".")                                 # ['orig', ..., start, len]
    win_len        = int(parts[-1])
    genomic_start  = int(parts[-2])
    chrom          = ".".join(parts[1:-2])                   # skip leading 'orig'
    return chrom, genomic_start, win_len


# ── Leave-one-window-out AUROC ─────────────────────────────────────────────────

def lowo_auroc(win_feats: dict) -> dict:
    """Leave-one-window-out AUROC for one (config, bin).

    Parameters
    ----------
    win_feats : {wid: {"syn": ndarray (n, f), "nat": ndarray (n, f)}}

    Returns
    -------
    {wid: float}  — AUROC for each held-out window (nan if unevaluable)
    """
    wids = sorted(win_feats)
    clf  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=1)
    results = {}

    for held_wid in wids:
        # Build train set from all windows except the held-out one
        tr_parts_X, tr_parts_y = [], []
        for wid in wids:
            if wid == held_wid:
                continue
            d = win_feats[wid]
            if d["syn"].shape[0]:
                tr_parts_X.append(d["syn"])
                tr_parts_y.append(np.ones(d["syn"].shape[0]))
            if d["nat"].shape[0]:
                tr_parts_X.append(d["nat"])
                tr_parts_y.append(np.zeros(d["nat"].shape[0]))

        d_te = win_feats[held_wid]
        n_syn_te = d_te["syn"].shape[0]
        n_nat_te = d_te["nat"].shape[0]

        if not tr_parts_X or n_syn_te == 0 or n_nat_te == 0:
            results[held_wid] = math.nan
            continue

        X_tr = np.vstack(tr_parts_X)
        y_tr = np.concatenate(tr_parts_y)

        if len(np.unique(y_tr)) < 2:
            results[held_wid] = math.nan
            continue

        X_te = np.vstack([d_te["syn"], d_te["nat"]])
        y_te = np.concatenate([np.ones(n_syn_te), np.zeros(n_nat_te)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]

        results[held_wid] = float(roc_auc_score(y_te, proba))

    return results


# ── Exact paired permutation test (sign-flip) ──────────────────────────────────

def paired_permtest(a: list, b: list) -> float:
    """Two-tailed exact paired permutation test via sign-flip enumeration.

    Enumerates all 2^n sign combinations of the pairwise differences
    d_i = a_i - b_i.  Returns the fraction with |mean| >= |observed mean|.
    """
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    n     = len(diffs)
    d_obs = abs(diffs.mean())
    count = 0
    total = 2 ** n
    for mask in range(total):
        signs = np.array([(1 if (mask >> i) & 1 else -1) for i in range(n)],
                         dtype=float)
        if abs((signs * diffs).mean()) >= d_obs - 1e-12:
            count += 1
    return count / total


# ── Benjamini–Hochberg correction ──────────────────────────────────────────────

def bh_correct(pvals: list) -> list:
    """BH-adjusted p-values.  nan inputs produce nan outputs; others are adjusted
    together as a group (so m equals the number of non-nan tests)."""
    valid_idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    adj       = [math.nan] * len(pvals)
    if not valid_idx:
        return adj
    vp    = [pvals[i] for i in valid_idx]
    m     = len(vp)
    order = np.argsort(vp)            # indices into vp sorted ascending
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    raw = np.array([min(p * m / r, 1.0) for p, r in zip(vp, ranks)])
    # Enforce monotonicity: cumulative minimum from highest rank downwards
    for i in range(m - 2, -1, -1):
        raw[order[i]] = min(raw[order[i]], raw[order[i + 1]])
    for j, i in enumerate(valid_idx):
        adj[i] = float(raw[j])
    return adj


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Distance-from-seed AUROC — LOWO-CV")
    ap.add_argument("--winlen-dir", required=True,
                    help="Root dir with seed3k/, seed10k/, seed20k/ subdirs")
    ap.add_argument("--manifest",   required=True,
                    help="seed_manifest_long.csv (natref paths + coordinates)")
    ap.add_argument("--outdir",     required=True)
    ap.add_argument("--tags",       nargs="+", default=["Publish_Human"])
    ap.add_argument("--bin-size",   type=int,  default=10000,
                    help="Distance bin width in bp (default 10 kb)")
    ap.add_argument("--chunk-size", type=int,  default=1024,
                    help="Classifier chunk size in bp (default 1024)")
    ap.add_argument("--kmer-k",     type=int,  default=6)
    ap.add_argument("--max-dist",   type=int,  default=100000,
                    help="Max distance from seed end in bp (default 100 kb)")
    # --n-folds is no longer used (LOWO replaces k-fold CV) but kept for
    # backward-compatibility with existing call sites.
    ap.add_argument("--n-folds",    type=int,  default=5,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    winlen_dir = Path(args.winlen_dir)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ──────────────────────────────────────────────────────
    tags_set = set(args.tags)
    manifest: dict = {}    # wid -> {natref, chrom, genomic_start, win_len}
    with open(args.manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["tag"] not in tags_set:
                continue
            wid = row["window_id"]
            chrom, genomic_start, win_len = parse_coords(row["src_record"])
            manifest[wid] = {
                "natref":        row["natref_fasta"],
                "chrom":         chrom,
                "genomic_start": genomic_start,
                "win_len":       win_len,
            }
    print(f"[info] {len(manifest)} windows for tags {args.tags}", flush=True)
    if not manifest:
        print("[error] no windows found — check --tags and manifest path")
        return

    bins       = list(range(0, args.max_dist, args.bin_size))
    wids_sorted = sorted(manifest)

    # Accumulators
    pw_rows: list = []    # rows for context_decay_per_window.csv
    # regional_auroc[config][wid][region] = list of per-bin AUROC values
    regional: dict = {
        cfg: {wid: {"near_seed": [], "long_range": []} for wid in wids_sorted}
        for cfg in CONFIGS_ORDERED
    }

    for config in CONFIGS_ORDERED:
        seed_len = SEED_LEN_FOR_CONFIG[config]
        subdir   = winlen_dir / config
        if not subdir.is_dir():
            print(f"[warn] {subdir} not found — skipping {config}", flush=True)
            continue

        syn_files = sorted(subdir.glob(f"*.{config}.syn.fa"))
        print(f"\n[{config}] seed_len={seed_len:,}  {len(syn_files)} syn.fa files",
              flush=True)

        # Load sequences once per config
        seqs: dict = {}
        for syn_fa in syn_files:
            wid = syn_fa.name[: -len(f".{config}.syn.fa")]
            if wid not in manifest:
                print(f"  [warn] {wid} not in manifest — skipping", flush=True)
                continue
            nat_path = Path(manifest[wid]["natref"])
            if not nat_path.exists():
                print(f"  [warn] natref missing: {nat_path}", flush=True)
                continue
            print(f"  load {wid} ...", end="", flush=True)
            syn_seq = load_seq(syn_fa)
            nat_seq = load_seq(nat_path)
            print(f" syn={len(syn_seq):,}  nat={len(nat_seq):,}", flush=True)
            seqs[wid] = {"syn": syn_seq, "nat": nat_seq}

        # Per-bin LOWO loop
        for bin_start in bins:
            bin_end  = bin_start + args.bin_size
            bkb      = bin_start / 1000

            # Extract chunks for every available window in this bin
            win_chunks: dict = {}
            for wid, s in seqs.items():
                abs_s = seed_len + bin_start
                abs_e = seed_len + bin_end
                if abs_e > len(s["syn"]) or abs_e > len(s["nat"]):
                    continue
                syn_ch = extract_chunks(s["syn"], abs_s, abs_e, args.chunk_size)
                nat_ch = extract_chunks(s["nat"], abs_s, abs_e, args.chunk_size)
                win_chunks[wid] = {"syn": syn_ch, "nat": nat_ch}

            if len(win_chunks) < 2:
                print(f"  {bkb:.0f}–{bin_end / 1000:.0f} kb: "
                      f"only {len(win_chunks)} window(s) — skipping", flush=True)
                continue

            # Precompute feature matrices (avoids recomputing kmer_vec per LOWO fold)
            win_feats = {
                wid: {
                    "syn": featurise(win_chunks[wid]["syn"], args.kmer_k),
                    "nat": featurise(win_chunks[wid]["nat"], args.kmer_k),
                }
                for wid in win_chunks
            }

            per_win = lowo_auroc(win_feats)

            valid = [v for v in per_win.values() if not math.isnan(v)]
            summary_str = (f"mean={np.mean(valid):.3f}  "
                           f"[{', '.join(f'{v:.3f}' for v in valid)}]"
                           if valid else "no valid AUROCs")
            print(f"  {bkb:>5.0f}–{bin_end / 1000:>5.0f} kb: {summary_str}",
                  flush=True)

            # Write per-window rows
            for wid, auroc in per_win.items():
                m_info = manifest[wid]
                other  = [w for w in win_chunks if w != wid]
                n_syn_tr = sum(len(win_chunks[w]["syn"]) for w in other)
                n_nat_tr = sum(len(win_chunks[w]["nat"]) for w in other)
                held_ch  = win_chunks[wid]

                pw_rows.append({
                    "config":           config,
                    "seed_len":         seed_len,
                    "bin_start_bp":     bin_start,
                    "bin_end_bp":       bin_end,
                    "bin_start_kb":     bkb,
                    "bin_end_kb":       bin_end / 1000,
                    "held_out_window":  wid,
                    "chrom":            m_info["chrom"],
                    "genomic_start":    m_info["genomic_start"],
                    "window_length_bp": m_info["win_len"],
                    "auroc":            f"{auroc:.4f}" if not math.isnan(auroc) else "",
                    "n_syn_train":      n_syn_tr,
                    "n_nat_train":      n_nat_tr,
                    "n_syn_test":       len(held_ch["syn"]),
                    "n_nat_test":       len(held_ch["nat"]),
                })

                # Accumulate into regional averages
                if not math.isnan(auroc):
                    if bkb in NEAR_SEED_BINS_KB:
                        regional[config][wid]["near_seed"].append(auroc)
                    if bkb in LONG_RANGE_BINS_KB:
                        regional[config][wid]["long_range"].append(auroc)

    # ── context_decay_per_window.csv ────────────────────────────────────────
    if pw_rows:
        p = outdir / "context_decay_per_window.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pw_rows[0].keys()))
            w.writeheader()
            w.writerows(pw_rows)
        print(f"\n[done] {len(pw_rows)} rows -> {p}")

    # ── context_decay_auroc.csv  (summary; backward-compat with plot script) ─
    summary_rows: list = []
    for (config, bin_bp), grp in groupby(
            sorted(pw_rows, key=lambda r: (r["config"], r["bin_start_bp"])),
            key=lambda r: (r["config"], r["bin_start_bp"])):
        grp = list(grp)
        aurocs = [float(r["auroc"]) for r in grp if r["auroc"]]
        if not aurocs:
            continue
        summary_rows.append({
            "config":         config,
            "seed_len":       grp[0]["seed_len"],
            "bin_start_bp":   bin_bp,
            "bin_end_bp":     grp[0]["bin_end_bp"],
            "bin_start_kb":   bin_bp / 1000,
            "bin_end_kb":     grp[0]["bin_end_kb"],
            "auroc_mean":     f"{np.mean(aurocs):.4f}",
            "auroc_std":      f"{np.std(aurocs, ddof=1):.4f}",
            "auroc_median":   f"{np.median(aurocs):.4f}",
            "n_windows":      len(aurocs),
            "n_syn_chunks":   sum(r["n_syn_test"] for r in grp),
            "n_nat_chunks":   sum(r["n_nat_test"] for r in grp),
        })

    if summary_rows:
        p = outdir / "context_decay_auroc.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[done] {len(summary_rows)} rows -> {p}")

    # ── context_decay_region_window.csv ────────────────────────────────────
    rw_rows: list = []
    for config in CONFIGS_ORDERED:
        for wid in wids_sorted:
            m_info = manifest[wid]
            for region in ("near_seed", "long_range"):
                vals   = regional[config][wid][region]
                mean_v = np.mean(vals) if vals else math.nan
                rw_rows.append({
                    "config":            config,
                    "seed_len":          SEED_LEN_FOR_CONFIG[config],
                    "region":            region,
                    "window_id":         wid,
                    "chrom":             m_info["chrom"],
                    "genomic_start":     m_info["genomic_start"],
                    "auroc_mean_region": f"{mean_v:.4f}" if not math.isnan(mean_v) else "",
                    "n_bins":            len(vals),
                })

    if rw_rows:
        p = outdir / "context_decay_region_window.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rw_rows[0].keys()))
            w.writeheader()
            w.writerows(rw_rows)
        print(f"[done] {len(rw_rows)} rows -> {p}")

    # ── Paired permutation tests ─────────────────────────────────────────────
    # For each (region, seed-length pair): collect the 5 per-window regional
    # mean AUROCs and run an exact sign-flip test.  6 tests total → BH correct.
    pairs   = [("seed3k", "seed10k"), ("seed3k", "seed20k"), ("seed10k", "seed20k")]
    regions = ["near_seed", "long_range"]

    # Build per-window regional mean table
    reg_mean: dict = {
        cfg: {
            wid: {
                region: (np.mean(regional[cfg][wid][region])
                         if regional[cfg][wid][region] else math.nan)
                for region in regions
            }
            for wid in wids_sorted
        }
        for cfg in CONFIGS_ORDERED
    }

    raw_pvals: list = []
    meta: list      = []    # (region, ca, cb, a_vals, b_vals, mean_diff)

    for region in regions:
        for ca, cb in pairs:
            a_all = [reg_mean[ca][w][region] for w in wids_sorted]
            b_all = [reg_mean[cb][w][region] for w in wids_sorted]
            clean = [(a, b) for a, b in zip(a_all, b_all)
                     if not math.isnan(a) and not math.isnan(b)]
            if len(clean) < 2:
                raw_pvals.append(math.nan)
                meta.append((region, ca, cb, [], [], math.nan))
                continue
            a_c = [x[0] for x in clean]
            b_c = [x[1] for x in clean]
            mean_diff = float(np.mean(np.array(a_c) - np.array(b_c)))
            p = paired_permtest(a_c, b_c)
            raw_pvals.append(p)
            meta.append((region, ca, cb, a_c, b_c, mean_diff))

    adj_pvals = bh_correct(raw_pvals)

    pt_rows: list = []
    for i, (region, ca, cb, a_c, b_c, mean_diff) in enumerate(meta):
        p_raw = raw_pvals[i]
        p_adj = adj_pvals[i]
        pt_rows.append({
            "region":              region,
            "config_a":            ca,
            "config_b":            cb,
            "n_windows":           len(a_c),
            "mean_auroc_a":        f"{np.mean(a_c):.4f}" if a_c else "",
            "mean_auroc_b":        f"{np.mean(b_c):.4f}" if b_c else "",
            "mean_diff_a_minus_b": f"{mean_diff:.4f}"    if not math.isnan(mean_diff) else "",
            "n_permutations":      2 ** len(a_c)         if a_c else "",
            "p_value":             f"{p_raw:.6f}"        if not math.isnan(p_raw) else "",
            "p_adj_bh":            f"{p_adj:.6f}"        if not math.isnan(p_adj) else "",
            "sig_005":             ("yes" if not math.isnan(p_adj) and p_adj < 0.05
                                    else "no"),
            "window_auroc_a":      ";".join(f"{v:.4f}" for v in a_c),
            "window_auroc_b":      ";".join(f"{v:.4f}" for v in b_c),
        })

    if pt_rows:
        p = outdir / "context_decay_permtest.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pt_rows[0].keys()))
            w.writeheader()
            w.writerows(pt_rows)
        print(f"[done] {len(pt_rows)} rows -> {p}")

        print("\n=== Permutation test results (BH-corrected) ===")
        for row in pt_rows:
            print(f"  {row['region']:12s}  {row['config_a']:8s} vs {row['config_b']:8s}: "
                  f"diff={row['mean_diff_a_minus_b']:>7s}  "
                  f"p={row['p_value']:>8s}  p_adj={row['p_adj_bh']:>8s}  "
                  f"sig={row['sig_005']}")


if __name__ == "__main__":
    main()
