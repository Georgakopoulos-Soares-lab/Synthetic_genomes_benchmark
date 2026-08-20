#!/usr/bin/env python3
"""
Reviewer #1, Major #1 (strategy 2): inference-time guided/constrained decoding
tested on the *actual* Evo 2 (evo2_7b), not a toy proxy.

We wrap Evo 2's own sampling step (vortex.model.generation.sample) with a
stateful constrainer that, at each generated nucleotide, biases the A/C/G/T
logits to (a) penalize extending homopolymer runs and (b) penalize k-mers that
are already over-represented in the generated sequence relative to the natural
target k-mer distribution. Because we patch the existing sampler, Evo 2's fast
cached-generation loop is preserved (near-native throughput).

Configs:
  baseline    : Evo 2 default decoding (T=1.0, top_k=4), no constraint
  constrained : same sampler + homopolymer + k-mer over-representation penalties

Outputs one FASTA per window per config, mirroring evo2_generate_sweep.py, so
the same downstream metric pipeline applies.

Run inside evo2.sif with PYTHONNOUSERSITE=1 (see decsweep launch pattern).
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch

ACCEPT = set("ACGTacgt")
_NT = {"A": 0, "C": 1, "G": 2, "T": 3}


def wrap80(s):
    return "\n".join(s[i:i + 80] for i in range(0, len(s), 80))


def load_seq(path, limit=None):
    buf = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            buf.append(line.strip())
    s = "".join(buf).upper()
    return s[:limit] if limit else s


def kmer_freq(seq, k):
    idx = np.array([_NT.get(c, -1) for c in seq], dtype=np.int64)
    idx = idx[idx >= 0]
    if idx.size < k:
        return np.ones(4 ** k) / (4 ** k)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(idx, k)
    codes = (win * powers).sum(axis=1)
    c = np.bincount(codes, minlength=4 ** k).astype(float)
    return c / c.sum()


class Constrainer:
    """Stateful per-batch logit constrainer applied at each sampling step."""

    def __init__(self, tok2nt, actg_token_ids, nat_kmer, k=4,
                 rep_pen=2.5, kmer_pen=1.5, overrep=1.5, mode="constrained"):
        self.tok2nt = tok2nt                 # {token_id: nt_index 0-3}
        self.actg = actg_token_ids           # [tid_A, tid_C, tid_G, tid_T]
        self.nat = nat_kmer + 1e-9
        self.k = k
        self.rep_pen = rep_pen
        self.kmer_pen = kmer_pen
        self.overrep = overrep
        self.mode = mode
        self.powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        self.reset(1)

    def reset(self, B):
        self.hist = [[] for _ in range(B)]
        self.kcounts = np.zeros((B, 4 ** self.k))
        self.ktot = np.zeros(B)

    def process(self, logits):
        if self.mode == "baseline":
            return logits
        # Evo 2 produces "inference tensors" (created under inference_mode) that
        # cannot be edited in place; clone to a normal tensor first.
        logits = logits.clone()
        B = logits.shape[0]
        for b in range(B):
            h = self.hist[b]
            if h:
                last = h[-1]
                rl = 1
                for j in range(len(h) - 2, -1, -1):
                    if h[j] == last:
                        rl += 1
                    else:
                        break
                logits[b, self.actg[last]] -= self.rep_pen * min(rl, 6) / 6.0
            if len(h) >= self.k - 1:
                ctx = h[-(self.k - 1):]
                base = int(np.dot(ctx, self.powers[1:]))
                for cand in range(4):
                    code = base + cand * int(self.powers[0])
                    cur = (self.kcounts[b, code] + 1e-9) / (self.ktot[b] + 1e-9)
                    ratio = cur / self.nat[code]
                    over = max(ratio - self.overrep, 0.0)
                    if over > 0:
                        logits[b, self.actg[cand]] -= self.kmer_pen * float(np.tanh(over))
        return logits

    def record(self, new_idx):
        arr = new_idx.detach().cpu().numpy().ravel()
        for b in range(arr.shape[0]):
            nt = self.tok2nt.get(int(arr[b]))
            if nt is None:
                continue
            h = self.hist[b]
            h.append(nt)
            if len(h) >= self.k:
                code = int(np.dot(h[-self.k:], self.powers))
                self.kcounts[b, code] += 1
                self.ktot[b] += 1


# global handle used by the patched sampler
CON = None


def install_patch():
    import vortex.model.generation as gen
    orig = gen.sample

    def patched(logits, top_k=1, top_p=0.0, temperature=1.0):
        if CON is not None:
            logits = CON.process(logits)
        idx = orig(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        if CON is not None:
            CON.record(idx)
        return idx

    gen.sample = patched
    return orig


def resolve_actg_ids(model):
    """Return {token_id: nt_index} and [tid_A,tid_C,tid_G,tid_T]."""
    tok = getattr(model, "tokenizer", None)
    ids = {}
    for nt in "ACGT":
        tid = None
        for meth in ("tokenize", "encode"):
            fn = getattr(tok, meth, None)
            if fn is None:
                continue
            try:
                out = fn(nt)
                if hasattr(out, "__len__") and len(out) >= 1:
                    tid = int(out[0] if not isinstance(out, (bytes, str)) else out[0])
                elif isinstance(out, int):
                    tid = out
                if tid is not None:
                    break
            except Exception:
                continue
        if tid is None:
            tid = ord(nt)  # byte tokenizer fallback (A=65,C=67,G=71,T=84)
        ids[nt] = tid
    actg = [ids["A"], ids["C"], ids["G"], ids["T"]]
    tok2nt = {ids["A"]: 0, ids["C"]: 1, ids["G"]: 2, ids["T"]: 3}
    return tok2nt, actg


def clean_dna(s):
    return "".join(ch.upper() for ch in s if ch in ACCEPT)


def extract_text(out):
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)) and out:
        seqs = out[0]
        if isinstance(seqs, str):
            return seqs
        if isinstance(seqs, (list, tuple)) and seqs:
            return seqs[0] if isinstance(seqs[0], str) else str(seqs[0])
    for attr in ("sequences", "seqs"):
        v = getattr(out, attr, None)
        if isinstance(v, (list, tuple)) and v:
            return v[0]
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True, choices=["baseline", "constrained"])
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--target-len", type=int, default=50000,
                    help="total sequence length (seed + generated).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only-tags", nargs="+", default=None)
    ap.add_argument("--kmer-k", type=int, default=4)
    ap.add_argument("--rep-pen", type=float, default=2.5)
    ap.add_argument("--kmer-pen", type=float, default=1.5)
    ap.add_argument("--overrep", type=float, default=1.5)
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.config
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading {args.model} ...", flush=True)
    from evo2 import Evo2
    model = Evo2(args.model)
    orig_sample = install_patch()
    tok2nt, actg = resolve_actg_ids(model)
    print(f"[info] ACTG token ids A/C/G/T = {actg}", flush=True)

    with open(args.manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.only_tags:
        rows = [r for r in rows if r["tag"] in set(args.only_tags)]

    global CON
    for r in rows:
        wid = r["window_id"]
        seed = load_seq(r["seed_fasta"])
        phylotag = r.get("phylotag", "") or ""
        prompt = (phylotag + seed) if phylotag else seed
        need_new = args.target_len - len(seed)
        out_fa = outdir / f"{wid}.{args.config}.syn.fa"
        if out_fa.exists() and out_fa.stat().st_size > 0:
            print(f"[skip] {out_fa}", flush=True)
            continue
        nat_kmer = kmer_freq(load_seq(r["natref_fasta"], args.target_len), args.kmer_k)
        CON = Constrainer(tok2nt, actg, nat_kmer, k=args.kmer_k,
                          rep_pen=args.rep_pen, kmer_pen=args.kmer_pen,
                          overrep=args.overrep, mode=args.config)
        CON.reset(1)
        print(f"[gen] {wid} config={args.config} need_new={need_new:,}", flush=True)
        t0 = time.time()
        out = model.generate([prompt], n_tokens=need_new,
                             temperature=args.temperature, top_k=args.top_k,
                             cached_generation=True)
        txt = clean_dna(extract_text(out))
        p = clean_dna(prompt)
        if txt.startswith(p[:200]) and len(txt) > len(p):
            txt = txt[len(p):]
        final = (seed + txt)[:args.target_len]
        with open(out_fa, "w") as oh:
            oh.write(f">{wid}.{args.config} T={args.temperature} top_k={args.top_k} "
                     f"rep_pen={args.rep_pen} kmer_pen={args.kmer_pen}\n{wrap80(final)}\n")
        dt = time.time() - t0
        print(f"[done] {wid}: {len(final):,} bp in {dt:.1f}s "
              f"({len(txt)/max(dt,1e-6):.1f} bp/s) -> {out_fa}", flush=True)

    print("[summary] done", flush=True)


if __name__ == "__main__":
    main()
