#!/usr/bin/env python3
"""
Alternative-decoding generation for the Evo2 revision experiment
(Reviewer #3 comment #7; Reviewer #1 minor #2).

Reads the seed manifest produced by extract_seed_windows.py and, for ONE
decoding configuration, generates a synthetic sequence per window by prompting
Evo2 with  <phylotag> + <seed (seed_len bp)>  and decoding to target_len.

Signature-safe: only passes kwargs that exist in the model.generate signature
(mirrors scripts/evo2_generate_byseq.py behaviour). Supports temperature,
top_k and top_p so we can compare the paper baseline (T=1, top_k=4) against
lower-temperature and nucleus (top_p) sampling.

GPU-only. Launched from the SLURM script decoding_sweep.sbatch inside the
evo2.sif container.
"""
import argparse
import csv
import inspect
import os
import sys
import time
from pathlib import Path

ACCEPT = set("ACGTacgt")


def wrap80(seq):
    return "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80))


def clean_dna(s):
    return "".join(ch.upper() for ch in s if ch in ACCEPT)


def load_seed(path):
    seq = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq)


def build_kwargs(sig_params, n_tokens, prompt_len, temperature, top_k, top_p,
                 max_seqlen_margin, force_prompt_threshold=None):
    """Only include kwargs Evo2.generate actually accepts."""
    kw = {}
    # length / token-count key (mirror evo2_generate_byseq.py key order)
    for k, v in (("n_tokens", n_tokens), ("num_tokens", n_tokens),
                 ("max_new_tokens", n_tokens), ("target_length", n_tokens),
                 ("length", n_tokens), ("tokens_to_generate", n_tokens)):
        if k in sig_params:
            kw[k] = int(v)
            break
    if "max_seqlen" in sig_params:
        kw["max_seqlen"] = int(prompt_len + n_tokens + max_seqlen_margin)
    # CRITICAL: without cached_generation the model pre-allocates a full
    # ~1M-token KV cache (=> multi-100GB OOM); the incremental path is required.
    if "cached_generation" in sig_params:
        kw["cached_generation"] = True
    if "temperature" in sig_params:
        kw["temperature"] = float(temperature)
    if top_k is not None and "top_k" in sig_params:
        kw["top_k"] = int(top_k)
    if top_p is not None and "top_p" in sig_params:
        kw["top_p"] = float(top_p)
    # Chunk-prefill long prompts to avoid Hyena FFT OOM on 10-20 kb seeds.
    if force_prompt_threshold and "force_prompt_threshold" in sig_params:
        kw["force_prompt_threshold"] = int(force_prompt_threshold)
    return kw


def extract_text(out):
    """Evo2.generate returns Tuple[List[str], List[float]] in this container
    (sequences, scores). Also tolerate object / list / str variants."""
    if isinstance(out, str):
        return out
    # attribute-style outputs (older APIs)
    for attr in ("sequences", "seqs", "text", "sequence"):
        v = getattr(out, attr, None)
        if isinstance(v, (list, tuple)) and v:
            return v[0] if isinstance(v[0], str) else str(v[0])
        if isinstance(v, str):
            return v
    # tuple/list: first element is the list of generated sequences
    if isinstance(out, (list, tuple)) and out:
        seqs = out[0]
        if isinstance(seqs, str):
            return seqs
        if isinstance(seqs, (list, tuple)) and seqs:
            return seqs[0] if isinstance(seqs[0], str) else str(seqs[0])
    return str(out)


def generate_one(model, gen_sig, prompt, need_new, temperature, top_k, top_p,
                 chunk_tokens, max_seqlen_margin, force_prompt_threshold=None):
    produced = []
    generated = 0
    t0 = time.time()
    ctx = prompt
    chunks = 1 if chunk_tokens <= 0 else (need_new + chunk_tokens - 1) // chunk_tokens
    for ci in range(chunks):
        want = need_new - generated if chunk_tokens <= 0 else min(chunk_tokens, need_new - generated)
        if want <= 0:
            break
        kw = build_kwargs(gen_sig, want, len(ctx), temperature, top_k, top_p,
                          max_seqlen_margin, force_prompt_threshold)
        t1 = time.time()
        # Evo2.generate expects prompt_seqs as a LIST; passing a bare string
        # makes it iterate characters -> huge batch -> multi-100GB KV cache OOM.
        out = model.generate([ctx], **kw)
        txt = clean_dna(extract_text(out))
        # strip echoed prompt if returned
        if txt.startswith(clean_dna(ctx)[:200]) and len(txt) > len(ctx):
            txt = txt[len(clean_dna(ctx)):]
        produced.append(txt)
        generated += len(txt)
        ctx = (ctx + txt)[-max(len(prompt), 30000):]
        dt = time.time() - t1
        print(f"[prog] chunk {ci + 1}/{chunks}: +{len(txt):,} bp in {dt:.1f}s "
              f"({len(txt) / max(dt, 1e-6):.1f} bp/s); total {generated:,}/{need_new:,}",
              flush=True)
        if len(txt) == 0:
            print("[warn] empty chunk; stopping", flush=True)
            break
    return "".join(produced)[:need_new]


def generate_batch(model, gen_sig, prompts, need_new_list, temperature, top_k,
                   top_p, max_seqlen_margin, force_prompt_threshold=None):
    """Single-shot batched generation for a group of windows.

    Evo2.generate accepts a list of prompts and decodes them in parallel on the
    GPU, which is far faster than one window at a time. n_tokens is shared, so
    we request max(need_new) and truncate each sequence to its own target.
    """
    n_tokens = max(need_new_list)
    max_prompt = max(len(p) for p in prompts)
    kw = build_kwargs(gen_sig, n_tokens, max_prompt, temperature, top_k, top_p,
                      max_seqlen_margin, force_prompt_threshold)
    t1 = time.time()
    out = model.generate(list(prompts), **kw)
    seqs = out[0] if isinstance(out, (list, tuple)) else \
        getattr(out, "sequences", out)
    if isinstance(seqs, str):
        seqs = [seqs]
    results = []
    for i, s in enumerate(seqs):
        txt = clean_dna(s if isinstance(s, str) else str(s))
        p = clean_dna(prompts[i])
        if txt.startswith(p[:200]) and len(txt) > len(p):
            txt = txt[len(p):]
        results.append(txt[:need_new_list[i]])
    dt = time.time() - t1
    tot = sum(len(r) for r in results)
    print(f"[prog] batch of {len(prompts)}: +{tot:,} bp in {dt:.1f}s "
          f"({tot / max(dt, 1e-6):.1f} bp/s aggregate)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config-name", required=True,
                    help="Label for this decoding config (e.g. lowtemp, nucleus).")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--seed-len-override", type=int, default=None,
                    help="Use this many bp of seed instead of manifest value "
                         "(for the conditioning-window experiment).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk-tokens", type=int, default=8192)
    ap.add_argument("--max-seqlen-margin", type=int, default=16)
    ap.add_argument("--only-tags", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="windows generated in parallel per generate() call.")
    ap.add_argument("--target-len-override", type=int, default=None,
                    help="override manifest target_len (e.g. 50000) for a "
                         "faster sweep at viral-genome scale.")
    ap.add_argument("--force-prompt-threshold", type=int, default=0,
                    help="Pass force_prompt_threshold to model.generate if "
                         "accepted; chunks long-prompt prefill to avoid Hyena "
                         "FFT OOM (e.g. 8192 for seed_len >= 10k).")
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.config_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] importing evo2 / loading model {args.model} ...", flush=True)
    from evo2 import Evo2
    model = Evo2(args.model)
    gen_sig = set(inspect.signature(model.generate).parameters.keys())
    print(f"[info] generate() accepts: {sorted(gen_sig)}", flush=True)
    print(f"[info] config={args.config_name} T={args.temperature} "
          f"top_k={args.top_k} top_p={args.top_p}", flush=True)

    with open(args.manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.only_tags:
        rows = [r for r in rows if r["tag"] in set(args.only_tags)]

    summary = []
    # build the list of pending windows (skip already-generated outputs)
    pending = []
    for r in rows:
        wid = r["window_id"]
        target_len = args.target_len_override or int(r["target_len"])
        seed_len = args.seed_len_override or int(r["seed_len"])
        seed = load_seed(r["seed_fasta"])[:seed_len]
        phylotag = r.get("phylotag", "") or ""
        prompt = (phylotag + seed) if phylotag else seed
        need_new = target_len - len(seed)
        out_fa = outdir / f"{wid}.{args.config_name}.syn.fa"
        if out_fa.exists() and out_fa.stat().st_size > 0:
            print(f"[skip] {out_fa} exists", flush=True)
            continue
        pending.append(dict(wid=wid, seed=seed, prompt=prompt,
                            need_new=need_new, target_len=target_len,
                            out_fa=out_fa))

    bs = max(1, args.batch_size)
    for bi in range(0, len(pending), bs):
        grp = pending[bi:bi + bs]
        print(f"[gen] batch {bi // bs + 1}: windows "
              f"{[g['wid'] for g in grp]} need_new~{grp[0]['need_new']:,}",
              flush=True)
        t0 = time.time()
        fpt = args.force_prompt_threshold or None
        news = generate_batch(model, gen_sig, [g["prompt"] for g in grp],
                              [g["need_new"] for g in grp], args.temperature,
                              args.top_k, args.top_p, args.max_seqlen_margin,
                              fpt)
        for g, new in zip(grp, news):
            final = (g["seed"] + new)[:g["target_len"]]
            with open(g["out_fa"], "w") as oh:
                oh.write(f">{g['wid']}.{args.config_name} T={args.temperature} "
                         f"top_k={args.top_k} top_p={args.top_p} "
                         f"seed_len={len(g['seed'])}\n{wrap80(final)}\n")
            print(f"[done] {g['wid']}: wrote {len(final):,} bp -> {g['out_fa']}",
                  flush=True)
            summary.append((g["wid"], len(final)))
        print(f"[time] batch {bi // bs + 1} took {time.time() - t0:.1f}s",
              flush=True)

    print(f"[summary] {len(summary)} sequences generated for "
          f"config={args.config_name}")


if __name__ == "__main__":
    main()
