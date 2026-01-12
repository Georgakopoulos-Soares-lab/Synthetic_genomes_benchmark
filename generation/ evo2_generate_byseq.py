#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evo2 window generator (repo-shipped, workplace-independent).

- Reads a prompt FASTA (typically a single extracted window).
- Generates a synthetic sequence of the same length.
- Writes explicit output paths: --out_fasta, --out_meta
- Signature-safe: only passes kwargs that exist in model.generate signature.
"""

import argparse, gzip, json, math, os, re, signal, contextlib
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

ACCEPT_NUCS = set("ACGTacgt")

def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    hdr = None
    buf = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf).replace(" ", "").replace("\t", "")
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if hdr is not None:
            yield hdr, "".join(buf).replace(" ", "").replace("\t", "")

def only_dna(s: str) -> str:
    return "".join(ch.upper() if ch in ACCEPT_NUCS else "" for ch in s)

def wrap80(seq: str) -> str:
    return "\n".join(seq[i:i+80] for i in range(0, len(seq), 80))

SPECIES_STOPWORDS = {
    "strain","isolate","subsp.","subsp","chromosome","chrom.","plasmid","dna","rna","complete","genome",
    "chromosom","segment","mitochondrion","chloroplast","organelle","unlocalized","unplaced","scaffold"
}

def species_from_header(header: str) -> str:
    toks = header.split()
    clean = []
    for t in toks:
        t0 = t.rstrip(",;")
        if t0.lower() in SPECIES_STOPWORDS:
            break
        clean.append(t0)
    for i in range(len(clean) - 1):
        g, s = clean[i], clean[i+1]
        if re.match(r"^[A-Z][a-zA-Z-]+$", g) and re.match(r"^[a-z][a-zA-Z-]+$", s):
            return f"{g} {s}"
    return ""

def make_phylotag(species_hint: str, static_tag: str, disable_gbif: bool) -> str:
    if disable_gbif:
        return static_tag or ""
    if species_hint:
        try:
            from evo2.utils import make_phylotag_from_gbif
            tag = make_phylotag_from_gbif(species_hint)
            if isinstance(tag, str) and tag.startswith("|") and tag.endswith("|"):
                return tag
        except Exception:
            pass
    return static_tag or ""

def load_model(model_name: str):
    from evo2 import Evo2
    return Evo2(model_name)

def _decode_generated(model, out) -> str:
    try:
        if isinstance(out, tuple) and len(out) >= 1:
            gen_ids = out[0]
            tok = getattr(model, "tokenizer", None)
            if tok is not None:
                ids = gen_ids[0].tolist()
                return tok.decode(ids, skip_special_tokens=True)
            dec = getattr(model, "decode", None)
            if callable(dec):
                return dec(gen_ids)
        if isinstance(out, list) and out and isinstance(out[0], str):
            return out[0]
        if isinstance(out, str):
            return out
        if isinstance(out, dict) and "text" in out:
            return str(out["text"])
    except Exception:
        pass
    return str(out)

def _sig_params(fn):
    import inspect
    try:
        return set(inspect.signature(fn).parameters.keys())
    except Exception:
        return set()

class Timeout(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: int):
    def _handler(signum, frame):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def _build_kwargs(model, prompt: str, n_tokens: int, temperature: float, top_k: int, max_seqlen_margin: int):
    params = _sig_params(model.generate)
    kwargs = {}

    if "prompt_seqs" in params:
        kwargs["prompt_seqs"] = [prompt]
    elif "input_string" in params:
        kwargs["input_string"] = prompt
    elif "prompts" in params:
        kwargs["prompts"] = [prompt]
    else:
        kwargs["input_string"] = prompt

    for k in ["n_tokens","num_tokens","max_new_tokens","target_length","length","tokens_to_generate"]:
        if k in params:
            kwargs[k] = int(n_tokens)
            break

    max_seqlen = len(prompt) + int(n_tokens) + int(max_seqlen_margin)
    if "max_seqlen" in params:
        kwargs["max_seqlen"] = int(max_seqlen)

    if "temperature" in params:
        kwargs["temperature"] = float(temperature)
    if "top_k" in params and top_k is not None:
        kwargs["top_k"] = int(top_k)

    if "stop_at_eos" in params:
        kwargs["stop_at_eos"] = False
    if "skip_special_tokens" in params:
        kwargs["skip_special_tokens"] = True
    if "verbose" in params:
        kwargs["verbose"] = False
    if "print_generation" in params:
        kwargs["print_generation"] = False
    if "actg_only" in params:
        kwargs["actg_only"] = True

    return kwargs

def generate_new(model, prompt_text: str, need_new: int, temperature: float, top_k: int,
                 max_seqlen_margin: int, chunk_tokens: int, per_chunk_timeout_s: int) -> str:
    if need_new <= 0:
        return ""
    if not chunk_tokens or chunk_tokens <= 0:
        chunk_tokens = need_new

    chunks = math.ceil(need_new / chunk_tokens)
    produced = []
    generated = 0

    for _ci in range(chunks):
        want = min(chunk_tokens, need_new - generated)
        kwargs = _build_kwargs(model, prompt_text, want, temperature, top_k, max_seqlen_margin)
        with time_limit(per_chunk_timeout_s):
            out = model.generate(**kwargs)
            txt = _decode_generated(model, out)

        dna = only_dna(txt)
        if len(dna) < want:
            dna += "N" * (want - len(dna))
        produced.append(dna[:want])
        generated += want

    return "".join(produced)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Prompt FASTA (usually one window).")
    ap.add_argument("--out_fasta", required=True, help="Output FASTA path (explicit).")
    ap.add_argument("--out_meta", required=True, help="Output JSON metadata path (explicit).")
    ap.add_argument("--logdir", default=".", help="Log directory (optional).")

    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--seed_len", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=4)

    ap.add_argument("--phylo-tag", default="")
    ap.add_argument("--disable_gbif", action="store_true")
    ap.add_argument("--device", default="cuda")  # informational

    ap.add_argument("--max-seqlen-margin", type=int, default=16)
    ap.add_argument("--chunk_tokens", type=int, default=0)
    ap.add_argument("--per_chunk_timeout_s", type=int, default=10000000)
    args = ap.parse_args()

    out_fasta = Path(args.out_fasta)
    out_meta = Path(args.out_meta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.input)
    records = list(iter_fasta(fasta_path))
    if not records:
        raise SystemExit(f"No sequences found in {fasta_path}")

    model = load_model(args.model)
    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")

    synth_entries = []
    meta_records = []

    for hdr, seq in records:
        L = len(seq)
        seed = seq[: min(args.seed_len, L)]
        species_hint = species_from_header(hdr)
        phylotag = make_phylotag(species_hint, args.phylo_tag, args.disable_gbif)
        prompt = (phylotag + "\n" if phylotag else "") + seed

        need = max(0, L - len(seed))
        new = generate_new(
            model, prompt, need,
            args.temperature, args.top_k,
            args.max_seqlen_margin,
            args.chunk_tokens,
            args.per_chunk_timeout_s,
        )
        final = seed + new

        final_dna = only_dna(final)
        if len(final_dna) < L:
            final_dna += "N" * (L - len(final_dna))
        final_dna = final_dna[:L]

        synth_entries.append((hdr, final_dna))
        meta_records.append({
            "header": hdr,
            "native_length": L,
            "seed_len": min(args.seed_len, L),
            "species_from_header": species_hint,
            "phylotag_used": bool(phylotag),
            "phylo_tag": phylotag or None,
        })

    with open(out_fasta, "w") as f:
        for i, (hdr, seq) in enumerate(synth_entries, start=1):
            f.write(f">synthetic_{i} model={args.model} source={fasta_path.name}\n")
            f.write(wrap80(seq) + "\n")

    meta = {
        "timestamp": ts_run,
        "input": str(fasta_path),
        "model": args.model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "device": args.device,
        "max_seqlen_margin": args.max_seqlen_margin,
        "generation": {
            "chunk_tokens": args.chunk_tokens,
            "per_chunk_timeout_s": args.per_chunk_timeout_s,
        },
        "records": meta_records,
        "output_fasta": str(out_fasta),
    }
    with open(out_meta, "w") as jf:
        json.dump(meta, jf, indent=2)

if __name__ == "__main__":
    main()
