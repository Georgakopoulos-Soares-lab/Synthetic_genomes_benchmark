#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_intervention_demo.py

Does intervening actually make generated sequence more natural?

Trains a small character-level DNA model on sequences you supply, then
generates from it under a 2x2 design and scores every arm with the same
metrics the benchmark suite uses:

                       plain decoding      constrained decoding
    standard training      baseline            constr_decode
    + structural loss      aux_loss            aux+constr

The model is deliberately small (a 2-layer GRU, a few hundred training steps)
so the whole comparison runs on a laptop CPU in minutes. It is a testbed for
the *interventions*, not a genomic language model: absolute numbers mean
nothing, only the differences between arms do. The same two interventions
apply unchanged to a real model -- see ``structural_losses.py`` for the loss
terms and ``constrained_decoding.py`` for the sampler wrapper, including its
Evo 2 adapter.

Metrics (lower is better for all of them)
-----------------------------------------
    kmer_jsd            Jensen-Shannon divergence of k-mer spectra against
                        the held-out natural sequences
    homopolymer_frac    |generated - natural| in homopolymer content
    low_complexity      |generated - natural| in 3-mer low-complexity score
    cpg_oe_error        |generated - natural| in CpG observed/expected
    detect_auroc        AUROC of a k-mer logistic regression separating
                        generated from held-out natural chunks. 0.5 means
                        indistinguishable; this is the metric that matters,
                        and the one the interventions find hardest to move.

Outputs
-------
    <outdir>/interventions.metrics.csv   one row per arm
    <outdir>/interventions.png           grouped bar chart (unless --no-plot)
    <outdir>/<arm>.generated.fa          the generated sequence per arm

Example
-------
    python generation/improve/run_intervention_demo.py \\
        --fasta data/Homo_sapiens/orig.*.fa \\
        --outdir results/interventions --steps 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts" / "benchmarks"))

import _seqio as S  # noqa: E402
from constrained_decoding import (  # noqa: E402
    ConstraintStack, HomopolymerPenalty, KmerOverrepresentationPenalty,
    NucleotideVocab, reference_kmer_freqs,
)
from structural_losses import (  # noqa: E402
    dinucleotide_kl_loss, homopolymer_loss, reference_dinucleotide,
)

ARMS = ["baseline", "aux_loss", "constr_decode", "aux+constr"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sequences(patterns: list[str], min_len: int) -> list[np.ndarray]:
    """Load every FASTA record matching the given paths/globs as index arrays."""
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.fa")) + sorted(p.glob("*.fasta")))
        elif any(ch in pattern for ch in "*?["):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(p)

    out = []
    for path in paths:
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        for _, seq in S.iter_fasta(path):
            idx = S.encode_acgt(seq.upper())
            idx = idx[idx >= 0].astype(np.int64)
            if idx.size >= min_len:
                out.append(idx)
    if not out:
        raise SystemExit(
            "no sequences loaded; pass --fasta with FASTA files, a directory, "
            "or a glob (quote the glob so the shell does not expand it)"
        )
    return out


def to_string(idx: np.ndarray) -> str:
    return "".join("ACGT"[i] for i in idx)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharGRU(nn.Module):
    """Minimal autoregressive character model over the 4-base alphabet."""

    def __init__(self, embed: int = 64, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(4, embed)
        self.gru = nn.GRU(embed, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 4)

    def forward(self, x, h=None):
        out, h = self.gru(self.emb(x), h)
        return self.head(out), h


def make_batch(seqs, batch, seqlen, rng):
    xb = np.empty((batch, seqlen + 1), dtype=np.int64)
    for b in range(batch):
        s = seqs[rng.integers(len(seqs))]
        start = int(rng.integers(0, s.size - seqlen - 1))
        xb[b] = s[start:start + seqlen + 1]
    t = torch.from_numpy(xb)
    return t[:, :-1], t[:, 1:]


def train(seqs, use_aux, ref_dinuc, args, rng, label):
    torch.manual_seed(args.seed)
    model = CharGRU(args.embed, args.hidden, args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(seqs, args.batch, args.seqlen, rng)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1))
        loss = ce
        if use_aux:
            loss = (ce
                    + args.lambda_homo * homopolymer_loss(logits, x)
                    + args.lambda_dinuc * dinucleotide_kl_loss(logits, x, ref_dinuc))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 4) == 0 or step == 1:
            print(f"[{label}] step {step}/{args.steps} ce={ce.item():.4f} "
                  f"loss={loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def generate(model, seeds, constrained, ref_kmer, args, rng):
    """Autoregressive sampling, optionally through the constraint stack."""
    model.eval()
    vocab = NucleotideVocab.identity()
    stack = None
    if constrained:
        stack = ConstraintStack([
            HomopolymerPenalty(vocab, strength=args.rep_penalty),
            KmerOverrepresentationPenalty(vocab, ref_kmer, k=args.constraint_k,
                                          strength=args.kmer_penalty),
        ])
        stack.reset(len(seeds))

    seed_arr = np.stack([s[:args.seed_len] for s in seeds]).astype(np.int64)
    x = torch.from_numpy(seed_arr)
    _, h = model(x)
    last = x[:, -1:]
    if stack is not None:
        for pos in range(seed_arr.shape[1]):
            stack.record(seed_arr[:, pos])

    generated = []
    for _ in range(args.gen_len):
        logits, h = model(last, h)
        lg = (logits[:, -1, :] / args.temperature).numpy()
        if stack is not None:
            lg = stack.process(lg)
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        nxt = np.array([rng.choice(4, p=p[b]) for b in range(p.shape[0])])
        if stack is not None:
            stack.record(nxt)
        generated.append(nxt)
        last = torch.from_numpy(nxt[:, None].astype(np.int64))
    arr = np.stack(generated, axis=1)
    return [arr[b] for b in range(arr.shape[0])]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def detect_auroc(nat_chunks, gen_chunks, k, seed):
    """AUROC of a k-mer logistic regression separating generated from natural."""
    if len(nat_chunks) < 5 or len(gen_chunks) < 5:
        return float("nan")
    x = np.vstack([np.vstack([S.kmer_freqs(c, k) for c in nat_chunks]),
                   np.vstack([S.kmer_freqs(c, k) for c in gen_chunks])])
    y = np.concatenate([np.zeros(len(nat_chunks)), np.ones(len(gen_chunks))])
    n_splits = int(min(5, len(nat_chunks), len(gen_chunks)))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for tr, te in skf.split(x, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x[tr], y[tr])
        scores[te] = clf.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, scores))


def evaluate(gen_seqs, nat_seqs, args):
    gen_str = [to_string(g) for g in gen_seqs]
    nat_str = [to_string(n[:args.gen_len]) for n in nat_seqs]
    gen_all, nat_all = "".join(gen_str), "".join(nat_str)

    rng = np.random.default_rng(args.seed)
    nat_chunks, gen_chunks = [], []
    for s in nat_str:
        nat_chunks += S.chunk_sequence(s, args.eval_chunk, 0, rng)
    for s in gen_str:
        gen_chunks += S.chunk_sequence(s, args.eval_chunk, 0, rng)

    return {
        "kmer_jsd": S.js_divergence(S.kmer_freqs(nat_all, args.eval_k),
                                    S.kmer_freqs(gen_all, args.eval_k)),
        "homopolymer_frac": abs(S.homopolymer_fraction(gen_all)
                                - S.homopolymer_fraction(nat_all)),
        "low_complexity": abs(S.low_complexity(gen_all) - S.low_complexity(nat_all)),
        "cpg_oe_error": abs(S.cpg_observed_expected(gen_all)
                            - S.cpg_observed_expected(nat_all)),
        "detect_auroc": detect_auroc(nat_chunks, gen_chunks, args.eval_k_clf,
                                     args.seed),
        "raw_homopolymer_frac": S.homopolymer_fraction(gen_all),
        "raw_low_complexity": S.low_complexity(gen_all),
        "raw_cpg_oe": S.cpg_observed_expected(gen_all),
    }


def make_plot(df, out_png, natural_row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # noqa: F401

    metrics = ["kmer_jsd", "homopolymer_frac", "low_complexity",
               "cpg_oe_error", "detect_auroc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.8))
    colors = ["#2c3e50", "#2980b9", "#e67e22", "#c0392b"]
    for ax, metric in zip(axes, metrics):
        vals = [float(df.loc[df["arm"] == a, metric].iloc[0]) for a in ARMS]
        ax.bar(range(len(ARMS)), vals, color=colors)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=40, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if metric == "detect_auroc":
            ax.axhline(0.5, color="0.35", ls="--", lw=1.2)
            ax.set_ylim(0.4, 1.02)
    fig.suptitle("Interventions vs baseline (lower is more natural); "
                 f"natural reference: {natural_row}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fasta", nargs="+", required=True,
                    help="FASTA files, a directory, or a quoted glob.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-eval", type=int, default=8,
                    help="Sequences held out as the natural reference.")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lambda-homo", type=float, default=0.5,
                    help="Weight of the homopolymer auxiliary loss.")
    ap.add_argument("--lambda-dinuc", type=float, default=1.0,
                    help="Weight of the dinucleotide KL auxiliary loss.")
    ap.add_argument("--seed-len", type=int, default=256,
                    help="Natural prompt length before generation starts.")
    ap.add_argument("--gen-len", type=int, default=5000,
                    help="Bases generated per evaluation sequence.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--constraint-k", type=int, default=4)
    ap.add_argument("--rep-penalty", type=float, default=2.5)
    ap.add_argument("--kmer-penalty", type=float, default=1.5)
    ap.add_argument("--eval-k", type=int, default=6,
                    help="k for the k-mer JSD.")
    ap.add_argument("--eval-k-clf", type=int, default=4,
                    help="k for the detectability classifier.")
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def main() -> int:
    import pandas as pd

    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    min_len = max(args.seqlen + 2, args.seed_len + 1)
    seqs = load_sequences(args.fasta, min_len)
    print(f"[info] loaded {len(seqs)} sequences "
          f"({sum(s.size for s in seqs) / 1e6:.2f} Mb)")
    if len(seqs) < args.n_eval + 2:
        raise SystemExit(
            f"need at least {args.n_eval + 2} sequences of >= {min_len} bp; "
            f"got {len(seqs)}. Lower --n-eval or --seqlen."
        )

    order = rng.permutation(len(seqs))
    eval_seqs = [seqs[i] for i in order[:args.n_eval]]
    train_seqs = [seqs[i] for i in order[args.n_eval:]]
    print(f"[info] {len(train_seqs)} training / {len(eval_seqs)} held-out "
          f"reference sequences")

    ref_text = "".join(to_string(s[:args.gen_len]) for s in eval_seqs)
    ref_dinuc = reference_dinucleotide(ref_text)
    ref_kmer = reference_kmer_freqs(ref_text, k=args.constraint_k)

    models = {}
    for use_aux, name in ((False, "standard"), (True, "aux_loss")):
        print(f"\n[info] training model: {name}")
        models[name] = train(train_seqs, use_aux, ref_dinuc, args, rng, name)

    rows = []
    for arm in ARMS:
        model = models["aux_loss" if arm.startswith("aux") else "standard"]
        constrained = "constr" in arm
        print(f"\n[info] generating: {arm} "
              f"({'constrained' if constrained else 'plain'} decoding)")
        gen = generate(model, eval_seqs, constrained, ref_kmer, args, rng)
        fa = outdir / f"{arm}.generated.fa"
        with open(fa, "w") as fh:
            for i, g in enumerate(gen):
                seq = to_string(g)
                fh.write(f">{arm}.seq{i:03d} len={len(seq)}\n")
                for j in range(0, len(seq), 80):
                    fh.write(seq[j:j + 80] + "\n")
        metrics = evaluate(gen, eval_seqs, args)
        rows.append({"arm": arm, "training": "aux_loss" if arm.startswith("aux")
                     else "standard",
                     "decoding": "constrained" if constrained else "plain",
                     **metrics})
        print(f"[ok] {arm}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if not k.startswith("raw_")
        ))

    df = pd.DataFrame(rows)
    csv_path = outdir / "interventions.metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[ok] wrote {csv_path}")

    nat_all = "".join(to_string(s[:args.gen_len]) for s in eval_seqs)
    natural_row = (f"homopolymer={S.homopolymer_fraction(nat_all):.4f} "
                   f"CpG O/E={S.cpg_observed_expected(nat_all):.3f}")

    print(f"\n=== interventions (lower is better; natural: {natural_row}) ===")
    header = ["arm", "kmer_jsd", "homopolymer_frac", "low_complexity",
              "cpg_oe_error", "detect_auroc"]
    print("  " + "".join(f"{h:>18s}" if h != "arm" else f"{h:<15s}"
                         for h in header))
    for _, r in df.iterrows():
        print(f"  {r['arm']:<15s}" + "".join(
            f"{float(r[h]):>18.4f}" for h in header[1:]
        ))
    best = df.loc[df["detect_auroc"].idxmin()]
    print(f"\n  hardest arm to detect: {best['arm']} "
          f"(AUROC {best['detect_auroc']:.4f} vs baseline "
          f"{float(df.loc[df['arm'] == 'baseline', 'detect_auroc'].iloc[0]):.4f})")

    if not args.no_plot:
        png = outdir / "interventions.png"
        make_plot(df, png, natural_row)
        print(f"[ok] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
