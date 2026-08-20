#!/usr/bin/env python3
"""
Reviewer #1, major comment #1: move from diagnosis to a constructive baseline.

We train a scaled-down ("toy") character-level genomic language model on a
compact bacteriophage dataset and test whether two of the reviewer's suggested
optimization strategies mitigate the context-collapse failure mode documented in
the paper, evaluated with the paper's own metric families (k-mer spectra, a
low-complexity/homopolymer proxy for non-B/structural degradation, and
shallow-classifier distinguishability).

Configurations compared
------------------------
  baseline      : standard cross-entropy training, plain temperature sampling
  aux_loss      : + explicit structural/k-mer auxiliary loss during training
                  (Reviewer option 1: penalise homopolymer continuation and
                   reward adherence to the natural dinucleotide distribution)
  constr_decode : baseline weights, but inference-time guided/constrained
                  decoding (Reviewer option 2: homopolymer-run penalty +
                  k-mer over-representation look-ahead penalty)
  aux+constr    : aux_loss weights with constrained decoding

Metrics (generated vs held-out natural)
  - k-mer Jensen-Shannon divergence (k=4)
  - homopolymer fraction (runs >= 5 bp) and low-complexity score
  - shallow classifier AUROC (LogReg on k=4 frequency vectors): lower = the
    generated set is harder to tell apart from natural (more realistic)

CPU only. Small model + limited steps so it runs on a single core in minutes.
Run with system python3 from /tmp.
"""

import os as _os

# Root of the analysis tree these revision scripts were run against on TACC
# Lonestar6. Set NONBDNA_ROOT to point them at a local copy.
_ROOT = _os.environ.get("NONBDNA_ROOT", "/work/11034/atzanakak/ls6/nonbdna")

import sys, time, math, glob, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(max(1, torch.get_num_threads()))
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PHAGE_DIR = Path(f"{_ROOT}/megadna/data/phage_fastas")
OUTDIR = Path(f"{_ROOT}/revisions/results")
FIGDIR = Path(f"{_ROOT}/revisions/figures")
OUTDIR.mkdir(parents=True, exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- hyperparameters (kept small for single-core CPU) ----
N_EVAL = 10          # held-out genomes used as natural reference
SEQLEN = 256         # training crop length
BATCH = 64
STEPS = 600
LR = 2e-3
EMBED = 64
HIDDEN = 256
LAYERS = 1
GEN_LEN = 6000       # length of each generated sequence
N_GEN = 6            # generated sequences per config
SEED_LEN = 200       # natural prefix used to condition generation
KMER_K = 4
EVAL_CHUNK = 2048    # chunk size for classifier / k-mer eval
LAM_HOMO = 0.5       # aux-loss weight: anti-homopolymer
LAM_DINUC = 0.5      # aux-loss weight: dinucleotide KL
REP_PEN = 2.5        # constrained decoding: homopolymer logit penalty
KMER_PEN = 1.5       # constrained decoding: k-mer over-representation penalty
OVERREP = 1.5        # trigger penalty when running freq > OVERREP x natural

_B2I = {65: 0, 67: 1, 71: 2, 84: 3}
I2B = np.array([65, 67, 71, 84], dtype=np.uint8)


def load_genomes(directory: Path):
    seqs = []
    for fp in sorted(glob.glob(str(directory / "*.fna"))):
        buf = []
        with open(fp) as fh:
            for line in fh:
                if line.startswith(">"):
                    continue
                buf.append(line.strip().upper())
        s = "".join(buf)
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        idx = np.full(a.shape, -1, np.int8)
        for code, i in _B2I.items():
            idx[a == code] = i
        idx = idx[idx >= 0].astype(np.int64)   # drop non-ACGT
        if idx.size > SEQLEN + 1:
            seqs.append(idx)
    return seqs


# ---------------- k-mer / structural utilities ----------------

def kmer_freq(idx: np.ndarray, k: int) -> np.ndarray:
    if idx.size < k:
        return np.ones(4 ** k) / (4 ** k)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(idx, k)
    codes = (win * powers).sum(axis=1)
    c = np.bincount(codes, minlength=4 ** k).astype(float)
    return c / c.sum()


def js_div(p, q):
    p = p + 1e-12; q = q + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def homopolymer_fraction(idx: np.ndarray, min_run: int = 5) -> float:
    if idx.size == 0:
        return 0.0
    runs = 0
    cur = 1
    covered = 0
    for i in range(1, idx.size):
        if idx[i] == idx[i - 1]:
            cur += 1
        else:
            if cur >= min_run:
                covered += cur
            cur = 1
    if cur >= min_run:
        covered += cur
    return covered / idx.size


def lowcomplexity(idx: np.ndarray, k: int = 3) -> float:
    f = kmer_freq(idx, k)
    f = f[f > 0]
    ent = -(f * np.log2(f)).sum()
    return float(1.0 - ent / (k * 2.0))   # normalised 1 - H/Hmax


# ---------------- model ----------------

class CharGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, EMBED)
        self.gru = nn.GRU(EMBED, HIDDEN, LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN, 4)

    def forward(self, x, h=None):
        e = self.emb(x)
        out, h = self.gru(e, h)
        return self.head(out), h


def make_batch(train_seqs):
    xb = np.empty((BATCH, SEQLEN + 1), dtype=np.int64)
    for b in range(BATCH):
        s = train_seqs[random.randrange(len(train_seqs))]
        start = random.randrange(0, s.size - SEQLEN - 1)
        xb[b] = s[start:start + SEQLEN + 1]
    t = torch.from_numpy(xb)
    return t[:, :-1], t[:, 1:]


def train_model(train_seqs, nat_dinuc, use_aux: bool, label: str):
    model = CharGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    nat_dinuc_t = torch.tensor(nat_dinuc, dtype=torch.float32)  # (4,4) rows=cur,cols=next
    t0 = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        x, y = make_batch(train_seqs)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1))
        loss = ce
        if use_aux:
            p = F.softmax(logits, dim=-1)           # (B,T,4) predicted next dist
            cur = x                                 # current base index (B,T)
            # anti-homopolymer: probability assigned to repeating current base
            p_same = p.gather(-1, cur.unsqueeze(-1)).squeeze(-1)  # (B,T)
            aux_homo = p_same.mean()
            # dinucleotide adherence: expected dinuc dist vs natural
            onehot_cur = F.one_hot(cur, 4).float()              # (B,T,4)
            # expected counts dinuc[a,b] = sum [cur==a] * p(next=b)
            dinuc = torch.einsum("bti,btj->ij", onehot_cur, p)
            dinuc = dinuc / dinuc.sum().clamp_min(1e-9)
            q = dinuc.clamp_min(1e-9)
            pnat = nat_dinuc_t.clamp_min(1e-9)
            aux_dinuc = (pnat * (pnat.log() - q.log())).sum()   # KL(nat || model)
            loss = ce + LAM_HOMO * aux_homo + LAM_DINUC * aux_dinuc
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0 or step == 1:
            print(f"[{label}] step {step}/{STEPS} ce={ce.item():.3f} "
                  f"loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return model


# ---------------- generation ----------------

@torch.no_grad()
def generate(model, seeds, constrained: bool, nat_kmer: np.ndarray, temp=1.0):
    """Batched autoregressive generation. seeds: list of np arrays (>= SEED_LEN)."""
    model.eval()
    B = len(seeds)
    powers = (4 ** np.arange(KMER_K - 1, -1, -1)).astype(np.int64)
    # prime hidden state on the seed
    seed_arr = np.stack([s[:SEED_LEN] for s in seeds]).astype(np.int64)
    x = torch.from_numpy(seed_arr)
    logits, h = model(x)
    last = x[:, -1:]
    gen = [seed_arr[:, i] for i in range(SEED_LEN)]
    # per-sequence running k-mer counts (numpy) for the constraint
    run_counts = np.zeros((B, 4 ** KMER_K), dtype=np.float64)
    run_total = np.zeros(B)
    hist = seed_arr.copy()  # keep recent context for k-mer / run logic
    nat = nat_kmer + 1e-9
    for t in range(GEN_LEN):
        logits, h = model(last, h)
        lg = logits[:, -1, :] / temp           # (B,4)
        if constrained:
            lg = lg.clone()
            prev = hist[:, -1]                  # current last base per seq
            # homopolymer penalty: discourage extending a run
            run_len = np.ones(B)
            for b in range(B):
                rl = 1
                j = hist.shape[1] - 2
                while j >= 0 and hist[b, j] == hist[b, -1]:
                    rl += 1; j -= 1
                run_len[b] = rl
            for b in range(B):
                lg[b, prev[b]] -= REP_PEN * min(run_len[b], 6) / 6.0
            # k-mer over-representation look-ahead
            if hist.shape[1] >= KMER_K - 1:
                ctx = hist[:, -(KMER_K - 1):]   # (B,k-1)
                base_code = (ctx * powers[1:]).sum(axis=1)  # (B,)
                for cand in range(4):
                    code = base_code + cand * powers[0]
                    cur_freq = (run_counts[np.arange(B), code] + 1e-9) / \
                               (run_total + 1e-9)
                    ratio = cur_freq / nat[code]
                    over = np.clip(ratio - OVERREP, 0, None)
                    lg[:, cand] -= torch.tensor(KMER_PEN * np.tanh(over),
                                                dtype=lg.dtype)
        probs = F.softmax(lg, dim=-1).numpy()
        nxt = np.array([np.random.choice(4, p=probs[b]) for b in range(B)])
        # update running k-mer counts
        if hist.shape[1] >= KMER_K - 1:
            ctx = hist[:, -(KMER_K - 1):]
            code = (ctx * powers[1:]).sum(axis=1) + nxt * powers[0]
            run_counts[np.arange(B), code] += 1
            run_total += 1
        gen.append(nxt)
        last = torch.from_numpy(nxt[:, None].astype(np.int64))
        hist = np.concatenate([hist, nxt[:, None]], axis=1)
        if hist.shape[1] > 64:                  # cap context memory
            hist = hist[:, -64:]
    arr = np.stack(gen, axis=1)                  # (B, SEED_LEN+GEN_LEN)
    return [arr[b, SEED_LEN:] for b in range(B)]  # drop the natural seed


# ---------------- evaluation ----------------

def chunk(idx, size):
    return [idx[i:i + size] for i in range(0, idx.size - size + 1, size)]


def classifier_auroc(nat_chunks, gen_chunks, k=KMER_K):
    Xn = np.stack([kmer_freq(c, k) for c in nat_chunks])
    Xg = np.stack([kmer_freq(c, k) for c in gen_chunks])
    X = np.vstack([Xn, Xg])
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xg))]
    if len(np.unique(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 5:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def evaluate(gen_seqs, eval_seqs, nat_kmer):
    gen_concat = np.concatenate(gen_seqs)
    nat_concat = np.concatenate(eval_seqs)
    gen_kmer = kmer_freq(gen_concat, KMER_K)
    jsd = js_div(nat_kmer.copy(), gen_kmer)
    homo = np.mean([homopolymer_fraction(g) for g in gen_seqs])
    lowc = np.mean([lowcomplexity(g) for g in gen_seqs])
    nat_chunks, gen_chunks = [], []
    for s in eval_seqs:
        nat_chunks += chunk(s, EVAL_CHUNK)
    for g in gen_seqs:
        gen_chunks += chunk(g, EVAL_CHUNK)
    auroc = classifier_auroc(nat_chunks, gen_chunks)
    return dict(kmer_jsd=jsd, homopolymer_frac=homo, lowcomplexity=lowc,
                classifier_auroc=auroc)


def main():
    print(f"[info] torch threads={torch.get_num_threads()}", flush=True)
    seqs = load_genomes(PHAGE_DIR)
    print(f"[info] loaded {len(seqs)} phage genomes, "
          f"{sum(s.size for s in seqs)/1e6:.2f} Mb ACGT", flush=True)
    random.shuffle(seqs)
    eval_seqs = seqs[:N_EVAL]
    train_seqs = seqs[N_EVAL:]
    print(f"[info] train={len(train_seqs)} eval={len(eval_seqs)}", flush=True)

    # natural references from EVAL set (unseen by the model)
    nat_concat = np.concatenate(eval_seqs)
    nat_kmer = kmer_freq(nat_concat, KMER_K)
    # natural dinucleotide distribution from TRAIN set (for aux loss)
    tr_concat = np.concatenate(train_seqs)
    d2 = kmer_freq(tr_concat, 2).reshape(4, 4)
    nat_dinuc = d2 / d2.sum()
    nat_homo = np.mean([homopolymer_fraction(s) for s in eval_seqs])
    nat_lowc = np.mean([lowcomplexity(s) for s in eval_seqs])
    print(f"[info] natural reference homopolymer={nat_homo:.4f} "
          f"lowcomplexity={nat_lowc:.4f}", flush=True)

    # seeds for generation: natural prefixes from eval genomes
    seeds = [eval_seqs[i % len(eval_seqs)][:SEED_LEN + GEN_LEN]
             if eval_seqs[i % len(eval_seqs)].size >= SEED_LEN
             else eval_seqs[i % len(eval_seqs)] for i in range(N_GEN)]
    seeds = [s for s in seeds if s.size >= SEED_LEN][:N_GEN]

    # ---- train two models ----
    base_model = train_model(train_seqs, nat_dinuc, use_aux=False, label="baseline")
    aux_model = train_model(train_seqs, nat_dinuc, use_aux=True, label="aux_loss")

    configs = {
        "baseline":      (base_model, False),
        "aux_loss":      (aux_model, False),
        "constr_decode": (base_model, True),
        "aux+constr":    (aux_model, True),
    }
    rows = []
    for name, (model, constrained) in configs.items():
        t0 = time.time()
        gen = generate(model, seeds, constrained, nat_kmer)
        m = evaluate(gen, eval_seqs, nat_kmer)
        m["config"] = name
        rows.append(m)
        print(f"[{name}] jsd={m['kmer_jsd']:.4f} homo={m['homopolymer_frac']:.4f} "
              f"lowc={m['lowcomplexity']:.4f} auroc={m['classifier_auroc']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # natural-natural reference row (split eval in half)
    half = len(eval_seqs) // 2
    nat_a, nat_b = eval_seqs[:half], eval_seqs[half:]
    na_chunks, nb_chunks = [], []
    for s in nat_a:
        na_chunks += chunk(s, EVAL_CHUNK)
    for s in nat_b:
        nb_chunks += chunk(s, EVAL_CHUNK)
    ref = dict(config="natural-natural",
               kmer_jsd=js_div(kmer_freq(np.concatenate(nat_a), KMER_K),
                               kmer_freq(np.concatenate(nat_b), KMER_K)),
               homopolymer_frac=nat_homo, lowcomplexity=nat_lowc,
               classifier_auroc=classifier_auroc(na_chunks, nb_chunks))
    rows.append(ref)

    # ---- write outputs ----
    import csv
    cols = ["config", "kmer_jsd", "homopolymer_frac", "lowcomplexity",
            "classifier_auroc"]
    out_csv = OUTDIR / "toy_glm_optimization.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"[done] wrote {out_csv}", flush=True)

    # ---- figure ----
    order = ["baseline", "aux_loss", "constr_decode", "aux+constr",
             "natural-natural"]
    rows_by = {r["config"]: r for r in rows}
    metrics = [("kmer_jsd", "k-mer JSD (k=4)\n(lower = better)"),
               ("homopolymer_frac", "Homopolymer fraction\n(lower = better)"),
               ("lowcomplexity", "Low-complexity score\n(lower = better)"),
               ("classifier_auroc", "Classifier AUROC\n(lower = harder to detect)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#888888", "#1b9e77", "#7570b3", "#d95f02", "#000000"]
    for ax, (key, title) in zip(axes, metrics):
        vals = [rows_by[c][key] for c in order]
        ax.bar(range(len(order)), vals, color=colors)
        if key == "classifier_auroc":
            ax.axhline(0.5, ls="--", color="red", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12)
        for i, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Toy genomic LM (phage): do optimization strategies mitigate "
                 "context collapse?", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = FIGDIR / "toy_glm_optimization.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
