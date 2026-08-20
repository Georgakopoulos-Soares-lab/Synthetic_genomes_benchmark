# Benchmarking generated genomes

A practical guide to evaluating sequence produced by a genomic language model —
or any generative DNA model — against matched natural sequence.

- [What this suite measures](#what-this-suite-measures)
- [Quick start](#quick-start-no-external-tools)
- [Preparing your data](#preparing-your-data)
- [Choosing windows](#choosing-windows-read-this-before-you-benchmark)
- [The benchmarks](#the-benchmarks)
- [Reading the results](#reading-the-results)
- [Trying to improve generation](#trying-to-improve-generation)
- [Full worked example](#full-worked-example)
- [Troubleshooting](#troubleshooting)

---

## What this suite measures

Generative DNA models are usually reported with perplexity or a downstream
task score. Neither tells you whether the sequence they emit *looks like a
genome*. This suite asks that question directly, from several angles, on
matched pairs: a natural window and the synthetic sequence generated for the
same locus.

The benchmarks fall into three groups.

**Composition and divergence** — how far is synthetic from natural, in the
statistics genomes are described by? `composition.py` (GC, CpG, homopolymers,
low complexity, entropy, nullomers, k-mer JSD, FCGR L1), `kmer_spectra.py`,
`fcgr.py`, `nullomers.py`.

**Calibration** — is that distance actually large? Real genomic windows already
differ from each other, so a raw divergence is uninterpretable on its own.
`natural_baseline.py` compares synthetic-vs-natural against
natural-vs-natural and reports the ratio.

**Detectability and drift** — can a model tell them apart, and does the gap
grow with distance from the prompt? `detectability.py` (shallow, CPU),
`scripts/classifier/` (deep), `context_decay.py`.

Plus the biology-specific benchmarks that need external tools: transcription
factor binding sites (`tfbs_fimo.py`, FIMO) and non-B DNA structure
(`nonbdna_*.py`, ZSeeker / G4Hunter / non-B GFA).

---

## Quick start (no external tools)

Four of the benchmarks are pure Python — numpy, pandas, scipy, scikit-learn,
matplotlib — and need no bioinformatics tooling. On the example data shipped
with the repository:

```bash
# 1. Build a manifest pairing natural and synthetic windows
python scripts/make_manifest.py --tag HumanExample \
    --orig-dir data/Homo_sapiens \
    --out manifests/pairs.HumanExample.csv --relative-to .

# 2. Run the dependency-free benchmarks
python scripts/run_benchmarks.py --tag HumanExample \
    --manifest manifests/pairs.HumanExample.csv \
    --composition --detectability --natural-baseline --null-check
```

Results land under `results/HumanExample/<benchmark>/`. This takes a couple of
minutes for 40 windows of 300 kb.

Each benchmark also runs standalone with its own options:

```bash
python scripts/benchmarks/composition.py --help
```

---

## Preparing your data

### The manifest

Every benchmark takes a **pairs manifest**: a CSV with at least

| column | meaning |
| --- | --- |
| `id` | unique name for the pair |
| `orig` | path to the natural window's FASTA |
| `syn` | path to the synthetic sequence generated for that locus |

Extra columns are carried along and ignored, except `seed_len`, which
`context_decay.py` reads as the length of the natural prompt.

```csv
id,orig,syn,seed_len
chr1_165826768,data/Homo_sapiens/orig.chr1.165826768.300000.fa,data/Homo_sapiens/syn_chr1:165826769-166126768.fasta,3000
```

Relative paths resolve against the working directory (run from the repository
root), or against `--data-root` if you pass it.

### Building one

`scripts/make_manifest.py` covers the three common situations:

```bash
# Files named by locus, e.g. orig.chr1.165826768.300000.fa / syn_chr1:165826769-166126768.fasta
python scripts/make_manifest.py --tag MyRun --orig-dir data/MyRun --out manifests/pairs.MyRun.csv

# Two directories that correspond positionally
python scripts/make_manifest.py --tag MyRun --mode sorted \
    --orig-dir natural/ --syn-dir generated/ --out manifests/pairs.MyRun.csv

# Two multi-record FASTAs, paired by record order
python scripts/make_manifest.py --tag MyRun --mode records \
    --orig-fasta natural.fa --syn-fasta generated.fa \
    --split-dir data/MyRun --out manifests/pairs.MyRun.csv
```

If your filenames encode the locus differently, pass your own patterns —
capture groups named `contig` and `start` (or `start1` for 1-based
coordinates) are used to build the matching key:

```bash
python scripts/make_manifest.py --tag MyRun --orig-dir data/ \
    --orig-pattern '^natural_(?P<contig>\w+)_(?P<start>\d+)\.fa$' \
    --syn-pattern  '^gen_(?P<contig>\w+)_(?P<start>\d+)\.fa$' \
    --out manifests/pairs.MyRun.csv
```

The script validates every pair before writing — missing files, empty
sequences, and windows whose lengths differ by more than `--max-length-ratio`
are reported. A flood of length warnings almost always means the pairing rule
is wrong, not that your generator is unusual.

### Unpaired data

If you have a pile of natural genomes and a pile of synthetic ones with no
locus correspondence, use the population benchmarks instead:

```bash
python scripts/megadna_population_based/run_population_benchmarks.py \
    --natural-fasta natural.fasta --synthetic-fasta synthetic.fasta \
    --tag my_dataset --outdir results/population --balance-within-bin
```

`detectability.py` also works without true pairing — it only uses the pairing
to group cross-validation folds — but `natural_baseline.py`, `composition.py`
and `context_decay.py` all assume matched loci.

---

## Choosing windows (read this before you benchmark)

The single most common way to get a misleading result is to compare windows
that differ from each other for reasons unrelated to the generator.

**Keep the natural windows homogeneous.** They are the reference distribution.
If they mix chromosomes with very different GC content, or mix 10 kb and 1 Mb
windows, the natural-vs-natural spread balloons and everything looks "within
natural variation". Both `natural_baseline.py` and `detectability.py` accept
`--null-check`, which pits the natural windows against each other and tells you
whether this is happening. Use it.

**Use enough windows.** The permutation tests enumerate all `2^n` matched-label
swaps when `n <= 15`, so the smallest achievable p-value is `1/2^n`: with 10
pairs you cannot go below 0.001, with 5 pairs not below 0.03. Twenty or more
pairs is comfortable. The `--null-check` control needs at least ten pairs
before its output means anything.

**Make the windows long enough.** `detectability.py` and `context_decay.py`
cut sequences into 1024 bp chunks; a 5 kb window yields four chunks. If you
lower `--chunk`, k-mer frequency estimates get noisier — at `--chunk 256` a
6-mer vector has 4096 dimensions and 251 observations.

**Mind soft-masking.** Lower-case bases are read as ordinary bases. Chunks
containing non-ACGT characters are dropped entirely, so a heavily N-masked
window contributes little. `composition.py` reports `orig_len`/`syn_len` so you
can see what was actually read.

---

## The benchmarks

### composition.py — start here

Per-window GC, CpG observed/expected, homopolymer fraction, low-complexity
score, 3-mer entropy and canonical nullomer fraction, for natural and synthetic
separately; plus each synthetic window's k-mer JSD and FCGR L1 against **its
own** natural counterpart. Paired Wilcoxon and exact sign-flip permutation
tests, BH-corrected.

```bash
python scripts/benchmarks/composition.py \
    --manifest manifests/pairs.MyRun.csv --outdir results/MyRun/composition
```

Useful options: `--kmer-k` (JSD order, default 6), `--fcgr-k` (default 6),
`--nullomer-k` (default 11), `--min-run` (homopolymer threshold, default 5).

Outputs `composition.per_pair.csv` (every metric per window),
`composition.summary.csv` (tests and effect sizes), `composition.png`.

### natural_baseline.py — is the divergence real?

Reduces each window to a feature, forms all pairwise distances, and compares
`median(syn-nat)` against `median(nat-nat)`. Significance comes from permuting
the natural/synthetic label *within each matched pair*, which is the only thing
exchangeable under the null — a rank test over pairwise distances treats
correlated values as independent and is anticonservative.

```bash
python scripts/benchmarks/natural_baseline.py \
    --manifest manifests/pairs.MyRun.csv --outdir results/MyRun/natural_baseline \
    --metrics fcgr kmer gc homopolymer low_complexity cpg_oe --null-check
```

Metrics: `fcgr`, `kmer`, `gc`, `homopolymer`, `low_complexity`, `cpg_oe`,
`nullomer_fraction`. You can add your own per-window scalars — non-B DNA
coverage, TFBS counts, anything — with `--scalar-csv` (columns
`id,metric,orig,syn`), and they go through the same test.

The headline number is `ratio`:

| ratio | reading |
| --- | --- |
| ~1.0 | synthetic sits inside natural window-to-window variation |
| 2 | twice as far from natural as natural windows are from each other |
| 10+ | a qualitative failure, not a subtle shift |

`beyond_natural_variation` is `True` when `q < 0.05` and the difference is
positive.

### detectability.py — how separable is it?

A logistic regression on k-mer frequencies, Markov-1 transitions, or GC alone,
cross-validated with folds grouped by pair. AUROC is reported as a function of
how many 1024 bp chunks are averaged per sequence.

```bash
python scripts/benchmarks/detectability.py \
    --manifest manifests/pairs.MyRun.csv --outdir results/MyRun/detectability \
    --features kmer4 markov1 gc --null-check
```

Pass several manifests to switch to leave-one-manifest-out cross-validation,
which tests whether the signal generalises across species rather than being
specific to one genome:

```bash
python scripts/benchmarks/detectability.py \
    --manifest manifests/pairs.Human.csv manifests/pairs.Mouse.csv manifests/pairs.Zebrafish.csv \
    --outdir results/multi/detectability
```

The `gc` feature set is a deliberate floor. If `gc` alone reaches AUROC 0.9,
the model has a GC problem and nothing more subtle needs explaining. If
`kmer4` is high but `gc` is at chance, the failure is in local composition.
And if a linear model on 4-mers already matches your deep detector, the
separability is elementary compositional drift — not evidence that the model
fails at long-range structure.

### context_decay.py — does it drift from the prompt?

For sequences generated by conditioning on a natural seed, this measures AUROC
within each distance bin past the end of that seed, using leave-one-window-out
cross-validation.

```bash
python scripts/benchmarks/context_decay.py \
    --manifest manifests/pairs.MyRun.csv --outdir results/MyRun/context_decay \
    --seed-len 3000 --bin-size 20000 --compare-regions
```

The seed length comes from `--seed-len` or a `seed_len` column in the manifest.
If your synthetic sequences were not generated from a natural prompt, this
benchmark does not apply.

Three shapes to recognise:

- **rising from ~0.5** — the model tracks its context and loses it with
  distance. `--compare-regions` puts a p-value on that increase.
- **flat and high** — the output never resembled the reference; conditioning
  is not the limiting factor.
- **flat and low** — no measurable decay over this window length. Try longer
  windows before concluding there is none.

To ask whether a longer prompt helps, generate at several seed lengths and run
the benchmark once per set, then compare the curves.

### The external-tool benchmarks

These need software installed (see [INSTALL.md](INSTALL.md)) and are wired into
the same entry point:

| Flag | Measures | Needs |
| --- | --- | --- |
| `--spectra` | k-mer spectra, Chor-normalised, with significance | — |
| `--fcgr` | Frequency Chaos Game Representation distance | — |
| `--nullomers` | absent k-mers via KMC set operations | KMC |
| `--tfbs` | transcription factor binding site abundance | MEME/FIMO |
| `--nonbdna` | Z-DNA, G-quadruplexes, inverted/direct/mirror repeats, STRs | ZSeeker, G4Hunter, non-B GFA |

A note on `--nullomers`: KMC counts **canonical** k-mers by default, collapsing
each k-mer with its reverse complement, so the number of observable classes is
`(4^k + P(k))/2`, not `4^k`. Dividing a canonical count by `4^k` floors any
"fraction absent" near 0.5 for odd k regardless of the sequence. The benchmark
now picks the denominator that matches the counting mode and refuses to emit a
fraction if the two disagree; use `--nullomers-count-mode both-strands` if you
want KMC's `-b` behaviour instead.

### The deep detector

`scripts/classifier/` holds the dilated 1D ResNet used in the paper, with
leave-one-tag-out cross-validation and calibration-based checkpoint selection.
It needs a GPU and a training corpus. Run `detectability.py` first: if a linear
model on 4-mers already saturates, a CNN will not tell you anything new.

---

## Reading the results

The benchmarks answer different questions and are meant to be read together.

| Question | Benchmark | Look at |
| --- | --- | --- |
| What is different? | `composition.py` | `mean_difference`, `cohens_dz`, `q_permutation` |
| Is it more different than natural windows are? | `natural_baseline.py` | `ratio`, `beyond_natural_variation` |
| Can a model tell them apart? | `detectability.py` | `auroc_mean` at the largest `n_eval_chunks` |
| Is it composition or structure? | `detectability.py` | `gc` vs `kmer4` vs `markov1` |
| Does it get worse further from the prompt? | `context_decay.py` | `context_decay.by_bin.csv`, `regions.csv` |

Two combinations worth naming:

**Significant but ratio ≈ 1.** `composition.py` reports `q < 0.05` while
`natural_baseline.py` reports a ratio near 1. The difference is real and
consistent but no larger than the variation between real windows. Report it as
a small shift, not a failure.

**Low divergence but high AUROC.** Every summary statistic looks natural and a
classifier still separates the two at AUROC 0.95. The model matches the
marginals and misses their joint structure — usually the more interesting
finding, and the one that summary statistics alone would have hidden.

Always run `--null-check` at least once on a new dataset. It costs one extra
run and distinguishes "the generator is detectable" from "my windows are
heterogeneous".

---

## Trying to improve generation

`generation/improve/` holds the two interventions we tested, in reusable form.
They target failures the benchmarks detect, and both are model-agnostic.

### Constraints at sampling time

`constrained_decoding.py` biases logits at each step without touching the
weights:

```python
from constrained_decoding import (
    NucleotideVocab, ConstraintStack, HomopolymerPenalty,
    KmerOverrepresentationPenalty, reference_kmer_freqs,
)

vocab = NucleotideVocab.from_tokenizer(model.tokenizer)
stack = ConstraintStack([
    HomopolymerPenalty(vocab, strength=2.5),
    KmerOverrepresentationPenalty(vocab, reference_kmer_freqs(natural_seqs, k=4), k=4),
])
stack.reset(batch_size)

for step in range(n):
    logits = stack.process(model_logits)
    tokens = sample(logits)          # your sampler, unchanged
    stack.record_tokens(tokens)
```

For Evo 2 there is a context manager that patches its sampler in place, keeping
the fast cached-generation loop:

```python
from constrained_decoding import evo2_constrained
with evo2_constrained(stack):
    output = model.generate(...)
```

### Auxiliary losses at training time

`structural_losses.py` provides two differentiable terms that drop into any
autoregressive training loop:

```python
from structural_losses import homopolymer_loss, dinucleotide_kl_loss, reference_dinucleotide

ref = reference_dinucleotide(natural_sequences)
loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
loss = loss + 0.5 * homopolymer_loss(logits, inputs)
loss = loss + 1.0 * dinucleotide_kl_loss(logits, inputs, ref)
```

Watch validation perplexity. If it degrades noticeably the weights are too
high, and you are trading sequence modelling for a cosmetic improvement in one
statistic.

### Measuring whether it worked

`run_intervention_demo.py` trains a small character-level model on your FASTA
and scores a 2×2 design — standard vs auxiliary-loss training, plain vs
constrained decoding — with the same metrics the benchmarks use:

```bash
python generation/improve/run_intervention_demo.py \
    --fasta data/Homo_sapiens --outdir results/interventions --steps 400
```

Takes a few minutes on CPU. The model is a testbed for the interventions, not a
genomic language model: only the differences between arms mean anything.

**What we found, and what to watch for.** Both interventions reliably move the
statistic they target. Neither reliably moves `detect_auroc`. In our own runs,
constrained decoding lowered the low-complexity score while making the output
*easier* for a classifier to spot, because pushing a model off its own
distribution to satisfy a marginal constraint introduces a new signature. So
judge an intervention by detectability, not by the metric it was designed to
fix. Both modules ship with `--self-test` if you want to confirm the mechanics
before trusting a result:

```bash
python generation/improve/constrained_decoding.py --self-test
python generation/improve/structural_losses.py --self-test
```

---

## Full worked example

Evaluating a new set of generated sequences end to end.

```bash
# 0. Environment
conda env create -f environment.yml && conda activate syn_bench

# 1. Manifest
python scripts/make_manifest.py --tag MyModel \
    --orig-dir data/natural --syn-dir data/generated \
    --seed-len 3000 --out manifests/pairs.MyModel.csv --relative-to .

# 2. Cheap benchmarks first — these catch most problems in minutes
python scripts/run_benchmarks.py --tag MyModel \
    --manifest manifests/pairs.MyModel.csv \
    --composition --detectability --natural-baseline --null-check

# 3. Confirm the null check passed, then read the headline numbers
column -s, -t results/MyModel/natural_baseline/natural_baseline.per_metric.csv
column -s, -t results/MyModel/detectability/detectability.summary.csv

# 4. Drift from the prompt, if the sequences were seeded
python scripts/run_benchmarks.py --tag MyModel \
    --manifest manifests/pairs.MyModel.csv \
    --context-decay --seed-len 3000 --bin-size 20000

# 5. The biology-specific benchmarks, once the tools are installed
python scripts/run_benchmarks.py --tag MyModel \
    --manifest manifests/pairs.MyModel.csv \
    --spectra --fcgr --nullomers --tfbs --nonbdna

# 6. Everything at once (context decay excluded: it needs a seed length)
python scripts/run_benchmarks.py --tag MyModel \
    --manifest manifests/pairs.MyModel.csv --all --null-check
```

---

## Troubleshooting

**"Manifest must include columns ['id', 'orig', 'syn']"** — the CSV header is
wrong or the file is not comma-separated. `make_manifest.py` writes a correct
one.

**"no usable pairs" / "dropping N pairs with missing FASTA files"** — the paths
do not resolve. They are relative to the working directory; run from the
repository root or pass `--data-root`.

**"no usable chunks; check --chunk against window length"** — windows are
shorter than one chunk, or every chunk contains a non-ACGT base. Lower
`--chunk`, or check for N-masking.

**"need at least 4 matched pairs" / "leave-one-window-out needs at least 3
windows"** — too few pairs. These tests have no power at that size.

**The null check fails.** Natural windows are separable from each other. Almost
always heterogeneous windows: different chromosomes, very different lengths, or
a mix of GC regimes. Restrict to a more homogeneous set and rerun. Do not
report the main result until this passes.

**The null check says "underpowered".** Fewer than about ten pairs. The control
is uninformative at that size; it is not a failure.

**p-values bottom out at the same value.** Exhaustive enumeration at `n <= 15`
pairs gives a minimum p of `1/2^n`. Use more windows.

**"distinct k-mers exceed the canonical k-mer space"** — KMC counted both
strands but the canonical denominator was assumed, or vice versa. Match
`--nullomers-count-mode` to how KMC was run.

**`ModuleNotFoundError: No module named '_seqio'`** — the new benchmarks import
a shared helper from their own directory. Invoke them by path
(`python scripts/benchmarks/composition.py`), not by copying the file
elsewhere.

**Slow at large k.** `--nullomer-k 11` builds a 4-million-element table per
process (cached after the first window). `--fcgr-k 8` gives 65,536-dimensional
vectors; `natural_baseline.py` forms all pairwise distances over them, which is
`O(n^2)` in the number of windows. Drop to `--fcgr-k 6` while iterating.
