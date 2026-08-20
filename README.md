# Synthetic Genomes Benchmark

Benchmarks for evaluating generated DNA against matched natural sequence.

Generative genomic models are usually reported with perplexity or a downstream
task score. Neither says whether the sequence they emit *looks like a genome*.
This suite asks that question directly, on matched pairs of windows: a natural
locus and the synthetic sequence generated for it.

**→ [docs/BENCHMARKS.md](docs/BENCHMARKS.md) is the guide.** It covers data
preparation, how to choose windows, what each benchmark measures, how to read
the output, and how to test an intervention.

## Quick start

Four benchmarks need nothing but numpy, pandas, scipy, scikit-learn and
matplotlib. On the example data in this repository:

```bash
python scripts/make_manifest.py --tag HumanExample \
    --orig-dir data/Homo_sapiens \
    --out manifests/pairs.HumanExample.csv --relative-to .

python scripts/run_benchmarks.py --tag HumanExample \
    --manifest manifests/pairs.HumanExample.csv \
    --composition --detectability --natural-baseline --null-check
```

Results appear under `results/HumanExample/`. See
[docs/INSTALL.md](docs/INSTALL.md) for the external tools the remaining
benchmarks need.

## What this repo contains

### The benchmarks — `scripts/benchmarks/`

`scripts/run_benchmarks.py` is the single entry point that runs any combination
of them from one command.

**No external tools required**

- **`composition.py`** — per-window GC, CpG observed/expected, homopolymer
  fraction, low-complexity score, entropy and canonical nullomer fraction, plus
  each synthetic window's k-mer JSD and FCGR L1 against its own natural
  counterpart. Paired Wilcoxon and exact sign-flip permutation tests. Start here.
- **`natural_baseline.py`** — the calibration step. Real genomic windows already
  differ from each other, so a raw divergence means little on its own. This
  compares synthetic-vs-natural distance against natural-vs-natural distance and
  reports the ratio, with significance from permuting the natural/synthetic
  label *within* each matched pair.
- **`detectability.py`** — how separable are the two sets to a shallow model?
  Logistic regression on k-mer frequencies, Markov-1 transitions, or GC alone,
  cross-validated with folds grouped by pair. Runs on CPU in minutes. If a
  linear 4-mer model already saturates, the separability is elementary
  compositional drift rather than a failure of long-range structure.
- **`context_decay.py`** — for sequence generated from a natural prompt, AUROC
  as a function of distance from the end of that seed, leave-one-window-out.
  Shows whether the model tracks its context and then loses it.

Both `natural_baseline.py` and `detectability.py` accept `--null-check`, which
reruns the analysis with natural windows pitted against each other. It should
come out at chance; if it does not, your windows are heterogeneous and the main
result is inflated.

**External tools required**

- **k-mer spectra** — distribution of short-word abundances, capturing local
  composition, repetitiveness and rare or over-represented patterns.
- **Frequency Chaos Game Representation (FCGR)** — spatial encoding of k-mer
  frequencies for a global comparison of compositional structure.
- **Nullomers** (KMC) — k-mers absent from a genome, and whether synthesis
  introduces or removes constrained patterns.
- **TFBS** (MEME/FIMO) — transcription factor motif abundance, i.e. whether
  regulatory signal survives generation.
- **Non-B DNA** (ZSeeker / G4Hunter / non-B GFA) — Z-DNA, G-quadruplexes,
  inverted, direct and mirror repeats, STRs.

### Deep detector — `scripts/classifier/`

Dilated 1D ResNet trained to distinguish natural from synthetic, with
leave-one-tag-out cross-validation and calibration-based checkpoint and
threshold selection. `run_distance.py` evaluates against bp distance from the
conditioning seed, `run_evalchunks.py` against the number of averaged chunks,
and `plot_distance_curve.py` / `plot_evalchunks_metric.py` draw the results.

### Population benchmarks — `scripts/megadna_population_based/`

For unpaired data: k-mer spectra, FCGR, non-B DNA and nullomers on two
multi-record FASTAs, stratified by genome length. No manifest needed.

```bash
python scripts/megadna_population_based/run_population_benchmarks.py \
    --natural-fasta natural.fasta --synthetic-fasta synthetic.fasta \
    --tag my_dataset --outdir results/population \
    --gfa-bin /path/to/gfa --balance-within-bin
```

### Generation — `generation/`

Samples windows from a reference FASTA and generates synthetic sequence with
Evo 2 through an Apptainer container, emitting a `pairs.<TAG>.csv` manifest that
feeds straight into the benchmarks.

### Trying to improve generation — `generation/improve/`

The two interventions we tested against the failures these benchmarks detect,
in reusable, model-agnostic form:

- **`constrained_decoding.py`** — logit constraints applied at sampling time: a
  homopolymer-run penalty and a k-mer over-representation penalty, plus a
  context manager that patches Evo 2's sampler in place.
- **`structural_losses.py`** — differentiable auxiliary training terms that drop
  into any autoregressive loop: an anti-homopolymer term and a dinucleotide KL
  against a natural reference.
- **`run_intervention_demo.py`** — trains a small character-level model on your
  FASTA and scores a 2×2 design (standard vs auxiliary-loss training, plain vs
  constrained decoding) with the same metrics the benchmarks use.

Both modules ship with `--self-test`. Judge an intervention by detectability,
not by the statistic it was designed to fix: in our runs constrained decoding
improved the targeted metric while making the output *easier* to detect.

### Paper provenance — `paper/`

The exact scripts behind the manuscript's figures, tables and statistics,
including the analyses added during peer review, with a map from each reviewer
comment to the script that answered it. Archival: they are wired to one species
list and one cluster. Several have generalised successors in
`scripts/benchmarks/` — [`paper/README.md`](paper/README.md) says which.

### Other

- `manifests/` — example `pairs.<TAG>.csv` files.
- `scripts/make_manifest.py` — builds one from your own FASTA files.
- `ref/jaspar/vertebrates/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme`
  — curated transcription factor motif database for the TFBS benchmark.
- `data/Homo_sapiens`, `Homo_sapiens_example.zip` — 40 matched 300 kb human
  windows, enough to run every dependency-free benchmark.

See `environment.yml` and [docs/INSTALL.md](docs/INSTALL.md) for setup.

## License

MIT — see [LICENSE](LICENSE).

## Reference

If you have found this work useful, please cite:

Tzanakakis, A., Mouratidis, I., & Georgakopoulos-Soares, I. (2026). Fundamental limitations of genomic language models for realistic sequence generation. *bioRxiv*. https://doi.org/10.64898/2026.01.17.700093
