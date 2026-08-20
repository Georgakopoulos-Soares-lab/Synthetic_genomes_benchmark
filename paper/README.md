# Paper provenance

The exact scripts that produced the figures, tables and statistics in

> Tzanakakis, A., Mouratidis, I., & Georgakopoulos-Soares, I. (2026).
> *Fundamental limitations of genomic language models for realistic sequence generation.*
> bioRxiv. https://doi.org/10.64898/2026.01.17.700093

including the analyses added during peer review.

**These are archival, not the recommended entry point.** They are wired to one
species list, one directory layout and one cluster, and several have since been
generalised into the reusable benchmarks under `scripts/benchmarks/`. If you
want to evaluate your own generated sequences, start from
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) and use the benchmarks. Come here
to see precisely how a published number was computed, or to reproduce the
manuscript.

```
paper/
├── analysis/     statistics, baselines and corrections
├── figures/      figure-generating scripts
├── generation/   Evo 2 sweep generation and constrained decoding
├── slurm/        cluster job scripts and launchers
└── manifests/    seed/window manifests for the generation experiments
```

## What supersedes what

Where a script has a generalised successor, use the successor.

| Paper script | Superseded by | Why the successor differs |
| --- | --- | --- |
| `analysis/natural_natural_fcgr_baseline.py`<br>`analysis/natural_natural_nullomer_baseline.py`<br>`analysis/natnat_nonb_baseline.py` | `scripts/benchmarks/natural_baseline.py` | These use a Mann-Whitney test over pairwise distances, which treats non-independent distances as independent. See the note below. |
| `analysis/natnat_fcgr_permutation.py`<br>`analysis/natnat_nonb_permutation.py` | `scripts/benchmarks/natural_baseline.py` | Same statistic, but manifest-driven and metric-agnostic instead of hard-wired to one species list and one metric. |
| `analysis/shallow_baseline_classifier.py`<br>`analysis/baseline_shallow_classifier.py` | `scripts/benchmarks/detectability.py` | Grouped cross-validation by pair, manifest-driven, with a natural-vs-natural negative control. |
| `analysis/context_decay_auroc.py` | `scripts/benchmarks/context_decay.py` | Same leave-one-window-out design, generalised off the fixed 3/10/20 kb seed configs. |
| `analysis/fix_nullomer_canonical.py` | `scripts/benchmarks/nullomers.py` | The correction is now applied at source rather than patched afterwards. |
| `analysis/toy_glm_optimization.py` | `generation/improve/` | Split into reusable pieces: the loss terms, the decoding constraints, and a demo that runs them on any FASTA. |

The two `natnat_*_permutation.py` scripts supersede the three
`natural_natural_*` / `natnat_nonb_baseline.py` scripts **within the paper
itself** — the permutation versions produced the reported statistics. Because
every synthetic window is the matched counterpart of a specific natural window,
one window contributes to many pairwise distances, so a rank test over those
distances inflates the effective sample size and is anticonservative. The
replacements use an exact matched-label swap (exhaustive `2^n` enumeration for
`n <= 15`, otherwise `B = 10,000` Monte Carlo permutations, seed 1337) on
`Δ = median(syn-nat) − median(nat-nat)`, use all available windows rather than a
capped sample, and — for non-B DNA — restrict to `pair_id`s shared between
`orig` and `syn`, which the earlier script did not enforce.

## Reviewer comment → script

### R1 major #1 — from diagnosis to mitigation

| Script | Purpose |
| --- | --- |
| `analysis/toy_glm_optimization.py` | Scaled-down character-level gLM on a bacteriophage set, comparing `baseline` / `aux_loss` / `constr_decode` / `aux+constr`. |
| `generation/evo2_constrained_decode.py` | The same constrained decoding applied to the real Evo 2 7B, by wrapping `vortex.model.generation.sample`. |
| `analysis/constrained_decode_metrics.py` | Scores baseline vs constrained vs natural on k-mer JSD (k=6), homopolymer fraction, FCGR L1 (k=6) and shallow-classifier AUROC. |

### R1 major #2 / R2 — is the CNN's AUROC explained by shallow features?

| Script | Purpose |
| --- | --- |
| `analysis/shallow_baseline_classifier.py` | Leave-one-tag-out logistic regression / SVM on k-mer, Markov-1 and GC-only features. |
| `analysis/baseline_shallow_classifier.py` | The same comparison as a function of chunks averaged per sequence (1–32). |
| `figures/plot_shallow_vs_cnn.py` | Shallow-vs-CNN AUROC figure and table. |
| `analysis/natural_misclassification_composition.py` | Whether natural windows scored as synthetic are compositional outliers. |

### R1 #3 + R2 (figure legibility) — Figure 4A

| Script | Purpose |
| --- | --- |
| `figures/plot_fig4A_heatmap.py` | Original three-block layout, enlarged labels, prokaryotic block separated. |
| `figures/plot_fig4A_subpanels.py` | Alternative rendering as three separated sub-panels. |

Both recompute the cell values (median paired `log2((orig+eps)/(syn+eps))` of
non-B DNA bp coverage) and per-cell paired Wilcoxon tests with BH-FDR within
each block.

### R1 minor #1 — supplementary phylotag table

`analysis/build_phylotag_table.py` collates every Evo 2 phylogenetic
conditioning tag used for generation and parses it into taxonomic rank columns.

### R3 #7 + R1 minor #2 — do the failure modes survive alternative decoding?

| Script | Purpose |
| --- | --- |
| `generation/extract_seed_windows.py` | Builds the seed and natural-reference FASTAs and the manifests in `manifests/`. |
| `generation/evo2_generate_sweep.py` | Generates one synthetic sequence per window per decoding configuration. |
| `analysis/run_sweep_metrics.py` | FCGR L1 (k=8), k-mer JSD (k=6) and canonical nullomer fraction per generated window. |
| `figures/plot_sweep_metrics.py` | Per-config comparison against the natural-natural variability band. |
| `analysis/decoding_sweep_audit.py` | Per-window table, summary, paired tests and audit figure. |

Decoding configurations: `baseline` (T=1.0, top_k=4, as in the paper),
`lowtemp` (T=0.7, top_k=4) and `nucleus` (T=1.0, top_p=0.9, top_k=0).

### R1 minor #2 — does a longer conditioning window delay context decay?

| Script | Purpose |
| --- | --- |
| `slurm/window_length.sbatch`, `slurm/run_seed_length_decay.sh` | Generate 100 kb of new sequence after 3 / 10 / 20 kb seeds, one seed length per GPU. |
| `analysis/context_decay_auroc.py` | Distance-from-seed AUROC with leave-one-window-out CV, plus paired sign-flip permutation tests between seed lengths for the near-seed (0–20 kb) and long-range (40–100 kb) regions, BH-corrected. |
| `figures/plot_context_decay.py` | Decay curves, one per seed length. |

### R2 major #2 — natural-natural baselines

| Script | Metric |
| --- | --- |
| `analysis/natural_natural_fcgr_baseline.py` | FCGR L1 (k=8) |
| `analysis/natural_natural_nullomer_baseline.py` | canonical nullomer fraction (k=9) |
| `analysis/natnat_nonb_baseline.py` | non-B DNA motif bp coverage (DR, GQ, IR, MR, STR) |
| `analysis/natnat_fcgr_permutation.py`, `analysis/natnat_nonb_permutation.py` | matched-label permutation replacements (these produced the reported numbers) |
| `figures/plot_natnat_baseline_summary.py` | Combined summary panel |

### R2 #3 / major #3 — canonical nullomer denominator (correction)

KMC counts canonical k-mers by default, collapsing each k-mer with its reverse
complement, but the original pipeline divided by the full `4^k` space. For odd k
there are no reverse-complement palindromes, so the canonical count can never
exceed `4^k / 2` and the reported nullomer fraction was floored near 0.5 — the
artifact flagged in Figure 3. The correct denominator is

```
canonical_classes(k) = (4^k + P(k)) / 2,   P(k) = 0 for odd k, 4^(k/2) for even k
```

No KMC rerun is needed: the observed distinct-k-mer count is already canonical.

| Script | Purpose |
| --- | --- |
| `analysis/fix_nullomer_canonical.py` | Recomputes counts and fractions from existing KMC outputs, keeping the original columns for provenance. Use this to migrate results produced before the fix. |
| `figures/plot_nullomer_canonical.py` | Plots both conventions side by side. |
| `figures/plot_fig3_nullomers_canonical.py` | Regenerates Figure 3 from the corrected fractions. |

`scripts/benchmarks/nullomers.py` now applies the correct denominator at source
and refuses to emit a fraction when the distinct-k-mer count exceeds the k-mer
space implied by the counting mode.

### R2 #4 — Figure 5 (TFBS hotspot organisation)

| Script | Purpose |
| --- | --- |
| `figures/plot_fig5B_tfbs_hotspot.py` | Paired boxplots and scatter of Fano factor, Gini coefficient and lag-1 spatial autocorrelation, with paired Wilcoxon tests. |
| `figures/plot_fig5_volcano_fano_gini.py` | Combined Figure 5: TFBS volcano plus the Fano/Gini panels. |

The corrected analysis shows natural windows have **higher** Fano and Gini than
synthetic — synthetic TFBS distributions are more uniform — which is the
opposite direction to the original manuscript sentence.

### R3 minor #5 — Figure 1 k-mer spectra on a log x-axis

`figures/plot_fig1_kmer_spectra_logx.py` recomputes the Chor-normalised k=7
spectra (contig-aware, strand-specific) and replots panels B–G with a
log-scaled abundance axis, where nearly all density sits below ~300 on the
original linear axis.

## Running these scripts

### Paths

They were run in place on TACC Lonestar6 against an analysis tree at
`/work/11034/atzanakak/ls6/nonbdna`. That path is preserved as the default so
the code matches what produced the submitted results, but every occurrence is
overridable:

| Variable | Default | Used by |
| --- | --- | --- |
| `NONBDNA_ROOT` | `/work/11034/atzanakak/ls6/nonbdna` | all Python scripts |
| `NONBDNA_BASE` | `/work/11034/atzanakak/ls6` | `slurm/*` (container, HF cache, Evo 2 source) |
| `NONBDNA_REVISIONS` | `$NONBDNA_BASE/nonbdna/revisions` | `slurm/*` |
| `REVISION_SCRIPTS` | `$NONBDNA_REVISIONS/scripts` | `slurm/*` |

```bash
export NONBDNA_ROOT=/path/to/your/analysis/tree   # must contain results/, data/
export REVISION_SCRIPTS="$PWD/paper/analysis"
```

The `#SBATCH -o/-e` log paths in `slurm/*.sbatch` are absolute and
site-specific; edit them for your cluster.

### Inputs

The analysis scripts read the outputs of the main pipeline under
`$NONBDNA_ROOT`:

- `results/harmonized/<TAG>/<TAG>.{orig,syn}.concat.fa` — matched natural and
  synthetic windows per species/tag
- `results/metrics/<TAG>/{g4hunter,zseeker,nonbgfa,tfbs_*}.metrics.csv` —
  per-window benchmark metrics
- `results/**/nullomers.metrics.csv` — KMC nullomer counts
- `data/generated/<TAG>/pairs.<TAG>.csv` — classifier training pairs

`manifests/*.csv` carry the window coordinates, Evo 2 phylotags and target
lengths for the generation experiments. Their `seed_fasta` / `natref_fasta`
columns point at files regenerated by `generation/extract_seed_windows.py`.

### Order

1. `generation/extract_seed_windows.py` → seed/reference FASTAs and manifests
2. `slurm/decoding_sweep.sbatch` (or `slurm/autostart_sweeps.sh`) → decoding sweep;
   `slurm/window_length.sbatch` / `slurm/run_seed_length_decay.sh` → seed-length sweep
3. `analysis/run_sweep_metrics.py`, `analysis/decoding_sweep_audit.py`,
   `analysis/context_decay_auroc.py` → metrics
4. `figures/plot_*.py` → figures

`slurm/autorun_r37.sh` and `slurm/autostart_sweeps.sh` are watchdog launchers
for a shared cluster; they poll for free GPUs and are kept because they record
the exact decoding parameters each sweep was launched with.

## Not included

The Enformer-based functional audit (`fig_enformer_expression`,
`enformer_audit/audit_*.csv`) was run outside this tree and its code is not part
of this directory.
