# Synthetic Genomes Benchmark

Benchmarks for comparing synthetic genomes against matched original genome windows.

## What this repo contains

- `scripts/run_benchmarks.py`: Main entry point that runs all implemented genome comparison benchmarks from a single command.

- `scripts/benchmarks/`: Individual benchmark implementations used to quantify different aspects of sequence similarity and divergence between original and synthetic genome windows:

  - **k-mer spectra analysis**  
    Compares the distribution of short sequence words (k-mers) between original and synthetic genomes, capturing differences in local sequence composition, repetitiveness, and abundance of rare or over-represented patterns.

  - **Frequency Chaos Game Representation (FCGR)**  
    Encodes genome sequences into high-dimensional spatial representations that summarize k-mer frequencies, enabling a global comparison of compositional structure between original and synthetic genomes.

  - **Nullomer analysis**  
    Identifies short k-mers that are completely absent from a genome and evaluates whether synthetic genomes introduce or remove biologically forbidden or highly constrained sequence patterns.

  - **Transcription factor binding site (TFBS) analysis**  
    Scans original and synthetic sequences for transcription factor binding motifs and compares their genomic abundance, assessing whether regulatory sequence signals are preserved during synthesis.

  - **Non-B DNA structure analysis**  
    Quantifies sequence motifs associated with alternative DNA conformations (e.g. Z-DNA, G-quadruplexes, inverted repeats), which are linked to genome stability and regulatory function, and evaluates their preservation in synthetic genomes.

- `manifests/`: Example `pairs.<TAG>.csv` files defining matched original and synthetic genome windows used as input to all benchmarks.

- `ref/jaspar/vertebrates/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme`:  
  Curated transcription factor motif database used for TFBS analysis.

See `environment.yml` and `docs/INSTALL.md` for setup instructions.

## Population-based benchmarks (no pairing required)
`scripts/megadna_population_based/run_population_benchmarks.py`: Runs k-mer spectra, FCGR, non-B DNA motifs, and nullomers on two multi-record FASTA files (natural vs synthetic), stratified by genome length. No paired manifest needed.

```bash
python scripts/megadna_population_based/run_population_benchmarks.py \
    --natural-fasta natural.fasta --synthetic-fasta synthetic.fasta \
    --tag my_dataset --outdir results/population \
    --gfa-bin /path/to/gfa --balance-within-bin
```

## Deep-learning classifier
`scripts/classifier/deep_detector_dilated_resnet1d.py`: Dilated 1D ResNet trained to distinguish natural from synthetic sequences. Uses leave-one-tag-out cross-validation with calibration-based checkpoint and threshold selection.

- `run_all_experiments.py`: Sweep over tags and configurations.
- `deep_detector_distance_eval.py`: Evaluate performance vs bp distance from the conditioning seed.
- `run_evalchunks.py`: Evaluate performance vs number of averaged chunks per sequence.
- `plot_distance_curve.py`, `plot_distance_curve_broken_overlay.py`, `plot_evalchunks_metric.py`: Plotting utilities for the above.

## Synthetic genome generation (Evo2)
Generation is kept separate from the benchmarks (see `generation/`). It samples windows from a reference FASTA and generates synthetic sequences using Evo2 via an Apptainer container, producing a `pairs.<TAG>.csv` manifest compatible with the benchmarks.

## Reference
If you have found this work useful, please cite:

Tzanakakis, A., Mouratidis, I., & Georgakopoulos-Soares, I. (2026). Fundamental limitations of genomic language models for realistic sequence generation. *bioRxiv*. https://doi.org/10.64898/2026.01.17.700093
