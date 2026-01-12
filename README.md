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



## Synthetic genome generation (Evo2)
Generation is kept separate from the benchmarks (see `generation/`). It samples windows from a reference FASTA and generates synthetic sequences using Evo2 via an Apptainer container, producing a `pairs.<TAG>.csv` manifest compatible with the benchmarks.