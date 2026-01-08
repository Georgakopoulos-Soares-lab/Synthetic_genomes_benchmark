# Synthetic Genomes Benchmark

Benchmarks for comparing synthetic genomes against matched original genome windows.

## What this repo contains
- `scripts/run_benchmarks.py`: main entrypoint (runs the wired benchmarks)
- `scripts/benchmarks/`: individual benchmark scripts (k-mer spectra, FCGR, nullomers, TFBS, non-B DNA, ...)
- `manifests/`: example `pairs.<TAG>.csv` manifests (id, orig, syn)
- `results/`: output directory (ignored by git)

See `environment.yml` and `INSTALL.md` for setup instructions.
