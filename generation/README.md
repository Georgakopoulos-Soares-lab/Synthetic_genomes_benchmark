# Synthetic genome generation (Evo2 via Apptainer)

This folder contains a reproducible pipeline to:
1) sample matched windows from a reference FASTA/FASTA.GZ
2) generate synthetic sequences for each window using Evo2 (Apptainer container)
3) write a manifest CSV: `id,orig,syn,contig,start,len,seed`


The generation code is **decoupled from any workplace paths**. Reproduction requires:
- cloning this GitHub repo
- downloading the Evo2 Apptainer container from Zenodo
- running the pipeline with paths specified in a YAML config


## Container (recommended)
We use the prebuilt Evo2 Apptainer container from Zenodo:

- Zenodo record: https://zenodo.org/records/15194473
- DOI: 10.5281/zenodo.15194473
- License: CC BY 4.0
- Files: `evo2.sif`, `sha256sum.txt`

Download and verify (example):
```bash
mkdir -p containers/evo2_zenodo
cd containers/evo2_zenodo
# download via the Zenodo UI or direct file links
sha256sum -c sha256sum.txt
```

Install (Python)

From the repo root:

pip install -r generation/requirements.txt


If your repo already uses conda, just add pyyaml to your environment.

Quickstart (HPC/Slurm)

Edit generation/configs/example.yaml paths

Submit:

sbatch generation/slurm/generate_windows.sbatch

Run locally (no Slurm)
python -m generation.src.nonbdna_gen.cli run \
  --config generation/configs/example.yaml \
  --workdir runs/example_local


Outputs are written under workdir/:

ref.fa (decompressed reference)

windows.tsv

windows/orig.*.fa

pairs.<GENOME_TAG>.csv

run_meta.<GENOME_TAG>.json

provenance.json

```txt
pyyaml>=6.0
## Trying to improve generation

`generation/improve/` holds the two interventions we tested against the failure
modes the benchmarks detect. Both are model-agnostic and independent of the
pipeline above:

- `constrained_decoding.py` — logit constraints at sampling time (homopolymer
  run penalty, k-mer over-representation penalty), with an `evo2_constrained`
  context manager that patches Evo 2's sampler in place while keeping its
  cached-generation loop.
- `structural_losses.py` — differentiable auxiliary training terms
  (`homopolymer_loss`, `dinucleotide_kl_loss`) for any autoregressive loop.
- `run_intervention_demo.py` — trains a small character-level model on your own
  FASTA and scores a 2x2 design (standard vs auxiliary-loss training, plain vs
  constrained decoding) using the benchmark metrics.

```bash
python generation/improve/constrained_decoding.py --self-test
python generation/improve/structural_losses.py --self-test
python generation/improve/run_intervention_demo.py --fasta data/Homo_sapiens \
    --outdir results/interventions --steps 400
```

See [../docs/BENCHMARKS.md](../docs/BENCHMARKS.md#trying-to-improve-generation)
for how to read the result. Judge an intervention by detectability, not by the
statistic it was designed to fix.
