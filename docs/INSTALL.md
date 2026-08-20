# Installation

The benchmarks come in two tiers. Everything in step 1 is enough for
`composition.py`, `natural_baseline.py`, `detectability.py`,
`context_decay.py`, `make_manifest.py` and `generation/improve/` — that is, for
a complete first evaluation of a new generator. Step 2 adds the external
bioinformatics tools that the k-mer spectra, nullomer, TFBS and non-B DNA
benchmarks call out to.

## 1. Conda environment

```bash
conda env create -f environment.yml
conda activate syn_bench
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

Verify, and run the self-tests of the intervention modules:

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('core ok')"
python generation/improve/constrained_decoding.py --self-test
python generation/improve/structural_losses.py --self-test
```

`environment.yml` installs a CPU PyTorch build, which is all
`generation/improve/` needs. The deep detector in `scripts/classifier/` wants a
GPU build matched to your CUDA version — install that separately, following
https://pytorch.org.

### Smoke test on the bundled data

```bash
python scripts/make_manifest.py --tag HumanExample \
    --orig-dir data/Homo_sapiens \
    --out manifests/pairs.HumanExample.csv --relative-to .

python scripts/benchmarks/composition.py \
    --manifest manifests/pairs.HumanExample.csv \
    --outdir results/HumanExample/composition --max-pairs 5
```

If `data/Homo_sapiens/` is missing, unpack `Homo_sapiens_example.zip` first.

## 2. External tools

Only needed for `--spectra`, `--nullomers`, `--tfbs` and `--nonbdna`.

### KMC — nullomer benchmark

```bash
conda install -c bioconda kmc
kmc -h && kmc_tools -h
```

### MEME Suite / FIMO — TFBS benchmark

```bash
conda install -c bioconda meme
fimo --version
```

### ZSeeker — Z-DNA

```bash
git clone https://github.com/Georgakopoulos-Soares-lab/ZSeeker.git
cd ZSeeker && pip install . && cd ..
zseeker --help
```

### G4Hunter-Companion — G-quadruplexes

```bash
git clone https://github.com/Georgakopoulos-Soares-lab/G4Hunter-Companion.git
cd G4Hunter-Companion && pip install . && cd ..
g4hunter --help
```

### non-B_gfa — inverted, direct and mirror repeats, STRs

```bash
git clone https://github.com/abcsFrederick/non-B_gfa.git
cd non-B_gfa && make && export PATH=$(pwd):$PATH && cd ..
non-B_gfa --help
```

`run_benchmarks.py` looks for the binary at `non-B_gfa/gfa` relative to the
repository root by default; override with `--gfa-bin`.

## 3. Full sanity check

```bash
python -c "import numpy, pandas, matplotlib, scipy, sklearn, torch"
kmc -h
fimo --version
zseeker --help
g4hunter --help
non-B_gfa --help
```

## Next

[BENCHMARKS.md](BENCHMARKS.md) — preparing your data, choosing windows, running
the benchmarks and reading the output.
