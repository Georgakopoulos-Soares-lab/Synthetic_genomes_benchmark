# Installation

## 1. Create conda environment (used in the paper)

```bash
conda env create -f environment.yml
conda activate syn_bench

#If the environment already exists:

conda env update -f environment.yml --prune

# 2. External Tools
# KMC
conda install -c bioconda kmc

# Check

kmc -h
kmc_tools -h


# MEME Suite / FIMO
conda install -c bioconda meme

#Check 

fimo --version

# ZSeeker (Z-DNA / non-B benchmark)

git clone https://github.com/Georgakopoulos-Soares-lab/ZSeeker.git
cd ZSeeker
pip install .
cd ..

# Check

zseeker --help

# G4Hunter-Companion (G-quadruplex benchmark)

git clone https://github.com/Georgakopoulos-Soares-lab/G4Hunter-Companion.git
cd G4Hunter-Companion
pip install .
cd ..

# Check

g4hunter --help

# non-B_gfa (non-B DNA benchmark)

git clone https://github.com/abcsFrederick/non-B_gfa.git
cd non-B_gfa
make
export PATH=$(pwd):$PATH
cd ..

# Check

non-B_gfa --help

# 3. Minimal sanity check

python -c "import numpy, pandas, matplotlib, scipy"
kmc -h
fimo --version
zseeker --help
g4hunter --help
non-B_gfa --help
