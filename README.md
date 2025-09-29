# SwiftTCR: Efficient computational docking protocol of TCRpMHC-I complexes using restricted rotation matrices

## Overview
**SwiftTCR** is a fast fourier transform based rigid-body docking tool designed to predict bindings between T-cell receptors (TCR) and peptide-MHC complexes.

Link to the paper: [SwiftTCR](https://www.biorxiv.org/content/10.1101/2024.05.27.596020v2.full)

## Features
- Predict binding interactions between TCRs and peptide-MHC.
- User-friendly command-line interface.
- Efficient clustering algorithms for data analysis.
- With 12 CPU cores it takes around 200 seconds.

## Getting Started

To get started with SwiftTCR, follow these steps:

1. **Clone or Download** this repository.
2. Navigate into the SwiftTCR folder.

### Piper

SwiftTCR is built on Piper (v0.0.4). For academic use, Piper can be obtained by contacting Sandor Vajda's lab (vajda@bu.edu) or George Jones (george.jones@stonybrook.edu). For industrial use, a license agreement must be obtained through Acpharis Inc. or Schrödinger LLC. <br/>
Once obtained put Piper in the tools folder, the path should look like this tools/piper.<br>
The piper folder should be named ```piper``` so that swifttcr can find the tool

### Installation

To quickly install all the necessary packages, you can use the provided `swifttcr_install.yml` file. Run the following commands:

```
conda env create -f swifttcr_install.yml
conda activate swifttcr 
```

### Running SwiftTCR
Use the following command to execute SwiftTCR:

```bash
python3 scripts/swift_tcr.py -r /your/input/peptide-mhc -l /your/input/tcr -o output_directory -op output_prefix -c number_of_cores -t clustering_threshold (default=3) -m amount_of_models_generated
```
<br />

**Example command:**
```bash
python3 scripts/swift_tcr.py -r example/input/pmhc_1/unbound_structures/3w0w/3w0w_pmhc_renumbered.pdb -l example/input/pmhc_1/unbound_structures/3w0w/3w0w_tcr.pdb -o example/output/ -op first_test -c 6 -t 3 -m 100
```

## Dependencies:
* Python 3.9.12
* [Pymol open source: 3.0.0](https://github.com/schrodinger/pymol-open-source)
* [anarci: 2021.02.04](https://github.com/oxpig/ANARCI) 
* [gradpose: 0.1](https://github.com/X-lab-3D/GradPose)
* [pdb-tools: 2.5.0](http://www.bonvinlab.org/pdb-tools/)
* [torch: 2.4.1](https://pytorch.org/)
* [pdb2sql: 0.5.3](https://github.com/DeepRank/pdb2sql)
* [Biopython: 1.84](https://biopython.org/)
* [PDB2PQR: 3.6.1](https://github.com/Electrostatics/pdb2pqr)
* [Matplotlib: 3.9.2](https://matplotlib.org/)
* [Plotly: 5.24.1](https://plotly.com/)

## Output SwiftTCR

### Output Structure Naming Convention initial placement

#### Peptide-MHC Chains
- **A** = MHC (Not IMGT numbered)
- **B** = β2m (Not IMGT numbered)
- **C** = Peptide (Not IMGT numbered)

#### TCR Chains
- **D** = TCR Alpha Chain (IMGT numbered)
- **E** = TCR Beta Chain (IMGT numbered)

### Output Structure Naming Convetion SwiftTCR

#### TCR-Peptide-MHC
- **A** = The ABC Chains of original Peptide-MHC combined
- **D** = The Alpha and Beta chains of TCR combined

### Structure of output folder
The output is a folder, named using the specified output prefix, created within the designated output directory. This folder contains the following files and subfolders:
```
output
    └── 3w0w
        ├── 3w0w
        ├── 3w0w_pmhc_renumbered_pnon.ms
        ├── 3w0w_pmhc_renumbered_pnon.pdb
        ├── 3w0w_pmhc_renumbered_pnon_rename.pdb
        ├── clustering.txt
        ├── ft.000.00
        ├── irmsd.csv
        ├── merged
        │   └── merged_0.pdb
        ├── renumbered_3w0w_tcr.pdb
        ├── renumbered_3w0w_tcr_pnon.ms
        ├── renumbered_3w0w_tcr_pnon.pdb
        └── rotated
            └── 3w0w.0.pdb
```

#### Merged folder
A Folder that contains the predicted structures of the TCR-peptide-MHC.

#### Rotated folder
A folder that contains the rotated structures that where made with the use of the PIPER energies.

#### irmsd.csv
A .csv file that contains the iRMSD between all the merged files.

#### ft. files
These files contain the energies calculated by PIPER and are sorted on lowest energies first.

#### pnon.pdb files
Input structures that have been prepared and been aligned to a reference structure. They also have the same chainIDs as the reference structures.

#### .ms files
These files have added attractions that is used to run PIPER.

#### clustering.txt
This file is the output that contains the top ranked models and how many neigbours that where found with near the model.

----

## Experimental features

### pMHC Class II Support
We are currently experimenting with support for pMHC Class II modeling. While this feature is not fully refined yet, users can try it by including the following command in their input:

```bash
-mhc2
```
This flag indicates that the program should model MHC Class II. The modeling process will use a reference MHC Class II structure, and a set of rotation matrices adjusted for MHC Class II complexes.

----

## Useful links

* Pandora : https://github.com/X-lab-3D/PANDORA
* TCRmodel2: https://github.com/piercelab/tcrmodel2
