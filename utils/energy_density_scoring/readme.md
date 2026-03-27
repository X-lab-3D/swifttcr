# Rank-based docking scoring pipeline

This pipeline performs rank-based scoring of docking models using pairwise interface RMSD (iRMSD) and docking energies.
Models are ranked based on their own energy and the energies of structurally similar neighbors.

---

# Input
Each docking case must be stored in a separate folder containing:

```
irmsd.csv
ft.000.00
```

### irmsd.csv
Pairwise interface RMSD values:

```
model1,model2,irmsd
merged_0.pdb,merged_1.pdb,2.1
merged_0.pdb,merged_2.pdb,3.5
```

### ft.000.00
Docking output file containing energies.
The **5th column (index 4)** is used as the model energy.

# Scoring formula

For each model:

Score(model) = E_model + sum(E_neighbor / (iRMSD + 1))

Lower score = better rank.

# Usage

Run the pipeline on a directory containing multiple docking folders:

```bash
python energy_density_scoring.py /path/to/data
```

Example structure:

```
data/
 ├── case1/
 │   ├── irmsd.csv
 │   └── ft.000.00
 ├── case2/
 │   ├── irmsd.csv
 │   └── ft.000.00
```

# Output
For each folder:

```
merged_with_energy.csv
final_ranking.csv
```

`final_ranking.csv` contains the ranked docking models:

```
rank,model,score
1,merged_12.pdb,-305.42
2,merged_4.pdb,-301.15
```
