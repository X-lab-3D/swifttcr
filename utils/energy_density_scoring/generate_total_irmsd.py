import os
import pandas as pd

def generate_total_irmsd(ft_file_path, merged_models):
    """
    Create total_irmsd CSV for rank-based scoring.
    
    ft_file_path: path to .ft.000.00 file (energy column at index 4)
    merged_models: list of model filenames like merged_0.pdb, merged_1.pdb, ...
    """
    energy_df = pd.read_csv(ft_file_path, sep="\s+", header=None)
    energies = energy_df[4].tolist()  # 5th column

    # build CSV
    records = []
    for i, model in enumerate(merged_models):
        records.append({"merged_id": model, "total_irmsd": energies[i]})

    output_csv = os.path.join(os.path.dirname(ft_file_path), "merged_with_energy.csv")
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv


