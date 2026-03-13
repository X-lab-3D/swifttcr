import os
import pandas as pd
import argparse
# Import specific functions from local files
from generate_total_irmsd import generate_total_irmsd
from energy_density_calc import energy_calc_main

def run_pipeline(base_path):
    """
    Automated pipeline to process multiple docking clusters.

    This script automates the scoring process across multiple subdirectories. 
    It performs the following logic for every folder found in the base_path:
    
    1. Detection: Scans for sub-folders containing 'irmsd.csv' and 'ft.000.00'.
    2. Model Mapping: Reads model names from the distance file to ensure consistency.
    3. Energy Extraction: Calls generate_total_irmsd() to link raw energy values 
       from the 'ft.000.00' file to specific models.
    4. Rank Calculation: Executes rank_based_main() which applies a weighted 
       scoring formula (Score = E_own + Sum(E_neighbor / (Dist + 1))).
    5. Output: Saves a 'final_ranking.csv' within each cluster folder.
    
    Usage:
    Run the script from the terminal by providing the path to your main data directory:
    python script_name.py /path/to/your/data_folders
    """
    if not os.path.exists(base_path):
        print(f"Error: The path {base_path} does not exist.")
        return

    # Retrieve all subdirectories
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for folder in directories:
        folder_path = os.path.join(base_path, folder)
        irmsd_path = os.path.join(folder_path, "irmsd.csv")
        ft_path = os.path.join(folder_path, "ft.000.00")

        if os.path.exists(irmsd_path) and os.path.exists(ft_path):
            print(f"--- Processing {folder} ---")
            
            # Extract unique model names from irmsd.csv
            df_irmsd = pd.read_csv(irmsd_path, header=None)
            models = sorted(list(set(df_irmsd[0].unique()) | set(df_irmsd[1].unique())))
            
            # Generate energy mapping 
            energy_csv = generate_total_irmsd(ft_path, models)
            
            # Calculate rank-based scores
            output_rank = os.path.join(cluster_dir, "final_ranking.csv")
            energy_calc_main(irmsd_path, energy_csv, output_file=output_rank)
            
            print(f"Success: {output_rank} created.\n")
        else:
            print(f"Skipping {cluster}: missing irmsd.csv or ft.000.00")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the energy-density scoring pipeline on a directory.")
    parser.add_argument("path", help="Path to the directory containing folders")
    
    args = parser.parse_args()
    run_pipeline(args.path)
