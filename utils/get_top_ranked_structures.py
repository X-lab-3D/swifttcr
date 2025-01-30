"""
Function: This script is used to get the top ranked models from a clustering file and rename them using pdb_tidy and pdb_merge. This script renames the chains of the PDB files to A, B, C, D, and E which is the original.
Date: 29-01-2025
Author: Nils Smit
"""

"""
Example usage:

python3 get_top_ranked_structures.py /path/to/clustering_file /path/to/input_directory /path/to/output_directory 4
"""

import os
import sys
import uuid
import subprocess
import multiprocessing

def run_command(command):
    """
    Helper function to run a command with subprocess.run and handle errors.
    
    Args:
        command (str): The shell command to be executed.
        
    Returns:
        str: The standard output of the command if successful.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(f"Error message: {e.stderr}")
        raise

def parse_clustering_file(clustering_path):
    """Parse a clustering file and return a list of model names.
    
    Args:
        clustering_path (str): Path to the clustering file.
        
    Returns:
        list: List of model names.
    """
    models = []
    with open(clustering_path, "r") as f:
        for line in f:
            if line.startswith("Cluster center: "):
                model_name = line.split("Cluster center: ")[-1].split("with")[0].strip()
                models.append(model_name)
    return models

def process_single_pdb(model_info):
    """Process a single PDB file with unique temporary filenames.
    
    Args:
        model_info (tuple): Tuple containing (model name, input directory, output directory).
    """
    model, input_dir, output_dir = model_info
    
    input_pdb = os.path.join(input_dir, model)
    output_pdb = os.path.join(output_dir, model)

    if not os.path.exists(input_pdb):
        print(f"Warning: {input_pdb} not found. Skipping.")
        return

    # Generate unique temporary filenames
    unique_id = uuid.uuid4().hex
    temp_A = f"temp_A_{unique_id}.pdb"
    temp_B = f"temp_B_{unique_id}.pdb"
    temp_C = f"temp_C_{unique_id}.pdb"
    temp_D = f"temp_D_{unique_id}.pdb"
    temp_E = f"temp_E_{unique_id}.pdb"

    command = (
        f"pdb_tidy {input_pdb} | pdb_selchain -A | pdb_selres -1000:1999 | pdb_chain -A > {temp_A}; "
        f"pdb_tidy {input_pdb} | pdb_selchain -A | pdb_selres -2000: | pdb_chain -B > {temp_B}; "
        f"pdb_tidy {input_pdb} | pdb_selchain -A | pdb_selres -1:999 | pdb_chain -C > {temp_C}; "
        f"pdb_tidy {input_pdb} | pdb_selchain -D | pdb_selres -1:1999 | pdb_chain -D > {temp_D}; "
        f"pdb_tidy {input_pdb} | pdb_selchain -D | pdb_selres -2000: | pdb_chain -E > {temp_E}; "
        f"pdb_merge {temp_A} {temp_B} {temp_C} {temp_D} {temp_E} | pdb_tidy > {output_pdb}; "
        f"rm {temp_A} {temp_B} {temp_C} {temp_D} {temp_E}"
    )

    run_command(command)

def process_pdb_files(models, input_dir, output_dir, num_cores):
    """Process PDB files using multiprocessing.
    
    Args:
        models (list): List of model names.
        input_dir (str): Path to the input directory containing PDB files.
        output_dir (str): Path to the output directory to save processed PDB files.
        num_cores (int): Number of CPU cores to use.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    model_info_list = [(model, input_dir, output_dir) for model in models]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(process_single_pdb, model_info_list, chunksize=len(models) // (num_cores * 2) or 1)

def get_top_ranked_models_and_rename_them():
    """ Main function that processes the PDB files.
    """
    if len(sys.argv) < 5:
        print("Usage: python script.py <clustering_file> <input_directory> <output_directory> <num_cores>")
        sys.exit(1)

    cluster_file = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    num_cores = int(sys.argv[4])

    models = parse_clustering_file(cluster_file)
    process_pdb_files(models, input_dir, output_dir, num_cores)
    print(f"Processed {len(models)} PDB files using {num_cores} cores and saved them to {output_dir}")

if __name__ == "__main__":
    get_top_ranked_models_and_rename_them()
