"""
Name: get_energies_from_clustering_data.py
Function: This script is used to extract all information from Piper ft files based on a clustering file.
Date: 08-11-2024
Author: Nils Smit
"""

"""
Example usage:
python get_energies_from_clustering_data.py /path/to/input_dir /path/to/output_file.tsv clustering_file_name energy_file_name
"""

import os
import re
import argparse

def parse_clustering_file(clustering_file_path):
    """Parse clustering file to get list of relevant merged file names in order."""
    merged_files_ordered = []
    with open(clustering_file_path, 'r') as file:
        for line in file:
            match = re.search(r'merged_(\d+)\.pdb', line)
            if match:
                merged_index = int(match.group(1))
                merged_files_ordered.append(f"merged_{merged_index}.pdb")  # Store as "merged_X.pdb" in order
    return merged_files_ordered

def parse_ft_file(ft_file_path, merged_files_ordered):
    """Parse ft file to get all information for specific merged files."""
    all_data = {}
    with open(ft_file_path, 'r') as file:
        for i, line in enumerate(file):
            if f"merged_{i}.pdb" in merged_files_ordered:  # Only process lines that match merged files in clustering order
                columns = line.strip().split()
                all_data[f"merged_{i}.pdb"] = columns  # Store all columns as a list of strings
    return all_data

def generate_output(input_dir, output_file, clustering_file_name, energy_file_name):
    """Generate the output TSV file with all information from ft file, in clustering order."""
    # Full path to the clustering file
    clustering_file_path = os.path.join(input_dir, clustering_file_name)
    
    # Full path to the ft file based on the base name of input directory
    ft_file_name = energy_file_name
    ft_file_path = os.path.join(input_dir, ft_file_name)
    
    # Parse clustering file to get the list of relevant merged files in order
    merged_files_ordered = parse_clustering_file(clustering_file_path)
    
    # Parse ft file to get all data for the relevant merged files
    all_data = parse_ft_file(ft_file_path, merged_files_ordered)
    
    # Define column names as per provided descriptions
    headers = [
        "Name", 
        "Rotation Index", 
        "Translation (x)", 
        "Translation (y)", 
        "Translation (z)", 
        "total weighted energy", 
        "repulsive vdW energy (unweighted)", 
        "attractive vdW energy (unweighted)", 
        "coulombic electrostatic energy (unweighted)", 
        "generalized Born approximation electrostatics energy (unweighted)", 
        "pairwise potential energy (unweighted)"
    ]
    
    # Write the output TSV file, following the order from clustering file
    with open(output_file, 'w') as out_file:
        # Write the header row with descriptive column names
        out_file.write("\t".join(headers) + "\n")
        
        # Write each row, in the order from clustering file
        for merged_file in merged_files_ordered:
            if merged_file in all_data:
                row_data = [merged_file] + all_data[merged_file]
                out_file.write("\t".join(row_data) + "\n")
            else:
                print(f"Warning: No data found for {merged_file}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract all information from Piper ft files.")
    parser.add_argument("input_dir", help="Directory containing the data")
    parser.add_argument("output_file", help="Name of the output TSV file")
    parser.add_argument("clustering_file_name", help="Name of the clustering file (located in the input directory)")
    parser.add_argument("energy_file_name", help="Name of the energy file (located in the input directory)")
    args = parser.parse_args()
    
    # Run the main function
    generate_output(args.input_dir, args.output_file, args.clustering_file_name, args.energy_file_name)
