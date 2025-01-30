"""
Name: get_energies_from_ft_file.py
Function: This script extracts energy information from a Piper ft file, matching each merged file by its index.
Date: 08-11-2024
Author: Nils Smit
"""

"""
Example usage:
python get_energies_from_ft_file.py /path/to/input_dir /path/to/output_file.tsv energy_file_name
"""

import os
import argparse

def parse_ft_file(ft_file_path, max_files=1000):
    """Parse ft file to get information for up to a maximum number of merged files by index."""
    all_data = {}
    with open(ft_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= max_files:  # Stop after processing the maximum number of merged files
                break
            merged_file_name = f"merged_{i}.pdb"
            columns = line.strip().split()
            all_data[merged_file_name] = columns  # Store all columns as a list of strings for each merged file
    return all_data

def generate_output(input_dir, output_file, energy_file_name):
    """Generate the output TSV file with all information from ft file, by merged file index."""
    # Full path to the ft file based on the base name of input directory
    ft_file_path = os.path.join(input_dir, energy_file_name)
    
    # Parse ft file to get all data for each merged file, up to 1000 files
    all_data = parse_ft_file(ft_file_path)
    
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
    
    # Write the output TSV file
    with open(output_file, 'w') as out_file:
        # Write the header row with descriptive column names
        out_file.write("\t".join(headers) + "\n")
        
        # Write each row, for each merged file in index order
        for merged_file, row_data in all_data.items():
            row_data = [merged_file] + row_data
            out_file.write("\t".join(row_data) + "\n")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract all information from Piper ft files.")
    parser.add_argument("input_dir", help="Directory containing the data")
    parser.add_argument("output_file", help="Name of the output TSV file")
    parser.add_argument("energy_file_name", help="Name of the energy file (located in the input directory)")
    args = parser.parse_args()
    
    # Run the main function
    generate_output(args.input_dir, args.output_file, args.energy_file_name)
