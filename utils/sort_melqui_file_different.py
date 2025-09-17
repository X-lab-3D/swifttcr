"""
Function: This script reorders the tuples in the new_melqui_plot_combined file based on a specified column in the corresponding .tsv files.
Date: 10-12-2024
Author: Nils Smit
"""

"""
Example usage:
python sort_melqui_file_different.py /path/to/tsv_directory /path/to/new_melqui_plot_combined /path/to/output_file column_name
"""

"""
file format tsv files (which are in the tsv_directory):
1ao7	0.571429	0.385714	0.5
1mi5	0.068493	0.082192	0.150685
1mwa	0.5625	0.453125	0.578125
"""

import pandas as pd
import os
import sys

def load_tsv_data(file_path):
    """ Load .tsv file and return it as a DataFrame 
    
    Args:
        file_path (str): Path to the .tsv file
        
    Returns:
        pd.DataFrame: DataFrame containing the data from the .tsv file
    """
    # Load .tsv file and return it as a DataFrame
    return pd.read_csv(file_path, sep='\t')

def load_new_melqui_data(file_path):
    """ Load the new_melqui_plot_combined file and return it as a dictionary
    
    Args:
        file_path (str): Path to the new_melqui_plot_combined file
        
    Returns:
        dict: Dictionary containing the data from the new_melqui_plot_combined file
    """
    # Load the new_melqui_plot_combined file
    melqui_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            file_id = parts[0]
            tuples = [eval(t) for t in parts[1:]]
            melqui_data[file_id] = tuples
    return melqui_data

def reorder_tuples(tsv_data, tuples, column_name):
    """ Reorder the tuples based on the specified column in the tsv data
    
    Args:
        tsv_data (pd.DataFrame): DataFrame containing the data from the .tsv file
        tuples (list): List of tuples to reorder
        column_name (str): Name of the column to sort by
        
    Returns:
        list: Reordered list of tuples
    """
    # Sort tuples based on a specified column in the tsv data
    # Add an index column to preserve original order and facilitate sorting
    tsv_data['OriginalIndex'] = range(len(tsv_data))
    sorted_tsv = tsv_data.sort_values(by=column_name, ascending=True).reset_index(drop=True)
    
    # Map original indices to sorted indices
    index_map = {row['OriginalIndex']: idx for idx, row in sorted_tsv.iterrows()}
    
    # Reorder tuples based on the sorted indices
    reordered_tuples = [tuples[index_map[i]] for i in range(len(tuples))]
    return reordered_tuples

def process_files(tsv_dir, melqui_file, output_file, sort_column):
    """ Process the .tsv files and the new_melqui_plot_combined file
    
    Args:
        tsv_dir (str): Directory containing the .tsv files
        melqui_file (str): Path to the new_melqui_plot_combined file
        output_file (str): Output file path
        sort_column (str): Column to sort by
    """
    # Load new_melqui_plot_combined file
    melqui_data = load_new_melqui_data(melqui_file)

    # Prepare output file
    with open(output_file, 'w') as out_file:
        for file_id, tuples in melqui_data.items():
            tsv_path = os.path.join(tsv_dir, f"{file_id}.tsv")
            
            if os.path.exists(tsv_path):
                # Load corresponding .tsv file
                tsv_data = load_tsv_data(tsv_path)
                
                if sort_column in tsv_data.columns:
                    # Reorder tuples based on the specified sorting column
                    reordered_tuples = reorder_tuples(tsv_data, tuples, sort_column)
                    
                    # Write to output file
                    out_file.write(f"{file_id}\t" + "\t".join(map(str, reordered_tuples)) + "\n")
                else:
                    print(f"Column '{sort_column}' not found in {file_id}.tsv")
            else:
                print(f"TSV file not found for {file_id}")


if __name__ == "__main__":
    # Parameters
    tsv_directory = sys.argv[1] # Directory containing the .tsv files
    melqui_file_path = sys.argv[2] # Path to the new_melqui_plot_combined file
    output_file_path = sys.argv[3] # Output file path
    sort_column_name = sys.argv[4] # Column to sort by

    # Run the processing function
    process_files(tsv_directory, melqui_file_path, output_file_path, sort_column_name)
