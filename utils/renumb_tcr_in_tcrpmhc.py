"""
Name: renumb_tcr_in_tcrpmhc.py
Function: Script that renumbers TCR in TCRpMHC complex and returns file with the complete structure. Input is a directory with .pdb files and output is all pdb files renumbered in given output directory.
Date: 03-12-2025
Author: Daniëlle Diepenbroek, Wieke Krösschell
"""

"""
Example usage:
python renumb_tcr_in_tcrpmhc.py /path/to/ANARCI/ImmunoPDB.py /path/to/pdb_dir/ /path/to/output_dir/ 
"""

#  Imports
import sys
import os
import warnings
import subprocess
from pathlib import Path

# Take the original ImmunoPDB.py file from ANARCI. The altered one from SwiftTCR does not work.
# immunopdb = "/home/wkrosschell/1_swifttcr/swifttcr/tools/ANARCI_master/Example_scripts_and_sequences/ImmunoPDB_renumber.py"

# Wrapper that runs the command
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
    

def split_tcr_pmhc(input):
    """
    Splits the TCR and pMHC into different files based on the chains.

    Args:
        input (Path): file path of the input pdb file 
    """

    command_sel_tcr = f"pdb_selchain -D,E {input} > tcr.pdb"
    #  Cleans up the tcr file
    command_tidy_tcr = f"pdb_sort tcr.pdb | pdb_tidy | pdb_delhetatm > tidy_tcr.pdb"
    command_sel_tcr_reres = f"pdb_reres -500 tidy_tcr.pdb > tcr_shifted.pdb"
    command_sel_pmhc = f"pdb_selchain -A,B,C {input} > pmhc.pdb"
    run_command(command_sel_tcr)
    run_command(command_tidy_tcr)
    run_command(command_sel_tcr_reres)
    run_command(command_sel_pmhc)
    

def run_anarci():
    """
    Runs immunoPDB from ANARCI to renumber the files.

    Args:
    immunopdb (Path): path to the ImmunoPDB.py
    """
    print(immunopdb)
    try:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module='Bio.PDB')
            result = subprocess.run([
                "python", immunopdb,
                "-i", "tcr_shifted.pdb",
                "-o", "renumb_tcr.pdb",
                "-s", "imgt",
                "--receptor", "tr"
            ], check=True)


    except subprocess.CalledProcessError as e:
        print("This file is ignored as it gives an error.")
        # print(result.stdout)
        return False


def merge_tcr_pmhc(output_dir, file_name):
    """
    Merges the renumbered TCR file and the original pMHC. It also removes the temporary made files.

    Args:
    output_dir (Path): path to the given output directory
    file_name (tuple): tuple with the basename and extension
    """
    renum_file_name = f"{file_name[0]}{file_name[1]}"
    command_merge = f"cat pmhc.pdb renumb_tcr.pdb | grep '^ATOM' > {os.path.join(output_dir, renum_file_name)}"
    run_command(command_merge)
    os.remove("tcr.pdb")
    os.remove("tidy_tcr.pdb")
    os.remove("pmhc.pdb")
    os.remove("tcr_shifted.pdb")
    os.remove("renumb_tcr.pdb")


def extract_pdb(input_dir, output_dir):
    """
    Extracts the PDB files from the given input directory and calls all other funtions.

    Args:
    input_dir (Path): given input directory
    output_dir (Path): given output direcotry
    immunopdb (Path): path to the ImmunoPDB.py
    """
    #  Put in run_command wrapper?
    for file in os.listdir(input_dir):
        if file.endswith(".pdb"):
            print(file)
            split_tcr_pmhc(os.path.join(input_dir, file))
            run_anarci()
            file_name = os.path.splitext(os.path.basename(file))
            merge_tcr_pmhc(output_dir, file_name)

        else:
            print(f"{file} is ignored as it is not a .pdb file.")


if __name__ == '__main__':
    global immunopdb
    immunopdb = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    extract_pdb(input_dir, output_dir)
