"""
Name: pipeline_handler.py
Function: This script is used to handle the arguments from the user. It checks if the files exist, if the file extensions are correct and if the amount of chains in the pdb files are as expected. The output is the arguments from the user.
Date: 25-09-2024
Author: Nils Smit
"""
import os
from argparse import ArgumentParser


def get_arguments():
    """Gets the arguments from the user using the command line

    Returns:
        args: The arguments from the user
    """
    parser = ArgumentParser(description="SwiftTCR")
    parser.add_argument("--pmhc", "-r", required=True, help="Path to mhc pdb file")
    parser.add_argument("--tcr", "-l", required=True, help="Path to tcr pdb file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--outprefix", "-op", required=True, help="Name of the output file")
    parser.add_argument("--cores", "-c", required=True, type=int ,help="Amount of cores to use")
    parser.add_argument("--threshold", "-t", required=False, type=int, help="Threshold for clustering default is 3", default=3)
    parser.add_argument("--models", "-m", required=False, type=int, help="Amount of models that are generated default is 1000", default=1000)
    parser.add_argument("--mhc_class_2", "-mhc2",action="store_true",
    help="Use this flag when the input is MHC class II. If not provided, it defaults to False.")
    args = parser.parse_args()
    return args


def check_files(receptor, ligand):
    """Checks if the files exist

    Args:
        receptor (str): Path to receptor pdb file
        ligand (str): Path to ligand pdb file
    """
    if not os.path.exists(receptor):
        print(f"Receptor file {receptor} does not exist")
        exit(1)
    if not os.path.exists(ligand):
        print(f"Ligand file {ligand} does not exist")
        exit(1)


def check_file_extensions(receptor, ligand):
    """Checks if the file extensions are correct
    
    Args: 
        receptor (str): Path to receptor pdb file
        ligand (str): Path to ligand pdb file
    """
    if not receptor.endswith(".pdb"):
        print("Receptor file must be a pdb file")
        exit(1)
    if not ligand.endswith(".pdb"):
        print("Ligand file must be a pdb file")
        exit(1)
