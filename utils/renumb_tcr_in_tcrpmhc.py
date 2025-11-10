import sys
import os
import warnings
import subprocess

#directory Immunopdb script of ANARCI
immunopdb = "/home/ddiepenbroek/ANARCI/Example_scripts_and_sequences/ImmunoPDB.py"

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
    command_sel_tcr = f"pdb_selchain -A,B {input} > tcr.pdb"
    command_sel_tcr_reres = f"pdb_reres -500 tcr.pdb > tcr_shifted.pdb"
    command_sel_pmhc = f"pdb_selchain -M,N,P {input} > pmhc.pdb"
    run_command(command_sel_tcr)
    run_command(command_sel_tcr_reres)
    run_command(command_sel_pmhc)
    
def run_anarci():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module='Bio.PDB')
        subprocess.run([
            "python", immunopdb,
            "-i", "tcr_shifted.pdb",
            "-o", "renumb_tcr.pdb",
            "-s", "imgt",
            "--receptor", "tr"
        ], check=True)
        
def merge_tcr_pmhc(output):
    command_merge = f"cat renumb_tcr.pdb pmhc.pdb | grep '^ATOM' > {output}"
    run_command(command_merge)
    os.remove("tcr_tidy.pdb")
    os.remove("tcr.pdb")
    os.remove("tcr_reres.pdb")
    os.remove("pmhc.pdb")
    os.remove("renumb_tcr.pdb")

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    split_tcr_pmhc(input)
    run_anarci()
    merge_tcr_pmhc(output)