"""
Name: swift_tcr.py
Function: This script is the main pipeline for the swiftTCR project. It takes a TCR and a p-MHC structure as input and runs a series of programs to predict the orientation of the TCR on the PMHC. And then it calculates the pairwise RMSD between the predicted structures and clusters them based on the RMSD values. The output is a text file with the cluster information and a directory with all the predicted structures of the TCR-p-MHC. 
Date: 25-09-2024
Author: Nils Smit
"""

import os.path
import os
import sys
import pdb2ms
import initial_placement
import subprocess
import postfilter
import apply_results
import merge_pdbs
import pairwise_rmsd
import clustering
import flip_alternative
import pipeline_handler
import warnings
import shutil
import time


# Add the project directory to the path so we can import the modules
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from tools.protein_prep import prepare
from tools.ANARCI_master.Example_scripts_and_sequences import ImmunoPDB

if __name__ == "__main__":
    # Get the arguments from the user
    args = pipeline_handler.get_arguments()
    
    # Gets the reference files and rotations based of if it is a MHC class 1 or 2
    if args.mhc_class_2:
        reference_ligand = os.path.realpath("ref/pmhc_2/2iam_r_u.pdb") 
        reference_receptor = os.path.realpath("ref/pmhc_2/2iam_l_u.pdb")
        rotations = os.path.realpath("rotations_and_restraints/pmhc_2/reduced_sampling_mhc2.prm")
    else:
        reference_ligand = os.path.realpath("ref/pmhc_1/2bnr_r_u.pdb")
        reference_receptor = os.path.realpath("ref/pmhc_1/2bnr_l_u.pdb")
        rotations = os.path.realpath("rotations_and_restraints/pmhc_1/filtered_cr_in_60.prm")
    
    restraint_path =  os.path.realpath("rotations_and_restraints/restraintsDE.json")
    amount_of_models = args.models
    output_path = os.path.join(os.path.realpath(args.output), args.outprefix)

    if not os.path.exists(os.path.realpath(args.output)):
        os.mkdir(os.path.realpath(args.output))

    # checks if the output directory exists, if not it creates it
    if not os.path.exists(os.path.realpath(output_path)):
        os.mkdir(os.path.realpath(output_path))
    
    # make sure the paths are absolute and creates the variables
    receptor_path = os.path.realpath(args.pmhc)
    ligand_path = os.path.realpath(args.tcr)
    cores = args.cores
    treshold = args.threshold
    output_path = output_path + "/"
    piper_path = os.path.realpath("tools/piper")
    
    # renumbers the TCR file
    out_ligand = "renumbered_"+ os.path.basename(ligand_path)
    out_ligand_path = os.path.join(output_path, out_ligand)
    
    # sets the amount of cores to use
    os.environ["OMP_NUM_THREADS"] = str(cores)

    # checks if the files exist and if the extensions are correct
    pipeline_handler.check_files(receptor_path, ligand_path)
    pipeline_handler.check_file_extensions(receptor_path, ligand_path)
    
    # Renumbers the TCR file and ignore the warnings that bio.pdb throws
    start_time_all = time.time()
    start_time_prep = time.time()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module='Bio.PDB')
        ImmunoPDB.immunopdb_main(ligand_path, out_ligand_path)
    
    # prepares the pdb files and moves them to the output directory
    receptor_pnon = prepare.prepare_main(receptor_path)
    shutil.move(receptor_pnon, os.path.join(output_path, os.path.basename(receptor_pnon)))
    receptor_pnon = os.path.join(output_path, os.path.basename(receptor_pnon))
    ligand_pnon = prepare.prepare_main(out_ligand_path)
    
    flip_alternative.reorder_residues_in_structure(receptor_pnon, receptor_pnon)    
    flip_alternative.reorder_residues_in_structure(ligand_pnon, ligand_pnon)
    
    print("Finished with preparing the files")
    
    end_time_prep = time.time()
    print(f"Time taken for preparing the files: {end_time_prep - start_time_prep} seconds")
        
    # # gets extra path names for later use
    receptor = os.path.basename(receptor_pnon)
    receptor_ms = os.path.basename(receptor_pnon).replace(".pdb", ".ms")

    ligand = os.path.basename(ligand_pnon)
    ligand_ms = os.path.basename(ligand_pnon).replace(".pdb", ".ms")
        
    # runs initial placement
    start_time_initial = time.time()
    initial_placement.initial_placement_main(receptor_pnon, ligand_pnon, output_path, reference_receptor, reference_ligand)
    end_time_initial = time.time()

    print(f"Time taken for initial placement: {end_time_initial - start_time_initial} seconds")
    
    print("Finished with aligning the target structures to the reference structures and changing the chainIDs")

    # gets the paths for the ms files
    output_receptor_path = os.path.join(output_path, receptor)
    output_ligand_path = os.path.join(output_path, ligand)
    
    # runs pdb2ms
    start_time_pdb2ms = time.time()
    pdb2ms.pdb2ms_main(output_receptor_path, output_ligand_path)
    end_time_pdb2ms = time.time()
    
    print(f"Time taken for pdb2ms: {end_time_pdb2ms - start_time_pdb2ms} seconds")
    
    print("Finished with adding attractions and creating the ms files")

    os.chdir(output_path)

    # runs piper
    start_time_piper = time.time()
    subprocess.run([
        piper_path + "/piper_attr",
        "-k1",
        "--msur_k=1.0",
        "--maskr=1.0",
        "-T", "FFTW_EXHAUSTIVE",
        "-p", piper_path + "/atoms04.prm",
        "-f", piper_path + "/coeffs04.prm",
        "-r", rotations,
        os.path.join(output_path, receptor_ms),
        os.path.join(output_path, ligand_ms)
    ]
    )
    
    print("Finished with running piper")
    
    end_time_piper = time.time()
    
    print(f"Time taken for piper: {end_time_piper - start_time_piper} seconds")

    # # runs postfilter
    # postfilter.post_filter_main(output_path, "ft.000.00", rotations, restraint_path, receptor, ligand, str(args.outprefix), cores)
    
    # print("Finished with filtering pipeline results")

    # checks if the rotated directory exists, if not it creates it
    if not os.path.exists("rotated"):
        os.mkdir("rotated")
    os.chdir("rotated")

    # runs apply_results
    time_start_apply_results = time.time()
    
    # for now postfilter is not used
    #apply_results.apply_results_main(amount_of_models, None, None,  args.outprefix, os.path.join(output_path, args.outprefix), os.path.join(output_path,rotations), os.path.join(output_path,ligand), cores)
    
    apply_results.apply_results_main(1000, None, None,  args.outprefix, os.path.join(output_path, "ft.000.00"), os.path.join(output_path,rotations), os.path.join(output_path,ligand), cores)
    time_end_apply_results = time.time()

    print(f"Time taken for apply_results: {time_end_apply_results - time_start_apply_results} seconds")    
    print("Finished with creating the rotated structures")

    os.chdir("..")

    # checks if the merged directory exists, if not it creates it
    if not os.path.exists("merged"):
        os.mkdir("merged")

    # runs merge_pdbs
    time_start_merge_pdbs = time.time()
    merge_pdbs.merge_pdbs_main(receptor,"rotated", "merged", cores)
    time_end_merge_pdbs = time.time()
    
    print(f"Time taken for merge_pdbs: {time_end_merge_pdbs - time_start_merge_pdbs} seconds")
    
    print("Finished with merging the pMHC and rotated TCR files")

    # runs pairwise_rmsd
    time_start_pairwise_rmsd = time.time()
    
    pairwise_rmsd.calc_rmsd("merged", "irmsd.csv", "A", "D", 10,  n_cores=cores)

    time_end_pairwise_rmsd = time.time()
    
    print(f"Time taken for pairwise_rmsd: {time_end_pairwise_rmsd - time_start_pairwise_rmsd} seconds")
    print("Finished with calculating the pairwise RMSD")
    
    # runs clustering
    time_start_clustering = time.time()
    output_file = os.path.join(output_path, 'clustering.txt')

    clustering.clustering_main(os.path.join(output_path, 'irmsd.csv'), treshold, output_file)

    print("Finished with clustering the structures")
    
    time_end_clustering = time.time()
    print(f"Time taken for clustering: {time_end_clustering - time_start_clustering} seconds")
    
    end_time_all = time.time()
    
    print(f"Total time taken: {end_time_all - start_time_all} seconds")