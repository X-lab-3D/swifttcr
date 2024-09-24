#!/usr/bin/env python
# Li Xue
# 16-Mar-2023 17:22
#REWRITE using 2 reference structures and do superposition take 2bnr_l_u, 2bnr_r_u from input_new/
"""Script to rotate the ligand and receptor based on alpha helices and conserved cys residues.

https://pdb2sql.readthedocs.io/en/latest/_modules/pdb2sql/superpose.html#superpose

Usage: python3 pca.py <receptor> <ligand> <output_dir>
"""

"""
In the future it is beter to renumber the pMHC to IMGT numbering so that we can look at only a specific residue in both the reference and target. This will make it faster to find which chain is superimposed to which chain in the reference.

Todo: explain why specific residues are chosen line 77-81
"""

# Have to import cmd because that stops the warning from PyMOL because now i think it can overwrite the cmd module and otherwise pymol will give a warning
import cmd
from pymol import cmd
from pathlib import Path
import numpy as np

 
def get_residue_coordinates(selection):
    """
    get the coordinates of the alpha carbon of the residue and chain in the selection
    """
    # Get the coordinates of the atoms in the selection
    model = cmd.get_model(selection)

    # Extract the atom coordinates
    for atom in model.atom:
        if atom.name == "CA":
            coordinates = [atom.coord[0], atom.coord[1], atom.coord[2]]
    return coordinates


def find_superposed_chain(ref_residue_coords, target_residues):
    """
    Find the closest residue to the reference residue in the target
    """
    closest_residue = None
    min_distance = float('inf')

    for atom in target_residues:
        if atom.name == "CA":
            target_coord = np.array(atom.coord)
            # Calculate the distance between the reference and target
            distance = np.linalg.norm(ref_residue_coords - target_coord)
            if distance < min_distance:
                min_distance = distance
                closest_residue = atom
    return closest_residue


def initial_placement_main(receptor, ligand, outputdir, reference_receptor, reference_ligand):
    receptor = Path(receptor)
    ligand = Path(ligand)
    outputdir = Path(outputdir)
    reference_receptor = Path(reference_receptor)
    reference_ligand = Path(reference_ligand)

    # Load receptor, ligand, and reference structures into PyMOL
    cmd.load(str(receptor), 'receptor')
    cmd.load(str(ligand), 'ligand')
    cmd.load(str(reference_receptor), 'ref_receptor')
    cmd.load(str(reference_ligand), 'ref_ligand')
    
    # Superpose ligand to reference ligand (align entire structures)
    results_lig = cmd.super('ligand', 'ref_ligand')
    rmsd_lig = results_lig[0]
    results_rec = cmd.super('receptor', 'ref_receptor')
    rmsd_rec = results_rec[0]
    
    # The specific residues are chosen because they are far away from the other chains which makes it easier to find the correct chain
    ref_residue_coords_a = get_residue_coordinates("ref_receptor and chain A and resi 216")
    ref_residue_coords_b = get_residue_coordinates("ref_receptor and chain B and resi 28")
    ref_residue_coords_c = get_residue_coordinates("ref_receptor and chain C and resi 2")
    ref_residue_coords_d = get_residue_coordinates("ref_ligand and chain D and resi 89")
    ref_residue_coords_e = get_residue_coordinates("ref_ligand and chain E and resi 90")
    
    # Get the atoms in the target structure
    target_residues_tcr = cmd.get_model("ligand").atom
    target_residues_pmhc = cmd.get_model("receptor").atom

    # Find the closest residue to the  of the reference residues
    closest_residue_a = find_superposed_chain(ref_residue_coords_a, target_residues_pmhc)
    closest_residue_b = find_superposed_chain(ref_residue_coords_b, target_residues_pmhc)
    closest_residue_c = find_superposed_chain(ref_residue_coords_c, target_residues_pmhc)
    closest_residue_d = find_superposed_chain(ref_residue_coords_d, target_residues_tcr)
    closest_residue_e = find_superposed_chain(ref_residue_coords_e, target_residues_tcr)
    
    print(f"Closest residue to reference chain A residue 216: residue {closest_residue_a.resi} in chain {closest_residue_a.chain}")
    print(f"Closest residue to reference chain B residue 28: residue {closest_residue_b.resi} in chain {closest_residue_b.chain}")
    print(f"Closest residue to reference chain C residue 2: residue {closest_residue_c.resi} in chain {closest_residue_c.chain}")
    print(f"Closest residue to reference chain D residue 89: residue {closest_residue_d.resi} in chain {closest_residue_d.chain}")
    print(f"Closest residue to reference chain E residue 90: residue {closest_residue_e.resi} in chain {closest_residue_e.chain}")

    # First change the chain of the reference residues to a temporary chain ID
    cmd.alter(f'receptor and chain {closest_residue_a.chain}', 'chain="X"')
    cmd.alter(f'receptor and chain {closest_residue_b.chain}', 'chain="Y"')
    cmd.alter(f'receptor and chain {closest_residue_c.chain}', 'chain="Z"')
    
    cmd.alter(f'receptor and chain X', 'chain="A"')
    cmd.alter(f'receptor and chain Y', 'chain="B"')
    cmd.alter(f'receptor and chain Z', 'chain="C"')

    # First change the chain of the reference residues to a temporary chain ID
    cmd.alter(f'ligand and chain {closest_residue_d.chain}', 'chain="Y"')
    cmd.alter(f'ligand and chain {closest_residue_e.chain}', 'chain="Z"')

    # change the chain of the closest residues to the chain of the reference
    # residues
    cmd.alter(f'ligand and chain Y', 'chain="D"')
    cmd.alter(f'ligand and chain Z', 'chain="E"')
    
    output_receptor_path = Path(outputdir, receptor.name)
    output_ligand_path = Path(outputdir, ligand.name)
    
    print(f"RMSD of ligand superposition: {rmsd_lig:.2f} Å")
    print(f"RMSD of receptor superposition: {rmsd_rec:.2f} Å")
    if rmsd_lig > 2.0:
        print(f"Warning: RMSD of ligand superposition is {rmsd_lig:.2f}")
    if rmsd_rec > 2.0:
        print(f"Warning: RMSD of receptor superposition is {rmsd_rec:.2f}")

    cmd.save(str(output_receptor_path), 'receptor')
    cmd.save(str(output_ligand_path), 'ligand')

    cmd.delete("all")
    