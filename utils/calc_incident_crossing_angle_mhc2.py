# -*- coding: utf-8 -*-
"""
Name: calc_incident_crossing_angle.py
Date:   Created on Thu Jun  1 12:44:45 2023
        Adapted to MHC-II on Wed April  8 2025
Author: Yannick Aarts, Wieke Krösschell

Function: 
The goal is to evaluate what ranges to use for crossing and incident angles by calculating the angles of multiple TCRpMHC complexes. To show the angles in a plot, use 'bokeh_plot.py'.

INPUT:
    - <path/to/directory>   :   Path to directory containing the TCRpMHC complexes in .pdb format
    - <meta_data_file>.tsv  :   File name containing the meta data of all structures. This summary file is free to download on STCRDab 
                                (https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred). After searching the database on desired structures, 
                                summary file is possible to download at the bottom of the page.
    - <path/to>/utils
                            :   Path to Utils directory from SwiftTCR to use script

OUTPUT:
    - angles.csv            :   CSV file containing the model name, crossing angle, incident angle, organism and MHC class. Placed in the directory of the input TCRpMHC complexes.

Example usage:
python calc_incident_crossing_angle.py /calc_angles/stcrdab_mhc2 /calc_angles/20251209_0612964_summary.tsv /swifttcr/utils/

"""

import numpy as np
from sklearn.decomposition import PCA
from pdb2sql import pdb2sql
from pathlib import Path
import pandas as pd
import time
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model_path_in', type=Path, help='Path to the model')
parser.add_argument('meta_data_file_in', type=Path, help='Path to the metadata TSV file')
parser.add_argument('utils_dir', type=Path, help='Path to utils directory in SwiftTCR')
args = parser.parse_args()

# Imports python file from SwiftTCR
sys.path.append(str(args.utils_dir))
print(str(args.utils_dir))
import generate_rotation_matrices_from_angle as incr 

def main():
    model_path = args.model_path_in
    meta_data_file = args.meta_data_file_in
    df = pd.read_csv(meta_data_file, sep='\t')   
    p = Path(model_path)
    crossing_angles = []
    incident_angles = []
    organisms = []
    models = []
    mhc_types = []
    failed_cases = []
    for model in p.iterdir():
        if ".pdb" in model.name:
            try:
                mhc_type = df[df['pdb'] == model.stem]['mhc_type'].values[0]
                
                if mhc_type == 'MH2':
                    print(model.stem, mhc_type)
                    Achain = df[df['pdb'] == model.stem]['Achain'].values[0]
                    Bchain = df[df['pdb'] == model.stem]['Bchain'].values[0]
                    mhc_chain1 = df[df['pdb'] == model.stem]['mhc_chain1'].values[0]
                    mhc_chain2 =  df[df['pdb'] == model.stem]['mhc_chain2'].values[0]
                    print(Achain, Bchain, mhc_chain1, mhc_chain2)
                    #beta_organism	alpha_organism	gamma_organism	delta_organism	antigen_organism	mhc_chain1_organism	mhc_chain2_organism
                    organism = define_organism(df, model)
                    (crossing_angle, incident_angle) = calc_crossing_incident_angle(model, mhc_type, Bchain, Achain, mhc_chain1, mhc_chain2)
                    print(crossing_angle, incident_angle)
                    crossing_angles.append(crossing_angle)
                    incident_angles.append(incident_angle)
                    organisms.append(organism)
                    mname = model.name.split("_")[0]
                    models.append(mname)
                    mhc_types.append(mhc_type)
                    print(mname, crossing_angle, incident_angle)
                
            except IndexError as e:
                failed_cases.append(model.stem)
                #print(df.columns)
                #print(df[df['pdb'] == model.stem].values)
            except ValueError as e:
                failed_cases.append(model.stem)
                #print(df.columns)
                #print(df[df['pdb'] == model.stem].values)

            
    data = {
    'modelname': models,
    'crossing_angle': crossing_angles,
    'incident_angle': incident_angles,
    'organism' : organisms,
    'mhc_type' : mhc_types
    }

    df = pd.DataFrame(data)
    df.to_csv(Path(p, 'angles2.csv'))
    if failed_cases is not []:
        print("Failed cases:")
        for f in failed_cases:
            print(f)

def define_organism(df, model):
    beta_organism = df[df['pdb'] == model.stem]['beta_organism'].values[0]
    alpha_organism = df[df['pdb'] == model.stem]['alpha_organism'].values[0]
    #gamma_organism = df[df['pdb'] == model.stem]['gamma_organism'].values[0]
    #delta_organism = df[df['pdb'] == model.stem]['delta_organism'].values[0]
    #antigen_organism = df[df['pdb'] == model.stem]['antigen_organism'].values[0]
    mhc_chain1_organism = df[df['pdb'] == model.stem]['mhc_chain1_organism'].values[0]
    mhc_chain2_organism = df[df['pdb'] == model.stem]['mhc_chain2_organism'].values[0]
    organism = "Default"
    if beta_organism == alpha_organism == mhc_chain1_organism == mhc_chain2_organism:
        organism = "TCR and mhc: " + beta_organism
    elif beta_organism == alpha_organism and mhc_chain1_organism == mhc_chain2_organism:
        organism = "TCR: " + beta_organism + " mhc: " + mhc_chain1_organism
    else:
        organism = "Hybrid"
    return organism

def calculate_angle_dot(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    crossing_angle_radians = np.arccos(dot_product)
    crossing_angle_degrees = np.degrees(crossing_angle_radians)
    return crossing_angle_degrees

def orient_vector(vector, reference_point):
    # Calculate the vector pointing from the reference point to the vector
    direction_vector = vector - reference_point

    # Calculate the dot product of the direction vector and the vector
    dot_product = np.dot(direction_vector, vector)

    # Change the sign of the vector based on the dot product
    oriented_vector = vector if dot_product >= 0 else -vector

    return oriented_vector


def calc_crossing_incident_angle(model, mhc_type, Bchain = 'E', Achain = 'D', mhc_chain1 = 'A', mhc_chain2 = 'B'):
    pdb = pdb2sql(model)
    if mhc_type == "MH1":
        #xyzCYS = pdb.get('x,y,z', chainID = [Achain, Bchain], resSeq = [23, 104], name = ['CA'])
        xyzCYS_D = pdb.get('x,y,z', chainID = [Achain], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
        xyzCYS_E = pdb.get('x,y,z', chainID = [Bchain], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
        helix1 = [i for i in range(50, 86)]
        #helix2 = [i for i in range(140, 176)]
        helix2 = [i for i in range(1050, 1086)]
        xyzMHC = pdb.get('x,y,z', chainID = [mhc_chain1], resSeq = helix1 + helix2, name = ['CA'])
        xyzTCR = pdb.get('x,y,z', chainID = [Achain, Bchain], resSeq = [i for i in range(1, 126)], name = ['CA'])
        #ref_point = pdb.get('x,y,z', chainID = ['B'], resSeq = ['73'])
        ref_pointMHC = pdb.get('x,y,z', chainID = [mhc_chain1], resSeq = ['88'], name = ['CA'])
        #ref_pointCYS = pdb.get('x,y,z', chainID = [Bchain], resSeq = ['86'], name = ['CA'])
        ref_pointTCR =pdb.get('x,y,z', chainID = [Achain], resSeq = ['125'], name = ['CA'])#97
    elif mhc_type == "MH2":
        xyzCYS_D = pdb.get('x,y,z', chainID = [Achain], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
        xyzCYS_E = pdb.get('x,y,z', chainID = [Bchain], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
        # helix = [i for i in range(50, 86)]
        # helix2 = [i for i in range(50, 86)]
        helix = [i for i in range(50, 78)]
        helix2 = [i for i in range(51, 86)]
        xyzMHC1 = pdb.get('x,y,z', chainID = [mhc_chain2], resSeq = helix, name = ['CA'])
        xyzMHC2 = pdb.get('x,y,z', chainID = [mhc_chain1], resSeq = helix2, name = ['CA'])
        print(len(xyzMHC1))
        print(len(xyzMHC2))
        xyzMHC = xyzMHC1 + xyzMHC2
        xyzTCR = pdb.get('x,y,z', chainID = [Achain, Bchain], resSeq = [i for i in range(1, 128)], name = ['CA'])
        ref_pointMHC = pdb.get('x,y,z', chainID = [mhc_chain1], resSeq = ['78']) 
        ref_pointTCR = pdb.get('x,y,z', chainID = [Achain], resSeq = ['97'], name = ['CA']) 


    #(ref_crossing, ref_incident) = calc_crossing_incident_angle_coordinates(xyzCYS, xyzMHC, xyzTCR, ref_pointMHC, ref_pointCYS, ref_pointTCR)
    (crossing_angle, incident_angle) = incr.calc_crossing_incident_angle(xyzCYS_D, xyzCYS_E, xyzMHC, xyzTCR, ref_pointMHC, ref_pointTCR) #(xyzCYS_D, xyzCYS_E, xyzMHC, xyzTCR, ref_pointMHC, ref_pointTCR)
    
    
    return (crossing_angle, incident_angle)

def calculate_incident_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    return  angle_degrees


def calculate_crossing_angle(vector1, vector2, ref_v):
    """
    Calculate the angle between v1 and v2.

    Using only dot_product(v1,v2), the order of v1 and v2 does not matter and we cannot distinguish theta and -theta (reversed docking of tcr).

    Therefore, we need to use right-hand rule:
        if cross_product(v1, v2) is pointing to ref_v, rotation angle from v1 to v2 is positive. Otherwise, the angle is negative.

    Parameters:
    - vector1 (numpy.ndarray): The first vector.
    - vector2 (numpy.ndarray): The second vector.

    Returns:
    float: The angle between the two vectors in degrees.

    This function calculates the angle between two vectors using the dot product method.
    It computes the dot product of the input vectors, then calculates the arccosine of the dot product to obtain the angle in radians,
    and finally converts the angle to degrees.
    """

    dot_product = np.dot(vector1, vector2)
    angle_radians = np.arccos(dot_product)

    # apply right-hand rule
    angle_radians = angle_radians if np.dot(np.cross(vector1, vector2), ref_v) >=0 else - angle_radians

    # convert to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def orient_vector_new(vector, direction_vector):
    # Calculate the dot product of the direction vector and the vector
    dot_product = np.dot(direction_vector, vector)
    # Change the sign of the vector based on the dot product
    oriented_vector = vector if dot_product >= 0 else -vector
    return oriented_vector
  

if __name__ =="__main__":
    main()
