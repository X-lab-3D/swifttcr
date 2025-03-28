import numpy as np
from numpy import pi, cos, sin
from scipy.spatial.transform import Rotation as R
import random
from pdb2sql import pdb2sql
from sklearn.decomposition import PCA
import sys
import pdb
import warnings


def write_rotation_matrix(matrices, outfilename):
    with open(outfilename, 'w') as f:
        for i, matrix in enumerate(matrices):
            f.write(str(i))
            f.write(' ' + matrix)
#            for value in matrix.reshape(1, 9)[0]:
#                f.write(' '+ '%.9f' % value)
            f.write('\n')
    print("Rotation matrices wrote to: ", outfilename)

"""

def euler_angles_in_range_old(x_range, y_range, z_range, steps):
    R_list = []
    xyz_list = []
    stepx = steps
    stepy = steps
    stepz = steps
    x_steps = [i * (1/stepx) for i in range(stepx + 1)]
    y_steps = [i * (1/stepy) for i in range(stepy + 1)]
    z_steps = [i * (1/stepz) for i in range(stepz + 1)]
    for x_step in x_steps:#steps should go from 0-1
        print(x_step)
        theta = 2 * math.pi * x_step - math.pi
        for y_step in y_steps:
            psi = math.acos(1-2* y_step) + 0.5 * math.pi
            if y_step < 0.5:
                if psi < math.pi:
                    psi = psi + math.pi
                else:
                    psi = psi - math.pi
            for z_step in z_steps:
                eta = 2 * math.pi * z_step - math.pi
                r = R.from_euler('xyz', [theta, psi, eta], degrees=False)
                x, y, z = r.as_euler('xyz', degrees = True)
                #print("Degrees: x: {}, y: {}, z: {}".format(xd,yd,zd))
                #print("R_euler x: {}, y: {}, z: {}".format(x, y, z))
                if x >= x_range[0] and x <= x_range[1] and y >= y_range[0] and y <= y_range[1] and z >= z_range[0] and z <= z_range[1]:
                    R_list.append(r.as_matrix())
                    xyz_list.append((x,y,z))
    return R_list, xyz_list
"""


def round_rot(rotation_matrix):
    rotation_matrix_rounded = np.round(rotation_matrix, 9)
    return rotation_matrix_rounded

#def to_string_xue(matrices):
#    #matrices.shape = (n, 3,3)
#
#    # Flatten the 2nd and 3rd dimensions
#    flattened = matrices.reshape(matrices.shape[0], -1)
#    flattened_str = [' '.join(row.astype(str)) for row in flattened]
#    return flattened_str

def to_string(matrix):
    # Convert the matrix to a flattened list of values as strings
    flattened_values = ["{:.9f}".format(val) for val in matrix.flatten()]

    # Join the flattened values into a single space-separated string
    return " ".join(flattened_values)

def from_string(string):
    # Split the input string into individual float values
    matrix_values = list(map(float, string.split()))

    # Reshape the list into a 3x3 NumPy array
    matrix = np.array(matrix_values).reshape(3, 3)

    return matrix
"""
def generate_matrices_old(steps):
    R_list = set()
    x_steps = [i * (1/steps) for i in range(steps + 1)]
    y_steps = [i * (1/steps) for i in range(steps + 1)]
    z_steps = [i * (1/steps) for i in range(steps + 1)]
    for x_step in x_steps:
        theta = 2 * math.pi * x_step - math.pi
        for y_step in y_steps:
            psi = math.acos(1-2* y_step) + 0.5 * math.pi
            if y_step < 0.5:
                if psi < math.pi:
                    psi = psi + math.pi
                else:
                    psi = psi - math.pi
            for z_step in z_steps:
                eta = 2 * math.pi * z_step - math.pi
                r = R.from_euler('xyz', [theta, psi, eta], degrees=False)

                #round r 9 decimals and convert to string
                r_string = to_string(r.as_matrix())
                R_list.add(r_string)

    return R_list
"""

def uniform_sampling(n_samples):

    """Generete rotation matrices uniformly.

    # n_samples: the number of samples between -180 and 180
    # n_samples = 36 meaning sampling every 10 degrees (360/n_samples)

    Returns: R_list (list): List of rotation matrices.

    Ref:
    1. Ken Shoemake, "Uniform random rotations", Graphics Gems III, 1992
    2. Kuffner, James J. “Effective sampling and distance metrics for 3D rigid
         body path planning”. IEEE International Conference on Robotics and
         Automation, 4 (2004): 3993-3998.
    """
    R_list = set()
    for s in np.linspace(0,1, n_samples):
        delta1 = np.sqrt(1-s)
        delta2 = np.sqrt(s)

        for t1 in np.linspace(0,1, n_samples):
            theta1 = 2*pi* t1

            for t2 in np.linspace(0,1, n_samples):
                theta2 = 2*pi* t2

                # quaternion Q = (w, x, y, z)
                w = cos(theta2) * delta2
                x = sin(theta1) * delta1
                y = cos(theta1) * delta1
                z = sin(theta2) * delta2

                r = R.from_quat([[w, x, y, z]])

                #round r 9 decimals and convert to string
                r_string = to_string(r.as_matrix())
                R_list.add(r_string)

    print(f"Uniform sampling done. Unique rotations: {len(R_list)}")
    return R_list

#def generate_matrices_(steps):
#
#    """Generete rotation matrices uniformly.
#
#    The number of matrices generated depends on steps, steps of 360 corresponds to 1 Euler angle in each direction.
#    Input: steps (int): the stepsize to iterate over.
#    Returns: R_list (list): List of rotation matrices.
#
#    Ref: Kuffner, James J. “Effective sampling and distance metrics for 3D rigid
#         body path planning”. IEEE International Conference on Robotics and
#         Automation, 4 (2004): 3993-3998.
#    """
#    R_list = set()
#    t = np.arange(0,1+1/steps, 1/steps)
#    Theta = 2 * np.pi * t - np.pi
#    Eta = 2 * np.pi * t - np.pi
#
#    t = np.arange(0,1, 2/steps)
#    psi1 = np.arccos(1-2* t)
#    t = np.arange(0,1+1/steps, 2/steps)
#    psi2 = np.arccos(1-2*t) -2*np.pi
#    Psi = np.concatenate((psi1,psi2))
#
#    for theta in Theta:
#        for eta in Eta:
#            for psi in Psi:
#                r = R.from_euler('xyz', [theta, psi, eta], degrees=False)
#
#                #round r 9 decimals and convert to string
#                r_string = to_string(r.as_matrix())
#                R_list.add(r_string)
#
#    return R_list
#


def calculate_incident_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    return  angle_degrees


def calculate_crossing_angle(v1, v2, ref_v):
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

    dot_product = np.dot(v1, v2)
    angle_radians = np.arccos(dot_product/np.linalg.norm(v1)/np.linalg.norm(v2))

    # apply right-hand rule
    angle_radians = angle_radians if np.dot(np.cross(v1, v2), ref_v) >=0 else - angle_radians

    # convert to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# def orient_vector(vector, reference_point):
#     # Calculate the vector pointing from the reference point to the vector
#     direction_vector = vector - reference_point
#
#     # Calculate the dot product of the direction vector and the vector
#     dot_product = np.dot(direction_vector, vector)
#
#     # Change the sign of the vector based on the dot product
#     oriented_vector = vector if dot_product >= 0 else -vector
#
#     return oriented_vector

def orient_vector(vector, direction_vector):
    # Calculate the dot product of the direction vector and the vector
    dot_product = np.dot(direction_vector, vector)

    # Change the direction of the vector based on direction_vector
    oriented_vector = vector if dot_product >= 0 else -vector
    return oriented_vector


def calc_crossing_incident_angle(xyzCYS_D, xyzCYS_E, xyzMHC, xyzTCR, ref_pointMHC, ref_pointTCR):

    centerMHC = np.mean(np.array(xyzMHC), axis = 0)
    centerTCR = np.mean(np.array(xyzTCR), axis = 0)
    #centerCYS = np.mean(np.array(xyzCYS), axis = 0) # not needed

    v_mhc = ref_pointMHC[0] - centerMHC
    v_tcr = ref_pointTCR[0] - centerTCR
    #v_cys = ref_pointCYS[0] - centerCYS # not needed

    # we do not need a PCA on these four Cys. We can just calcualte the
    # geometric centers of Cys in chain D and E, respectively,
    # and get the subtraction between them.
    #pcaCYS = PCA()
    #pcaCYS.fit(xyzCYS)
    #pca1CYS = pcaCYS.components_[0]
    #pca1CYS = orient_vector(pca1CYS, v_cys)
    Vec_CYS =  np.mean(np.array(xyzCYS_E), axis = 0) - np.mean(np.array(xyzCYS_D), axis = 0) #  a vector between the centroids of the conserved CYS in TCR chains

    pcaMHC = PCA()
    pcaMHC.fit(xyzMHC)
    pca1MHC = orient_vector(pcaMHC.components_[0], v_mhc) #MHC binding groove vector
    pca2MHC = orient_vector(pcaMHC.components_[1], v_mhc) #the waist of the MHC binding groove
    pca3MHC = np.cross(pca1MHC, pca2MHC) #the normal vector of MHC plane

    pcaTCR = PCA()
    pcaTCR.fit(xyzTCR)
    pca2TCR = orient_vector(pcaTCR.components_[1], v_tcr) #TCR inter-domain vector

    crossing_angle_degrees = calculate_crossing_angle(pca1MHC, Vec_CYS, pca3MHC)

    incident_angle_degrees = calculate_incident_angle(pca2TCR, pca3MHC)

    return (crossing_angle_degrees, incident_angle_degrees)

def filter_crossing_incident_angle(R_list, reference_rec, reference_lig, cross_min, cross_max, incident_min, incident_max):
    """Filter a list of rotation matrices based on the incident and crossing angle of a reference structure.

    Input: R_list (list), list of rotation matrices.
    reference (str), path to reference structure.
    cross_min (float), minimum value for crossing angle.
    cross_max (float), maximum value for crossing angle.
    incident_min (float), minimum value for incident angle.
    incident_max (float), maximum value for incident angle.
    Returns: R_list_filtered (list), filtered list with rotation matrices.
    """
    R_list_filtered = set()
    rot_mat_cr_in_list = []

    #get relevant coordinates from reference structure to calculate crossing and incident angle.
    ref_model_rec = pdb2sql(reference_rec)
    ref_model_lig = pdb2sql(reference_lig)
    xyzCYS_D = ref_model_lig.get('x,y,z', chainID = ['D'], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
    xyzCYS_E = ref_model_lig.get('x,y,z', chainID = ['E'], name = ['CA'], resName = ['CYS'], resSeq = [23, 104])
    helix1 = [i for i in range(50, 86)]
    helix2 = [i for i in range(138, 176)]
    xyzMHC = ref_model_rec.get('x,y,z', chainID = ['A'], resSeq = helix1 + helix2, name = ['CA'])
    xyzTCR = ref_model_lig.get('x,y,z', chainID = ['D', 'E'], resSeq = [i for i in range(1, 128)], name = ['CA']) # TCR variable domain

    # ref_points are used to determine the direction of PCs
    ref_pointMHC = ref_model_rec.get('x,y,z', chainID = ['A'], resSeq = ['86']) # a residue at the C-ter end of the MHC alpha1 helix. Is used for determine the direction of PC1_mhc and PC2_mhc
    ref_pointTCR = ref_model_lig.get('x,y,z', chainID = ['D'], resSeq = ['97'], name = ['CA']) # a residue in TCR chain D near the constant domain. Used for determine the direction of PC2_TCR (TCR inter-domain vector)
    #ref_pointCYS = ref_model_lig.get('x,y,z', chainID = ['E'], resSeq = ['86'], name = ['CA']) # not needed

    for i, r_string in enumerate(R_list):
        rot_mat = R.from_matrix(from_string(r_string))
        rot_xyzCYS_D = rot_mat.apply(xyzCYS_D)
        rot_xyzCYS_E = rot_mat.apply(xyzCYS_E)
        rot_xyzTCR = rot_mat.apply(xyzTCR)
        #rot_refCYS = rot_mat.apply(ref_pointCYS) # not needed
        rot_refTCR = rot_mat.apply(ref_pointTCR)
        (crossing_angle, incident_angle) = calc_crossing_incident_angle(rot_xyzCYS_D, rot_xyzCYS_E, xyzMHC ,rot_xyzTCR , ref_pointMHC , rot_refTCR)
        if crossing_angle >= cross_min and crossing_angle <= cross_max and incident_angle >= incident_min and incident_angle <= incident_max:
                        R_list_filtered.add(to_string(rot_mat.as_matrix()))
                        #print("Added ", crossing_angle, incident_angle)
                        rot_mat_cr_in_list.append((rot_mat, crossing_angle, incident_angle, i))
    print(f"Reduced sampling Done. Unique rotations: {len(R_list_filtered)}")
    return R_list_filtered, rot_mat_cr_in_list

def rotate(xyz, rot_mat, inverse = False):
    r = R.from_matrix(rot_mat)
    return r.apply(xyz, inverse)

def write_naive_rotationset(prmFL = 'naive_sampling.prm', n_samples = 60):
    # n_samples: the number of samples between -180 and 180
    # n_samples = 36 meaning sampling every 10 degrees (360/n_samples)
    R_list = set()
    samples = np.linspace(-180, 180, n_samples)

    for x in samples:
        for y in samples:
            for z in samples:
                r = R.from_euler('xyz', [x, y, z], degrees=True)
                r_string = to_string(r.as_matrix())
                R_list.add(r_string)
    print(f"naive sampling Done. Unique rotations: {len(R_list)}")
    write_rotation_matrix(R_list, prmFL)


def main():

    #reference_rec = '/home/jaarts/experiments/generated_rotations_from_angle/input_new/1ao7_l_u.pdb'
    #reference_lig = '/home/jaarts/experiments/generated_rotations_from_angle/input_new/1ao7_r_u.pdb'
    reference_rec = 'ref_pdb/2bnr_l_u.pdb' # rec: pMHC
    reference_lig = 'ref_pdb/2bnr_r_u.pdb' # lig: TCR
    steps = 60
    cross_min = 15
    cross_max = 90
    incident_min = 0
    incident_max = 35
    print(f"steps: {steps}")
    print(f"cross_min = {cross_min}, cross_max={cross_max}, incident_min = {incident_min}, incident_max ={incident_max}")

    # write rotation matrices for uniform sampling
    R_list = uniform_sampling(steps)
    write_rotation_matrix(R_list, f"data/uniform_sampling.prm")

    # write rotation matrices for reduced sampling
    R_list_filtered, rot_mat_cr_in_list = filter_crossing_incident_angle(R_list, reference_rec, reference_lig, cross_min, cross_max, incident_min, incident_max)
    write_rotation_matrix(R_list_filtered, "data/reduced_sampling.prm")

    # write rotation matrices for naive sampling
    write_naive_rotationset('data/naive_sampling.prm', steps)



if __name__ == "__main__":
    main()


"""
    for i, rot_mat in enumerate(R_list):
        rot_xyz_CYS = rot_mat.apply(xyzCYS)
        rot_xyzTCR = rot_mat.apply(xyzTCR)
        rot_refCYS = rot_mat.apply(ref_pointCYS)
        (crossing_angle, incident_angle) = calc_crossing_incident_angle_coordinates(rot_xyz_CYS, xyzMHC ,rot_xyzTCR , ref_point , rot_refCYS)
        if i % 1000 == 0:
            print(i)
        if crossing_angle >= cross_min and crossing_angle <= cross_max and incident_angle >= incident_min and incident_angle <= incident_max:
            R_list_filtered.append(rot_mat.as_matrix())
"""
