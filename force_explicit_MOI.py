# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:54:24 2017

@author: kaley_000
"""

import numpy as np
from matplotlib import pyplot as plt

from ase import Atoms, Atom
from ase.io import read

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def get_moments_of_inertia(positions, masses, center, vectors=False):
    """Get the moments of inertia along the principal axes.

    The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor. Periodic boundary
    conditions are ignored. Units of the moments of inertia are
    amu*angstrom**2.

    atoms: ASE Atoms object
    center: A list of three floats representing the center for calculating
    moments of inertia.
    vectors: If True, will return the eigenvectors also
    (along with the eigenvalues).
    """
    #positions = atoms.get_positions()
#    com = atoms.get_center_of_mass()
#    positions -= com  # translate center of mass to origin
    center = np.array(center)
    positions -= center  # translate center
    #masses = atoms.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(masses)):
        x, y, z = positions[i]
        m = masses[i]

        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    I = np.array([[I11, I12, I13],
                  [I12, I22, I23],
                  [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(I)
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals


def transform_forces(pdirections, forces):
    """
    Transforms forces from one coordinate basis to the other.

    pdirections: Is a list of three vectors, each vector is a list of
    three floats corresponding to a single principle direction. forces is
    a vector of three floats representing the forces in the three directions.
    This function will return the force vector in the basis coordinates of the
    principle directions.
    forces: The 3*3 matrix of forces in the original coordinate system.
    """

    transformed_forces = np.zeros(shape=3)

    transformed_force = forces[0] * np.dot([1, 0, 0], pdirections[0])
    transformed_force += forces[1] * np.dot([0, 1, 0], pdirections[0])
    transformed_force += forces[2] * np.dot([0, 0, 1], pdirections[0])
    transformed_forces[0] = transformed_force

    transformed_force = forces[0] * np.dot([1, 0, 0], pdirections[1])
    transformed_force += forces[1] * np.dot([0, 1, 0], pdirections[1])
    transformed_force += forces[2] * np.dot([0, 0, 1], pdirections[1])
    transformed_forces[1] = transformed_force

    transformed_force = forces[0] * np.dot([1, 0, 0], pdirections[2])
    transformed_force += forces[1] * np.dot([0, 1, 0], pdirections[2])
    transformed_force += forces[2] * np.dot([0, 0, 1], pdirections[2])
    transformed_forces[2] = transformed_force

    return transformed_forces


def calculate_image_center(atoms, watchindex, pdirs):
    """Calculates the center of the image in the pdir basis coordinates."""

    # The first moment of the image around the center atom is calculated.
    atoms = atoms.copy()
    from ase.calculators.neighborlist import NeighborList
    _nl = NeighborList(cutoffs=([6.5 / 2.] * len(atoms)),
                       self_interaction=False,
                       bothways=True,
                       skin=0.)
    _nl.update(atoms)

    position = atoms.positions[watchindex]

    # Step 1: Calculating neighbors of atom.
    n_indices, n_offsets = _nl.get_neighbors(watchindex)
    Rs = [atoms.positions[n_index] +
          np.dot(n_offset, atoms.get_cell()) - position
          for n_index, n_offset in zip(n_indices, n_offsets)]

    xtilde = 0.
    ytilde = 0.
    ztilde = 0.
    for rs in Rs:
        xtilde += rs[0] * np.dot([1, 0, 0], pdirs[0])
        xtilde += rs[1] * np.dot([0, 1, 0], pdirs[0])
        xtilde += rs[2] * np.dot([0, 0, 1], pdirs[0])
        ytilde += rs[0] * np.dot([1, 0, 0], pdirs[1])
        ytilde += rs[1] * np.dot([0, 1, 0], pdirs[1])
        ytilde += rs[2] * np.dot([0, 0, 1], pdirs[1])
        ztilde += rs[0] * np.dot([1, 0, 0], pdirs[2])
        ztilde += rs[1] * np.dot([0, 1, 0], pdirs[2])
        ztilde += rs[2] * np.dot([0, 0, 1], pdirs[2])

    return xtilde, ytilde, ztilde


def forcesIntoMOIBasis(atoms, index):
    forces = atoms.get_forces()
    positions = atoms.get_positions()

    # find MOIs
    pvalues, pdirs = get_moments_of_inertia(positions=positions,
                                            masses=atoms.get_masses(),
                                            center=positions[index],
                                            vectors=True)

    # Normalizing the vectors
    pdirs[0] /= np.linalg.norm(pdirs[0])
    pdirs[1] /= np.linalg.norm(pdirs[1])
    pdirs[2] /= np.linalg.norm(pdirs[2])
    
    # Calculating the image center in the pdirs coordinate basis.
    xtilde, ytilde, ztilde = calculate_image_center(atoms=atoms,
                                                    watchindex=index,
                                                    pdirs=pdirs)

    # If the first moment is negative, then the principle direction is flipped.
    # We want the first moment to be positive in all three directions.
    if xtilde < 0.:
        pdirs[0] *= -1.
    if ytilde < 0.:
        pdirs[1] *= -1.
    if ztilde < 0.:
        pdirs[2] *= -1.

    # Forces on atom index 'index' are transformed into the principle directions.
    transformed_forces = transform_forces(pdirections=pdirs,
                                          forces=forces[index])
    
    return transformed_forces, pdirs

def forcesIntoXYZBasis(positions, forces, index):
    forces = 23
    return forces



def f_Botu(r):
    """ Botu damping function """
    Rc = 8
    if r < Rc:
        return 0.5*(np.cos(np.pi * r / Rc) + 1)
    else:
        return 0
    
def Vki_MOI(k, i, symb, atoms, pdirs):
    """ Botu fingerprint """
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    #symbs = atoms.get_chemical_symbols()
    for j in range(0, len(atoms)):
        if (j != i):# and (symbs[j] == symb):
            r_i = atoms.positions[i]
            r_j = atoms.positions[j]
            
            r_i_tilde = transform_forces(pdirections=pdirs,
                                         forces=r_i)
            r_j_tilde = transform_forces(pdirections=pdirs,
                                         forces=r_j)
            
            r_ij = atoms.get_distance(i,j)
            r_kij = r_i_tilde[k] - r_j_tilde[k]
            # ^ r_i - r_j or other way around? absolute value??
            for m in range(0,8):
                Vk[m] += r_kij / r_ij * np.exp(-np.square(r_ij/eta[m]))*f_Botu(r_ij)
    return Vk

def arrayFingerprintForces(atom1,atom2,atomslist,returnPDirs=False):
    """ makes an array of tuples with (1) fingerprint, (2) forces """
    arrayFF = []
    arrayPDirs = []
    
    for atoms in atomslist:
        symbs = atoms.get_chemical_symbols()
        
        for i in range(0,len(symbs)):
            if symbs[i] == atom1:
                forceMOI, pdirs = forcesIntoMOIBasis(atoms,i)
                fingerprint = np.zeros(shape=(3,8))
                for k in range(0,3):
                    fingerprint[k] = Vki_MOI(k=k,i=i,symb=atom2,atoms=atoms,pdirs=pdirs)
                arrayFF.append((fingerprint,forceMOI))
                arrayPDirs.append(pdirs)
    
    if returnPDirs:
        return arrayFF, arrayPDirs
    else:
        return arrayFF

def putInXY(tupleData):
    """ turns an array of tuples (fingerprint, forces) into x and y arrays """
    n = len(tupleData)
    x = np.zeros(shape=(n,3*8))
    y = np.zeros(shape=(n,3))
    
    for i in range(0,n):
        x[i] = tupleData[i][0].flatten()
        y[i] = tupleData[i][1].flatten()
    
    return x, y

def splitUpXYZ(y):
    """ 
    splits up an array into the x, y, and z components and returns three 
    arrays (one for each component)
    """
    n = len(y)
    y_x = np.zeros(shape=n)
    y_y = np.zeros(shape=n)
    y_z = np.zeros(shape=n)
    for i in range(0,n):
        y_x[i] = y[i][0]
        y_y[i] = y[i][1]
        y_z[i] = y[i][2]
    
    return y_x, y_y, y_z

""" begin script """

# number of images in training set
n = 100

# placing all training images into a list of images
trainH2O = [None] * n
for i in range(0, n):
    # training data: every other image from total data
    trainH2O[i] = read('water.extxyz', 50+i*2)
    
# placing all testing images into a list of images
testH2O = [None] * (n - 1)
for i in range(0, n - 1):
    # testing data: images not included in training data
   testH2O[i] = read('water.extxyz', 50+i*2+1)
   
""" making fingerprints """

#""" oxygen (from hydrogen) """
#trainO = arrayFingerprintForces('O','H',trainH2O)
#xO, yO = putInXY(trainO)
#testO, pDirsO = arrayFingerprintForces('O','H',testH2O,returnPDirs=True)
#xTestO, yTestO = putInXY(testO)
#
#""" hydrogen 2 """
#trainH = arrayFingerprintForces('H','H',trainH2O)
#xH, yH = putInXY(trainH)
#testH, pDirsH = arrayFingerprintForces('H','H',testH2O,returnPDirs=True)
#xTestH, yTestH = putInXY(testH)

""" platinum """
# number of images in training set
n = 20

# placing all training images into a list of images
trainPts = [None] * n
for i in range(0, n):
    # training data: every other image from total data
    trainPts[i] = read('defect-trajectory.extxyz', 50+i*2)
    
# placing all testing images into a list of images
testPts = [None] * (n - 1)
for i in range(0, n - 1):
    # testing data: images not included in training data
   testPts[i] = read('defect-trajectory.extxyz', 50+i*2+1)

trainPt = arrayFingerprintForces('Pt','Pt',trainPts)
xPt, yPt = putInXY(trainPt)
testPt, pDirsPt = arrayFingerprintForces('Pt','Pt',testPts,returnPDirs=True)
xTestPt, yTestPt = putInXY(testPt)

# all the data in stored in tuples
# each tuple has (1) fingerprints and (2) the forces

""" making kernel and gaussian process objects """
kernel = RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)


#""" train O """
#x, y = xO, yO
#xT, yT = xTestO, yTestO
#pDirs = pDirsO

#""" train H """
#x, y = xH, yH
#xT, yT = xTestH, yTestH
#pDirs = pDirsH

""" train Pt """
x, y = xPt, yPt
xT, yT = xTestPt, yTestPt
pDirs = pDirsPt


""" training """
gp.fit(x, y)
y_pred, sigma = gp.predict(xT, return_std=True)

n = len(y_pred)
yP_1, yP_2, yP_3 = splitUpXYZ(y_pred)
yT_1, yT_2, yT_3 = splitUpXYZ(yT)

""" transforming forces back into xyz basis """
y_xyz = np.zeros(shape=(n,3))
y_Txyz = np.zeros(shape=(n,3))
for i in range(0,n):
    R = np.linalg.inv(pDirs[i])
    y_xyz[i] = transform_forces(pdirections=R, forces=y_pred[i])
    y_Txyz[i] = transform_forces(pdirections=R, forces=yT[i])

yP_x, yP_y, yP_z = splitUpXYZ(y_xyz)
yT_x, yT_y, yT_z = splitUpXYZ(y_Txyz)

#%%

""" to plot: """
P = yP_z
T = yT_z

errMOI = abs((P-T)/T)
print(np.mean(errMOI))

matplotlib.rcParams.update({'font.size': 18, 'figure.autolayout': True})

""" plot predicted and calculated """
fig = plt.figure(figsize=(11, 20))
a = fig.add_subplot(2,1,1)
a.errorbar(x=range(0,n),y=P, yerr=sigma, ecolor='g', 
           capsize=7, elinewidth=2, label="Predicted", linestyle='', marker='o')
a.plot(range(0,n),T,'ro', label="Calculated")
#a.plot(range(0,n),Botu,'go', label="Botu")

a.set_title('MOI: Predicted and Calculated F_x for Platinum')
a.set_xlabel('Image Number')
a.set_ylabel('Force (eV/Angstrom)')

handles, labels = a.get_legend_handles_labels()
a.legend(handles, labels)

""" plot errors """
b = fig.add_subplot(2,1,2)
b.plot(range(0,n), errMOI, linestyle='--', marker='o')
b.set_yscale('log')

b.set_title('MOI: Relative Error b/w Predicted and Calculated F_x for Platinum')
b.set_xlabel('Image Number')
b.set_ylabel('Relative Error = |pred - calc|/|calc|')


#plt.savefig('Pt_Fx_MOI.png')
