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
        mUndamped = masses[i]
        r = np.sqrt(x**2 + y**2 + z**2)
        m = mUndamped * f_c(r)

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
    """  
    transforms the forces on atom number [index] from the xyz basis 
    into the MOI basis
    
    atoms: list of atoms
    index: the atom number for which the forces will be transformed
    """
    
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
    
    # Calculating the center of mass in the pdirs coordinate basis.
    xtilde, ytilde, ztilde = calculate_image_center(atoms=atoms,
                                                    watchindex=index,
                                                    pdirs=pdirs)

    # If the coordinates of the center of mass are negative, 
    # then the principle direction is flipped.
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

def f_c(r):
    """ damping function """
    Rc = 8
    if r < Rc:
        return 0.5*(np.cos(np.pi * r / Rc) + 1)
    else:
        return 0
    
def Vki_MOI(k, i, atoms, pdirs):
    """ Botu fingerprint """
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    for j in range(0, len(atoms)):
        if (j != i):
            r_i = atoms.positions[i]
            r_j = atoms.positions[j]
            
            # transform the distances into MOI basis
            r_i_tilde = transform_forces(pdirections=pdirs,
                                         forces=r_i)
            r_j_tilde = transform_forces(pdirections=pdirs,
                                         forces=r_j)
            
            r_ij = atoms.get_distance(i,j)
            r_kij = r_i_tilde[k] - r_j_tilde[k]
            for m in range(0,8):
                Vk[m] += r_kij / r_ij * np.exp(-np.square(r_ij/eta[m]))*f_c(r_ij)
    return Vk

def arrayFingerprintForces(atom1,atomslist,returnPDirs=False):
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
                    fingerprint[k] = Vki_MOI(k=k,i=i,atoms=atoms,pdirs=pdirs)
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

def splitInto12(y):
    n = len(y)
    nSmall = np.int(n/12)
    a1 = np.zeros(shape=nSmall)
    a2 = np.zeros(shape=nSmall)
    a3 = np.zeros(shape=nSmall)
    a4 = np.zeros(shape=nSmall)
    a5 = np.zeros(shape=nSmall)
    a6 = np.zeros(shape=nSmall)
    a7 = np.zeros(shape=nSmall)
    a8 = np.zeros(shape=nSmall)
    a9 = np.zeros(shape=nSmall)
    a10 = np.zeros(shape=nSmall)
    a11 = np.zeros(shape=nSmall)
    a12 = np.zeros(shape=nSmall)
    for i in range(0,nSmall):
        j = i*12
        a1[i] = y[j]
        a2[i] = y[j+1]
        a3[i] = y[j+2]
        a4[i] = y[j+3]
        a5[i] = y[j+4]
        a6[i] = y[j+5]
        a7[i] = y[j+6]
        a8[i] = y[j+7]
        a9[i] = y[j+8]
        a10[i] = y[j+9]
        a11[i] = y[j+10]
        a12[i] = y[j+11]
    return a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12

""" begin script """

""" 
to train/test:
    O = 1
    H = 2
    Pt = 3
"""
trainAtom = 2


""" placing data in lists """

if trainAtom == 1 or trainAtom == 2:
    # number of images in training set
    n = 50
    # number of images in testing set
    nT = 100
    
    # placing all training images into a list of images
    trainH2O = [None] * n
    for i in range(0, n):
         # training data: every other image from total data
         trainH2O[i] = read('water.extxyz', 50+4*i)
    
    # placing all testing images into a list of images
    testH2O = [None] * nT
    for i in range(0, nT):
        # testing data: images not included in training data
        testH2O[i] = read('water.extxyz', 50+2*i+1)
        
if trainAtom == 3:
    # number of images in training set
    n = 5
    # number of images in testing set
    nT = 20

    # placing all training images into a list of images
    trainPts = [None] * n
    for i in range(0, n):
        # training data: every other image from total data
        trainPts[i] = read('defect-trajectory.extxyz', 1000+i*8)
    
    # placing all testing images into a list of images
    testPts = [None] * (nT)
    for i in range(0, nT):
        # testing data: images not included in training data
        testPts[i] = read('defect-trajectory.extxyz', 1000+i*2+1)


""" fingerprinting """

if trainAtom == 1:
    """ oxygen """
    trainO = arrayFingerprintForces('O',trainH2O)
    x, y = putInXY(trainO)
    testO, pDirs = arrayFingerprintForces('O',testH2O,returnPDirs=True)
    xT, yT = putInXY(testO)
    atomName = 'Oxygen'

if trainAtom == 2:
    """ hydrogen """
    trainH = arrayFingerprintForces('H',trainH2O)
    x, y = putInXY(trainH)
    testH, pDirs = arrayFingerprintForces('H',testH2O,returnPDirs=True)
    xT, yT = putInXY(testH)
    atomName = 'Hydrogen'
    
if trainAtom == 3:
    """ platinum """
    trainPt = arrayFingerprintForces('Pt',trainPts)
    x, y = putInXY(trainPt)
    testPt, pDirs = arrayFingerprintForces('Pt',testPts,returnPDirs=True)
    xT, yT = putInXY(testPt)
    atomName = 'Platinum'


""" making kernel and gaussian process objects """
kernel = RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

""" training and predicting """
gp.fit(x, y)
y_pred, sigma = gp.predict(xT, return_std=True)

n = len(y_pred)
#yP_1, yP_2, yP_3 = splitUpXYZ(y_pred)
#yT_1, yT_2, yT_3 = splitUpXYZ(yT)

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
""" plotDirec: 1 = x, 2 = y, 3 = z """
plotDirec = 3

if plotDirec == 1:
    P = yP_z
    T = yT_z
    direcName = 'F_x'
if plotDirec == 2:
    P = yP_y
    T = yT_y
    direcName = 'F_y' 
if plotDirec == 3:
    P = yP_z
    T = yT_z
    direcName = 'F_z' 

errMOI = abs((P-T)/T)
print('x errors')
print("%.4f" % np.mean(abs((yP_x-yT_x))))
print("%.5f" % np.median(abs((yP_x-yT_x))))
print('y errors')
print("%.4f" % np.mean(abs((yP_y-yT_y))))
print("%.5f" % np.median(abs((yP_y-yT_y))))
print('z errors')
print("%.4f" % np.mean(abs((yP_z-yT_z))))
print("%.5f" % np.median(abs((yP_z-yT_z))))

fz_MOI = yP_z
fy_MOI = yP_y
fx_MOI = yP_x

if trainAtom == 3:
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = splitInto12(P)
    a1T, a2T, a3T, a4T, a5T, a6T, a7T, a8T, a9T, a10T, a11T, a12T = splitInto12(T)

plotBool = False
parityBool = True

matplotlib.rcParams.update({'font.size': 18, 'figure.autolayout': True})

if plotBool:
    """ plot predicted and calculated """
    fig = plt.figure(figsize=(11, 20))
    a = fig.add_subplot(2,1,1)
    a.errorbar(x=range(0,n),y=P, yerr=sigma, ecolor='g', 
               capsize=7, elinewidth=2, label="Predicted", linestyle='', marker='o')
    a.plot(range(0,n),T,'ro', label="Calculated")
    #a.plot(range(0,n),Botu,'go', label="Botu")

    a.set_title('MOI: Predicted and Calculated %s for %s' % (direcName, atomName))
    a.set_xlabel('Atom Number')
    a.set_ylabel('Force (eV/Angstrom)')

    handles, labels = a.get_legend_handles_labels()
    a.legend(handles, labels)

    """ plot errors """
    b = fig.add_subplot(2,1,2)
    b.plot(range(0,n), errMOI, linestyle='--', marker='o')
    b.set_yscale('log')
    
    b.set_title('MOI: Relative Error b/w Predicted and Calculated %s on %s' % (direcName, atomName))
    b.set_xlabel('Atom Number')
    b.set_ylabel('Relative Error = |pred - calc|/|calc|')
    
if parityBool:
    """ makes a parity plot """
    fig = plt.figure(figsize=(10,10))
    a = fig.add_subplot(1,1,1)
    a.plot(T,T,'-r')
    a.plot(T,P,linestyle='',marker='o')
    
    a.set_title('MOI Parity Plot: %s on %s' % (direcName, atomName))
    a.set_xlabel('Calculated Force (eV/Angstrom)')
    a.set_ylabel('Predicted Force (eV/Angstrom)')
#    a.plot(a1T,a1,'go')
#    a.plot(a2T,a2,'ro')
#    a.plot(a3T,a3,'bo')
#    a.plot(a4T,a4,'mo')
#    a.plot(a5T,a5,'yo')
#    a.plot(a6T,a6,'bo')
#    a.plot(a7T,a7,'co')
#    a.plot(a8T,a8,marker='o',color='0.5',linestyle='')
#    a.plot(a9T,a9,marker='o',color='#ffc0cb',linestyle='')
#    a.plot(a10T,a10,marker='o',color='#11a51b',linestyle='')
#    a.plot(a12T,a12,marker='o',color='#11dbad',linestyle='')
#    a.plot(a11T,a11,marker='o',color='k',linestyle='')

#plt.savefig('MOI_parity_Pt_Fy.png')
#plt.savefig('Pt_Fx_MOI.png')
