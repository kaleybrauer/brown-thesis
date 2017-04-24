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


def f_Botu(r):
    """ Botu damping function """
    Rc = 8
    if r < Rc:
        return 0.5*(np.cos(np.pi * r / Rc) + 1)
    else:
        return 0
    
def Vki_Botu(k, i, symb, atoms):
    """ Botu fingerprint """
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    #symbs = atoms.get_chemical_symbols()
    for j in range(0, len(atoms)):
        if (j != i):# and (symbs[j] == symb):
            r_i = atoms.positions[i]
            r_j = atoms.positions[j]
            
            r_ij = atoms.get_distance(i,j)
            r_kij = r_i[k] - r_j[k]
            # ^ r_i - r_j or other way around? absolute value??
            for m in range(0,8):
                Vk[m] += r_kij / r_ij * np.exp(-np.square(r_ij/eta[m]))*f_Botu(r_ij)
    return Vk

def Vki_dG(k, i, atoms):
    """ fingerprint based on deriv of G^II """
    Rc = 8
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    lam = np.logspace(-1,2,num=8)
    zeta = np.logspace(-1,2,num=8)
    for j in range(0,len(atoms)):
        if (j != i):
            for l in range(0,len(atoms)):
                if l != i and l != j:
                    r_i = atoms.positions[i]
                    r_j = atoms.positions[j]
                    r_l = atoms.positions[l]
            
                    r_ij = atoms.get_distance(i,j)
                    r_il = atoms.get_distance(i,l)
                    r_jl = atoms.get_distance(j,l)
                    
                    cosTheta = np.dot(r_i-r_j,r_i,r_l)/(r_ij*r_il)
                    rk_ij = r_i[k] - r_j[k] 
                    rk_il = r_i[k] - r_l[k] 
                    
                    for m in range(0,8):
                        Vk[m] += (1+lam[m]*cosTheta)**(zeta[m]-1)*\
                                np.exp(-eta[m]*(r_ij**2 + r_il**2 + r_jl**2)/Rc)*\
                                (lam[m]*zeta[m]*\
                                (rk_ij*(r_ij*r_il-r_il**2*cosTheta)\
                                + rk_il*(r_ij*r_il-r_ij**2*cosTheta))/(r_ij + r_il)**2\
                                - 2*eta[m]/Rc*(rk_ij + rk_il)*(1+lam[m]*cosTheta))
    return Vk            
            

def arrayFingerprintForces(atom1,atom2,atomslist):
    """ makes an array of tuples with (1) fingerprint, (2) forces """
    arrayFF = []
    
    for atoms in atomslist:
        symbs = atoms.get_chemical_symbols()
        forces = atoms.get_forces()
        
        for i in range(0,len(symbs)):
            if symbs[i] == atom1:
                fingerprint = np.zeros(shape=(3,8))
                for k in range(0,3):
                    fingerprint[k] = Vki_Botu(k,i,atom2,atoms)
                arrayFF.append((fingerprint,forces[i]))
                
    return arrayFF

def putInXY(tupleData):
    """ 
    turns an array of tuples (fingerprint, forces) into x and y arrays 
    x = n by 24 array; x[i] = fingerprint for all three forces for atom i
    y = n by 3 array; y[i] = all three forces on atom i
    """
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
   
""" beginning training """

""" oxygen (from hydrogen) """
trainO = arrayFingerprintForces('O','H',trainH2O)
xO, yO = putInXY(trainO)
testO = arrayFingerprintForces('O','H',testH2O)
xTestO, yTestO = putInXY(testO)

#""" hydrogen 1 """
#trainHO = arrayFingerprintForces('H','O',trainH2O)
#xHO, yHO = putInXY(trainHO)
#testHO = arrayFingerprintForces('H','O',testH2O)
#xTestHO, yTestHO = putInXY(testHO)

""" hydrogen 2 """
trainHH = arrayFingerprintForces('H','H',trainH2O)
xHH, yHH = putInXY(trainHH)
testHH = arrayFingerprintForces('H','H',testH2O)
xTestHH, yTestHH = putInXY(testHH)

# all the data in stored in tuples
# each tuple has (1) fingerprints and (2) the forces

""" making kernel and gaussian process objects """
kernel = RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

""" training oxygen """
gp.fit(xHH, yHH)
y_pred, sigma = gp.predict(xTestHH, return_std=True)

n = len(y_pred)
yP_x, yP_y, yP_z = splitUpXYZ(y_pred)
yT_x, yT_y, yT_z = splitUpXYZ(yTestHH)
y_x, y_y, y_z = splitUpXYZ(yO)

#%%

# to plot:
P = yP_z
T = yT_z

errBotu = abs((P-T)/T)

matplotlib.rcParams.update({'font.size': 18, 'figure.autolayout': True})

""" plot predicted and calculated """
fig = plt.figure(figsize=(11, 20))
a = fig.add_subplot(2,1,1)
a.errorbar(x=range(0,n),y=P, yerr=sigma, ecolor='g', 
           capsize=7, elinewidth=2, label="Predicted", linestyle='--', marker='o')
a.plot(range(0,n),T,'ro', label="Calculated")
#a.plot(range(0,len(yO)),y_y,'go',label="Training Data")

a.set_title('Botu: Predicted and Calculated F_z for Hydrogen')
a.set_xlabel('Image Number')
a.set_ylabel('Force (eV/Angstrom)')

handles, labels = a.get_legend_handles_labels()
a.legend(handles, labels)

""" plot errors """
b = fig.add_subplot(2,1,2)
b.plot(range(0,n), errBotu, linestyle='--', marker='o')
b.set_yscale('log')

b.set_title('Botu: Relative Error b/w Predicted and Calculated F_z for Hydrogen')
b.set_xlabel('Image Number')
b.set_ylabel('Relative Error = |pred - calc|/|calc|')

#b.plot([i[15] for i in xTestO],P,'o')

#plt.savefig('H_Fz_Botu3.png')
#plt.savefig('botu3_traintest_firsthalf.png')
