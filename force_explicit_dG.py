# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:04:56 2017

@author: kaley_000
"""

import numpy as np
import matplotlib
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
    
def Vki_I(k, i, atoms):
    """ Botu fingerprint """
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    for j in range(0, len(atoms)):
        if (j != i):
            r_i = atoms.positions[i]
            r_j = atoms.positions[j]
            
            r_ij = atoms.get_distance(i,j)
            r_kij = r_i[k] - r_j[k]
            for m in range(0,8):
                Vk[m] += r_kij /r_ij * np.exp(-np.square(r_ij/eta[m]))*f_Botu(r_ij)
    return Vk

def Vki_II(k, i, atoms):
    """ fingerprint based on deriv of G^II """
    Rc = 8
    Vk = np.zeros(shape=8)
    eta = np.logspace(-1,2,num=8)
    #lam = np.logspace(-2,-1,num=8)
    #zeta = np.logspace(-2,-1,num=8)
    lam = 0.1*np.ones(shape=8)
    zeta = 0.1*np.ones(shape=8)
    """ lambda and zeta values??? """
    for j in range(0,len(atoms)):
        if (j != i):
            for l in range(j+1,len(atoms)):
                if l != i:
                    r_i = atoms.positions[i]
                    r_j = atoms.positions[j]
                    r_l = atoms.positions[l]
            
                    r_ij = atoms.get_distance(i,j)
                    r_il = atoms.get_distance(i,l)
                
                    cosTheta = np.dot(r_i-r_j,r_i-r_l)/(r_ij*r_il)
                    rk_ij = r_i[k] - r_j[k] 
                    rk_il = r_i[k] - r_l[k] 
                    
                    
                    for m in range(0,8):
                        Vk[m] += (1+lam[m]*cosTheta)**(zeta[m]-1)*\
                                np.exp(-eta[m]*(r_ij**2 + r_il**2)/Rc)*\
                                (lam[m]*zeta[m]*\
                                (rk_ij*(r_ij*r_il-r_il**2*cosTheta)\
                                + rk_il*(r_ij*r_il-r_ij**2*cosTheta))/(r_ij + r_il)**2\
                                - 2*eta[m]*(rk_ij + rk_il)*(1+lam[m]*cosTheta)/Rc)\
                                *f_Botu(r_ij)**f_Botu(r_il)
                    
    for m in range(0,8):
        Vk[m] = Vk[m]*2**(1-zeta[m])
        
    return Vk                   

def arrayFingerprintForces(atom1,atomslist):
    """ makes an array of tuples with (1) fingerprint, (2) forces """
    arrayFF = []
    
    for atoms in atomslist:
        symbs = atoms.get_chemical_symbols()
        forces = atoms.get_forces()
        
        for i in range(0,len(symbs)):
            if symbs[i] == atom1:
                fingerprint = np.zeros(shape=(3,8*2))
                for k in range(0,3):
                    fingerprintI = Vki_I(k,i,atoms)
                    fingerprintII = Vki_II(k,i,atoms)
                    fingerTotal = np.zeros(shape=8*2)
                    for j in range(0,8):
                        fingerTotal[j] = fingerprintI[j]
                        fingerTotal[j+8] = fingerprintII[j]
                    fingerprint[k] = fingerTotal
                arrayFF.append((fingerprint,forces[i]))
                
    return arrayFF

def putInXY(tupleData):
    """ 
    turns an array of tuples (fingerprint, forces) into x and y arrays 
    x = n by 24 array; x[i] = fingerprint for all three forces for atom i
    y = n by 3 array; y[i] = all three forces on atom i
    """
    n = len(tupleData)
    x = np.zeros(shape=(n,3*2*8))
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


""" --------------------- begin script --------------------- """

""" 
to train/test:
    O = 1
    H = 2
    Pt = 3
"""
trainAtom = 3


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
    testO = arrayFingerprintForces('O',testH2O)
    xT, yT = putInXY(testO)
    atomName = 'Oxygen'

if trainAtom == 2:
    """ hydrogen """
    trainH = arrayFingerprintForces('H',trainH2O)
    x, y = putInXY(trainH)
    testH = arrayFingerprintForces('H',testH2O)
    xT, yT = putInXY(testH)
    atomName = 'Hydrogen'
    
if trainAtom == 3:
    """ platinum """
    trainPt = arrayFingerprintForces('Pt',trainPts)
    x, y = putInXY(trainPt)
    testPt = arrayFingerprintForces('Pt',testPts)
    xT, yT = putInXY(testPt)
    atomName = 'Platinum'


""" making kernel and gaussian process objects """
kernel = RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

""" training and testing """
gp.fit(x, y)
y_pred, sigma = gp.predict(xT, return_std=True)

n = len(y_pred)
yP_x, yP_y, yP_z = splitUpXYZ(y_pred)
yT_x, yT_y, yT_z = splitUpXYZ(yT)
y_x, y_y, y_z = splitUpXYZ(y)

fz_dG = yP_z
fy_dG = yP_y
fx_dG = yP_x
#%%

# to plot:
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

errdG = abs((P-T)/T)
print('x errors')
print("%.4f" % np.mean(abs((yP_x-yT_x))))
print("%.5f" % np.median(abs((yP_x-yT_x))))
print('y errors')
print("%.4f" % np.mean(abs((yP_y-yT_y))))
print("%.5f" % np.median(abs((yP_y-yT_y))))
print('z errors')
print("%.4f" % np.mean(abs((yP_z-yT_z))))
print("%.5f" % np.median(abs((yP_z-yT_z))))

plotBool = False
parityBool = True
plotErrs = False


matplotlib.rcParams.update({'font.size': 18, 'figure.autolayout': True})

if plotBool:
    """ plot predicted and calculated """
    fig = plt.figure(figsize=(11, 20))
    a = fig.add_subplot(2,1,1)
    a.errorbar(x=range(0,n),y=P, yerr=sigma, ecolor='g', 
               capsize=7, elinewidth=2, label="Predicted", linestyle='', marker='o')
    a.plot(range(0,n),T,'ro', label="Calculated")
    #a.plot(range(0,len(yO)),y_x,'go',label="Training Data")

    a.set_title('dG: Predicted and Calculated %s for %s' % (direcName, atomName))
    a.set_xlabel('Image Number')
    a.set_ylabel('Force (eV/Angstrom)')

    handles, labels = a.get_legend_handles_labels()
    a.legend(handles, labels)

    """ plot errors """
    b = fig.add_subplot(2,1,2)
    b.plot(range(0,n), errdG, linestyle='--', marker='o',label="dG")
    b.set_yscale('log')
    
    b.set_title('dG: Relative Error b/w Predicted and Calculated %s for %s' % (direcName, atomName))
    b.set_xlabel('Image Number')
    b.set_ylabel('Relative Error = |pred - calc|/|calc|')


if plotErrs:
    """ parity plot for all three schemes """
    fig = plt.figure(figsize=(10,30))
    a = fig.add_subplot(3,1,1)
    a.plot(yT_x,fx_MOI,linestyle='', marker='^',color='g',label="MOI",markersize=8)
    a.plot(yT_x,fx_Botu,linestyle='', marker='*',color='m',label="Botu",markersize=12)
    a.plot(yT_x,fx_dG,linestyle='', marker='o',label="dG",markersize=6)
    a.plot(yT_x,yT_x)
    
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles, labels)
    
    a.set_title('Parity Plot: F_x on Oxygen')
    a.set_xlabel('Calculated Force (eV/Angstrom)')
    a.set_ylabel('Predicted Force (eV/Angstrom)')
    
    # y component
    b = fig.add_subplot(3,1,2)
    b.plot(yT_y,fy_MOI,linestyle='', marker='^',color='g',label="MOI",markersize=8)
    b.plot(yT_y,fy_Botu,linestyle='', marker='*',color='m',label="Botu",markersize=12)
    b.plot(yT_y,fy_dG,linestyle='', marker='o',label="dG",markersize=6)
    b.plot(yT_y,yT_y)
    
    handles, labels = a.get_legend_handles_labels()
    b.legend(handles, labels)
    
    b.set_title('Parity Plot: F_y on Oxygen')
    b.set_xlabel('Calculated Force (eV/Angstrom)')
    b.set_ylabel('Predicted Force (eV/Angstrom)')
    
    # z component
    c = fig.add_subplot(3,1,3)
    c.plot(yT_z,fz_MOI,linestyle='', marker='^',color='g',label="MOI",markersize=8)
    c.plot(yT_z,fz_Botu,linestyle='', marker='*',color='m',label="Botu",markersize=12)
    c.plot(yT_z,fz_dG,linestyle='', marker='o',label="dG",markersize=6)
    c.plot(yT_z,yT_z)
    
    handles, labels = a.get_legend_handles_labels()
    c.legend(handles, labels)
    
    c.set_title('Parity Plot: F_z on Hydrogen')
    c.set_xlabel('Calculated Force (eV/Angstrom)')
    c.set_ylabel('Predicted Force (eV/Angstrom)')
    
#    """ plot of relative errors for different schemes """
#    fig = plt.figure(figsize=(15,10))
#    b = fig.add_subplot(1,1,1)
#    b.set_yscale('log')
#    plt.plot(range(0,n), errdG, linestyle='', marker='o',label="dG",color='m',markersize=9)
#    b.plot(range(0,n), errMOI, linestyle='', marker='^',color='g',label="MOI",markersize=10)
#    b.plot(range(0,n), errBotu, linestyle='', marker='*',label="Botu",markersize=14)
#
#    handles, labels = b.get_legend_handles_labels()
#    b.legend(handles, labels)
#    b.set_title('Relative Error b/w Predicted and Calculated F_z for Platinum')
#    
#    b.set_xlabel('Atom Number')
#    b.set_ylabel('Relative Error = |pred - calc|/|calc|')

    
if parityBool:
    """ makes a parity plot """
    fig = plt.figure(figsize=(10,10))
    a = fig.add_subplot(1,1,1)
    a.plot(T,T,'-r')
    a.plot(T,P,linestyle='',marker='o')
    
    a.set_title('dG Parity Plot: %s on %s' % (direcName, atomName))
    a.set_xlabel('Calculated Force (eV/Angstrom)')
    a.set_ylabel('Predicted Force (eV/Angstrom)')

#plt.savefig('Pt_Fz_dG.png')
#plt.savefig('dG_parity_Pt_Fy.png')
#plt.savefig('parity_H.png')