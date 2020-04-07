#!/usr/bin/python3

import numpy as np
from scipy import interpolate


def FirstInperlotation(filename, a0, b0, index):
    #Load the impedance file
    Impedance = np.loadtxt(filename)

    #Gets the component for different poisson's ratio.
    K25 = np.transpose(Impedance[:, index[ 0: 5]])
    K35 = np.transpose(Impedance[:, index[ 5:10]])
    K45 = np.transpose(Impedance[:, index[10:15]])

    # Data axes
    x0 = Impedance[:, 0]
    y0 = np.array([0.1, 0.25, 0.50, 1.0, 2.0])

    #Create interpolator 
    Fun_25 = interpolate.interp2d(x0, y0, K25, kind='linear')
    Fun_35 = interpolate.interp2d(x0, y0, K35, kind='linear')
    Fun_45 = interpolate.interp2d(x0, y0, K45, kind='linear')

    #Interpolate the values
    z0 = Fun_25(a0, b0)
    z1 = Fun_35(a0, b0)
    z2 = Fun_45(a0, b0)

    z = np.array([z0, z1, z2])
    
    return z

def SecondInterpolation(b, nu):
    #Solves a linear system.
    A = np.array([[1.0, 0.25, 0.25**2], [1.0, 0.35, 0.35**2], [1.0, 0.45, 0.45**2]]); 
    x = np.linalg.solve(A, b)

    #Interpolate for Foisson's ratio.
    z = x[0] + x[1]*nu + x[2]*nu**2

    return z

def ComputeImpedance(filename, a0, b0, nu):
    #Computes real part
    index = range(1,30,2)
    zp    = FirstInperlotation(filename, a0, b0, index)
    kij   = SecondInterpolation(zp, nu)

    #Computes imaginary part
    index = range(2,31,2)
    zp    = FirstInperlotation(filename, a0, b0, index)
    cij   = SecondInterpolation(zp, nu)

    Impedance = np.array([kij, cij])

    return Impedance


#INPUTS 
hb = 15.0
Tb = 0.50
B  = 40.0
D  = 27.0
Vs = 250
f0 = 2.0
nu = 0.4275

#DIMENSIONLESS VECTOR
PI = np.array([B/Vs/Tb, hb/B, D/B, nu, f0*B/Vs])
print(PI)

#DIMENIONLESS FREQUENCY
a0 = 1.111 #model.predict(PI)

kxx = ComputeImpedance("Normalized_Impedace_Functions/Kxx.txt", a0, PI[2], PI[3])
kxt = ComputeImpedance("Normalized_Impedace_Functions/Kxt.txt", a0, PI[2], PI[3])
ktt = ComputeImpedance("Normalized_Impedace_Functions/Ktt.txt", a0, PI[2], PI[3])

print(kxx)
print(kxt)
print(ktt)

#[[ 0.72240007]
# [ 1.70080473]]
#[[ 0.13824467]
# [ 0.48635204]]
#[[ 1.28825378]
# [ 1.02496305]]
