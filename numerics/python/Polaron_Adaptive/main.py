"""
Edited by: Eugen Dizer
Last modified: 27.03.2023

Welcome to main.py! This program is the executable file to calculate
Real-time propagators after a chosen number of iterations.
"""

# Load needed modules
import adaptive
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.interpolate import LinearNDInterpolator
from oneLoop import *
from specFunc import *

# Set parameters for the calculation
iterations = 5
T          = 0.2
h          = 1
nu         = 0 # 1 / (8*np.pi)
Mu         = 0.964
mu         = 0
alpha      = 0
threshold  = 40
epsilon    = 0.01

wmax = 1.5*threshold
pmax = wmax/2


# Load spectral function classes from oneLoop.py and specFunc.py
oneloop  = OneLoop()
impurity = SpecFunc("fermion")
molecule = SpecFunc("boson")


# Update the classes for the first calcualtion
oneloop.T  = T
oneloop.h  = h
oneloop.nu = nu
oneloop.Mu = Mu
oneloop.mu = mu
oneloop.alpha = alpha

# Initialize impurity and molecule spectral function
impurity.mu      = mu
impurity.epsilon = epsilon
impurity.spec    = impurity.spec0
molecule.mu      = nu
molecule.epsilon = epsilon

"""
# Define vacuum boson self energy from Punk
def selfImBoson(w, p):
    l = (w+Mu+mu-p**2/2)/2
    k = (w-(Mu-mu))/2
    return np.heaviside(l,1)*np.heaviside(k+p*np.real(np.sqrt(l+0j)),1) \
    *( 2*np.real(np.sqrt(l+0j))*np.heaviside(k-p*np.real(np.sqrt(l+0j)),1) \
    + (np.real(np.sqrt(l+0j)) + k/p)*np.heaviside(-k+p*np.real(np.sqrt(l+0j)),1) ) / (16*np.pi)
def f(x, p):
    return np.where(x > 0, \
    0.5*np.log(np.abs( ((1-np.real(np.sqrt(x+0j)))**2-p**2/4)/((1+np.real(np.sqrt(x+0j)))**2-p**2/4) )), \
    np.pi - np.arctan((1+p/2)/np.sqrt(np.abs(x))) - np.arctan((1-p/2)/np.sqrt(np.abs(x))))
def selfReBoson(w, p):
    l = (w+Mu+mu)/2
    return h**2/(2*np.pi)**2 * (1+np.sqrt(mu)+np.sqrt(np.abs(l)) * \
    (0.5*np.log(np.abs((1-np.sqrt(np.abs(l)))/(1+np.sqrt(np.abs(l))))) + 0.5*np.log(np.abs((np.sqrt(mu)-np.sqrt(np.abs(l)))/(np.sqrt(mu)+np.sqrt(np.abs(l))))))*np.heaviside(l, 0) \
    + (np.pi/2 - np.arctan(1/np.sqrt(np.abs(l))) -np.arctan(np.sqrt(mu/np.abs(l))))*np.heaviside(-l, 0))
    #l = (w+Mu+mu-p**2/2)/2
    #return -h**2*( 1 - (1-l-p**2/4)/(2*p)*np.log(np.abs((l-(1-p/2)**2)/(l-(1+p/2)**2))) \
    #+ np.sqrt(np.abs(l)) * f(l, p) ) / (2*np.pi)**2/2
"""


# Print information about number of cores
cores = mp.cpu_count()
print("Number of cores here: ", cores)


# Initialize the iteration for the given particle typ
def setParticles(typ):
    if typ == "boson":
        oneloop.param = 0
        oneloop.rhoA  = impurity.spec

    elif typ == "fermion":
        oneloop.param = 1
        oneloop.rhoA  = molecule.spec

    else:
        print("Wrong particle / propagator type!")
        return 0


# Define self energy to depend only on one variable x for multiprocessing
# Variable transform to have large grid!!!
# Maybe do variable transform for bosons to resolve sharp onset structure...
def SelfImag(xy):
    x, y = xy
    return oneloop.SelfImag(x, y)

def SelfReal(xy):
    x, y = xy
    return oneloop.SelfReal(x, y)


def calcAdaptive(func, bounds):
    # Adaptive calculation of a function
    learner = adaptive.Learner2D(func, bounds=bounds)
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.0007)
    runner.ioloop.run_until_complete(runner.task)

    points = list(learner.data)
    values = np.array([dict(learner.data)[i] for i in points])
    points = np.asarray(points)
    FuncInter = LinearNDInterpolator(points, values, fill_value=0)

    return FuncInter


def calcSelfImag(typ):
    # Adaptive calculation of the imaginary part of the self energy.
    SelfImInter = calcAdaptive(SelfImag, bounds=[(-wmax,wmax), (0,pmax)])

    # Prepare imaginary part for calculation of the real part
    if typ == "boson":
        def newSelfIm(w, p):
            # Subtract SelfIm0 to make SelfRe calculation easier
            SelfIm0 = oneloop.SelfImBoson0(w, p)
            return (SelfImInter(w, p) - SelfIm0) * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0)

    elif typ == "fermion":
        def newSelfIm(w, p):
            return SelfImInter(w, p) * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0)

    oneloop.selfIm = newSelfIm
    return SelfImInter


def calcSelfReal(typ):
    # Adaptive calculation of the real part of the self energy.
    SelfReInter = calcAdaptive(SelfReal, bounds=[(-wmax,wmax), (0,pmax)])
    return SelfReInter

"""
def exactSelfImag(xy):
    x, y = xy
    return oneloop.SelfImBosonT(x, y)
"""


def setNewSpec(typ, SelfReInter, SelfImInter):
    if typ == "boson":
        def newSelfIm(w, p):
            SelfIm0 = oneloop.SelfImBoson0(w, p)+oneloop.SelfImBosonT(w, p)
            return (SelfImInter(w, p)-SelfIm0*np.heaviside(pmax-p,0)) \
            * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0) \
            * (1-2*(p-pmax/2)/pmax*np.heaviside(p-pmax/2,0))*np.heaviside(pmax-p,0) + SelfIm0
        def newSelfRe(w, p):
            return oneloop.SelfReBoson0(w, p) + SelfReInter(w, p) * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0) \
            * (1-2*(p-pmax/2)/pmax*np.heaviside(p-pmax/2,0))*np.heaviside(pmax-p,0)
        molecule.selfIm = newSelfIm
        molecule.selfRe = newSelfRe
        molecule.spec   = molecule.newG

    elif typ == "fermion":
        def newSelfIm(w, p):
            return -SelfImInter(w, p) * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0) \
            * (1-2*(p-pmax/2)/pmax*np.heaviside(p-pmax/2,0))*np.heaviside(pmax-p,0)
        def newSelfRe(w, p):
            return -SelfReInter(w, p) * (1-(np.abs(w)-threshold)/(wmax-threshold)*np.heaviside(np.abs(w)-threshold,0))*np.heaviside(wmax-np.abs(w),0) \
            * (1-2*(p-pmax/2)/pmax*np.heaviside(p-pmax/2,0))*np.heaviside(pmax-p,0)
        impurity.selfIm = newSelfIm
        impurity.selfRe = newSelfRe
        impurity.spec   = impurity.newG



def iteration(typ, i):
    # Here the calculation of the spectral functions begins
    t0 = time.time()
    print(f"Start {typ} iteration: ", i)

    setParticles(typ)

    # Calculate imagniary part adaptively
    SelfImInter = calcSelfImag(typ)
    t1 = time.time() - t0
    print("SelfImag done, time elapsed: ", t1)

    # Calculate real part adaptively
    SelfReInter = calcSelfReal(typ)
    t1 = time.time() - t0
    print("SelfReal done, time elapsed: ", t1)

    # Update spectral function for next iteration
    setNewSpec(typ, SelfReInter, SelfImInter)

    t1 = time.time() - t0
    print("Time elapsed: ", t1)

    # Save information about self energy in .pickle file
    with open(f"data/Test/{typ}/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={i}.pickle", "wb") as f:
        pickle.dump(SelfReInter, f)
        pickle.dump(SelfImInter, f)



if __name__ == "__main__":
    for i in range(iterations):
        iteration("boson", i+1)
        iteration("fermion", i+1)