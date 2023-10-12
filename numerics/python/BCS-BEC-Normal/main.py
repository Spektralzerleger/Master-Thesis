"""
Edited by: Eugen Dizer
Last modified: 15.06.2023

Welcome to main.py! This program is the executable file to calculate
Real-time propagators after a chosen number of iterations.
"""

# Load needed modules
import adaptive
import numpy as np
import pickle
import time
import multiprocessing as mp
from scipy.interpolate import LinearNDInterpolator
from oneLoop import *
from specFunc import *

# Set parameters for the calculation
iterations = 4
T          = 1
h          = 1
nu         = 0 # 0.5 / (8*np.pi)
mu         = 0.13146


wmax_ferm   = 160
pmax_ferm   = 8
prec_ferm   = 0.0008
wmax_bos    = 36
pmax_bos    = 8
prec_bos    = 0.0008


# Load spectral function classes from oneLoop.py and specFunc.py
oneloop  = OneLoop()
fermion  = SpecFunc("fermion")
boson    = SpecFunc("boson")

# Update the oneloop class for the calculation
oneloop.T  = T
oneloop.h  = h
oneloop.nu = nu
oneloop.mu = mu
oneloop.pmax = pmax_ferm
oneloop.wmax_ferm = wmax_ferm
oneloop.wmax_bos  = wmax_bos

# Load initial self energy to make convergence easier
def get_self_energy(typ, iteration):
    data = []
    with open(f"data/{typ}/T={T}_mu={mu}_nu={0}_h={round(h, 2)}_iter={iteration}.pickle", "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    SelfReInter = data[0]
    SelfImInter = data[1]
    return SelfReInter, SelfImInter

SelfReInter1, SelfImInter1 = get_self_energy("fermion", 9)
#SelfReInter2, SelfImInter2 = get_self_energy("boson", 7)

# Initialize spectral functions
fermion.mu       = mu
fermion.pmax     = pmax_ferm
boson.mu         = nu #2*mu-nu


# Print information about number of cores
cores = mp.cpu_count()
print("Number of cores here: ", cores)


# Initialize the iteration for the given particle typ
def setParticles(typ):
    if typ == "boson":
        oneloop.param = 0
        oneloop.rhoA = fermion.spec
        oneloop.rhoB = fermion.spec

    elif typ == "fermion":
        oneloop.param = 1
        oneloop.rhoA = boson.spec
        oneloop.rhoB = fermion.spec


# Adaptive calculation of a function
def calcAdaptive(func, bounds, precision):
    learner = adaptive.Learner2D(func, bounds=bounds)
    runner = adaptive.Runner(learner, goal=lambda l: l.loss() < precision)
    runner.ioloop.run_until_complete(runner.task)

    points = list(learner.data)
    values = np.array([dict(learner.data)[i] for i in points])
    points = np.asarray(points)
    FuncInter = LinearNDInterpolator(points, values, fill_value=0)
    print(len(values))

    return FuncInter


# Adaptive calculation of the imaginary part of the boson self energy for the 1st iteration (Pole-Pole)
def SelfImBos(xy):
    x, y = xy
    return oneloop.SelfImBoson0(x+y**2/2, y)+oneloop.SelfImBosonT(x+y**2/2, y)

def calcSelfImag0():
    SelfImInter = calcAdaptive(SelfImBos, bounds=[(-wmax_bos,wmax_bos), (0,pmax_bos)], precision=prec_bos)
    return SelfImInter


# Adaptive calculation of the imaginary part of the fermion self energy for the 1st iteration (Pole-Tail)
def SelfImag1(xy):
    x, y = xy
    return oneloop.SelfImag1(x+y**2/2, y)

def calcSelfImag1():
    SelfImInter = calcAdaptive(SelfImag1, bounds=[(-wmax_ferm,wmax_ferm), (0,pmax_ferm)], precision=prec_ferm)
    oneloop.selfIm = SelfImInter
    return SelfImInter


# Adaptive calculation of the imaginary part of the self energy for further iterations
def SelfImag(xy):
    x, y = xy
    return oneloop.SelfImag(x+y**2/2, y)

def calcSelfImag(typ):
    if typ == "boson":
        SelfImInter = calcAdaptive(SelfImag, bounds=[(-wmax_bos,wmax_bos), (0,pmax_bos)], precision=prec_bos)
        def newSelfIm(w, p):
            # Subtract SelfImBos to make SelfRe calculation easier for the bosons
            SelfImBos = (oneloop.SelfImBoson0(w+p**2/2, p)+oneloop.SelfImBosonT(w+p**2/2, p))*np.heaviside(wmax_bos-np.abs(w),1)
            return SelfImInter(w, p) - SelfImBos
    elif typ == "fermion":
        SelfImInter = calcAdaptive(SelfImag, bounds=[(-wmax_ferm,wmax_ferm), (0,pmax_ferm)], precision=prec_ferm)
        def newSelfIm(w, p):
            return SelfImInter(w, p)

    oneloop.selfIm = newSelfIm
    return SelfImInter


# Adaptive calculation of the real part of the self energy
def SelfReal(xy):
    x, y = xy
    return oneloop.SelfReal(x, y)

def calcSelfReal(typ):
    if typ == "boson":
        SelfReInter = calcAdaptive(SelfReal, bounds=[(-wmax_bos,wmax_bos), (0,pmax_bos)], precision=prec_bos)
    elif typ == "fermion":
        SelfReInter = calcAdaptive(SelfReal, bounds=[(-wmax_ferm,wmax_ferm), (0,pmax_ferm)], precision=prec_ferm)

    return SelfReInter


# Adaptive calculation of the real part of the boson self energy for 1st iteration (larger grid)
def calcSelfReal0():
    SelfReInter = calcAdaptive(SelfReal, bounds=[(-20,wmax_ferm), (0,pmax_ferm)], precision=prec_bos)
    return SelfReInter


# Define updated spectral function
def setNewSpec(typ, SelfReInter, SelfImInter):
    if typ == "boson":
        def newSelfIm(w, p):
            SelfImBos = oneloop.SelfImBoson0(w, p)+oneloop.SelfImBosonT(w, p)
            SelfImStrip = SelfImBos*np.heaviside(wmax_bos-np.abs(w-p**2/2),1)
            return (SelfImInter(w-p**2/2, p)+SelfImInter(w-p**2/2, pmax_bos)*np.heaviside(p-pmax_bos,0)-SelfImStrip) + SelfImBos
        SelfReBosonT, _ = get_self_energy("boson", 1)
        def newSelfRe(w, p):
            return oneloop.SelfReBoson0(w, p) + SelfReBosonT(w-p**2/2, p) \
            + SelfReInter(w-p**2/2, p)+SelfReInter(w-p**2/2, pmax_bos)*np.heaviside(p-pmax_bos,0)

        boson.selfIm = newSelfIm
        boson.selfRe = newSelfRe
        boson.spec   = boson.newSpec

    elif typ == "fermion":
        def newSelfIm(w, p):
            return -SelfImInter(w-p**2/2, p)
        def newSelfRe(w, p):
            return -SelfReInter(w-p**2/2, p)

        fermion.selfIm = newSelfIm
        fermion.selfRe = newSelfRe
        fermion.spec   = fermion.newSpec


# Here the calculation of the spectral functions begins!
# First iteration of the boson spectral function
def first_iteration():
    t0 = time.time()
    print(f"Start boson iteration: ", 1)

    # Set particle parameters
    oneloop.param = 0

    # Calculate imaginary part
    SelfImInter = calcSelfImag0()
    t1 = time.time() - t0
    print("SelfImag done, time elapsed: ", t1)

    # Calculate real part
    def SelfIm(w, p):
        return oneloop.SelfImBosonT(w+p**2/2,p)
    oneloop.selfIm = SelfIm
    SelfReInter = calcSelfReal0()
    t1 = time.time() - t0
    print("SelfReal done, time elapsed: ", t1)

    # Update spectral function
    def newSelfIm(w, p):
        return oneloop.SelfImBoson0(w, p) + oneloop.SelfImBosonT(w, p)
    def newSelfRe(w, p):
        return oneloop.SelfReBoson0(w, p) + SelfReInter(w-p**2/2, p)
    boson.selfIm = newSelfIm
    boson.selfRe = newSelfRe
    boson.spec   = boson.newSpec
    t1 = time.time() - t0
    print("Time elapsed: ", t1)

    with open(f"data/boson/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter=1.pickle", "wb") as f:
        pickle.dump(SelfReInter, f)
        pickle.dump(SelfImInter, f)


# First iteration of the fermion spectral function
def second_iteration():
    t0 = time.time()
    print(f"Start fermion iteration: ", 1)

    # Set particle parameters
    oneloop.param = 1
    oneloop.rhoA = boson.spec

    # Calculate imaginary part
    SelfImInter = calcSelfImag1()
    t1 = time.time() - t0
    print("SelfImag done, time elapsed: ", t1)

    # Calculate real part
    SelfReInter = calcSelfReal("fermion")
    t1 = time.time() - t0
    print("SelfReal done, time elapsed: ", t1)

    # Update spectral function
    setNewSpec("fermion", SelfReInter, SelfImInter)
    t1 = time.time() - t0
    print("Time elapsed: ", t1)

    with open(f"data/fermion/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter=1.pickle", "wb") as f:
        pickle.dump(SelfReInter, f)
        pickle.dump(SelfImInter, f)


# General function for further iterations
def iteration(typ, i):
    t0 = time.time()
    print(f"Start {typ} iteration: ", i)

    # Set particle parameters
    setParticles(typ)

    # Calculate imaginary part
    SelfImInter = calcSelfImag(typ)
    t1 = time.time() - t0
    print("SelfImag done, time elapsed: ", t1)

    # Calculate real part
    SelfReInter = calcSelfReal(typ)
    t1 = time.time() - t0
    print("SelfReal done, time elapsed: ", t1)

    # Update spectral function
    setNewSpec(typ, SelfReInter, SelfImInter)
    t1 = time.time() - t0
    print("Time elapsed: ", t1)

    with open(f"data/{typ}/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={i}.pickle", "wb") as f:
        pickle.dump(SelfReInter, f)
        pickle.dump(SelfImInter, f)



if __name__ == "__main__":
    setNewSpec("fermion", SelfReInter1, SelfImInter1)
    #iteration("boson", 8)
    #setNewSpec("boson", SelfReInter2, SelfImInter2)
    #first_iteration()
    #second_iteration()
    for i in range(iterations-1):
        iteration("boson", i+10)
        iteration("fermion", i+10)
        #iteration("fermion", i+2)
