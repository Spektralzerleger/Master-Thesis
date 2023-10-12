### Welcome to openPickle.py! In this file, *.pickle files will be opened and
### the information plotted.

### First, load some modules
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from oneLoop import *


### The parameters here can be used to open pickles faster.
iteration = 3
T         = 1
h         = 1
nu        = 0 # 0.039788735772973836
mu        = -2
Delta     = 0
epsilon   = 0.008


# Load class oneLoop.py for vacuum self energy
oneloop  = OneLoop()
oneloop.T  = T
oneloop.h  = h
oneloop.nu = nu
oneloop.mu = mu

# Define zero function
def zero_func(w, q):
    return 0

### First we define a general data import function
def get_self_energy(typ, prop, iter):
    data = []
    try:
        with open(f"data/{typ}/{prop}/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={iter}.pickle", "rb") as f:
          while True:
              try:
                  data.append(pickle.load(f))
              except EOFError:
                  break
        SelfReInter = data[0]
        SelfImInter = data[1]
    except IOError:
        SelfReInter = zero_func
        SelfImInter = zero_func

    return SelfReInter, SelfImInter


### Then we load the data from as many pickle files as we want.
SelfReNorm1, SelfImNorm1 = get_self_energy("fermion", "normal", iteration)
SelfReAnom1, SelfImAnom1 = get_self_energy("fermion", "anomalous", iteration)
SelfReNorm2, SelfImNorm2 = get_self_energy("boson", "normal", iteration)
SelfReAnom2, SelfImAnom2 = get_self_energy("boson", "anomalous", iteration)


### Define the inverse propagators for the fermions
def gamma2_11(typ, w, p):
    if typ == "fermion":
        return w-p**2+mu + SelfReNorm1(w,p) + SelfImNorm1(w,p)*1j + epsilon*1j
    elif typ == "boson":
        return nu - (SelfReNorm2(w,p) + oneloop.SelfReBoson0(w,p)) - SelfImNorm2(w,p)*1j + epsilon/10*1j

def gamma2_22(typ, w, p):
    if typ == "fermion":
        return w+p**2-mu - SelfReNorm1(-w,p) + SelfImNorm1(-w,p)*1j + epsilon*1j
    elif typ == "boson":
        return nu - (SelfReNorm2(-w,p) + oneloop.SelfReBoson0(-w,p)) - SelfImNorm2(-w,p)*1j + epsilon/10*1j

def gamma2_12(typ, w, p):
    if typ == "fermion":
        return Delta + SelfReAnom1(w,p) + SelfImAnom1(w,p)*1j
    elif typ == "boson":
        return SelfReAnom2(w,p) + SelfImAnom2(w,p)*1j

### Define spectral functions for the fermions
def specNorm(typ, w, p):
    return np.imag(-gamma2_22(typ,w,p)/( gamma2_11(typ,w,p)*gamma2_22(typ,w,p) - gamma2_12(typ,w,p)**2 ) ) / np.pi

def specAnom(typ, w, p):
    return np.imag(-gamma2_12(typ,w,p)/( gamma2_11(typ,w,p)*gamma2_22(typ,w,p) - gamma2_12(typ,w,p)**2 ) ) / np.pi


### Now we generate the actual plot.
w = np.linspace(-5,15,200)
p = np.linspace(0,8,200)

plt.xlabel(r'$k / k_F$')
plt.ylabel(r'$\omega / \varepsilon_F$')
#plt.pcolormesh(p, w, np.array([[SelfImNorm1(x, y) for y in p] for x in w]), shading="nearest")
plt.pcolormesh(p, w, np.array([[specNorm("fermion", x, y) for y in p] for x in w]), shading="nearest", vmin=0.02, vmax=2, norm=colors.LogNorm())
#plt.pcolormesh(p, w, np.array([[specAnom("boson", x, y) for y in p] for x in w]), shading="nearest")
#plt.pcolormesh(p, w, np.array([[np.imag( -1/gamma2_11("boson", x, y) ) / np.pi for y in p] for x in w]), shading="nearest")
plt.colorbar()
#plt.plot(w, np.array([specAnom("fermion", x, 0) for x in w]))

### And show the plot.
plt.show()
