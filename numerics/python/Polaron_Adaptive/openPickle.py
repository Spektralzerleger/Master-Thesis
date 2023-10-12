### Welcome to openPickle.py! In this file, *.pickle files will be opened and
### the information plotted.

### First, load some modules
import numpy as np
import pickle
#import adaptive
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
from oneLoop import *


### The parameters here can be used to open pickles faster.
iteration = 5
T         = 0
h         = 1
nu        = 0.039788735772973836
mu        = 0
Mu        = 1


# Load class oneLoop.py for vacuum self energy
oneloop  = OneLoop()
oneloop.T  = T
oneloop.h  = h
oneloop.nu = nu
oneloop.Mu = Mu
oneloop.mu = mu
#oneloop.alpha = alpha


### First we define a general data import function     ### maybe remove "normal" from data structure?
def get_self_energy(typ, iter):
    data = []
    with open(f"data/{typ}/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={iter}.pickle", "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    SelfReInter = data[0]
    SelfImInter = data[1]
    return SelfReInter, SelfImInter


### Then we load the data from as many pickle files as we want.
SelfReInter1, SelfImInter1 = get_self_energy("fermion", iteration)
SelfReInter2, SelfImInter2 = get_self_energy("boson", iteration)


### Define the inverse propagators
def Gamma2(typ, w, p):
    if typ == "fermion":
        return w-p**2+mu + SelfReInter1(w,p) + SelfImInter1(w,p)*1j + 0.001j
    if typ == "boson":
        return nu - (SelfReInter2(w,p) + oneloop.SelfReBoson0(w,p)) - SelfImInter2(w,p)*1j + 0.001j

### Define spectral function
def spec(typ, w, p):
    return np.imag( -1 / Gamma2(typ,w,p) ) / np.pi


### Now we generate the actual plot.
w = np.linspace(-2,4,200)
p = np.linspace(0,2,200)

plt.xlabel(r'$k / k_F$')
plt.ylabel(r'$\omega / \varepsilon_F$')
#plt.pcolormesh(p, w, np.array([[SelfImInter1(x, y) for y in p] for x in w]), shading="nearest")
plt.pcolormesh(p, w, np.array([[spec("fermion",x, y) for y in p] for x in w]), shading="nearest", vmin=0.02, vmax=2, norm=colors.LogNorm())
#plt.pcolormesh(p, w, np.array([[spec("boson",x, y) for y in p] for x in w]), shading="nearest", vmin=0.02, vmax=50)
plt.colorbar()


"""
def func(xy):
    x, y = xy
    return graphSpec2Inter(x, y)

# Adaptive calculation of a function
learner = adaptive.Learner2D(func, bounds=[(-3,3),(0,3)])
runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
runner.ioloop.run_until_complete(runner.task)

points = list(learner.data)
values = np.array([dict(learner.data)[i] for i in points])
points = np.asarray(points)

x = points[:, 0]
y = points[:, 1]

# Show sampling point as triangulation
fig, ax = plt.subplots(figsize=(13, 10))
triang = tri.Triangulation(x, y)
tripc = ax.tripcolor(triang, values, shading='flat', edgecolors='w')
cbar = fig.colorbar(tripc)  # ticks = cticks)
"""

### And show the plot.
plt.show()
