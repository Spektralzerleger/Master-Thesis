### Welcome to analysisPickle.py! In this file, *.pickle files will be opened and
### the information plotted.

### First, load some modules
import scipy as sp
from scipy.integrate import quad, dblquad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from oneLoop import *


### The parameters here can be used to open pickles faster.
iteration = 1
T         = 0
h         = 1
nu        = 0.039788735772973836 #0.019894367886486918
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


### First we define a general data import function
def get_self_energy(typ, iter):
    data = []
    with open(f"data/{typ}/normal/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={iter}.pickle", "rb") as f:
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

### Define spectral function of majority atoms
def spec0(w, p):
    return np.imag( -1 / ( w-p**2+Mu + 0.01j ) ) / np.pi


### Define several analysis functions
def norm_Check(typ):
    return quad(lambda x: spec(typ, x, 3), -30, 30, full_output=1)[:2]

def number_density(): ### with spec func, careful with q**2 ...
    return dblquad(lambda q, x: q**2*spec("fermion", x, q)/(np.exp((x+0.5)/T)+1), -100, 100, 0, 10)

def number_density2(): ### with delta peak
    return quad(lambda q: q**2/(np.exp(q**2/T)+1), 0, 10, full_output=1)

def number_density3(w): ### with spec func, careful with q**2 ...
    return quad(lambda q: q**2*spec("fermion", w, q)/(np.exp((w+0.5)/T)+1), 0, 10, full_output=1)[0]

def ejectionSpec(w):  ### careful with q**2 ...
    return quad(lambda q: q**2*spec("fermion", q**2-w, q)/(np.exp((q**2-w)/T)+1), 0, 10, full_output=1)[0]


### Extract quasiparticle properties through formulas! From 2204.06984, Scazza (2022)
def qp_formulas():
    w = np.linspace(-6,-1,200)
    mu_vals = [x+SelfReInter1(x,0) for x in w]
    fpole = UnivariateSpline(w, mu_vals)
    pole = fpole.roots()[0]
    re_vals = [-SelfReInter1(x,0) for x in w]
    f = UnivariateSpline(w, re_vals)
    derivative = f.derivatives(pole)[1]
    Z = 1 / ( 1 - derivative )
    Gamma = Z * SelfImInter1(pole,0)
    return [Z, pole, Gamma]

### Extract quasiparticle properties through fit! Improve SelfRe and SelfIm to match with formulas...? Fix weight of bosons!!
def qp_fit():
    w = np.linspace(-3.5,-2,200)

    def specQP(w, Z, mu, Gamma):
        return np.imag( - Z / ( w-mu + Gamma*1j ) ) / np.pi

    data = spec("fermion", w, 0)

    ### Here fit routine
    popt, pcov = curve_fit(specQP, w, data, p0=[1, 0, 0.1])
    
    plt.plot(w, data)
    plt.plot(w, specQP(w, *popt))
    plt.show()
    return popt


### Results from spectra analysis:
Ts    = [0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
Peak1 = [0.60, 0.65, 0.73, 0.71, 0.64, 0.48, 0.11]
Peak5 = [0.68, 0.71, 0.73, 0.63, 0.49, 0.29, 0.13]
FWHM1 = [0.17, 0.23, 0.52, 1.05, 1.31, 1.53, 1.38]
FWHM5 = [0.21, 0.36, 0.77, 1.28, 1.41, 1.42, 1.27]
Mus   = [1.0, 0.964, 0.743, 0.333, -0.019, -0.429, -1.123]


### Now we generate the actual plot.
w = np.linspace(-2,2,200)
p = np.linspace(0,2,200)
plt.xlabel(r'$k / k_F$')
plt.ylabel(r'$\omega / \varepsilon_F$')

def plot_SelfRe(typ):
    if typ == "fermion":
        plt.pcolormesh(p, w, np.array([[SelfReInter1(x, y) for y in p] for x in w]), shading="nearest")
    elif typ == "boson":
        plt.pcolormesh(p, w, np.array([[SelfReInter2(x, y) for y in p] for x in w]), shading="nearest")
    plt.colorbar()
    plt.show()

def plot_SelfIm(typ):
    if typ == "fermion":
        plt.pcolormesh(p, w, np.array([[SelfImInter1(x, y) for y in p] for x in w]), shading="nearest")
    elif typ == "boson":
        plt.pcolormesh(p, w, np.array([[SelfImInter2(x, y) for y in p] for x in w]), shading="nearest")
    plt.colorbar()
    plt.show()

def plot_SpecFunc(typ):
    plt.pcolormesh(p, w, np.array([[spec(typ, x, y) for y in p] for x in w]), shading="nearest", vmin=0.02, vmax=2, norm=colors.LogNorm())
    plt.colorbar()
    plt.show()

def plot_ejectionSpec():
    w = np.linspace(0,1.5,200)
    plt.xlabel(r'$\omega / \varepsilon_F$')
    plt.ylabel(r'$I(\omega)$')
    plt.plot(w, np.array([ejectionSpec(x) for x in w]))
    plt.show()

def plot_numberDensity():
    plt.plot(w, np.array([number_density3(x) for x in w]))
    plt.show()


def plot_peakPos():
    plt.xlabel(r'$T / T_F$')
    plt.ylabel(r'$\omega_{\mathrm{peak}} / \varepsilon_F$')
    plt.ylim(-0.5, 2)
    plt.plot(Ts, Peak1, label="n=1")
    plt.plot(Ts, Peak5, label="n=5")
    plt.legend()
    plt.show()

def plot_FWHM():
    plt.xlabel(r'$T / T_F$')
    plt.ylabel(r'$\mathrm{FWHM} / \varepsilon_F$')
    plt.ylim(0, 2)
    plt.plot(Ts, FWHM1, label="n=1")
    plt.plot(Ts, FWHM5, label="n=5")
    plt.legend()
    plt.show()

def plot_Mu():
    plt.xlabel(r'$T / T_F$')
    plt.ylabel(r'$\mu / \varepsilon_F$')
    plt.ylim(-2, 2)
    plt.plot(Ts, Mus)
    plt.show()


#print(number_density())
#print(number_density2())
#print(norm_Check("boson"))
#plot_SpecFunc("fermion")
#plot_ejectionSpec()
#plot_numberDensity()
#plot_FWHM()
#print(qp_formulas())
print(qp_fit())