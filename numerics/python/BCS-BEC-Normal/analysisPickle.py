### Welcome to analysisPickle.py! In this file, *.pickle files will be opened and
### the information plotted.

### First, load some modules
import scipy as sp
from scipy.integrate import quad, dblquad
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.special import hyp2f1
import numpy as np
import vegas
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from oneLoop import *


### The parameters here can be used to open pickles faster.
iteration = 9
T         = 1
h         = 1
nu        = 0 #-0.019894367886486918 #0.039788735772973836
mu        = 0.13146
epsilon   = 0
wmax_bos  = 12
pmax_bos  = 8.8


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
def get_self_energy(typ, iter):
    data = []
    try:
        with open(f"data/{typ}/T={T}_mu={mu}_nu={nu}_h={round(h, 2)}_iter={iter}.pickle", "rb") as f:
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
SelfRe1, SelfIm1 = get_self_energy("fermion", iteration)
SelfRe2, SelfIm2 = get_self_energy("boson", iteration)


# Define new self energy for boson
def newSelfIm(w, p):
    SelfImBos = oneloop.SelfImBoson0(w, p)+oneloop.SelfImBosonT(w, p)
    SelfImStrip = SelfImBos*np.heaviside(wmax_bos-np.abs(w-p**2/2),1)
    return (SelfIm2(w-p**2/2, p)+SelfIm2(w-p**2/2, pmax_bos)*np.heaviside(p-pmax_bos,0)-SelfImStrip) + SelfImBos
SelfReBosonT, _ = get_self_energy("boson", 1)
def newSelfRe(w, p):
    return oneloop.SelfReBoson0(w, p) + SelfReBosonT(w-p**2/2, p) \
    + SelfRe2(w-p**2/2, p)+SelfRe2(w-p**2/2, pmax_bos)*np.heaviside(p-pmax_bos,0)


### Define the inverse propagators for the fermions
def gamma2(typ, w, p):
    if typ == "fermion":
        return w-p**2+mu + SelfRe1(w-p**2/2,p) + SelfIm1(w-p**2/2,p)*1j + epsilon*1j
    elif typ == "boson":
        return nu - newSelfRe(w, p) - newSelfIm(w, p)*1j + epsilon*1j


### Define spectral functions for the fermions
def spec(typ, w, p):
    return np.imag( -1 / gamma2(typ,w,p) ) / np.pi


### Define several analysis functions
def norm_Check(typ):
    return quad(lambda x: spec(typ, x, 3), -30, 30, full_output=1)[:2]

def norm_vegas():
    @vegas.batchintegrand
    def Norm(x):
        lambda1   = x[:,0]
        l = lambda1 / (1-lambda1**2)
        return (1+lambda1**2)/(1-lambda1**2)**2 * spec("fermion", l, 1)

    integ = vegas.Integrator([[-1,1]])
    result = integ(Norm, nitn=20, neval=2e5, adapt_to_errors=True)
    return result.mean


def number_momentum(p):
    return quad(lambda x: spec("fermion", x, p)/(np.exp(x/T)+1), -100, 100, full_output=1)[0]


def number1D(w):
    return quad(lambda q: q**2*spec("fermion", w, q)/(np.exp(w/T)+1), 0, 8, full_output=1)[0] / np.pi**2

def number_vegas():              ######## See how integration limits effect result, also try with Johannes spec funcs...
    @vegas.batchintegrand        ######## Investigate why number density doesnt match with lattice. maybe better real-part? or bose function?
    def NumDensFermion(x):
        momentum  = x[:,0]
        lambda1   = x[:,1]
        q = momentum / (1-momentum)
        l = lambda1 / (1-lambda1**2)
        return 1/(1-momentum)**2 * (1+lambda1**2)/(1-lambda1**2)**2 * q**2*spec("fermion", l, q)/(np.exp(l/T)+1) / np.pi**2

    integ = vegas.Integrator([[0,1],[-1,1]])
    result = integ(NumDensFermion, nitn=10, neval=2e5, adapt_to_errors=True)
    return result.mean

def number_density0():
    return quad(lambda q: q**2/(np.exp((q**2-mu)/T)+1), 0, 9, full_output=1)[0] / np.pi**2


def ejectionSpec(w):
    return quad(lambda q: q**2*spec("fermion", q**2-w-mu, q)/(np.exp((q**2-w-mu)/T)+1), 0, 10, full_output=1)[0]


def norm_spec_vegas():
    @vegas.batchintegrand
    def diff(x):
        momentum  = x[:,0]
        lambda1   = x[:,1]
        q = momentum / (1-momentum)
        l = lambda1 / (1-lambda1**2)
        return 1/(1-momentum)**2 * (1+lambda1**2)/(1-lambda1**2)**2 * np.abs( np.imag( -1 / (l-q**2+mu + SelfRe1(l,q) + SelfIm1(l,q)*1j + epsilon*1j) ) - np.imag( -1 / (l-q**2+mu + SelfRe12(l,q) + SelfIm12(l,q)*1j + epsilon*1j) ) ) / np.pi

    integ = vegas.Integrator([[0,0.5],[-1,1]])
    integ(diff, nitn=20, neval=2e5, adapt_to_errors=True)
    result = integ(diff, nitn=20, neval=2e5, adapt_to_errors=True)
    return result.mean



def plot_SelfRe(typ):
    if typ == "fermion":
        #plt.pcolormesh(p, w, np.array([[SelfRe1(x, y) for y in p] for x in w]), shading="nearest")
        plt.plot(w, np.array([(SelfIm1(-1+x, 1) - SelfIm1(-1-x, 1))/x for x in w]))
    elif typ == "boson":
        plt.pcolormesh(p, w, np.array([[SelfRe2(x, y) for y in p] for x in w]), shading="nearest")
    plt.colorbar()
    plt.show()

def plot_SelfIm(typ):
    if typ == "fermion": #-SelfIm2(x,y)
        plt.pcolormesh(p, w, np.array([[-SelfIm2(x, y)-(-oneloop.SelfImBoson0(x+y**2/2,y)-oneloop.SelfImBosonT(x+y**2/2,y)) for y in p] for x in w]), shading="nearest")
        #plt.pcolormesh(p, w, np.array([[SelfIm2(x,y)*8*np.pi+(-oneloop.SelfImBoson0(x+y**2/2,y)-oneloop.SelfImBosonT(x+y**2/2,y))*8*np.pi*np.heaviside(25-x,1) for y in p] for x in w]), shading="nearest")
        #plt.pcolormesh(p, w, np.array([[oneloop.SelfReBosonT(x+y**2/2,y) for y in p] for x in w]), shading="nearest")
        #p=6
        #data = SelfIm2(w,p) -(oneloop.SelfImBoson0(w+p**2/2,p)+oneloop.SelfImBosonT(w+p**2/2,p))*np.heaviside(wmax_bos-w,1)
        #plt.plot(w, data) #-oneloop.SelfImFermionT(x, y)
        """
        # fit large frequency behavior
        def tail(w, a):
            return a / w**0.5
        popt, pcov = curve_fit(tail, w[-60:], data[-60:], p0=[-1])
        print(popt[0])
        l = 149.9999
        #print(quad(lambda x: tail(x+l,*popt)/x, 150-l, np.inf, full_output=1)[0]/np.pi)
        #print(popt[0] * (150-l)**(-popt[1]) * hyp2f1(popt[1],popt[1],1+popt[1],-l/(150-l)) / popt[1])
        plt.plot(w[-60:], tail(w[-60:], *popt))
        #print(SelfRe1(0,1))
        """
    elif typ == "boson":
        plt.pcolormesh(p, w, np.array([[-oneloop.SelfImFermionT(x, y) for y in p] for x in w]), shading="nearest")
        #plt.plot(w, np.array([-oneloop.SelfImFermionT(x, 0) for x in w]))
    plt.colorbar()
    plt.show()


def plot_SpecFunc(typ):
    plt.pcolormesh(p, w, np.array([[spec(typ, x, y) for y in p] for x in w]), shading="nearest")
    #plt.plot(w, spec(typ, w, 0))
    plt.colorbar()
    plt.show()

def plot_numberDensity():
    plt.plot(w, np.array([number1D(x) for x in w]))
    plt.show()

def plot_number():
    # marc data
    p_latt018 = [0,
1.19700592090261,
1.69282200758136,
2.07327507196409,
2.39401184180521,
2.67658660860796,
2.93205372530167,
3.38564401516272,
3.59101776270782,
3.59101776270782,
3.78526508275959,
3.97001951146778,
4.14655014392817,
4.31586622484834,
4.47878604595738,
4.78802368361042,
4.93538184637118,
5.07846602274408,
5.07846602274408,
5.2176278440343,
5.35317321721593,
5.48537023984651,
5.48537023984652,
5.61445543600354,
5.86410745060333,
5.98502960451303,
6.10355654856849,
6.21982521589226,
6.44607415917634,
6.44607415917634,
6.77128803032545,
6.87627550090196,
6.97968394262809,
7.18203552541563,
7.6645776244821,
8.29310028785634]
    n_latt018 = [0.536806390960361,0.178806988794015,0.042033590332054,0.011451482245716,0.004463099950462,0.002400016213234,0.001529572073883,0.000798977905302,0.000607043378588,0.000623210174109,0.000493276945455,0.000411956782129,0.00033861434774,0.000285612266422,0.000250865137449,0.000210594866372,0.000173669440825,0.000143142887313,0.000158558745987,0.000130120509718,0.00012559151577,0.000118811834254,0.000116285549799,9.85616525474979E-05,8.67262813914894E-05,8.05730085808127E-05,7.09911521245565E-05,6.77766219457144E-05,5.50158088427686E-05,5.42934571483257E-05,4.92669150408387E-05,4.11755290370668E-05,4.19064774824925E-05,3.54493351829354E-05,2.56044954662389E-05,1.27014933052006E-05]
    p_latt1 = [0.000000000000000000e+00,
6.566750017881460844e-01,
9.286786936001726334e-01,
1.137394467157452294e+00,
1.313350003576292169e+00,
1.468369943123090549e+00,
1.608518681222188995e+00,
1.857357387200345267e+00,
1.970025005364437920e+00,
1.970025005364438364e+00,
2.076588688145684536e+00,
2.177944590137241399e+00,
2.177944590137241843e+00,
2.274788934314904587e+00,
2.367675390262566903e+00,
2.457052871150407825e+00,
2.626700007152584337e+00,
2.707540394075192314e+00,
2.707540394075192314e+00,
2.786036080800517567e+00,
2.786036080800518011e+00,
2.862379971543919321e+00,
2.936739886246181097e+00,
3.009262902679441076e+00,
3.080078777469198670e+00]
    n_latt1 = [7.787705167584629828e-01,
6.218477498228901501e-01,
3.994663965108507520e-01,
2.047184900258857687e-01,
9.399708706727372132e-02,
4.511601283977970805e-02,
2.457610420182811464e-02,
1.043683809620509807e-02,
7.764340442515879626e-03,
7.791780486933162948e-03,
6.116068483898835610e-03,
4.953453558936593212e-03,
4.901860359002846619e-03,
4.023515954643466785e-03,
3.421910230668396505e-03,
2.940704760240061156e-03,
2.252185629582200473e-03,
1.933766433747139508e-03,
1.956639515089713972e-03,
1.727327569430058517e-03,
1.725145772762368612e-03,
1.552055021667944233e-03,
1.361720865082157242e-03,
1.231338794806309750e-03,
1.131258192569057708e-03]
    # density data
    n = np.array([number_momentum(q) for q in p])
    # fit contact OR: plot 3*pi^2*n*k^4 to obtain
    #def contact_fit(p, C):
    #    return C / p**4
    #popt, pcov = curve_fit(contact_fit, p[-50:], n[-50:])
    #print(popt)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(p_latt1, n_latt1, color='red', marker='o', linewidth=0, markersize=2) #*3*np.pi**2*p**4
    plt.plot(p, n) #*3*np.pi**2*p**4
    #plt.plot(p[-50:], contact_fit(p[-50:], popt))
    plt.show()


def plot_density_eos():
    # experiment zwierlein
    with open(f"/scratch/dizer/spectralpropertiesucoldatoms/spectral-bcs-bec/numerics/data/Fig4.dat", "r") as f:
        lines = f.readlines()[1:]
    betamu_exp = []
    density_exp = []
    for x in lines:
        if x.split(' ')[0] == "\n":
            continue
        else:
            density_exp.append(float(x.split(' ')[0]))
            betamu_exp.append(float(x.split(' ')[4][:-1]))
    # euclidean frank
    betamu_frank = [-1.28440, -0.98211, -0.69072, -0.40866, -0.13482, 0.13146, 0.39020, 0.64057, 0.88030, 1.10460, 1.30420, 1.45940, 1.61820, 1.90060, 2.19500]
    density_frank = [1.37040, 1.48650, 1.62370, 1.77880, 1.94580, 2.11640, 2.28230, 2.43640, 2.57390, 2.69190, 2.78790, 2.85700, 2.92320, 3.03220, 3.13920]
    # real dizer
    betamu_dizer = [-0.5, -0.18, 0.13146, 1]
    density_dizer = [1.69, 1.86, 2.03, 2.51]
    fig, ax = plt.subplots()
    plt.xlabel(r'$\beta\mu$')
    plt.ylabel(r'$n / n_0$')
    plt.xlim(-1.5,2.5)
    plt.ylim(0.9,3.4)
    ax.plot(betamu_exp, density_exp, color='red', marker='o', linewidth=0, markersize=2, label="experiment (MIT)")
    ax.plot(betamu_frank, density_frank, color='green', marker='^', linewidth=0, markersize=4, label="self-consistent T-matrix")
    ax.plot(betamu_dizer, density_dizer, color='blue', marker='s', linewidth=0, markersize=4, label="this work (dummy)")
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(which='both', direction='in', right=True, top=True)
    plt.legend()
    #plt.show()
    plt.savefig("density_eos.pdf", format="pdf", bbox_inches="tight")


def plot_contact():
    # experiment zwierlein
    Ts_exp = []
    contact_exp = []
    # euclidean frank
    Ts_frank = [0.16000, 0.18080, 0.20460, 0.23250, 0.25100, 0.27140, 0.30180, 0.34280, 0.39680, 0.46750, 0.56010, 0.68160, 0.84110, 1.04970, 1.32150, 1.67500]
    contact_frank = np.array([0.09100, 0.09000, 0.08930, 0.08890, 0.08870, 0.08850, 0.08830, 0.08790, 0.08750, 0.08670, 0.08550, 0.08350, 0.08050, 0.07630, 0.07080, 0.06430])
    # real dizer
    Ts_dizer = [0.18, 0.2, 0.24, 0.26, 0.28, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.68, 0.84, 1.0, 1.2, 1.5]
    contact_dizer = np.array([0.09100, 0.09000, 0.08930, 0.08890, 0.08870, 0.08850, 0.08830, 0.08790, 0.08750, 0.08670, 0.08550, 0.08350, 0.08050, 0.07630, 0.07080, 0.06430])
    plt.xlabel(r'$T/T_F$')
    plt.ylabel(r'$C/k_F^4$')
    plt.xlim(0,2)
    plt.ylim(1.5,3)
    #plt.plot(betamu_exp, density_exp, color='red', marker='o', linewidth=0, markersize=2, label="experiment (MIT)")
    plt.plot(Ts_frank, contact_frank*3*np.pi**2, color='green', marker='^', linewidth=0, markersize=4, label="Luttinger-Ward")
    #plt.plot(Ts_dizer, contact_dizer*3*np.pi**2, color='blue', marker='s', linewidth=0, markersize=4, label="this work (dummy)")
    plt.legend()
    plt.show()


def plot_rf_spectra():
    # experiment zwierlein
    T1 = np.array([[-1.40000000e+00,4.61739354e-03],
    [-1.10000000e+00,6.47511228e-03],
    [-9.00000000e-01,-9.79390862e-03],
    [-6.00000000e-01,-1.32817747e-03],
    [-4.00000000e-01,1.43736293e-02],
    [-3.00000000e-01,1.05527372e-02],
    [-2.00000000e-01,1.59537521e-02],
    [-1.00000000e-01,3.57767254e-02],
    [-0.00000000e+00,3.89515411e-02],
    [ 1.00000000e-01,4.62387175e-02],
    [ 2.00000000e-01,7.66308364e-02],
    [ 3.00000000e-01,9.13990682e-02],
    [ 4.00000000e-01,1.10428114e-01],
    [ 5.00000000e-01,1.83146901e-01],
    [ 6.00000000e-01,2.73320129e-01],
    [ 7.00000000e-01,3.95037584e-01],
    [ 8.00000000e-01,1.13255497e+00],
    [ 9.00000000e-01,1.01022452e+00],
    [ 1.00000000e+00,7.43985047e-01],
    [ 1.10000000e+00,5.83413404e-01],
    [ 1.20000000e+00,4.88763539e-01],
    [ 1.40000000e+00,3.33605924e-01],
    [ 1.70000000e+00,2.10690399e-01],
    [ 1.90000000e+00,1.88640954e-01],
    [ 2.00000000e+00,1.61953225e-01],
    [ 2.30000000e+00,1.19459916e-01],
    [ 2.40000000e+00,1.08702976e-01]])
    T2 = np.array([[-1.30000000e+00,-7.77001538e-04],
    [-9.00000000e-01,4.10911980e-03],
    [-5.00000000e-01,1.17778058e-03],
    [-2.00000000e-01,1.42228652e-02],
    [-1.00000000e-01,2.96905337e-02],
    [ 0.00000000e+00,7.13172146e-02],
    [ 1.00000000e-01,1.45442056e-01],
    [ 2.00000000e-01,1.83081248e-01],
    [ 3.00000000e-01,1.79563274e-01],
    [ 4.00000000e-01,2.83695285e-01],
    [ 5.00000000e-01,4.55474493e-01],
    [ 6.00000000e-01,6.54646392e-01],
    [ 7.00000000e-01,8.73287799e-01],
    [ 8.00000000e-01,7.35582800e-01],
    [ 9.00000000e-01,5.80070174e-01],
    [ 1.10000000e+00,4.19337104e-01],
    [ 1.20000000e+00,3.40083040e-01],
    [ 1.30000000e+00,2.72264295e-01],
    [ 1.40000000e+00,2.17635999e-01],
    [ 1.60000000e+00,1.93938654e-01],
    [ 2.00000000e+00,1.31848609e-01],
    [ 2.40000000e+00,9.23146085e-02]])
    T3 = np.array([[-1.3,0.00266086],
    [-0.9,-0.01195982],
    [-0.5,0.00914833],
    [-0.2,0.03526364],
    [-0.1,0.10305197],
    [ 0,0.13369773],
    [ 0.1,0.24228997],
    [ 0.2,0.33270118],
    [ 0.3,0.33844616],
    [ 0.4,0.46978314],
    [ 0.5,0.63036956],
    [ 0.6,0.75213898],
    [ 0.7,0.77321324],
    [ 0.8,0.63477125],
    [ 0.9,0.48544308],
    [ 1,0.36744922],
    [ 1.1,0.33056322],
    [ 1.2,0.29398894],
    [ 1.3,0.21861416],
    [ 1.4,0.16254641],
    [ 1.6,0.15107061],
    [ 1.9,0.09288562],
    [ 2.3,0.09217179]])
    T4 = np.array([[-1.2,0.01823865],
    [-0.8,0.02301478],
    [-0.7,0.03445714],
    [-0.5,0.05908955],
    [-0.3,0.1193441 ],
    [-0.2,0.14493539],
    [-0.1,0.21751716],
    [ 0,0.27325636],
    [ 0.1,0.37844392],
    [ 0.2,0.42086078],
    [ 0.3,0.47871112],
    [ 0.4,0.54876651],
    [ 0.5,0.58685019],
    [ 0.6,0.57406906],
    [ 0.7,0.49979907],
    [ 0.9,0.42842804],
    [ 1,0.35361638],
    [ 1.1,0.27395644],
    [ 1.2,0.21587764],
    [ 1.3,0.18261424],
    [ 1.5,0.15622343],
    [ 1.8,0.11252386],
    [ 2.2,0.082609  ],
    [ 2.4,0.07287785]])
    T5 = np.array([[-1.2,0.0265641],
    [-0.8,0.04915284],
    [-0.7,0.06923517],
    [-0.5,0.11764507],
    [-0.4,0.15893188],
    [-0.3,0.25083083],
    [-0.2,0.28822305],
    [-0.1,0.4083916 ],
    [ 0,0.45676468],
    [ 0.1,0.46649803],
    [ 0.2,0.52867469],
    [ 0.3,0.52365265],
    [ 0.4,0.48218442],
    [ 0.5,0.51063882],
    [ 0.6,0.47482439],
    [ 0.7,0.4212466 ],
    [ 0.8,0.35346843],
    [ 1.1,0.26206552],
    [ 1.2,0.20554714],
    [ 1.3,0.16809965],
    [ 1.4,0.15647495],
    [ 1.8,0.11170946],
    [ 2.2,0.07857011],
    [ 2.4,0.06845923]])
    T6 = np.array([[-1.2,0.03277127],
    [-0.9,0.06711509],
    [-0.8,0.06578074],
    [-0.7,0.09636195],
    [-0.6,0.12031079],
    [-0.5,0.16875646],
    [-0.4,0.21402316],
    [-0.3,0.27589339],
    [-0.2,0.40107495],
    [-0.1,0.48176214],
    [ 0,0.60036718],
    [ 0.1,0.55629078],
    [ 0.2,0.53149413],
    [ 0.3,0.44897084],
    [ 0.4,0.43620787],
    [ 0.5,0.41587736],
    [ 0.6,0.35745466],
    [ 0.7,0.3361848 ],
    [ 0.8,0.29705811],
    [ 0.9,0.2889556 ],
    [ 1,0.25371975],
    [ 1.1,0.23702872],
    [ 1.2,0.21376959],
    [ 1.3,0.18092041],
    [ 1.5,0.15532288],
    [ 1.7,0.13604306],
    [ 1.9,0.12288275],
    [ 2.1,0.09280388],
    [ 2.3,0.08044752]])
    T7 = np.array([[-1.3,-0.00460556],
    [-0.8,0.06478889],
    [-0.6,0.11496899],
    [-0.5,0.15628273],
    [-0.4,0.19921627],
    [-0.3,0.27858763],
    [-0.2,0.40172931],
    [-0.1,0.63911551],
    [ 0,0.70083813],
    [ 0.1,0.55996663],
    [ 0.2,0.51297973],
    [ 0.3,0.4667882 ],
    [ 0.5,0.42506285],
    [ 0.6,0.40140226],
    [ 0.7,0.3935764 ],
    [ 0.8,0.28391481],
    [ 0.9,0.31038633],
    [ 1.1,0.22032704],
    [ 1.3,0.18086127],
    [ 1.5,0.14543162],
    [ 1.7,0.10094652],
    [ 2,0.06792136],
    [ 2.2,0.08467273],
    [ 2.4,0.07992705]])
    T8 = np.array([[-1.3,0.03081578],
    [-1.2,0.03975699],
    [-1,0.07252148],
    [-0.8,0.08533146],
    [-0.6,0.16133193],
    [-0.5,0.21035965],
    [-0.4,0.30915642],
    [-0.3,0.37487214],
    [-0.2,0.59258612],
    [-0.1,0.72658928],
    [ 0,0.63718183],
    [ 0.2,0.46712868],
    [ 0.3,0.41963554],
    [ 0.4,0.41207953],
    [ 0.5,0.32623141],
    [ 0.7,0.27403741],
    [ 0.9,0.18626304],
    [ 1.1,0.17074721],
    [ 1.2,0.14738876],
    [ 1.4,0.12653094],
    [ 1.6,0.0929384 ],
    [ 1.7,0.07164437],
    [ 2.1,0.06566712],
    [ 2.4,0.05649455]])
    T9 = np.array([[-1.3,0.05130505],
    [-1.1,0.07261904],
    [-0.9,0.11141929],
    [-0.6,0.18658798],
    [-0.4,0.3113893 ],
    [-0.3,0.49994599],
    [-0.2,0.64345918],
    [-0.1,0.77176516],
    [ 0,0.62042968],
    [ 0.2,0.53649474],
    [ 0.3,0.43775624],
    [ 0.4,0.34639528],
    [ 0.5,0.32167291],
    [ 0.7,0.22444108],
    [ 1,0.15473098],
    [ 1.4,0.11266569],
    [ 1.9,0.0633382 ],
    [ 2.3,0.0729541 ]])
    # real dizer
    fig, ax = plt.subplots(figsize=(6,12))
    plt.xlabel(r'$\omega/\varepsilon_F$')
    plt.xlim(-1.6,2.6)
    plt.ylim(-8.5,1.5)
    ax.set_ylabel(r'$I(\omega)$', ha='left', y=0.875, labelpad=2.6)
    ax.vlines(0, -8.5, 1.5, color="gray", linestyles='dashed', linewidth=0.9, alpha=0.8)
    ax.plot(T1[:,0], T1[:,1]-0, marker='o', linewidth=0, markersize=4)
    ax.plot(T2[:,0], T2[:,1]-1, marker='o', linewidth=0, markersize=4)
    ax.plot(T3[:,0], T3[:,1]-2, marker='o', linewidth=0, markersize=4)
    ax.plot(T4[:,0], T4[:,1]-3, marker='o', linewidth=0, markersize=4)
    ax.plot(T5[:,0], T5[:,1]-4, marker='o', linewidth=0, markersize=4)
    ax.plot(T6[:,0], T6[:,1]-5, marker='o', linewidth=0, markersize=4)
    ax.plot(T7[:,0], T7[:,1]-6, marker='o', linewidth=0, markersize=4)
    ax.plot(T8[:,0], T8[:,1]-7, marker='o', linewidth=0, markersize=4)
    ax.plot(T9[:,0], T9[:,1]-8, marker='o', linewidth=0, markersize=4)
    #ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_yticks([0,0.5,1])
    # Create a second y-axis
    y2 = ax.twinx()
    # Copy the y limits from the left axis
    y2.set_ylim(ax.get_ylim())
    # Set the right y-yick positions
    y2.set_yticks([0,-1,-2,-3,-4,-5,-6,-7,-8])
    # Set the right y-tick labels
    y2.set_yticklabels([0.10,0.12,0.17,0.30,0.44,0.64,1.03,1.48,2.02])
    #ax.tick_params(which='both', direction='in', right=True, top=True)
    ax.annotate(r"$T/T_F$", xy=(1.017, 0.91), xycoords='axes fraction')
    plt.show()

def plot_peak_position():
    # experiment zwierlein
    Ts = np.array([0.104,0.123,0.17,0.304,0.444,0.639,1.03,1.479,2.019])
    peaks_exp = [-0.8,-0.7,-0.6,-0.5,-0.3,-0.05,-0.04,-0.03,-0.06]
    # real dizer
    plt.xlabel(r'$T/T_F$')
    plt.ylabel(r'$E_p/\varepsilon_F$')
    plt.xlim(0,2.1)
    plt.ylim(-1.3,0.6)
    plt.plot(Ts, peaks_exp, color='red', marker='o', linewidth=0, markersize=2, label="experiment (MIT)")
    plt.legend()
    plt.show()

def plot_life_time():
    # experiment zwierlein
    Ts = np.array([0.104,0.123,0.17,0.304,0.444,0.639,1.03,1.479,2.019])
    gammas_exp = [0.5,0.55,0.75,1.24,1.3,1.1,1.03,0.9,0.8]
    # real dizer
    plt.xlabel(r'$T/T_F$')
    plt.ylabel(r'$\Gamma/\varepsilon_F$')
    plt.xlim(0,2.1)
    plt.ylim(0,2)
    plt.plot(Ts, gammas_exp, color='red', marker='o', linewidth=0, markersize=2, label="experiment (MIT)")
    plt.legend()
    plt.show()


def plot_ejectionSpec():
    w = np.linspace(0,20,100)
    # ejection spectra data
    I = np.array([ejectionSpec(x) for x in w])
    # fit contact
    def contact_fit(w, C):
        return C * w**(-3/2) / (2*np.sqrt(2)*np.pi**2)
    popt, pcov = curve_fit(contact_fit, w[-50:], I[-50:])
    print(popt)
    plt.xlabel(r'$\omega / \varepsilon_F$')
    plt.ylabel(r'$I(\omega)$')
    plt.yscale("log")
    plt.plot(w, I)
    plt.plot(w[-80:], contact_fit(w[-80:], popt))
    plt.show()

def integrate_ejectionSpec():
    w = np.linspace(-55,55,200)
    f = interpolate.interp1d(w, np.array([number1D(x) for x in w]))
    return quad(lambda x: f(x), -55, 55, full_output=1)[0]


def plot_self_real_enss(typ):
    with open(f"/scratch/dizer/spectralpropertiesucoldatoms/spectral-bcs-bec/numerics/data/ufg-enss/ufg25As{typ}10", "r") as f:
        lines = f.readlines()[2:]

    real=[]
    imag=[]
    for x in lines:
        if x.split(' ')[0] == "\n":
            continue
        else:
            real.append(float(x.split(' ')[2]))
            imag.append(float(x.split(' ')[3]))

    real = np.array(real)
    imag = np.array(imag)
    data_real = real.reshape((401,51))
    data_imag = imag.reshape((401,51))

    w = np.linspace(-100,100,401)
    p = np.linspace(0,10,51)
    x = np.repeat(w, 51).reshape((401,51))
    y = np.tile(p, 401).reshape((401,51))
    data_spec = np.imag( -1/(x-y**2+mu-data_real-1j*data_imag) )
    values = data_spec.flatten()
    x = np.repeat(w, 51)
    y = np.tile(p, 401)
    points = list(zip(x, y))
    FuncInter = LinearNDInterpolator(points, values, fill_value=0)
    ImagInter = LinearNDInterpolator(points, imag, fill_value=0)

    @vegas.batchintegrand
    def NumDensFermion(x):
        momentum  = x[:,0]
        lambda1   = x[:,1]
        q = momentum / (1-momentum)
        l = lambda1 / (1-lambda1**2)
        return 1/(1-momentum)**2 * (1+lambda1**2)/(1-lambda1**2)**2 * q**2*FuncInter(l, q)/(np.exp(l/T)+1) / np.pi**2

    #integ = vegas.Integrator([[0,0.7],[-0.99,0.99]])
    #result = integ(NumDensFermion, nitn=10, neval=2e5, adapt_to_errors=True)
    #print(result.mean / number_density0())

    def self_energy_re(w, p):
        return quad(lambda x: (ImagInter(w+x, p) - ImagInter(w-x, p))/x/np.pi, 0.0001, 280, points=(10,30), limit=100, epsrel=0.001, full_output=1)[0]

    w = np.linspace(-100,100,401)
    p = np.linspace(0,10,51)
    #plt.pcolormesh(p, w, data_imag, shading="nearest")
    #plt.pcolormesh(p, w, data_imag-np.array([[-SelfIm1(x, y) for y in p] for x in w]), shading="nearest")
    plt.pcolormesh(p, w, data_imag+np.array([[(-oneloop.SelfImBoson0(x, y)-oneloop.SelfImBosonT(x, y))*8*np.pi for y in p] for x in w]), shading="nearest") ### data = np.array([[SelfEnergy(x,y) for y in p] for x in w])
    #plt.pcolormesh(p, w, data_real+np.array([[(-oneloop.SelfReBoson0(x, y)-oneloop.SelfReBosonT(x, y))*8*np.pi for y in p] for x in w]), shading="nearest")
    #plt.pcolormesh(p, w, np.array([[self_energy_re(x,y) for y in p] for x in w]), shading="nearest")
    #plt.pcolormesh(p, w, np.array([[FuncInter(x,y) for y in p] for x in w]), shading="nearest")
    plt.colorbar()
    plt.show()


#plot_self_real_enss("bos")

### Now we generate the actual plot.
w = np.linspace(-16,16,401)
p = np.linspace(0,6,100)
plt.xlabel(r'$k / k_F$')
plt.ylabel(r'$\omega / \varepsilon_F$')

#print(norm_Check("fermion"))
#print(norm_vegas())
#print(number_vegas() / number_density0())
#print(number_density0())
#plot_SelfRe("boson")
plot_SelfIm("fermion")
#plot_Test()
#plot_SpecFunc("fermion")
#plot_SpecFunc("boson")
#plot_numberDensity()
#plot_number()
#plot_density_eos()
#plot_contact()
#plot_rf_spectra()
#plot_life_time()
#plot_peak_position()
#plot_ejectionSpec()
#print(number_vegas())
#print(integrate_ejectionSpec())