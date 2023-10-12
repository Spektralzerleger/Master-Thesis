"""
Edited by: Eugen Dizer
Last modified: 27.03.2023

Welcome to specFunc.py! This file contains all information of a propagator.
"""

# Load some packages
import numpy as np


# Define zero function
def zero_func(w, q):
    return 0


class SpecFunc:

    def __init__(self, typ):
        self.typ     = typ
        self.mu      = 0
        self.epsilon = 0.01
        self.selfIm  = zero_func       # currently used self energy
        self.selfRe  = zero_func
        self.spec    = zero_func       # currently used spectral function


    # Define bare spectral function
    def spec0(self, w, p):               ### edit! careful, when you want mass imbalance...
        if self.typ == "fermion":
           return np.imag( 1 / (w - p**2 + self.mu - self.epsilon*1j) ) / np.pi
        elif self.typ == "boson":
           return np.imag( 1 / (w - p**2/2 + self.mu - self.epsilon*1j) ) / np.pi

    # Define bare inverse propagator     ### edit boson...
    def InvProp(self, w, p):
        if self.typ == "fermion":
           return w - p**2 + self.mu + self.epsilon*1j
        elif self.typ == "boson":
           return self.mu + self.epsilon/10*1j # w - p**2/2 + self.mu


    # Define new Gamma2       ### can also do different signs for fermions and bosons here...
    def Gamma2(self, w, p):
        return self.InvProp(w, p) - self.selfRe(w, p) - 1j*self.selfIm(w, p)

    def newG(self, w, p):   ### Investigate Spec representation!!!! especially for bosons...
        return np.imag( -1 / self.Gamma2(w, p) ) / np.pi


