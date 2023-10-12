"""
Edited by: Eugen Dizer
Last modified: 15.06.2023

Welcome to specFunc.py! This file contains all information of a propagator.
"""

# Load needed packages
import numpy as np


# Define zero function
def zero_func(w, q):
    return 0


class SpecFunc:

    def __init__(self, typ):
        self.typ = typ
        self.mu = 0
        self.epsilon = 0
        self.selfIm = zero_func       # currently used self energy
        self.selfRe = zero_func
        self.spec   = zero_func       # currently used spectral function
        self.pmax   = 0               # momentum cutoff


    # Define bare inverse propagator
    def InvProp(self, w, p):
        if self.typ == "fermion":
           return w - p**2 + self.mu
        elif self.typ == "boson":
           return self.mu # w - p**2/2 + self.mu


    # Define new Gamma2
    def Gamma2(self, w, q):      ### can also incorporate different signs for bosons and fermions here
        if self.typ == "fermion":
           return self.InvProp(w, q) - self.selfRe(w, q) - 1j*self.selfIm(w, q) + self.epsilon*1j
        elif self.typ == "boson":
           return self.InvProp(w, q) - self.selfRe(w, q) - 1j*self.selfIm(w, q) + self.epsilon*1j


    def newSpec(self, w, q):
        if self.typ == "fermion":
           return np.imag( -1 /  self.Gamma2(w, q) ) / np.pi * np.heaviside(self.pmax-q,0)
        elif self.typ == "boson":
           return np.imag( -1 /  self.Gamma2(w, q) ) / np.pi #* np.heaviside(1000-w,0)
