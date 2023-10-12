"""
Edited by: Eugen Dizer
Last modified: 15.06.2023

Welcome to oneLoop.py! This file contains the numerical routines
to calculate the one loop integrals appearing in the self energy.
"""

# Load some packages
import numpy as np
import vegas
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import hyp2f1

np.seterr(all="ignore") # use this to suppress numpy warnings like overflow or division by zero

# Define Fermi distribution
def nF(T, x):
    return 1/(1+np.exp(x/T))

# Define Bose distribution
def sign(x):
    return np.heaviside(x,1)-np.heaviside(-x,0)
def poles(x):
    return x + 0.001 *sign(x)*np.heaviside(0.01-np.abs(x),0)
def nB(T, x):
    return 1/poles(np.exp(x/T)-1)


class OneLoop:

    def __init__(self):
        self.T = 1
        self.h = 1
        self.nu = 1
        self.mu = 1
        self.rhoA = 0         # currently used spectral functions
        self.rhoB = 0
        self.selfIm = 0       # currently used self energy
        self.selfRe = 0
        self.param = 0        # used to decide which particle is considered: boson:0, fermion:1
        self.pmax  = 0        # momentum and frequency cutoffs
        self.wmax_ferm = 0
        self.wmax_bos  = 0



    # Define non-selfconsistent Boson self energy (imaginary part)
    def SelfImBoson0(self, w, p):
        y = 1/2 * (w - p**2/2 + 2*self.mu)
        q = np.sqrt(y * np.heaviside(y, 0))

        VacuumPart = -self.h**2/(8*np.pi) * q

        return VacuumPart

    def SelfImBosonT(self, w, p):
        y = 1/2 * (w - p**2/2 + 2*self.mu)
        q = np.sqrt(y * np.heaviside(y, 0))
        q1 = (q + p/2)**2
        q2 = (q - p/2)**2

        ThermPart1 = self.h**2/(8*np.pi) * np.where(p == 0, q * nF(self.T, q1-self.mu), \
            np.divide(self.T, 2*p, where=p!=0) * np.log( nF(self.T, self.mu-q1)/nF(self.T, self.mu-q2) ) )

        ThermPart2 = self.h**2/(8*np.pi) * np.where(p == 0, q * nF(self.T, q1-self.mu), \
            np.divide(self.T, 2*p, where=p!=0) * np.log( nF(self.T, self.mu-q1)/nF(self.T, self.mu-q2) ) )

        return ThermPart1 + ThermPart2

    # Define non-selfconsistent Fermion self energy (imaginary part)
    def SelfImFermionT(self, w, p):
        mphi = 2*self.mu - self.nu
        y = 2 * (w - p**2 - self.mu + mphi)
        q = np.sqrt(-y * np.heaviside(-y, 0))
        Q1 = (q + p)**2
        Q2 = (q - p)**2
        q1 = (q - 2*p)**2/2
        q2 = (q + 2*p)**2/2

        ThermPart1 = -self.h**2/(2*np.pi) * np.where(p == 0, q * nF(self.T, Q1-self.mu), \
            np.divide(self.T, 4*p, where=p!=0) * np.log( nF(self.T, self.mu-Q1)/nF(self.T, self.mu-Q2) ) )

        ThermPart2 = -self.h**2/(2*np.pi) * np.where(p == 0, q * nB(self.T, q1-mphi), \
            np.divide(self.T, 4*p, where=p!=0) * np.real(np.log( nB(self.T, mphi-q1)/nB(self.T, mphi-q2) + 0j )) )

        return ThermPart1 + ThermPart2


    # Define 1st iteration of Fermion self energy (Pole-Tail) (imaginary part)
    def SelfImag1(self, w, p):

        @vegas.batchintegrand
        def PolePartFermion(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            q = momentum / (1-momentum)
            pq2 = q**2-2*p*q*angle+p**2
            return 1/(1-momentum)**2 \
            *np.pi*self.h**2 * q**2*self.rhoA(w+pq2-self.mu, q) \
            *( nB(self.T, w+pq2-self.mu) + nF(self.T, pq2-self.mu) )/(2*np.pi)**2

        integ = vegas.Integrator([[0,1],[-1,1]])
        integ(PolePartFermion, nitn=10, neval=2e5, adapt_to_errors=True)
        result = integ(PolePartFermion, nitn=10, neval=2e5, adapt_to_errors=True)
        return result.mean


    # Define general self energy iteration function (imaginary part)
    def SelfImag(self, w, p):

        ### This is the Boson Tail-Tail contribution
        @vegas.batchintegrand
        def ContPartBoson(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            lambda1   = x[:,2]
            q = momentum / (1-momentum)
            l = lambda1 / (1-lambda1**2)
            pq2 = q**2-2*p*q*angle+p**2
            return -1/(1-momentum)**2 * (1+lambda1**2)/(1-lambda1**2)**2 \
            *np.pi*self.h**2 * q**2*self.rhoA(w-l, np.sqrt(pq2))*self.rhoB(l, q) \
            *( 1 - nF(self.T, w-l) - nF(self.T, l) )/(2*np.pi)**2

        ### These are the Boson Pole-Tail contributions
        @vegas.batchintegrand
        def TailPartBoson1(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            q = momentum / (1-momentum)
            pq2 = q**2-2*p*q*angle+p**2
            return -1/(1-momentum)**2 \
            *np.pi*self.h**2 * q**2*self.rhoA(w-q**2+self.mu, np.sqrt(pq2)) * np.heaviside(q-self.pmax,0) \
            *( 1 - nF(self.T, w-q**2+self.mu) - nF(self.T, q**2-self.mu) )/(2*np.pi)**2

        @vegas.batchintegrand
        def TailPartBoson2(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            q = momentum / (1-momentum)
            pq2 = q**2-2*p*q*angle+p**2
            return -1/(1-momentum)**2 \
            *np.pi*self.h**2 * q**2*self.rhoB(w-pq2+self.mu, q) * np.heaviside(np.sqrt(pq2)-self.pmax,0) \
            *( 1 - nF(self.T, w-pq2+self.mu) - nF(self.T, pq2-self.mu) )/(2*np.pi)**2   

        ### This is the Boson Pole-Pole contribution
        @vegas.batchintegrand
        def PolePartBoson(x):
            momentum  = x[:,0]
            q = momentum / (1-momentum)
            x = (p**2-w-2*self.mu+2*q**2)/poles(2*q*p)
            return -1/(1-momentum)**2 * np.heaviside(1-np.abs(x),0)/poles(2*p) * np.heaviside(q-self.pmax,0) \
            *np.pi*self.h**2 * q * ( 1 - nF(self.T, w-q**2+self.mu) - nF(self.T, q**2-self.mu) )/(2*np.pi)**2

        ######################################################

        ### This is the Fermion Tail-Tail contribution
        @vegas.batchintegrand
        def ContPartFermion(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            lambda1   = x[:,2]
            q = momentum / (1-momentum)
            l = lambda1 / (1-lambda1**2)
            pq2 = q**2-2*p*q*angle+p**2
            return 1/(1-momentum)**2 * (1+lambda1**2)/(1-lambda1**2)**2 \
            *np.pi*self.h**2 * q**2*self.rhoA(w+l, q)*self.rhoB(l, np.sqrt(pq2)) \
            *( nB(self.T, w+l) + nF(self.T, l) )/(2*np.pi)**2

        ### This is the Fermion Pole-Tail contribution
        @vegas.batchintegrand
        def PolePartFermion(x):
            momentum  = x[:,0]
            angle     = x[:,1]
            q = momentum / (1-momentum)
            pq2 = q**2-2*p*q*angle+p**2
            return 1/(1-momentum)**2 \
            *np.pi*self.h**2 * q**2*self.rhoA(w+pq2-self.mu, q)*np.heaviside(np.sqrt(pq2)-self.pmax,1) \
            *( nB(self.T, w+pq2-self.mu) + nF(self.T, pq2-self.mu) )/(2*np.pi)**2

        #######################################################

        if(self.param==0):
            ### Tail-Tail
            integ = vegas.Integrator([[0,1],[-1,1],[-1,1]])
            integ(ContPartBoson, nitn=10, neval=1e6, adapt_to_errors=True)
            result = integ(ContPartBoson, nitn=12, neval=1e6, adapt_to_errors=True)
            """ This part is not so important for large cutoff
            ### 2x Pole-Tail
            integ2 = vegas.Integrator([[0,1],[-1,1]])
            integ2(TailPartBoson1, nitn=10, neval=2e5, adapt_to_errors=True)
            result2 = integ2(TailPartBoson1, nitn=10, neval=2e5, adapt_to_errors=True)
            integ3 = vegas.Integrator([[0,1],[-1,1]])
            integ3(TailPartBoson2, nitn=10, neval=2e5, adapt_to_errors=True)
            result3 = integ3(TailPartBoson2, nitn=10, neval=2e5, adapt_to_errors=True)
            ### Pole-Pole
            integ4 = vegas.Integrator([[0,1]])
            integ4(PolePartBoson, nitn=10, neval=2e5, adapt_to_errors=True)
            result4 = integ4(PolePartBoson, nitn=10, neval=2e5, adapt_to_errors=True)
            """
            return result.mean # + result2.mean + result3.mean + result4.mean
        elif(self.param==1):
            ### Tail-Tail
            integ = vegas.Integrator([[0,1],[-1,1],[-1,1]])
            integ(ContPartFermion, nitn=10, neval=2e5, adapt_to_errors=True)
            result = integ(ContPartFermion, nitn=12, neval=2e5, adapt_to_errors=True)
            ### Pole-Tail
            integ2 = vegas.Integrator([[0,1],[-1,1]])
            integ2(PolePartFermion, nitn=10, neval=2e5, adapt_to_errors=True)
            result2 = integ2(PolePartFermion, nitn=12, neval=2e5, adapt_to_errors=True)
            return result.mean + result2.mean


    # Define non-selfconsistent Boson self energy (real part)
    def SelfReBoson0(self, w, p):
        y = 1/2 * (w - p**2/2 + 2*self.mu)
        q = np.sqrt(-y * np.heaviside(-y, 0))

        VacuumPart = self.h**2/(8*np.pi) * q

        return VacuumPart

    def SelfReBosonT(self, w, p):
        y = 1/2 * (w - p**2/2 + 2*self.mu)
        q = np.sqrt(-y * np.heaviside(-y, 0))
        q1 = np.sqrt(self.mu)-p/2
        q2 = np.sqrt(self.mu)+p/2

        Part1 = self.h**2/(8*np.pi**2) * np.heaviside(self.mu,0) * np.where(p == 0, \
                2*np.sqrt(self.mu) - 2*np.sqrt(np.abs(y))*( np.arctanh(np.sqrt(self.mu/np.abs(y)))*np.heaviside(y,0)+np.arctan(np.sqrt(self.mu/np.abs(y)))*np.heaviside(-y,0) ), \
                np.sqrt(self.mu) - np.divide(self.mu-y-p**2/4, 2*p, where=p!=0)*np.log(np.abs((y-q1**2)/(y-q2**2))) + \
                np.sqrt(np.abs(y))* ( (np.real(-np.arctanh((q1+0j)/np.sqrt(np.abs(y))) - np.arctanh((q2+0j)/np.sqrt(np.abs(y))) ))*np.heaviside(y,0) + \
                (-np.arctan(q2/np.sqrt(np.abs(y))) - np.arctan(q1/np.sqrt(np.abs(y))))*np.heaviside(-y,0) ) )

        Part2 = self.h**2/(8*np.pi**2) * np.heaviside(self.mu,0) * np.where(p == 0, \
                2*np.sqrt(self.mu) - 2*np.sqrt(np.abs(y))*( np.arctanh(np.sqrt(self.mu/np.abs(y)))*np.heaviside(y,0)+np.arctan(np.sqrt(self.mu/np.abs(y)))*np.heaviside(-y,0) ), \
                np.sqrt(self.mu) - np.divide(self.mu-y-p**2/4, 2*p, where=p!=0)*np.log(np.abs((y-q1**2)/(y-q2**2))) + \
                np.sqrt(np.abs(y))* ( (np.real(-np.arctanh((q1+0j)/np.sqrt(np.abs(y))) - np.arctanh((q2+0j)/np.sqrt(np.abs(y))) ))*np.heaviside(y,0) + \
                (-np.arctan(q2/np.sqrt(np.abs(y))) - np.arctan(q1/np.sqrt(np.abs(y))))*np.heaviside(-y,0) ) )

        return Part1 + Part2 if 0 < self.mu else 0


    # Recover real part from Kramers-Kronig relation
    def SelfReal(self, w, p):
        def self_energy_re(w, p):
            return quad(lambda x: (self.selfIm(w+x, p) - self.selfIm(w-x, p))/x/np.pi, 0.0001, 350, points=(10,30,100), limit=100, epsrel=0.001, full_output=1)[0]

        # Fit large frequency behavior, and calculate the contribution b = 0.5
        def tail(x, a, b):
            return a / x**b
        def tail_contrib(y, a, b):
            return a*y**(-b)*hyp2f1(b,b,1+b,-w/y)/b/np.pi

        if(self.param==0):
            return self_energy_re(w, p)
        elif(self.param==1):
            x = np.linspace(0.8*self.wmax_ferm, self.wmax_ferm-0.0001, 60)
            data = self.selfIm(x, p)
            popt, pcov = curve_fit(tail, x, data, p0=[0.5,0.5])
            a, b = popt
            y    = self.wmax_ferm - w
            tail_contribution = tail_contrib(y,a,b) if y > 0 else tail_contrib(y+0.0001,a,b)
            return self_energy_re(w, p) + tail_contribution

