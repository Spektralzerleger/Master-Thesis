"""
Edited by: Eugen Dizer
Last modified: 27.03.2023

Welcome to oneLoop.py! This file contains the numerical routines
to calculate the one loop integrals appearing in the self energy.
"""

# Load some packages
import numpy as np
import vegas
from scipy.integrate import quad

np.seterr(all="ignore") # use this to suppress numpy warnings like overflow or division by zero

# Define Fermi distribution
def nF(T, x):
    return 1/(1+np.exp(x/T))


def sign(x):
    return np.heaviside(x,1)-np.heaviside(-x,0)
def poles(x):
    return x + 0.001 *sign(x)*np.heaviside(0.1-np.abs(x),0)

# Define Bose distribution
def nB(T, x):
    return 1/poles(np.exp(x/T)-1)


class OneLoop:

    def __init__(self):
        self.T      = 1
        self.h      = 1
        self.nu     = 1
        self.Mu     = 1
        self.mu     = 0
        self.alpha  = 0
        self.rhoA   = 0        # currently used spectral function
        self.selfIm = 0        # currently used imaginary part of self energy
        self.param  = 0        # used to decide which particle is considered: boson:0, fermion:1


    # Define non-selfconsitent Boson self energy
    def SelfImBoson0(self, w, p):
        y = (1+self.alpha)/2 * (w - (1-self.alpha)*p**2/2 + self.Mu + self.mu)
        q = np.sqrt(y * np.heaviside(y, 0))

        VacuumPart = -(1+self.alpha)*self.h**2/(8*np.pi) * q

        return VacuumPart

    def SelfImBosonT(self, w, p):
        y = (1+self.alpha)/2 * (w - (1-self.alpha)*p**2/2 + self.Mu + self.mu)
        q = np.sqrt(y * np.heaviside(y, 0))
        q1 = (1-self.alpha)/(1+self.alpha) * (q + (1+self.alpha)*p/2)**2
        q2 = (1-self.alpha)/(1+self.alpha) * (q - (1+self.alpha)*p/2)**2
        Q1 = (q + (1-self.alpha)*p/2)**2
        Q2 = (q - (1-self.alpha)*p/2)**2

        if self.T < 0.01:
            ThermPart1 = (1+self.alpha)*self.h**2/(8*np.pi) * np.where(p == 0, q * np.heaviside(self.mu-q1, 0), \
                np.divide(np.heaviside(self.mu-q2, 0), 2*(1-self.alpha)*p, where=p!=0) * (self.mu-q2-(self.mu-q1)*np.heaviside(self.mu-q1, 0)) )

            ThermPart2 = (1+self.alpha)*self.h**2/(8*np.pi) * np.where(p == 0, q * np.heaviside(self.Mu-Q1, 0), \
                np.divide(np.heaviside(self.Mu-Q2, 0), 2*(1-self.alpha)*p, where=p!=0) * (self.Mu-Q2-(self.Mu-Q1)*np.heaviside(self.Mu-Q1, 0)) )
        else:
            ThermPart1 = (1+self.alpha)*self.h**2/(8*np.pi) * np.where(p == 0, q * nF(self.T, q1-self.mu), \
                np.divide(self.T, 2*(1-self.alpha)*p, where=p!=0) * np.log( nF(self.T, self.mu-q1)/nF(self.T, self.mu-q2) ) )

            ThermPart2 = (1+self.alpha)*self.h**2/(8*np.pi) * np.where(p == 0, q * nF(self.T, Q1-self.Mu), \
                np.divide(self.T, 2*(1-self.alpha)*p, where=p!=0) * np.log( nF(self.T, self.Mu-Q1)/nF(self.T, self.Mu-Q2) ) )

        return ThermPart1 + ThermPart2

    def SelfImFermionT(self, w, p): ### This should be the formula for fermions... include also nB contribution... check!
        y = 2/(1+self.alpha) * (w - (1-self.alpha)/(1+self.alpha)*p**2 - self.Mu + self.nu)
        q = np.sqrt(-y * np.heaviside(-y, 0))
        Q1 = (q + (1-self.alpha)/(1+self.alpha)*p)**2
        Q2 = (q - (1-self.alpha)/(1+self.alpha)*p)**2

        if self.T < 0.01:
            ThermPart1 = -self.h**2/(2*np.pi*(1+self.alpha)) * np.where(p == 0, q * np.heaviside(self.Mu-Q1, 0), \
                np.divide(np.heaviside(self.Mu-Q2, 0)*(1+self.alpha), 4*(1-self.alpha)*p, where=p!=0) * (self.Mu-Q2-(self.Mu-Q1)*np.heaviside(self.Mu-Q1, 0)) )
        else:
            ThermPart1 = -self.h**2/(2*np.pi*(1+self.alpha)) * np.where(p == 0, q * nF(self.T, Q1-self.Mu), \
                np.divide(self.T*(1+self.alpha), 4*(1-self.alpha)*p, where=p!=0) * np.log( nF(self.T, self.Mu-Q1)/nF(self.T, self.Mu-Q2) ) )

        return ThermPart1


    def SelfImag(self, w, p):  ### Include T=0? Reproduce results of R. Schmidt and T. Enss, done!

        if self.T < 0.01:        
            @vegas.batchintegrand
            def ContPartBoson(x):
                momentum  = x[:,0]
                angle     = x[:,1]
                q = momentum / (1-momentum)
                pq2 = q**2-2*p*q*angle+p**2
                return -1/(1-momentum)**2 \
                *np.pi*self.h**2 * q**2*self.rhoA(w-q**2+self.Mu, np.sqrt(pq2)) \
                *np.heaviside(q**2-self.Mu, 0)/(2*np.pi)**2

            @vegas.batchintegrand
            def ContPartFermion(x):
                momentum  = x[:,0]
                angle     = x[:,1]
                q = momentum / (1-momentum)
                pq2 = q**2-2*p*q*angle+p**2
                return 1/(1-momentum)**2 \
                *np.pi*self.h**2 * q**2*self.rhoA(w+pq2-self.Mu, q) \
                *np.heaviside(self.Mu-pq2, 0)/(2*np.pi)**2

        else:        
            @vegas.batchintegrand
            def ContPartBoson(x):
                momentum  = x[:,0]
                angle     = x[:,1]
                q = momentum / (1-momentum)
                pq2 = q**2-2*p*q*angle+p**2
                return -1/(1-momentum)**2 \
                *np.pi*self.h**2 * q**2*self.rhoA(w-q**2+self.Mu, np.sqrt(pq2)) \
                *( 1 - nF(self.T, q**2-self.Mu) - nF(self.T, w-q**2+self.Mu) )/(2*np.pi)**2 # (1-np.tanh((q**2-self.Mu)/(2*self.T)))/2

            @vegas.batchintegrand
            def ContPartFermion(x):
                momentum  = x[:,0]
                angle     = x[:,1]
                q = momentum / (1-momentum)
                pq2 = q**2-2*p*q*angle+p**2
                return 1/(1-momentum)**2 \
                *np.pi*self.h**2 * q**2*self.rhoA(w+pq2-self.Mu, q) \
                *( nB(self.T, w+pq2-self.Mu) + nF(self.T, pq2-self.Mu) )/(2*np.pi)**2 #(-1+np.tanh((pq2-self.Mu)/(2*self.T)))/2/(2*np.pi)**2


        if(self.param==0):
            integ = vegas.Integrator([[0,1],[-1,1]])
            result = integ(ContPartBoson, nitn=20, neval=2e5, adapt_to_errors=True)
            return result.mean
        elif(self.param==1):
            integ = vegas.Integrator([[0,1],[-1,1]])
            result = integ(ContPartFermion, nitn=20, neval=2e5, adapt_to_errors=True)
            return result.mean
        else:
            print("Somethings wrong I can feel it!")
            return 0


    def SelfReBoson0(self, w, p):
        y = (1+self.alpha)/2 * (w - (1-self.alpha)*p**2/2 + self.Mu + self.mu)
        q = np.sqrt(-y * np.heaviside(-y, 0))

        VacuumPart = (1+self.alpha)*self.h**2/(8*np.pi) * q

        return VacuumPart

    def SelfReal(self, w, p):

        # Recover real part from Kramers-Kronig relation
        # Improve with points option etc?
        lim=100
        def self_energy_re(w, p):
            return quad(lambda x: (self.selfIm(w+x, p) - self.selfIm(w-x, p))/x/np.pi, 0.0001, 40, limit=lim, epsrel=0.001, full_output=1)[0]

        @vegas.batchintegrand
        def vegas_int(x):
            y  = x[:,0]
            q = y / (1-y)
            return 1/(1-y)**2 * (self.selfIm(w+q, p) - self.selfIm(w-q, p))/q/np.pi

        #integ = vegas.Integrator([[0.00001,1]])
        #result = integ(vegas_int, nitn=20, neval=2e5, adapt_to_errors=True)
        return self_energy_re(w, p) #result.mean

