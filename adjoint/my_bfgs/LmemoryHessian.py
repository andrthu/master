import numpy as np
from my_vector import *
from diagonalMatrix import DiagonalMatrix

class InvertedHessian():
    """
    Parent class for inverted hessian used in L-BFGS algorithm
    """

    def update(self,yk,sk,beta_scale={'scale':False}):
        """
        Method for updating the inverted hessian
        
        Arguments:
        * yk : Difference in gradient between iterations
        * sk : Difference in control between iterations
        """
        if self.mem_lim==0:
            return

        if len(self) == self.mem_lim:
            self.y = self.y[1:]
            self.s = self.s[1:]
            self.rho = self.rho[1:]
        
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append(self.make_rho(yk,sk))
        if beta_scale['scale']:
            
            #self.beta = (yk.dot(sk)/(yk.dot(yk)))**(-1)
            #print 'beta:',self.beta
            n = len(sk)
            N = n-beta_scale['m']
            yk1=yk[:N+1]
            yk2=yk[N+1:]
            sk1=sk[:N+1]
            sk2=sk[N+1:]

            beta1 = (yk1.dot(sk1)/(yk1.dot(yk1)))
            beta2 = (yk2.dot(sk2)/(yk1.dot(yk1)))
            beta = (yk.dot(sk)/(yk.dot(yk)))
            #print beta
            diag = np.zeros(n) 
            diag[:N+1]=beta**(-1)
            diag[N+1:]=beta**(-1)
            self.Hint=DiagonalMatrix(n,diag=diag)

    def __getitem__(self,k):
        
        if k==0:
            return self.Hint
        return (self.rho[k-1],self.y[k-1],self.s[k-1])

    def __len__(self):
        return len(self.y)
        
    def make_rho(self,yk,sk):

        raise NotImplementedError, 'InvertedHessian.make_rho() not implemented'
    
    def saveHessian(self):
        H=[self.y,self.s,self.rho]

        return H
    
    def matvec(self,x,k = -1):


        raise NotImplementedError, 'InvertedHessian.matvec() not implemented '

class MuLMIH(InvertedHessian):
    """
    Inverted hessian for MuLbfgs
    """

    def __init__(self,Hint,mu,H=None,mem_lim=10,beta=1,save_number=-1):
        """
        Initialazing the MuLMIH

        Valid options are
        
        * Hint : Initial approximation of inverted hessian, typically 1 
        * mu : scaling variable mu
        * H : Other inverted hessian that you want to incorporate
        * mem_lim : number of iterations the hessian remembers
        * beta : scaling for Hinit
        * save_number : number of values saved from H
        """
        self.Hint    = Hint
        self.mu      = mu
        self.mem_lim = mem_lim
        self.y       = []
        self.s       = []
        self.rho     = []
        self.beta    = beta

        if H!=None:
            if save_number==-1 or save_number>mem_lim:
                if len(H.y) > mem_lim:
                    start = len(H.y)-mem_lim
                    self.y   = H.y[start:]
                    self.s   = H.s[start:]
                    self.rho = H.rho[start:]
                else:                
                    self.y   = H.y
                    self.s   = H.s
                    self.rho = H.rho
            else:
                start = len(H.y)-save_number
                self.y   = H.y[start:]
                self.s   = H.s[start:]
                self.rho = H.rho[start:]
                

    def make_rho(self,yk,sk):
        
        return MuRho(sk,yk)

    def matvec(self,x,k = -1):
        """
        Recursive method for multiplying a vector with the inverted hessian
        
        Method is based on formula
        H(k+1) = [1-r(k)*y(k)*s(k)]H(k)[1-r(k)*y(k)*s(k)] + r(k)*s(k)*s(k)

        Arguments:
        * x : the vector you want to multiply the hessian with
        * k : counting variable that decides level of recurion
        """
        
        
        if k == -1:
            k = len(self)
        if k==0:
            return SimpleVector(self.beta * (x.matDot(self.Hint)))
        rhok, yk, sk = self[k]
        
        mu = self.mu
        
        
        
        YK = (yk.data[0] + mu*yk.data[1]).copy()
        SK = (sk.data[0] + mu*sk.data[1]).copy()

        
        
        t = x - float(rhok.func(mu) * sk.muVecVec(x,mu)) * YK
        t = self.matvec(t,k-1)
        t = t - float(rhok.func(mu) * yk.muVecVec(t,mu)) * SK
        t = t + float(rhok.func(mu) * sk.muVecVec(x,mu)) * SK
        return t

class LimMemoryHessian(InvertedHessian):
    """
    Normal Inverted Hessian
    """
    def __init__(self,Hint,mem_lim=10,beta=1):
        """
        Initialazing the LimMemoryHessian

        Valid options are:
        
        * Hint : Initial approximation of inverted hessian, typically 1
        * mem_lim : number of iterations the hessian remembers
        * beta : scaling for Hinit
        """


        self.Hint=Hint
        self.mem_lim=mem_lim
        self.y = []
        self.s = []
        self.rho = []
        self.beta = beta

    
    def make_rho(self,yk,sk):

        return 1./(yk.dot(sk))
    
    def matvec(self,x,k = -1):
        """
        Recursive method for multiplying a vector with the inverted hessian

        Method is based on formula
        H(k+1) = [1-r(k)*y(k)*s(k)]H(k)[1-r(k)*y(k)*s(k)] + r(k)*s(k)*s(k)

        Arguments:
        * x : the vector you want to multiply the hessian with
        * k : counting variable that decides level of recurion
        """
        
        if k == -1:
            k = len(self)
        if k==0:
            return SimpleVector(self.beta * (x.matDot(self.Hint)))
        rhok, yk, sk = self[k]
        
        
        
        t = x - float(rhok * x.dot(sk)) * yk
        t = self.matvec(t,k-1)
        t = t - float(rhok * yk.dot(t)) * sk
        t = t + float(rhok * x.dot(sk)) * sk
        return t

class NumpyLimMemoryHessian(InvertedHessian):
    """
    Normal Inverted Hessian
    """
    def __init__(self,Hint,mem_lim=10,beta=1,PPCH=None):
        """
        Initialazing the LimMemoryHessian

        Valid options are:
        
        * Hint : Initial approximation of inverted hessian, typically 1
        * mem_lim : number of iterations the hessian remembers
        * beta : scaling for Hinit
        """


        self.Hint=Hint
        self.mem_lim=mem_lim
        self.y = []
        self.s = []
        self.rho = []
        self.beta = beta
        self.PPCH = PPCH
    
    def make_rho(self,yk,sk):

        return 1./(yk.dot(sk))
    
    def matvec(self,x,k = -1):
        """
        Recursive method for multiplying a vector with the inverted hessian

        Method is based on formula
        H(k+1) = [1-r(k)*y(k)*s(k)]H(k)[1-r(k)*y(k)*s(k)] + r(k)*s(k)*s(k)

        Arguments:
        * x : the vector you want to multiply the hessian with
        * k : counting variable that decides level of recurion
        """
        
        if k == -1:
            k = len(self)
        if k==0:
            if self.PPCH!=None:
                
                return self.beta * self.Hint(self.PPCH(x))
                                  
            else:
                return self.beta * self.Hint(x)#(x.dot(self.Hint))
        rhok, yk, sk = self[k]
        
        
        
        t = x - (float(rhok * x.dot(sk)) * yk)        
        t = self.matvec(t,k-1)        
        t = t - (float(rhok * yk.dot(t)) * sk)
        t = t + (float(rhok * x.dot(sk)) * sk)
        return t
