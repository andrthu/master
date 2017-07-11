import numpy as np

from my_vector import SimpleVector, MuVector,MuVectors
from LmemoryHessian import LimMemoryHessian,NumpyLimMemoryHessian
from mpiVector import MPIVector

class LbfgsOptimizationControl():

    def __init__(self,x0,J_f,grad_J,H,H2=None,scaler=None):

        self.x = x0.copy()
        self.J_func = J_f
        self.grad_J = grad_J
        self.length = len(x0)
        self.lsiter = 0
        self.counter = None

        self.dJ = grad_J(x0)
        self.H = H
        self.H2 = H2
        self.scaler = scaler
        #self.vec = vec
        self.mu = None
        self.jump_diff = None

        self.niter = 0
        
        self.dictonary = {'control'    : SimpleVector(self.x),
                          'iterations' : self.niter,
                          'lbfgs'      : self.H}

    def update_dict(self):
        self.dictonary['control']   = SimpleVector(self.x)
        self.dictonary['iteration'] = self.niter
        self.dictonary['lbfgs']     = self.H

    def __getitem__(self,key):
        self.update_dict()
        return self.dictonary[key]
        

    def update(self,x,dJ):

        self.x = x.copy()
        self.dJ = dJ.copy()

        self.niter +=1

    def split_update(self,N,v,lamda):
        self.x[:N+1] = v.copy()
        self.x[N+1:]=lamda[:]
        self.dJ = self.grad_J(self.x)
        self.niter += 1

    def rescale(self):
        
        x = self.x
        N = self.scaler.N

        x[N+1:] = self.scaler.gamma*x[N+1:].copy()
        self.x = x
        self.J_func = self.scaler.old_J
        self.grad_J = self.scaler.old_grad
        #self.dJ=self.grad_J(self.x)
        

    def val(self):
        return self.J_func(self.x)
    def add_mu(self,mu):
        self.mu = mu

    def add_FuncGradCounter(self,arr):
        self.counter = FuncGradCounter(arr)
    def mpi_grad_norm(self):
        return self.dJ.l2_norm()
        
        
class FuncGradCounter():
    def __init__(self,arr):
        self.nfunc = arr[0]
        self.ngrad = arr[1]
    def __call__(self):
        return self.nfunc,self.ngrad


    
