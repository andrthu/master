import numpy as np
class PenaltyScaler():

    def __init__(self,J,grad_J,x0,m,factor=1):

        self.J = J
        self.grad_J = grad_J
        self.x0 = x0
        self.m = m
        self.N = len(x0)-m
        self.factor = factor
        self.gamma = self.find_gamma()
        self.old_J = J
        self.old_grad = grad_J
        
        
    def find_gamma(self):
        
        grad = self.grad_J(self.x0)
        N = self.N
        gamma = self.factor*np.max(abs(grad[:N+1]))/np.max(abs(grad[N+1:]))
        #gamma = ((self.m-1)/(N+1.))*np.sum(abs(grad[:N+1]))/np.sum(abs(grad[N+1:]))
        #gamma = np.sqrt(gamma)
        print gamma,np.max(abs(grad[N+1:])),np.min(abs(grad[N+1:]))
        print np.max(abs(grad[:N+1])),np.max(abs(grad[N+1:]))
        print  ((self.m-1)/(N+1.))*np.sum(abs(grad[:N+1]))/np.sum(abs(grad[N+1:]))
        return gamma

    def var(self,x):
        y = np.zeros(len(x))
        
        y[:self.N+1] = x[:self.N+1]
        y[self.N+1:] = x[self.N+1:]/self.gamma
        return y
    
    def grad(self,g):

        def new_grad(x):

            grad = self.grad_J(x)
            grad[self.N+1:] = self.gamma*grad[self.N+1:]
            return grad
        return new_grad

    def func_var(self,x):
        y = x.copy()
        y[self.N+1:] = self.gamma*y[self.N+1:]
        return y

    
