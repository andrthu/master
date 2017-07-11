import numpy as np
from linesearch.strong_wolfe import *
from my_vector import SimpleVector

def do_ls(J,d_J,x,p):
    
    x_new = x.copy()
    """
    def update_x_new(alpha):
        if update_x_new.alpha_new != alpha:
            x_new = x.copy()
            x_new = x_new + alpha*p
            update_x_new.alpha_new = alpha
    
    update_x_new.alpha_new=0
    """
    

    def phi(alpha):
        #update_x_new(alpha)
        x_new=x.copy()
        x_new.axpy(alpha,p)
        return J(x_new.array())
    
    def phi_dphi(alpha):
        #update_x_new(alpha)
        x_new = x.copy()
        
        x_new.axpy(alpha,p)
        
        f = J(x_new.array())
        djs = p.dot(SimpleVector(d_J(x_new.array())))
        
        return f,float(djs)
    
    phi_dphi0 = J(x.array()),float(p.dot(SimpleVector(d_J(x.array()))))
    #print phi_dphi0
    sw =  StrongWolfeLineSearch(start_stp=1.0,xtol=0.00001,ignore_warnings=False)

    alpha = sw.search(phi, phi_dphi, phi_dphi0)

    #update_x_new(alpha)
    x_new=x.copy()
    x_new.axpy(alpha,p)
    
    return x_new, float(alpha)

    
    
    
def bfgs(J,x0,d_J,tol,beta=1,max_iter=1000):
    
    
    n=x0.size()
    x = SimpleVector(np.zeros(n))
    
    I = np.identity(n)
    H = beta*I
    
    df0 =SimpleVector(d_J(x0.array()))
    df1 = SimpleVector(np.zeros(n))
    
    iter_k=0
    
    p  = SimpleVector(np.zeros(n))
    


    while np.sqrt(np.sum((df0.array())**2))/n>tol and iter_k<max_iter:

        
        p.set(-df0.matDot(H))
        
        #print p
        x,alfa = do_ls(J,d_J,x0,p)
        
        #print x, alfa
        

        df1.set(d_J(x.array()))
        
        s = x-x0
        y = df1-df0

        rho =1./(y.dot(s))
        V = I-rho*s.vecVecMat(y)
        
        H = V.dot(H).dot(V.T) + rho*s.vecVecMat(s)
        
        x0=x.copy()
        df0=df1.copy()
        #print df0.array()
        #print H
        iter_k=iter_k+1



    return x


if __name__ == "__main__":

    def J(x):

        s=0
        for i in range(len(x)):
            s = s + (x[i]-1)**2
        return s


    def d_J(x):

        return 2*(x-1)

    x0=SimpleVector(np.linspace(1,30,30))
    tol = 0.000001
    x=bfgs(J,x0,d_J,tol,beta=1,max_iter=1000)
    
    print x.array()
