from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem,Problem1

from ODE_pararealOCP import PararealOCP


class RungeKuttaProblem(PararealOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):
 
        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a

        


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        k1 = a*y[i] + u[j]
        k2 = a*(y[i]+0.5*dt*k1) +0.5*(u[j]+u[j+1])
        k3 = a*(y[i]+0.5*dt*k2)+0.5*(u[j]+u[j+1])
        k4 = a*(y[i]+dt*k3) + u[j+1]

        return y[i] +dt*(k1+2*k2+2*k3+k4)/6.
        
    def adjoint_update(self,l,y,i,dt):
        a = self.a

        k1 = a*l[-(i+1)]
        k2 = a*(l[-(i+1)]+0.5*dt*k1)
        k3 = a*(l[-(i+1)]+0.5*dt*k2) 
        k4 = a*(l[-(i+1)]+dt*k3)

        return l[-(i+1)] + dt*(k1+2*(k2+k3)+k4)/6.
        #return l[-(i+1)]/(1.-dt*a) 


    def adjoint_propogator_update(self,l,rhs,i,dt):
        a = self.a
        return (l[-(i+1)]+rhs[-(i+1)])/(1-dt*a)
    """
    def ODE_ppc_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)
        
        
    def adjoint_ppc_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a) 
    """
if __name__=='__main__':
    import matplotlib.pyplot as plt

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)




    

    problem = RungeKuttaProblem(1,1,1,1,J,grad_J)
    
    problem2 = Problem1(1,1,1,1,J,grad_J)



    N = 100

    e1 = lambda x: np.exp(x)
    t = np.linspace(0,1,N+1)
    u = np.zeros(N+1)

    y = problem.ODE_solver(u,N)
    
    #plt.plot(t,e1(t))
    #plt.plot(t,y)
    print max(abs(y-e1(t)))
    #plt.show()

    u2 = np.zeros(N+1)+1

    e2 = lambda x : 2*np.exp(x) - 1

    y2 = problem.ODE_solver(u2,N)

    print max(abs(y2-e2(t)))


    res = problem.solve(N)
    res2= problem2.solve(N)
    plt.plot(t,res.x)
    plt.plot(t,res2.x)
    print max(abs(res.x-res2.x))
    plt.show()
