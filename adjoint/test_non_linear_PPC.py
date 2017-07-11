import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
from parallelOCP import v_comm_numbers

from non_linear import non_lin_state


def quad_PPC_test():
    
    y0=1.
    yT=-2.
    T=1.
    a = 0.001
    F = lambda x: a*x**2
    DF = lambda x: a*2*x

    problem = non_lin_state(y0,yT,T,F,DF)
    

    N = 10000
    opt = {'jtol':1e-10}
    opt2 ={'jtol':1e-6}
    res = problem.solve(N,Lbfgs_options=opt)
    
    print res.counter()

    mu_list = [.01*N,0.1*N,N]
    tol=[1e-5,1e-6,1e-8]
    M = [64,164]

    for m in M:

        res2 = problem.PPCLBFGSsolve(N,m,mu_list,tol_list=tol,options=opt2)[-1]
        #res2 = problem.penalty_solve(N,m,mu_list,Lbfgs_options=opt2)[-1]
        plt.plot(res2.x[:N+1])
        print m,res2.counter(),max(abs(res.x-res2.x[:N+1]))
        

    y = problem.ODE_solver(res.x,N)
    plt.plot(res.x)
    plt.plot(y,'--')
    plt.show()

if __name__ == '__main__':

    quad_PPC_test()
