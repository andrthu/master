from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import Problem1
import matplotlib.pyplot as plt


y0 = 1.3
yT = 3
a = 0.4
T = 0.7

def J(u,y,yT,T):
    t = np.linspace(0,T,len(u))

    I = trapz(u**2,t)

    return 0.5*(I + (y-yT)**2)

def grad_J(u,p,dt):
    return dt*(u+p)

problem = Problem1(y0,yT,T,a,J,grad_J)

N = 1000
m = 3
res =problem.penalty_solve(N,m,[1],algorithm='split_lbfgs')
print res.niter
res2 = problem.penalty_solve(N,m,[1],algorithm='my_lbfgs')
print res2['iteration']
plt.plot(res.x[:N+1])
plt.plot(res2['control'].array()[:N+1])
plt.show()


