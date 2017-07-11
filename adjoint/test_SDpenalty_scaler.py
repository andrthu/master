import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem,Problem3
from scipy.integrate import trapz
import matplotlib.pyplot as plt
def J(u,y,yT,T,alp):
    t = np.linspace(0,T,len(u))

    I = trapz(u**2,t)
        
    return 0.5*(I + alp*(y-yT)**2)

def grad_J(u,p,dt,alp):
    return dt*(u+p)



T =  1
y0 = 1.2
a =  0.9
yT = 5
alpha = 0.5

N = 500
m = 10
mu = 1
mu_list = [1,10]
opt = {'maxiter':500}
problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
res1=problem.penalty_solve(N,m,mu_list,algorithm='my_steepest_decent')

JJ,grad_JJ = problem.generate_reduced_penalty(1./500,N,m,mu)
x0 = np.zeros(N+m)
x0[N+1:] = 0
solver2 = SteepestDecent(JJ,grad_JJ,x0,
                         options=opt,scale={'m':m})
res2 = solver2.solve()

solver3 = SteepestDecent(JJ,grad_JJ,x0,options=opt)
#res3 = solver3.solve()
res3 = problem.penalty_solve(N,m,mu_list,algorithm='my_steepest_decent',
                             scale=True)
#print res1.niter,res2.niter,res3.niter
for i in range(len(mu_list)):
    print res1[i].niter,res3[i].niter
plt.figure()
plt.plot(res1[-1].x[N+1:],'>r')
#plt.plot(res2.x[N+1:],'b--')
plt.plot(res3[-1].x[N+1:])
plt.plot(res2.x[N+1:]*res2.scaler.gamma,'b--')
plt.show()

plt.figure()
plt.plot(res1[-1].x[:N+1],'r')
plt.plot(res2.x[:N+1],'b-')
plt.plot(res3[-1].x[:N+1],'g--')
plt.show()
