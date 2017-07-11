from  optimalContolProblem import *
from ODE_pararealOCP import PararealOCP
import numpy as np
from scipy.integrate import trapz
from taylorTest import finite_diff,general_taylor_test
import pandas as pd
from parallelOCP import u_part,interval_partition
class StateIntegralProblem(OptimalControlProblem):

    def __init__(self,y0,T,yT,a,J,grad_J,z=None,options=None,implicit=True):

        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=implicit)

        self.a = a
        if z == None:
            self.z = lambda x : 0*x
            self.help_z = 0
        else:
            self.z = z
            self.help_z=1

    def ODE_update(self,y,u,i,j,dt):
        a = self.a

        
        return (y[i]+dt*u[j+1])/(1.-dt*a)
        


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        z = self.z
        help_z = self.help_z
        #print len(l),len(y),len(self.t)
        return (l[-(i+1)]+dt*(y[-(i+1)]-z(self.t[-help_z*(i+1)])))/(1.-dt*a)


        
    def func_state_part(self,u,N):
        return self.ODE_solver(u,N)
    def penalty_func_state_part(self,y,Y):
        return Y

    def decompose_time(self,N,m):
        t = np.linspace(0,self.T,N+1)

        T_z = []

        for i in range(m):
            ti = interval_partition(N+1,m,i)
            s = u_part(N+1,m,i)
            for j in range(len(ti)):
                ti[j] = t[s+j]
            T_z.append(ti.copy())

        return t,T_z

def create_stateIntegralProblem(y0,T,a,z):

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))
        dt = float(T)/(len(u)-1)
        I1 = trapz((u)**2,t)
        I2 = sum(dt*(y-z(t))**2)
        #I2 = trapz((y-z(t))**2,t)
        return 0.5*(I1+I2) + 0.5*(y[-1]-yT)**2

    def grad_J(u,p,dt):

        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0.5*dt*(u[0]) 
        grad[-1] = 0.5*dt*(u[-1]) + dt*p[-2]
        return grad
        
    yT = 0
    problem = StateIntegralProblem(y0,T,yT,a,J,grad_J,z=z)
    return problem

def test_stateI():
    
    y0 = 1
    a = 1
    T = 1
    z = lambda x : (1-x)*10
    
    problem = create_stateIntegralProblem(y0,T,a,z)

    N = 1000

    res = problem.solve(N)
    print res.counter()

    y = problem.ODE_solver(res.x,N)
    import matplotlib.pyplot as plt

    plt.plot(res.x)
    plt.plot(y,'--')
    plt.show()

def test_penaty_stateI():
    y0 = 1
    a = 1
    T = 1
    z = lambda x : (1-x)*10
    
    problem = create_stateIntegralProblem(y0,T,a,z)

    N = 1000
    m = 5
    
    res = problem.penalty_solve(N,m,[1,100,1000,10000])[-1]
    print res.counter()

    y,Y = problem.ODE_penalty_solver(res.x,N,m)
    import matplotlib.pyplot as plt

    plt.plot(res.x[:N+1])
    plt.plot(Y,'--')
    plt.show()

def taylorTest():

    y0 = 1
    a = 1
    T = 1
    z = lambda x : 10*(1-x)
    
    problem = create_stateIntegralProblem(y0,T,a,z)


    N = 100
    dt = float(T)/(N)
    
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(8):

        print J(u+h/(10**i))-J(u)

    problem.t = np.linspace(0,T,N+1)
    def grad_J(x):
        l = problem.adjoint_solver(x,N)
        return problem.grad_J(x,l,dt)
    
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(8):
        eps = 1./(10**i)
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)
        eps_list.append(eps)
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        #print max(abs(grad_fd[:]-grad[:]))
    
    data2 = pd.DataFrame(table,index=eps_list)
    #data2.to_latex('report/draft/discertizedProblem/taylorTest1.tex')
    
    print data2
    import matplotlib.pyplot as plt
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.legend(['num grad','finite diff grad'])
    plt.xlabel('gradient index')
    plt.ylabel('gradient value')
    plt.show()
    return

def tayorTest2():

    y0 = 1
    a = 1
    T = 1
    z = lambda x : (1-x)*10
    
    problem = create_stateIntegralProblem(y0,T,a,z)

    general_taylor_test(problem)


if __name__=='__main__':
    
    #test_stateI()
    #taylorTest()
    #test_penaty_stateI()
    tayorTest2()
