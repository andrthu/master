import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taylorTest import lin_problem


def unstable():

    a = -.2
    T = 100

    y0=300
    yT= 100

    problem,_ = lin_problem(y0,yT,T,a,c=10,implicit=False)
    
    N = 10000
    m = 30
    res = problem.PPCLBFGSsolve(N,m,[100,1000],options={'jtol':1e-5})[-1]
    #res = problem.penalty_solve(N,m,[100,1000],Lbfgs_options={'jtol':1e-6})[-1]
    res2 = problem.solve(N,Lbfgs_options={'jtol':1e-10})
    print res2.counter(),res.counter(),max(abs(res2.x-res.x[:N+1]))
    Y  =problem.ODE_solver(np.zeros(m+1),m)
    t = np.linspace(0,T,N+1)
    #Y = problem.ODE_solver(u,N)
    plt.plot(t,res.x[:N+1])
    plt.plot(t,res2.x,'--')
    #plt.plot(np.linspace(0,T,m+1),Y)
    plt.show()

def backwards(x,val):
    
    y = np.zeros(len(x))
    y = x.copy()
    for i in range(1,len(x)):
        y[-(i+1)] = y[-(i+1)] + val*y[-i]

    return y

def foreward(x,val):
    
    y = np.zeros(len(x))
    y = x.copy()
    for i in range(1,len(x)):
        y[i] = y[i] + val*y[i-1]

    return y



def test_PC_creator():
    m = 100
    
    T = 100
    a = -float(m)/T -0.1
    y0=300
    yT= 100

    problem,_ = lin_problem(y0,yT,T,a,c=10,implicit=True)
    
    
    N =10000
    x = 10*(np.random.random(m-1)-0.5)
    res = problem.PPCLBFGSsolve(N,m,[1])
    pc = problem.PC_maker4(N,m)
    val = 1./(1-a*T/float(m))
    #val = (1+a*T/float(m))
    y=pc(x.copy())
    y1=backwards(x,val)
    y2 = foreward(y1,val)
    plt.plot(x)
    plt.plot(y1)
    plt.plot(y2)
    plt.plot(y,'.')
    plt.show()


    
    print res.counter()
    plt.plot(res.x[:N+1])
    #plt.show()

if __name__ == '__main__':
    #unstable()
    test_PC_creator()
