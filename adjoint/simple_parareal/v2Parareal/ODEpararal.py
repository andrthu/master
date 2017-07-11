import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def implicit_solver(y0,a,dt,N,f=None):
    """
    Solves the equation y'-ay = f y(0)=y0 using implicit euler
    and N steps of timestep dt 
    """
    
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(N+1)

    for i in range(N):
        y[i+1] = (y[i] +dt*f[i])/(dt*a+1)

    return y
def int_par_len(n,m,i):
    """
    With fine resolution n and m time decomposition intervals,
    functions find number of points in interval i
    """
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = N+1
        else:
            state = N
    else:
        if rest - i >0:
            state = N+2
        else:
            state = N+1
    return state

def propagator_iteration(y,c_y,a,y0,N,M,dt,dT,f=None,fc=None):
    """
    updates the interval initialvalues and the fine solution using
    the new initial values. 
    """
    """
    S = np.zeros(M+1)
    for i in range(M):
        S[i+1] = (y[i][-1] - c_y[i+1])/dT
    
    return y,c_y
    """
    S = np.zeros(M+1)
    for i in range(M):
        S[i+1] = (y[i][-1] - c_y[i+1])/dT

    delta = implicit_solver(0,a,dT,M,f=S)
    for i in range(M):
        c_y[i+1] = y[i][-1] + delta[i+1]

        
    y=[]
    for i in range(M):
        y.append(implicit_solver(c_y[i],a,dt,int_par_len(N+1,M,i)-1,f=f))
    
    return y,c_y
    #"""

def parareal_solver(y0,a,T,M,N,order=3,f=None,fc=None):
    """
    implementation of the parareal scheme for our simple ODE
    using N+1 as fine resolution and M time decompositions, and
    doing k=order iterations
    """
    
    dt = float(T)/N
    dT =  float(T)/M
    coarse_y = implicit_solver(y0,a,dT,M,f=fc)
    
    y=[]
    for i in range(M):
        y.append(implicit_solver(coarse_y[i],a,dt,int_par_len(N+1,M,i)-1))    
    
    for k in range(order-1):
        y,coarse_y=propagator_iteration(y,coarse_y,a,y0,N,M,dt,dT,f=f,fc=fc)
        
    Y = gather_y(y,N)
    return Y

def gather_y(y,N):
    """
    Gathers tje y arrays into one array Y
    """
    Y = np.zeros(N+1)
    
    start = len(y[0])
    Y[:start] = y[0][:]
    
    for i in range(len(y)-1):
        
        end = start + len(y[i+1])-1
        
        Y[start:end] = y[i+1][1:]
        start = end
    return Y

if __name__ == '__main__':
    N = 1000
    M = 10
    T = 1.
    dt = T/N
    y0 = 1
    a = 2
    t = np.linspace(0,T,N+1)
    f = np.zeros(N+1)-30#np.sin(2*np.pi*t)

    y = implicit_solver(y0,a,dt,N,f=f)
    Y = parareal_solver(y0,a,T,M,N,order=4,f=f,fc=None)
    plt.plot(t,y,'r--')
    plt.plot(t,Y)
    plt.show()
