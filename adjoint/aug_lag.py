from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps

from scipy.optimize import minimize

def solver(y0,a,n,u,T):
    dt = float(T)/(2*n)
    
    lam = u[-1]
    y1 = zeros(2*n+1)
    y2 = zeros(2*n+1)
    y1[0]=y0
    y2[n]=lam
    for i in range(n):
        y1[i+1] = (y1[i] +dt*u[i+1])/(1.-dt*a)
        y2[n+i+1]  =(y2[n+i] +dt*u[n+i+1])/(1.-dt*a)
        #y2[i]=y0
        #y1[n+i+1]=y0
    return y1,y2

def adjoint_solver(y0,a,n,u,T,yT,my):
    dt = float(T)/(2*n)
    
    y1,y2=solver(y0,a,n,u,T)
    l1=zeros(2*n+1)
    l2=zeros(2*n+1)
    
    l2[-1]=y2[-1] -yT
    l1[n] = my*(y1[n]-u[-1])
    print l1[n], y1[n],u[-1]

    
    
    for i in range(n):
        l2[-(i+2)]=(1+dt*a)*l2[-(i+1)]
        l1[n-i-1]= (1+dt*a)*l1[n-i]
        
    return l1,l2


def Functional2(y1,y2,u,yT,T,my):
    t = linspace(0,T,len(u)-1)

    return 0.5*(simps(u[:-1]**2,t) + (y2[-1]-yT)**2 +my*(y1[(len(u)-2)/2]-u[-1])**2)

def J_red(u,a,y0,yT,T,my):
    y1,y2=solver(y0,a,n,u,T)
    return Functional2(y1,y2,u,yT,T,my)

def mini_solver(y0,a,T,yT,n,my0):
    t=linspace(0,T,2*n+1)
    x = zeros(2*n+2)
    for k in range(5):
        def J(u):
            y1,y2=solver(y0,a,n,u,T)
            return Functional2(y1,y2,u,yT,T,10**(k)*my0)
        
        def grad_J(u):
            l1,l2 = adjoint_solver(y0,a,n,u,T,yT,10**(k)*my0)
            g =zeros(len(u))
            eps=zeros(len(u)-1)
            eps[len(u)/2-1] =l1[len(u)/2-1] 
            g[:-1] = (u[:-1]+l1+l2-eps)/(len(u)-2)
            g[-1] = l2[len(u)/2-1]-l1[len(u)/2-1]
            return g
    
        res = minimize(J,x,method='L-BFGS-B', jac=grad_J,
                        options={'gtol': 1e-6, 'disp': True})

        u=res.x
        x=u
        print res.x
        print res.message

        
        y1,y2 = solver(y0,a,n,u,T)
        eps=zeros(len(u)-1)
        eps[len(u)/2-1] = y1[len(u)/2-1]
        y=y1+y2-eps
        plot(t,u[:-1])
        plot(t,y)
        print J(u)
        show()
 

if __name__ == '__main__':
    y0 =1
    a = 1
    n = 500   
    u = zeros(2*n+2)
    u[-1]=1
    T=1
    my0=1
    yT = 10
    
    t=linspace(0,T,2*n+1)
    
    y1,y2=solver(y0,a,n,u,T)
    l1,l2=adjoint_solver(y0,a,n,u,T,yT,my0)
    #plot(t,l1)
    #plot(t,l2)
    #plot(t,y1)
    #plot(t,y2)
    show()
    print l1[len(u)/2-1],l2[len(u)/2]
    mini_solver(y0,a,T,yT,n,my0)
