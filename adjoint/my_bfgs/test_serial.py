from my_vector import *
from lbfgs import Lbfgs,MuLbfgs
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import trapz
from scipy.optimize import minimize


def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = np.zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    
    return y

def adjoint_solver(y0,a,n,u,T,yT):
    dt = float(T)/n
    
    y=solver(y0,a,n,u,T)
    l=np.zeros(n+1)
    l[-1]=y[-1] -yT

    
    
    for i in range(n):
        l[-(i+2)]=(1+dt*a)*l[-(i+1)] 
    return l

def Func(u,a,y0,yT,T):
    
    t = np.linspace(0,T,len(u))
    y = solver(y0,a,len(u)-1,u,T)
    return 0.5*(trapz(u**2,t) + (y[-1]-yT)**2)

def L2_grad(u,l,dt):
    return dt*(l+u)

def l2_error(u,v,dt):

    S = np.sum( (u-v)**2)
    
    return dt * np.sqrt(S)

def opti(y0,a,T,yT,n,F,sol,adj,gr,printplot=False):

    t=np.linspace(0,T,n+1)
    dt=float(T)/n
    
    x0     = SimpleVector(np.zeros(n+1))
    szi_x0 = np.zeros(n+1)
    
    

    def J(u):
        return F(u,a,y0,yT,T)

    def grad_J(u):
        l = adj(y0,a,len(u)-1,u,T,yT)
        return gr(u,l,dt)
    
    def Mud_J(u):
        
        y  = SimpleVector(grad_J(u))
        y2 = SimpleVector(np.zeros(len(u)))
        s  = SimpleVector(u)
        s2 = SimpleVector(np.zeros(len(u)))
        
        
        return MuVector([y,y2]),MuVector([s,s2])

    res1 = minimize(J,szi_x0,method='L-BFGS-B', jac=grad_J,
                   options={'gtol': 1e-6, 'disp': False})

    options={"beta":1,"mem_lim" : 10,"return_data":True,"jtol": 1e-6,}

    S1 = Lbfgs(J,grad_J,x0,options=options)
    
    S2 = MuLbfgs(J,grad_J,x0,Mud_J,options=options)

    res2 = S1.solve()
    res3 = S2.solve()
    x1 = res1.x
    x2 = res2['control'].array()
    x3 = res3['control'].array()
    
    if printplot == True:
        print res1.nit,res2['iteration'],res3['iteration']
        
        print l2_error(x1,x2,dt)
        print l2_error(x1,x3,dt)
        print l2_error(x2,x3,dt)

        plot(t,x1)
        plot(t,x2)
        plot(t,x3)
        legend(['scipy','lbfgs','mu'])
        show()

    return res1,res2,res3

if __name__ == '__main__':

    opti(1,1,1,1,1000,Func,solver,adjoint_solver,L2_grad,printplot=True)

    
    
    
    T = 1
    yT = 1
    y0 =1

    N = [50,100,300,800,1000,10000]
    a = [-1,0.1,1,2,10]
    
    
    tol = 1e-12

    for i in range(len(N)):

        t = np.linspace(0,T,N[i]+1)
        dt = 1./N[i]
        for j in range(len(a)):
            try:
                res1,res2,res3 = opti(y0,a[j],T,yT,N[i],Func,solver,
                                      adjoint_solver,L2_grad)
            
                x1 = res1.x
                x2 = res2['control'].array()
                x3 = res3['control'].array()

                if res1.nit!=res2['iteration'] or res1.nit!=res3['iteration']:
                    print
                    print "iteration difference" 
                    print N[i], a[j]
                    print res1.nit,res2['iteration'],res3['iteration']
        
                    print l2_error(x1,x2,dt)
                    print l2_error(x1,x3,dt)
                    print l2_error(x2,x3,dt)
                    print
                    
                    plot(t,x1)
                    plot(t,x2)
                    plot(t,x3)
                    legend(['scipy','lbfgs','mu'])
                    show()

                elif l2_error(x1,x2,dt)>tol or l2_error(x1,x3,dt)>tol:
                    print
                    print "error difference" 
                    print N[i], a[j]
                    print res1.nit,res2['iteration'],res3['iteration']
        
                    print l2_error(x1,x2,dt)
                    print l2_error(x1,x3,dt)
                    print l2_error(x2,x3,dt)
                    print
                    
                    plot(t,x1)
                    plot(t,x2)
                    plot(t,x3)
                    legend(['scipy','lbfgs','mu'])
                    show()

                    
            except:
                print
                print N[i], a[i]
                print
            
