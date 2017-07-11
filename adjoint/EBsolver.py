from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps, trapz
from scipy import linalg, random

#Backward Euler solver for y'=ay +u y(0)=y0, endtime T and
#n discretization points.
def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    
    return y

#solving the adjoint equation -p=ap, p(T)=y(T)-yT
def adjoint_solver(y0,a,n,u,T,yT):
    dt = float(T)/n
    
    y=solver(y0,a,n,u,T)
    l=zeros(n+1)
    l[-1]=y[-1] -yT

    
    
    for i in range(n):
        l[-(i+2)]=(1+dt*a)*l[-(i+1)] 
        #l[-(i+2)]=l[-(i+1)]/(1-dt*a)
    return l

#Functional 0.5*(integral(u**2) +(y(T)-yT)**2)    
def Functional2(y,u,yT,T):
    t = linspace(0,T,len(u))

    return 0.5*(trapz(u**2,t) + (y[-1]-yT)**2) 

#Reduced Functinal dependent on u.
def J_red(u,a,y0,yT,T):
    return Functional2(solver(y0,a,len(u)-1,u,T),u,yT,T)


  
#finite fiffrence thing.    
def finite_diff(u,a,y0,yT,T,J):
    eps = 1./10000

    grad_J = zeros(len(u))

    for i in range(len(u)):
        e = zeros(len(u))
        e[i]=eps
        J1 = J(u,a,y0,yT,T)
        J2 = J(u+e,a,y0,yT,T)        
        grad_J[i] = (J2-J1)/eps

    return grad_J

#testing adjoint_solver for case T=a=y0=yT=1 and u=0. Returns max error
#for diffrent dt, and computes convergence rate. Also plots for N=50.
def test_exact():
    
    #define constants
    N=[50,100,500,1000]
    T=1
    a=1
    yT=1
    y0=1
    
    #exact solution
    l_exact= lambda x : (exp(1)-1)*exp(1-x)

    #array for storing
    error = zeros(len(N))
    h_val = zeros(len(N))

    #solve for diffrent dt
    for i in range(len(N)):

         
        u=zeros(N[i]+1)
        t = linspace(0,T,N[i]+1)
        l = adjoint_solver(y0,a,N[i],u,T,yT)

        
        #plot
        if i==0:
            plot(t,l)
            plot(t,l_exact(t),'g--')
            legend(['Numerical adjoint','Exact adjoint'])
            title('Numerical adjoint for ' + str(N[i]) + ' points')
            xlabel('t')
            ylabel('adjoint')
            show()
            
        h_val[i]=1./N[i]
        error[i]=max(abs(l-l_exact(t)))
        
    #Do least square stuff.    
    Q = vstack([log(h_val),ones(len(N))]).T
    LS=linalg.lstsq(Q, log(error))[0]
    return error,LS

def test_finiteDiff():
     
    #define constants
    N=[50,100,500,1000]
    T=1
    a=1
    yT=1
    y0=1
    
    #expression for u
    U = lambda x: exp(x) +x

    #array for storing
    error = zeros(len(N))
    h_val = zeros(len(N))

    #solve for diffrent dt
    for i in range(len(N)):

         
        
        t = linspace(0,T,N[i]+1)
        u = U(t)
        l = adjoint_solver(y0,a,N[i],u,T,yT)

        rel_grad = u+l
        rel_grad[0] = 0.5*u[0]
        rel_grad[-1]=0.5*u[-1]+l[-1]

        rel_gradFD = N[i]*finite_diff(u,a,y0,yT,T,J_red)
        
        #plot
        if i==0:
            plot(t,rel_grad)
            plot(t,rel_gradFD,'g--')
            legend(['adjoint approach','Finite difference approach'])
            title('Scaled numerical gradients for ' + str(N[i]) + ' points')
            xlabel('t')
            ylabel('Gradient')
            show()
            
        h_val[i]=1./N[i]
        error[i]=max(abs((rel_grad-rel_gradFD)))

        
    #Do least square stuff.    
    Q = vstack([log(h_val),ones(len(N))]).T
    LS=linalg.lstsq(Q, log(error))[0]
    return error,LS
    

if __name__ == '__main__':

    print test_exact()
    print test_finiteDiff()
    """
    n=1000
    m=10
    T=1
    t=linspace(0,T,n+1)
    u=zeros(n+1)
    y0=1
    a=2
    my=1000000
    lam = zeros(m-1)+1
    yT=1
    
    y = solver(y0,a,n,u,T)
    l = adjoint_solver(y0,a,n,u,T,yT)

    plot(t,y)
    plot(t,l)
    legend(['state','adjoint'])
    show()

    
    n=100
    t = linspace(0,1,n+1)
    T=1
    y0=1
    a=1
    u=exp(t)
    yT=10

    k=-1
    eps = zeros(n+1)
    eps[k] = 1

    
    
    y = solver(y0,a,n,u,T)
    print Functional2(y,u,yT,T)
    #print y[-1]-yT
    plot(t,y)

    l = adjoint_solver(y0,a,n,u,T,yT)
    l2 = n*finite_diff(u,a,y0,yT,T,J_red)
    #print l2
    u[-1]=0.5*u[-1]
    plot(t,u+l)
    plot(t,l2)

    
    print 2*n*trapz((l+u)*eps,t),(u[k]+l[k]), l2[0]
    show()
    """






