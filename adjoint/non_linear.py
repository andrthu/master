from  optimalContolProblem import *
from ODE_pararealOCP import PararealOCP
import numpy as np
from scipy.integrate import trapz

#from taylorTest import non_lin_state#(y0,yT,T,F,DF)

class Explicit_quadratic(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        #return (y[i] +dt*u[j+1])/(1.-dt*a)
        
        
        return y[i] - dt*(a*y[i]**2 - u[j])


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]
        #return (1-dt*a)*l[-(i+1)]
        return (1-a*2*dt*y[-(i+2)])*l[-(i+1)]


    def Penalty_Gradient2(self,u,N,m,mu):

        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
        a = self.a
        g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

        for j in range(m-1):
            g[Nc+1+j]= (l[j+1][0]*(1-dt*2*a*u[Nc+1+j]) - l[j][-1])
                    
        return g


class Explicit_sine(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        #return y[i] +dt*(a*y[i] +u[j])
        return y[i] + dt*(a*np.sin(y[i]) + u[j])


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]

        return (1+dt*a*np.cos(y[-(i+2)]))*l[-(i+1)]
    
class ExplicitNonLinear(PararealOCP):

    def __init__(self,y0,yT,T,F,DF,J,grad_J,options=None):

        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.F  = F
        self.DF = DF

    def ODE_update(self,y,u,i,j,dt):
        F = self.F
        
        
        return y[i] - dt*(F(y[i]) - u[j])


    def adjoint_update(self,l,y,i,dt):
        DF = self.DF
        

        return (1-dt*DF(y[-(i+2)]))*l[-(i+1)]


    def adjoint_ppc_update(self,l,y,i,dt):
        return self.adjoint_update(l,y,i,dt)


    def ODE_ppc_update(self,y,u,i,j,dt):
        
        DF = self.DF
        
        return (1-dt*DF(u[i]))*y[i]

    def NL_PC_maker(self,N,m,step=1,mu=1.):
        #"""
        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x.copy()[:]
            
            dT = float(self.T)/m
            dt = float(self.T)/N
            
            
            for i in range(1,m-1):
                lam = np.zeros(step+1) + x[-(i+1)]
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-i],dT,step=step,lam=lam)

            
            for i in range(1,m-1):
                lam = np.zeros(step+1) + x[i]
                S[i] = S[i] + self.ODE_step(S[i-1],dT,step=step,lam=lam)
            #print S
            #time.sleep(1)
            S = S
            x[:]=S.copy()[1:-1]
            return x
        #"""
        #pc = lambda x:x
        return pc

    def non_lin_PC_maker(self,N,m,step=1,mu=1.):      

        def pc(x):
            Nc = len(x)-m
            lam_pc = self.NL_PC_maker(N,m,step,mu)
            lam = x.copy()[Nc+1:]
            lam2 = lam_pc(lam)
            x[Nc+1:]= lam2.copy()[:]
            return x
        return pc




    PC_creator = non_lin_PC_maker

def non_lin_state(y0,yT,T,F,DF):
    
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[:-1] = dt*(u[:-1]+p[1:])
        grad[0] = 0.5*dt*(u[0])+dt*p[1] 
        grad[-1] = 0.5*dt*(u[-1]) 
        return grad
        

    problem = ExplicitNonLinear(y0,yT,T,F,DF,J,grad_J)
    

    return problem



def test_nonLinear():

    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 2
    T  = 1
    N1 = 2000
    N2 = 5000
    m=10
    a=0.1
    #"""
    #F  = lambda x : a*np.exp(-x)*np.cos(x*np.pi)
    
    #DF = lambda x : -a*np.exp(-x)*(np.pi*np.sin(np.pi*x)+np.cos(np.pi*x))

    F = lambda x : np.sin(x)
    DF = lambda x: np.cos(x)
    """
    F = lambda x : np.exp(-x**2)
    DF = lambda x : -2*x*np.exp(-x**2)
    """
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = non_lin_state(y0,yT,T,F,DF)

    opt = {"mem_lim":10,'jtol':1e-5}
    opt2 = {'jtol':1e-6}
    #res1=problem.plot_solve(N1,opt=opt,state= True)
    res1=problem.PPCLBFGSsolve(N1,m,[N1],options=opt,)
    res2 = problem.penalty_solve(N1,m,[N1],Lbfgs_options=opt2)
    #res2=problem.solve(N2,Lbfgs_options=opt)
    res3=problem.solve(N1,Lbfgs_options=opt2,)
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(res1['control'][:N1+1],'--')
    plt.plot(res2.x[:N1+1])
    plt.plot(res3.x,'-.')
    #plt.plot(t2,res2['control'],"r--")
    print res1.counter(),res2.counter(),res3.counter()
    """
    try:
        res3=problem.penalty_solve(N1,m,[N1],Lbfgs_options=opt)
        print res1['iteration'],res3['iteration']
        plt.plot(t1,res3['control'][:N1+1])
    except:
        #print res1['iteration'],res2['iteration']
        print res1['iteration']
    """
    plt.show()
    
def test_sine():

    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 2
    T  = 3
    a  = 1
    N1 = 1000
    N2 = 2000
    m=2
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        grad = np.zeros(len(u))
        grad[:-1] = dt*(u[:-1]+p[1:])
        grad[0] = 0.5*dt*(u[0])+dt*p[1] 
        grad[-1] = 0.5*dt*(u[-1]) 
        return grad
        #return dt*(u+p)


    problem = Explicit_sine(y0,yT,T,a,J,grad_J)

    opt = {"mem_lim":10,'jtol':1e-7}
    res1=problem.solve(N1,Lbfgs_options=opt)
    res2=problem.solve(N2,Lbfgs_options=opt)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(t1,res1['control'])
    plt.plot(t2,res2['control'],"r--")

    
    try:
        res3=problem.penalty_solve(N1,m,[N1,N1**2],Lbfgs_options={'jtol':1e-5})[-1]
        #print res1['iteration'],res2['iteration'],res3['iteration']
        print res3.counter(),res1.counter()
        print
        print "number of iterations for m=%d and N=%d: %d"%(1,N1,res1['iteration'])
        print "number of iterations for m=%d and N=%d: %d"%(1,N2,res2['iteration'])
        print "number of iterations for m=%d and N=%d: %d"%(m,N1,res3['iteration'])
        print
        plt.plot(t1,res3['control'][:N1+1])
        plt.legend(['m=1,N='+str(N1),'m=1,N='+str(N2),'m='+str(m)+',N='+str(N1)])
    except:
        print 'lol'
    plt.show()
    
def test_quad():
    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 5
    T  = 1
    a  = 1
    N1 = 1000
    N2 = 2000
    m=30
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)

    opt = {"mem_lim":80}
    res1=problem.solve(N1,Lbfgs_options=opt)
    res2=problem.solve(N2,Lbfgs_options=opt)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(t1,res1['control'])
    plt.plot(t2,res2['control'],"r--")

    
    
    res3=problem.penalty_solve(N1,m,[0.5*N1],Lbfgs_options=opt)
    #res3=problem.PPCLBFGSsolve(N1,m,[0.5*N1],options=opt)
    #print res1['iteration'],res2['iteration'],res3['iteration']
    print res3.counter(),res1.counter()
    print
    print "number of iterations for m=%d and N=%d: %d"%(1,N1,res1['iteration'])
    print "number of iterations for m=%d and N=%d: %d"%(1,N2,res2['iteration'])
    print "number of iterations for m=%d and N=%d: %d"%(m,N1,res3['iteration'])
    print
    plt.plot(t1,res3['control'][:N1+1])
    plt.legend(['m=1,N='+str(N1),'m=1,N='+str(N2),'m='+str(m)+',N='+str(N1)],loc=2)
    plt.show()
    
if __name__ == "__main__":

    test_quad()
    #test_sine()
    #test_nonLinear()
    
"""
terminal> python non_linear (test_quad())

number of iterations for m=1 and N=1000: 11
number of iterations for m=1 and N=2000: 11
number of iterations for m=30 and N=1000: 53


"""
