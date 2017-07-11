from ODE_pararealOCP import PararealOCP
import numpy as np
from scipy.integrate import trapz
from parallelOCP import u_part,interval_partition
class CrankNicolsonOCP(PararealOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):
 
        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a

    

    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        return (y[i]*(1+0.5*dt*a)+0.5*dt*(u[j]+u[j+1]))/(1-0.5*dt*a)
        
    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return (1+0.5*dt*a)*l[-(i+1)]/(1-0.5*dt*a)

    def implicit_gather2(self,l,N,m):
        L=np.zeros(N+1)
        #"""
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            if i!=0:
                L[start] = 0.5*(l[i][0]+l[i-1][-1])
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
        return L

class CrankNicolsonStateIntOCP(PararealOCP):
    def __init__(self,y0,yT,T,a,J,grad_J,z=None,options=None):
 
        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=True)

        self.a = a
        
        if z == None:
            self.z = lambda x : 0*x
            self.help_z = 0
        else:
            self.z = z
            self.help_z=1

        self.dt = 1./100

    def initial_adjoint(self,y):
        """
        Initial condition for adjoint equation. Depends on the Functional
        """
        return self.dt*y
        return y - self.yT + self.dt*y

    def initial_penalty(self,y,u,my,N,i,m=1):
        """
        initial conditian for the adjoint equations, when partitioning in time
        """
        return my*(y[i][-1]-u[-m+i+1]) +(self.T/float(N))*y[i][-1]



    def implicit_gather(self,l,N,m):
        L=np.zeros(N+1)
        L[0] = l[0][0]
        start = 0
        L[start:start+len(l[0])-1] = l[0][1:]
        start += len(l[0])-1
        for i in range(1,m):
            L[start:start+len(l[i])-1] = l[i][1:]
            start += len(l[i])-1
        return L
        
        


    def ODE_update(self,y,u,i,j,dt):
        a = self.a

        
        return (y[i]*(1+0.5*dt*a)+0.5*dt*(u[j]+u[j+1]))/(1-0.5*dt*a)
        


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        z = self.z
        help_z = self.help_z
        #return (1+0.5*dt*a)*l[-(i+1)]/(1-0.5*dt*a)
        A = 1+0.5*a*dt
        B = 1.-0.5*dt*a

        return (A*l[-(i+1)]/B+dt*(y[-(i+2)]-z(self.t[-help_z*(i+1)])))


        
    def func_state_part(self,u,N):
        self.dt = self.T/float(N)
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

    def Gradient(self,u,N):
        l = self.adjoint_solver(u,N)
        dt = float(self.T)/N
        y=self.ODE_solver(u,N)
        return self.grad_J(u,l,dt,y)

    def Penalty_Gradient(self,u,N,m,mu):
        
        y,Y = self.ODE_penalty_solver(u,N,m)
        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
            
        g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt,Y)

        for j in range(m-1):
            g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
        return g

    Penalty_Gradient2 = Penalty_Gradient

def create_simple_CN_problem(y0,yT,T,a,c=0):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u-c)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
        

        B = 1-a*dt*0.5
        A = 1+0.5*a*dt

        t = np.linspace(0,T,len(u))
        grad1 = np.zeros(len(u))
        grad1[:-1] = dt*(u[:-1]-c+p[1:]/B)
        grad1[0] = 0.5*dt*(u[0]-c)+dt*p[1]/B
        grad1[-1] = 0.5*dt*(u[-1])

        grad2 = np.zeros(len(u))
        grad2[1:] = dt*(u[1:]-c+p[1:]/B)
        grad2[0] = 0.5*dt*(u[0]-c)
        grad2[-1] = 0.5*dt*(u[-1]-c)+dt*p[-1]/B

        return 0.5*(grad1+grad2)
        

        



    problem = CrankNicolsonOCP(y0,yT,T,a,J,grad_J)
    return problem

def create_noise_CN_problem(y0,yT,T,a,f):

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u-f(t))**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
        

        B = 1-a*dt*0.5
        A = 1+0.5*a*dt

        t = np.linspace(0,T,len(u))
        grad1 = np.zeros(len(u))
        grad1[:-1] = dt*(u[:-1]-f(t[:-1])+p[1:]/B)
        grad1[0] = 0.5*dt*(u[0]-f(t[0]))+dt*p[1]/B
        grad1[-1] = 0.5*dt*(u[-1] -f(t[-1]))

        grad2 = np.zeros(len(u))
        grad2[1:] = dt*(u[1:]-f(t[1:])+p[1:]/B)
        grad2[0] = 0.5*dt*(u[0]-f(t[0]))
        grad2[-1] = 0.5*dt*(u[-1]-f(t[-1]))+dt*p[-1]/B

        return 0.5*(grad1+grad2)
        

        



    problem = CrankNicolsonOCP(y0,yT,T,a,J,grad_J)
    return problem





def create_state_CN_problem(y0,yT,T,a,z):

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))
        dt = float(T)/(len(u)-1)
        I1 = trapz((u)**2,t)
        #I2 = trapz((y-z(t))**2,t)
        I2 = sum(dt*(y-z(t))**2)
        return 0.5*(I1+I2) #+ 0.5*(y[-1]-yT)**2

    def grad_J(u,p,dt,y):


        B = 1-a*dt*0.5
        A = 1+0.5*a*dt
        
        

        t = np.linspace(0,T,len(u))
        grad1 = np.zeros(len(u))
        grad1[:-1] = dt*(u[:-1]+p[1:]/B)
        grad1[0] = 0.5*dt*(u[0])+dt*p[1]/B
        grad1[-1] = 0.5*dt*(u[-1])

        grad2 = np.zeros(len(u))
        grad2[1:] = dt*(u[1:]+p[1:]/B)
        grad2[0] = 0.5*dt*(u[0])
        grad2[-1] = 0.5*dt*u[-1]+dt*p[-1]/B#dt*(p[-1]+dt*y[-1])/B#dt*p[-1]/B

        return 0.5*(grad1+grad2)
        
    
     
    problem =CrankNicolsonStateIntOCP(y0,yT,T,a,J,grad_J,z=z)
    return problem

def test_CN():

    a = 1
    y0=1
    T=1
    yT=0

    problem = create_simple_CN_problem(y0,yT,T,a)
    problem2,_ = lin_problem(y0,yT,T,a)
    N = 10000
    opt = {'jtol':0,'maxiter':50}
    res = problem.solve(N,Lbfgs_options=opt )
    res2 = problem2.solve(N,Lbfgs_options=opt)

    res3 = problem.penalty_solve(N,10,[N,N**2],Lbfgs_options=opt)[-1]
    res4 = problem.PPCLBFGSsolve(N,10,[N,N**2],options=opt)[-1]

    import matplotlib.pyplot as plt
    print res.counter(),res2.counter(),res3.counter(),res4.counter()
    plt.plot(res.x,'--')
    plt.plot(res2.x)
    plt.plot(res3.x[:N+1])
    plt.plot(res4.x[:N+1],'o')
    plt.show()

def taylorTestCN():
    from taylorTest import general_taylor_test,lin_problem
    a = -3.9
    y0=3.2
    T=1
    yT=11.5

    z = lambda x : 0*x
    
    problem = create_simple_CN_problem(y0,yT,T,a)
    #problem =create_state_CN_problem(y0,yT,T,a,z)
    
    general_taylor_test(problem)


def test_noise():

    y0 = 3.2
    yT=11.5
    a = -0.097
    T=100

    f = lambda x : 0.3*np.sin(x)
    
    
    problem = create_noise_CN_problem(y0,yT,T,a,f)

    N = 1000
    m = [1,2,4,8,16,32,64,128]
    mu =[10*N,5*N,5*N,10*N,10*N,20*N,20*N]
    tol=[1e-6,3e-6,5e-6,1e-6,1e-6,1e-5,1e-5]
    t = np.linspace(0,T,N+1)
    res = problem.solve(N,Lbfgs_options={'jtol':1e-7})
    ue,_,_ =problem.sin_prop_exact_solution(N,0.3)

    L = res.counter()[0]+res.counter()[1]
    table ={'D':[],'L':[],'S':[]}
    

    table['D'].append(np.sqrt(trapz((res.x[1:-1]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t)))
    table['L'].append(L)
    table['S'].append(1)
    #"""
    for i in range(1,len(m)):
        
        res2 = problem.PPCLBFGSsolve(N,m[i],[mu[i-1]],options={'jtol':tol[i-1]})
        

        err = np.sqrt(trapz((res2.x[1:N]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))
        count = res2.counter()
        L2 = count[0]+count[1]
        table['D'].append(err)
        table['L'].append(L2)
        table['S'].append(m[i]*(L/float(L2)))

    import pandas as pd

    data = pd.DataFrame(table,index=m)
    print data

    #data.to_latex('report/whyNotEqual/unsmooth.tex')
    

    ue,t,_ = problem.simple_problem_exact_solution(N)
    #"""
    
    #import matplotlib.pyplot as plt


    """
    plt.figure()
    plt.plot(t,ue)
    plt.ylabel('v',fontsize=20)
    plt.xlabel('t',fontsize=20)
    plt.savefig('report/draft/draft2/smooth.png')
    plt.figure()
    plt.plot(t,res.x)
    plt.xlabel('t',fontsize=20)
    plt.ylabel('v',fontsize=20)
    plt.savefig('report/draft/draft2/unsmooth.png')
    plt.show()
    
    
    print res.counter(),res2.counter()
    print max(abs(res.x[1:-1]-res2.x[1:N]))
    plt.plot(t,res.x)

    plt.plot(t,res2.x[:N+1],'r--')
    plt.show()
    

    ue,t,_ = problem.sin_prop_exact_solution(N,0.3)

    plt.plot(t,ue,'r--')
    plt.plot(t,res.x)
    plt.show()
    """
if __name__=='__main__':

    #test_CN()
    taylorTestCN()
    #test_noise()
