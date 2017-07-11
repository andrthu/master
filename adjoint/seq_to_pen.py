import numpy as np
from optimalContolProblem import OptimalControlProblem,Problem1
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import pandas as pd

def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))


def non_lin_problem(y0,yT,T,a):
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
            
        grad=dt*(u+p)
        #grad[0] = dt*0.5*u[0]
        #grad[-1] = dt*(0.5*u[-1]+p[-1])#grad[-1]
        return grad
    


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    return problem

def test():

    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)

    N = 1000
    m = 10
    mu = 1
    

    res = problem.solve(N,Lbfgs_options={'jtol':1e-7})

    u = res.x
    
    t = np.linspace(0,T,N+1)

    y = problem.ODE_solver(u,N)

    u0 = np.zeros(N+m)

    u0[:N+1]=u[:]

    
    
    opt = {'jtol':0,'maxiter':20}

    if m == 2:

        u0[N+1] = y[N/2]

        
    P_seq = problem.adjoint_solver(u,N)
    p,P = problem.adjoint_penalty_solver(u0,N,m,40000000000) 
    
    plt.plot(t,P_seq,'r--')
    plt.plot(t,P)
    plt.title('Adjoints for sequential control solution')
    plt.xlabel('time')
    plt.ylabel('adjoints')
    plt.legend(['non-penalty','penalty'])
    #plt.savefig('report/whyNotEqual/adjoint.png')
    #plt.show()

    #mu_list = [40000000000,80000000000]
    mu_list = [10,100,1000,10000,20000,40000,80000,160000,320000,640000,1280000]
    mu_list = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e12]
    tol_list = [1e-3,1e-3,5e-4,1e-4,1e-2,5e-6,5e-6,5e-6,2e-6,4e-7,4e-6,1e-9]
    res2 = problem.PPCLBFGSsolve(N,m,mu_list,options=opt)

   #res2 = problem.penalty_solve(N,m,mu_list,tol_list=tol_list,x0=u0,Lbfgs_options=opt)

    for i in range(len(res2)):
        err = l2_diff_norm(res2[i]['control'].array()[:N+1],u,t)
        print err,res2[i].niter,mu_list[i]#,tol_list[i]
        #max(abs(res2[-1]['control'].array()[:N+1]-u))
    
    print l2_diff_norm(res2[-1]['control'].array()[:N+1],u,t)#max(abs(res2[-1]['control'].array()[:N+1]-u))

    plt.plot(t,u,'r--')
    plt.plot(t,res2[-1]['control'].array()[:N+1])
    plt.title('Optimal control for penalty and non-pnalty solvers')
    plt.xlabel('time')
    plt.ylabel('control')
    plt.legend(['non-penalty','penalty'],loc=4)
    #plt.savefig('report/whyNotEqual/control.png')
    #plt.show()

def increasing_mu():
    

    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)

    N = 1000
    m = 3
    mu = 1
    

    res = problem.solve(N)

    u = res.x
    
    t = np.linspace(0,T,N+1)

    y = problem.ODE_solver(u,N)
    
    u0 = np.zeros(N+m)

    u0[:N+1]=u[:]

    
    
    opt = {'jtol':1e-4}

    if m == 2:

        u0[N+1] = y[N/2]

    
    if m == 3:
        
        u0[N+1] = y[N/3]
        u0[N+2] = y[2*N/3+1]
        P_seq = problem.adjoint_solver(u,N)
        p,P = problem.adjoint_penalty_solver(u0,N,m,10) 
    
        plt.plot(t,P_seq,'r--')
        plt.plot(t,P)
        plt.show()
    

    table = {'mu':[],'||u-u_mu||':[]}
   
    for i in range(1,11):
    
    
        res2 = problem.penalty_solve(N,m,[10**i],x0=u0,Lbfgs_options=opt)
        err = l2_diff_norm(res2['control'].array()[:N+1],u,t)
        table['mu'].append(10**i)
        table['||u-u_mu||'].append(err)


    data = pd.DataFrame(table)
    print data

    #data.to_latex('report/whyNotEqual/mu_error.tex')

if __name__ =='__main__':
    test()
    #increasing_mu()
