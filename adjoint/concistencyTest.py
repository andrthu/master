import numpy as np
from optimalContolProblem import OptimalControlProblem,Problem1
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from scipy.integrate import trapz,simps,romb
import matplotlib.pyplot as plt
import pandas as pd
from runge_kutta_OCP import RungeKuttaProblem

def l2_diff_norm(u1,u2,t):
    #return max(abs(u1-u2))
    return np.sqrt(trapz((u1-u2)**2,t))

def l2_norm(u,t):
    #return max(abs(u))
    return np.sqrt(trapz((u)**2,t))

def lin_problem(y0,yT,T,a):
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)
        
        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
            
        grad=dt*(u+p)
        grad[0] = dt*0.5*u[0]+p[-1]*dt
        grad[-1] = dt*(0.5*u[-1])#grad[-1]
        return grad
    
        

    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    #problem = RungeKuttaProblem(y0,yT,T,a,J,grad_J)
    return problem

def test(N=1000):

    y0=1
    yT=-10
    T=1
    a=1

    problem = lin_problem(y0,yT,T,a)
    """

    """
    
    
    seq_res=problem.solve(N,Lbfgs_options={'jtol':0,'maxiter':50})

    opt = {'jtol':0,'maxiter':60}

    m = [2,4,8,16,32,64]
    m = [2,4,16,64]
    t = np.linspace(0,T,N+1)
    mu_list = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e13,1e15,1e17,1e18,1e20]
    mu_list =[1,1e1,1e2,1e3,2e3,5e3,1e4,2e4,5e4,7e4,1e5,2e5,5e5,7e5,1e6,2e6,5e6,7e6,1e7,2e7,5e7,7e7,1e8,2e8,5e8]
    table={}
    seq_norm = l2_norm(seq_res.x,t)
    for i in range(len(m)):
        res = problem.PPCLBFGSsolve(N,m[i],mu_list,options=opt)

        error = []
        for j in range(len(res)):
            err = l2_diff_norm(res[j].x[:N+1],seq_res.x,t)/seq_norm
            
            error.append(err)

        table.update({m[i]:error})

    data = pd.DataFrame(table,index=mu_list)
        
    print data
    print seq_norm
    data.to_latex('report/consitency_tables/different_m_Neql'+str(N)+'.tex')

def test2():
    
    y0=3.3
    yT=10
    T=1
    a=10.4

    problem = lin_problem(y0,yT,T,a)
    import matplotlib.pyplot as plt
    m = 2

    N = [101,501,801,1001]#,2000,5000,10000,50000]
    #N = [100,50000,]
    mu_list = [1,1e1,1e2,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e13]
    #mu_list = [1e5,1e6,1e7,1e8,1e9,2e9,5e9,1e10,1e11]
    #mu_list = [1,1e1,1e2,1e3,2e3,5e3,1e4,2e4,3e4,5e4,1e5,2e5,7e5,1e6]
    seq_opt = {'jtol': 1e-4,'maxiter':60}
    pen_opt = {'jtol' :1e-10,'maxiter':60}

    table = {}
    for i in range(len(N)):
        
        t = np.linspace(0,T,N[i]+1)

        seq_res = problem.solve(N[i],Lbfgs_options=seq_opt)
        #seq_res = problem.scipy_solver(N[i])
        #res  = problem.scipy_penalty_solve(N[i],m,mu_list)
        res = problem.PPCLBFGSsolve(N[i],m,mu_list,options=pen_opt)
        #res = problem.penalty_solve(N[i],m,mu_list,Lbfgs_options=pen_opt)
        seq_norm = l2_norm(seq_res.x,t)
        
        error = []
        for j in range(len(res)):
            err = l2_diff_norm(seq_res.x,res[j].x[:N[i]+1],t)/seq_norm 
            error.append(err)
        table.update({N[i]:error})
        #print table

        if i ==2:
            plt.plot(t,seq_res.x,'--r')
            for j in range(len(res)):
                print 'he'
                if j==0:
                    plt.plot(t,res[j].x[:N[i]+1],'--b')
                else:
                    plt.plot(t,res[j].x[:N[i]+1])

    plt.show()
    data = pd.DataFrame(table,index=mu_list)
    print data
    #data.to_latex('report/consitency_tables/different_N_meql'+str(m)+'.tex')
    print '||u||: ', seq_norm


def jump_diff_test(y0=3.3,yT=10,T=1,a=10.4):

    

    problem = lin_problem(y0,yT,T,a)
    import matplotlib.pyplot as plt
    N = 100000

    M = [2,3,4,5,6]
    res1 = problem.solve(N,Lbfgs_options={'jtol':1e-5})
    plt.plot(res1.x,'r--')
    jumps = []
    errors = []
    table = {'err':[],'jump':[],'iter':[]}
    for m in M:
        
        jumps2 = []
        errors2 = []
        res = problem.penalty_solve(N,m,[N],tol_list=[1e-5,1e-5],Lbfgs_options={'jtol':1e-1})
        res = [res]
        for i in range(len(res)):
            jumps2.append(res[i].jump_diff)
            errors2.append(max(abs(res[i].x[:N+1]-res1.x))/max(abs(res1.x)))
        errors.append(errors2)
        jumps.append(jumps2)
        table['err'].append(errors2)
        table['jump'].append(jumps2)
        table['iter'].append(res[-1].counter())
        plt.plot(res[-1].x[:N+1])
    data = pd.DataFrame(table)
    print res1.counter()
    print data
    plt.show()
    #print jumps
if __name__ == '__main__':
    #test(100000)
    #test2()
    jump_diff_test(y0=1,a=1,yT=1)
