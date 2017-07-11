from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from parallelOCP import partition_func,v_comm_numbers,u_part,interval_partition
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time

from optimalContolProblem import OptimalControlProblem, Problem1
from pathos.multiprocessing import ProcessingPool


class TPOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,
                Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J



    def thread_parallel_ODE_solver(self,u,N,m):


        variables = []

        for i in range(m):

            variables.append((i,u,N,m))

        p = ProcessingPool(m)
        y = p.map(self.p_ode_solver,variables)

        return y

    def p_ode_solver(self,variables):

        i = variables[0]
        u = variables[1]
        N = variables[2]
        m = variables[3]
        
        dt = float(self.T)/N

        y = interval_partition(N+1,m,i)

        if i == 0:
            y[0] = self.y0
        else:
            y[0] = u[N+i]

        start = u_part(N+1,m,i)
        
        for j in range(len(y)-1):
            
            y[j+1] = self.ODE_update(y,u,j,start+j,dt)

        return y
    
    def initial_penalty(self,y,u,mu,N,i):
        
        return mu*(y[i][-1]-u[N+i+1])


    def thread_parallel_adjoint_solver(self,u,N,m,mu):

        y = self.thread_parallel_ODE_solver(u,N,m)

        variables = []

        for i in range(m):

            variables.append((i,y,N,m,mu,u))


        p = ProcessingPool(m)

        l = p.map(self.p_adjoint_solver,variables)

        return l


    def p_adjoint_solver(self,variables):

        i  = variables[0]
        y  = variables[1]
        N  = variables[2]
        m  = variables[3]
        mu = variables[4]
        u  = variables[5]
        dt = float(self.T)/N

        l = interval_partition(N+1,m,i)

        if i == m-1:
            l[-1] = self.initial_adjoint(y[-1][-1])
        else:
            l[-1] = self.initial_penalty(y,u,mu,N,i)

        for j in range(len(l)-1):
            l[-(j+2)] = self.adjoint_update(l,y[i],j,dt)

        return l


    
    def thread_parallel_penalty_grad(self,u,N,m,mu):


        p = self.thread_parallel_adjoint_solver(u,N,m,mu)

        P = np.zeros(N+1)
        grad = np.zeros(N+m)
        
        start = 0
        for j in range(m):
            end = start + len(p[j]) - 1
            P[start:end] = p[j][:-1]
            start = end
            

        grad[:N+1] = self.grad_J(u[:N+1],P,float(self.T)/N)

        for j in range(1,m):
            grad[N+j] = p[j][0]-p[j-1][-1]

        return grad

    def thread_parallel_penalty_functional(self,u,N,m,mu):

        y = self.thread_parallel_ODE_solver(u,N,m)

        

        J_val = self.J(u[:N+1],y[-1][-1],self.yT,self.T)
        
        penalty = 0
        for i in range(m-1):
            penalty += 0.5*(y[i][-1]-u[N+1+i])**2

        return J_val + penalty


    def thread_parallel_penalty_solve(self,N,m,mu_list,x0=None,
                                      Lbfgs_options=None,
                                      algorithm='my_lbfgs',scale=False):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        x = None
        if algorithm=='my_lbfgs':
            x0 = self.Vec(x0)
        Result = []

        for i in range(len(mu_list)):
            
            def J(u):                
                return self.thread_parallel_penalty_functional(u,N,m,mu_list[i])

            def grad_J(u):

                return self.thread_parallel_penalty_grad(u,N,m,mu_list[i])
            
            #J,grad_J = self.generate_reduced_penalty(dt,N,m,mu_list[i])
            if algorithm=='my_lbfgs':
                self.update_Lbfgs_options(Lbfgs_options)

                if scale:
                    scaler={'m':m,'factor':self.Lbfgs_options['scale_factor']}
                    
                    solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options,
                                   scale=scaler)
                else:
                    solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options)
            
                res = solver.solve()
                Result.append(res)
                x0 = res['control']
                print J(x0.array())
            elif algorithm=='my_steepest_decent':

                self.update_SD_options(Lbfgs_options)
                SDopt = self.SD_options
                if scale:
                    
                    scale = {'m':m,'factor':SDopt['scale_factor']}
                    Solver = SteepestDecent(J,grad_J,x0.copy(),
                                             options=SDopt,scale=scale)
                    res = Solver.solve()
                    res.rescale()
                else:
                    Solver = PPCSteepestDecent(J,grad_J,x0.copy(),
                                               lambda x: x,options=SDopt)
                    res = Solver.split_solve(m)
                x0 = res.x.copy()
                Result.append(res)
            elif algorithm=='slow_steepest_decent':
                self.update_SD_options(Lbfgs_options)
                SDopt = self.SD_options
                Solver = SteepestDecent(J,grad_J,x0.copy(),
                                        options=SDopt)
                res = Solver.solve()
                x0 = res.x.copy()
                Result.append(res)
                
            elif algorithm == 'split_lbfgs':
                self.update_Lbfgs_options(Lbfgs_options)
                Solver = SplitLbfgs(J,grad_J,x0,m,options=self.Lbfgs_options)

                res = Solver.solve()
                x0 = res.x.copy()
                Result.append(res)
        if len(Result)==1:
            return res
        else:
            return Result

         
class TPProblem1(TPOCP):
    """
    optimal control with ODE y=ay'+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        TPOCP.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)] 


def generate_problem(y0,yT,T,a):
    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)

   
    
    
    pproblem = TPProblem1(y0,yT,T,a,J,grad_J)
    return pproblem

def test_solvers():

    y0=1
    yT=1
    T=1
    a=1

    problem = generate_problem(y0,yT,T,a)

    N = 500
    mu=1
    m = 2

    u = np.linspace(0,T,N+m)

    #grad = problem.thread_parallel_penalty_grad(u,N,m,mu)
    #print grad
    t0 =time.time()
    problem.thread_parallel_adjoint_solver(u,N,m,mu)
    t1=time.time()
    problem.adjoint_penalty_solver(u,N,m,mu)
    t2 = time.time()
    print t1-t0,t2-t1,(t2-t1)/(t1-t0)

    t0 = time.time()
    problem.penalty_solve(N,m,[mu])
    t1  = time.time()
    problem.thread_parallel_penalty_solve(N,m,[mu])
    t2 = time.time()

    print t1-t0,t2-t1

if __name__ == '__main__':
    
    test_solvers()


