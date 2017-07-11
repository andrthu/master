from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time
import sys

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector
from parallelOCP import interval_partition,v_comm_numbers
from ODE_pararealOCP import PararealOCP

class MpiVectorOCP(PararealOCP):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,parallel_grad_J=None,
                 Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,Lbfgs_options,options)

        self.parallel_J=parallel_J
        self.parallel_grad_J = parallel_grad_J
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.m = self.comm.Get_size()
        self.local_end = 0

    def initial_control2(self,N,m=1):
        comm = self.comm
        rank = self.rank
        return MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
        
    def parallel_ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points       
        """ 
        rank = self.rank
        
        T = self.T        
        dt = float(T)/N
        y = interval_partition(N+1,m,rank)

        if rank == 0:
            y[0] = self.y0     
            j_help = 0
            u = u.local_vec.copy()
        else:
            y[0] = u[-1]        #### OBS!!!! ####
            j_help = -1
            u = u.local_vec.copy()
            u_h = np.zeros(len(u)+1)
            u_h[1:]=u[:]
            u=u_h
            
        
        y_len = len(y)
        
        for j in range(y_len-1):
            y[j+1] = self.ODE_update(y,u,j,j,dt)#self.ODE_update(y,u,j,j+j_help,dt)
        self.local_end = y[-1]
        return y

    def parallel_penalty_functional(self,u,N,mu):

        comm = self.comm

        m = self.m
        rank = self.rank
        y = self.parallel_ODE_penalty_solver(u,N,m)

        return self.parallel_J(u,y,self.yT,self.T,N,mu,comm)

    def initial_penalty(self,y,u,mu,N,rank):

        return mu*(y[-1]-u)
        
    def parallel_adjoint_penalty_solver(self,u,N,m,mu):
        """
        Solving the adjoint equation using finite difference, and partitioning
        the time interval.

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """
        comm = self.comm
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        rank = self.rank
        
        l = interval_partition(N+1,m,rank)#partition_func(N+1,m)
        y= self.parallel_ODE_penalty_solver(u,N,m)
        
        if rank == m-1:
            comm.send(y[0],dest=rank-1)
            lam = None
        elif rank!=0:
            lam = comm.recv(source=rank+1)
            comm.send(y[0],dest=rank-1)
        else:
            lam = comm.recv(source=rank+1)

        if rank == m-1:
            l[-1] = self.initial_adjoint(y[-1])
        else:
            l[-1]=self.initial_penalty(y,lam,mu,N,rank) #my*(y[i][-1]-u[N+1+i])
            
        
        for j in range(len(l)-1):
            l[-(j+2)] = self.adjoint_update(l,y,j,dt)
        
        return l

    def penalty_grad(self,u,N,m,mu):

        comm = self.comm
        rank = self.rank
        p = self.parallel_adjoint_penalty_solver(u,N,m,mu)           

            
            
        grad = self.parallel_grad_J(u,p,float(self.T)/N,comm)
            
                       
        if rank == 0:
            my_lam = np.array([0])
            comm.send(p[-1],dest=1)
        elif rank!= m-1:
            p_n = comm.recv(source=rank-1)
            comm.send(p[-1],dest=rank+1)

            my_lam = np.array([p[0]-p_n])
            grad[-1] = p[0]-p_n
        else:
            p_n = comm.recv(source=rank-1)
            my_lam = np.array([p[0]-p_n])
            grad[-1] = p[0]-p_n
            
        return MPIVector(grad,comm)
        #return grad
    def parallel_penalty_solve(self,N,m,mu_list,tol_list=None,x0=None,Lbfgs_options=None):

        comm = self.comm
        rank = self.rank
        if x0==None:
            x0= self.initial_control2(N,m=m)
        initial_counter = self.counter.copy()
        self.update_Lbfgs_options(Lbfgs_options)
        Result = []
        for i in range(len(mu_list)):

            
            
            def J(u):
                self.counter[0]+=1
                return self.parallel_penalty_functional(u,N,mu_list[i])
            def grad_J(u):
                self.counter[1]+=1
                return self.penalty_grad(u,N,m,mu_list[i])

            solver = SplitLbfgs(J,grad_J,x0,options=self.Lbfgs_options,mpi=True)

            res = solver.mpi_solve()
            x0 = res.x
            
            res.add_FuncGradCounter(self.counter-initial_counter)
            Result.append(res)
            val =self.find_jump_diff(res.x)
            if self.rank == 0:
                print 'jump diff:',val
        return Result

    def PC_maker4(self,N,m,comm,step=1):
        
        
        
        
        def pc(x):

            rank = self.rank
            lam = np.zeros(m)
            loc_lam = np.zeros(1)
            loc_lam[0] = x[-1]
            start = tuple(np.linspace(0,m-1,m))
            length = tuple(np.zeros(m)+1)

            comm.Allgatherv(loc_lam,[lam,length,start,MPI.DOUBLE])
            lam = lam[1:]
            lam_pc = self.PC_maker2(N,m,step)            
            lam2 = lam_pc(lam)
            if rank!=0:
                x[-1] = lam2[rank-1]
            return x
        return pc

    def parallel_PPCLBFGSsolve(self,N,m,mu_list,tol_list=None,x0=None,options=None,scale=False):
        dt=float(self.T)/N
        comm = self.comm
        rank = self.rank
        if x0==None:
            x0 = self.initial_control2(N,m=m)
        
        Result = []
        PPC = self.PC_maker4(N,m,comm,step=1)
        initial_counter = self.counter.copy()
        for i in range(len(mu_list)):
            
            def J(u):
                self.counter[0]+=1
                return self.parallel_penalty_functional(u,N,mu_list[i])
            def grad_J(u):
                self.counter[1]+=1
                return self.penalty_grad(u,N,m,mu_list[i])

            self.update_Lbfgs_options(options)
            Lbfgsopt = self.Lbfgs_options
            if tol_list!=None:
                try:
                    opt = {'jtol':tol_list[i]}
                    self.update_Lbfgs_options(opt)
                    Lbfgsopt = self.Lbfgs_options
                except:
                    print 'no good tol_list'
            
            Solver = SplitLbfgs(J,grad_J,x0,m=m,Hinit=None,
                                options=Lbfgsopt,ppc=PPC,mpi=True)
            res = Solver.mpi_solve()
            x0 = res.x
            res.add_FuncGradCounter(self.counter-initial_counter)
            Result.append(res)
            val = self.find_jump_diff(res.x)
            if self.rank==0:
                print 'jump diff:',val
        return Result


    def parallel_SD_penalty_solve(self,N,m,mu_list,tol_list=None,x0=None,options=None):
        comm = self.comm
        rank = self.rank
        if x0==None:
            x0= self.initial_control2(N,m=m)
        initial_counter = self.counter.copy()
        self.update_Lbfgs_options(options)
        Result = []
        PPC = self.PC_maker4(N,m,comm,step=1)
        for i in range(len(mu_list)):

            
            
            def J(u):
                self.counter[0]+=1
                return self.parallel_penalty_functional(u,N,mu_list[i])
            def grad_J(u):
                self.counter[1]+=1
                return self.penalty_grad(u,N,m,mu_list[i])

            #solver = SplitLbfgs(J,grad_J,x0,options=self.Lbfgs_options,mpi=True)
            solver = PPCSteepestDecent(J,grad_J,x0,PPC,options=self.Lbfgs_options)
            res = solver.mpi_solve()
            x0 = res.x
            
            res.add_FuncGradCounter(self.counter-initial_counter)
            Result.append(res)
            val =self.find_jump_diff(res.x)
            if self.rank == 0:
                print 'jump diff:',val
        
        return Result
        
    def find_jump_diff(self,x):
        
        comm = self.comm
        rank = self.rank
        m = self.m
        if rank == 0:
            my_lam = np.array([0])
            comm.send(self.local_end,dest=1)
            my_diff =np.array(0)
            all_diff = np.zeros(m)
        elif rank!= m-1:
            nab_end = comm.recv(source=rank-1)
            comm.send(self.local_end,dest=rank+1)

            my_diff = np.array(nab_end-x[-1])
            all_diff = np.zeros(m)
        else:
            nab_end= comm.recv(source=rank-1)
            my_diff = np.array(nab_end-x[-1])
            all_diff = np.zeros(m)

        start = tuple(np.linspace(0,m-1,m))
        length = tuple(np.zeros(m)+1)

        comm.Allgatherv(my_diff,[all_diff,length,start,MPI.DOUBLE])

        return max(abs(all_diff))


class simpleMpiVectorOCP(MpiVectorOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,parallel_J=None,parallel_grad_J=None,options=None):

        MpiVectorOCP.__init__(self,y0,yT,T,J,grad_J,
                              parallel_J=parallel_J,
                              parallel_grad_J=parallel_grad_J,
                              options=options)

        self.a = a

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i]+dt*u[j+1])/(1.-dt*a)

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a)


def generate_problem(y0,yT,T,a):
    

    def J(u,y,yT,T):
        #t = np.linspace(0,T,len(u))
        dt = T/float(len(u)-1)
        I = dt*np.sum(u[1:]**2)
        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        #grad[0] = 0.5*dt*(u[0]) 
        #grad[-1] = 0.5*dt*(u[-1]) + dt*p[-2]
        return grad
        

    def parallel_J(u,y,yT,T,N,mu,comm):
        m = comm.Get_size()
        rank = comm.Get_rank()
        
        
        dt = float(T)/N
        

        if rank == m-1:
            comm.send(u[-1],dest=rank-1)
            lam = None
        elif rank!=0:
            lam = comm.recv(source=rank+1)
            comm.send(u[-1],dest=rank-1)
        else:
            lam = comm.recv(source=rank+1)

        
        s = np.zeros(1)
        if rank == m-1:
            s[0] = 0.5*((y[-1]-yT)**2) 
            
            I = dt*np.sum(u[:-1]**2)
            #I+= dt*0.5*u[-2]**2
            s[0] += 0.5*I 
        elif rank!=0:
            s[0] = 0.5*dt*np.sum(u[:-1]**2)
            s[0] += 0.5*mu*(y[-1]-lam)**2

        else:
            s[0] = 0.5*dt*np.sum(u[1:]**2)            
            #s[0] += 0.5*0.5*dt*u[0]**2
            s[0] += 0.5*mu*(y[-1]-lam)**2
        
        S =np.zeros(1)
        #comm.Barrier()
        comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

        return S[0]


    def parallel_grad_J(u,p,dt,comm):

        rank = comm.Get_rank()
        m = comm.Get_size()
        if rank ==0:

            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:] + p[:-1])
            #grad[0] = 0.5*dt*u[0]

        elif rank!=m-1:
            grad = np.zeros(len(u))

            grad[:-1]=dt*(u[:-1]+p[:-1])
        else:

            grad = np.zeros(len(u))
            grad[:-1] = dt*(u[:-1]+p[:-1])
            #grad[-2] = dt*0.5*u[-2] + dt*p[-2]
        return grad
        

        

    
    
    pproblem = simpleMpiVectorOCP(y0,yT,T,a,J,grad_J,parallel_J=parallel_J,parallel_grad_J=parallel_grad_J)
    problem = Problem1(y0,yT,T,a,J,grad_J)

    return problem,pproblem

def generate_problem_c(y0,yT,T,a,c):
    

    def J(u,y,yT,T):
        #t = np.linspace(0,T,len(u))
        dt = T/float(len(u)-1)
        I = dt*np.sum((u[1:]-c)**2)
        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]-c+p[:-1])
        #grad[0] = 0.5*dt*(u[0]) 
        #grad[-1] = 0.5*dt*(u[-1]) + dt*p[-2]
        return grad
        

    def parallel_J(u,y,yT,T,N,mu,comm):
        m = comm.Get_size()
        rank = comm.Get_rank()
        
        
        dt = float(T)/N
        

        if rank == m-1:
            comm.send(u[-1],dest=rank-1)
            lam = None
        elif rank!=0:
            lam = comm.recv(source=rank+1)
            comm.send(u[-1],dest=rank-1)
        else:
            lam = comm.recv(source=rank+1)

        
        s = np.zeros(1)
        if rank == m-1:
            s[0] = 0.5*((y[-1]-yT)**2) 
            
            I = dt*np.sum((u[:-1]-c)**2)
            #I+= dt*0.5*u[-2]**2
            s[0] += 0.5*I 
        elif rank!=0:
            s[0] = 0.5*dt*np.sum((u[:-1]-c)**2)
            s[0] += 0.5*mu*(y[-1]-lam)**2

        else:
            s[0] = 0.5*dt*np.sum((u[1:]-c)**2)            
            #s[0] += 0.5*0.5*dt*u[0]**2
            s[0] += 0.5*mu*(y[-1]-lam)**2
        
        S =np.zeros(1)
        #comm.Barrier()
        comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

        return S[0]


    def parallel_grad_J(u,p,dt,comm):

        rank = comm.Get_rank()
        m = comm.Get_size()
        if rank ==0:

            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:] -c + p[:-1])
            #grad[0] = 0.5*dt*u[0]

        elif rank!=m-1:
            grad = np.zeros(len(u))

            grad[:-1]=dt*(u[:-1]-c+p[:-1])
        else:

            grad = np.zeros(len(u))
            grad[:-1] = dt*(u[:-1]-c+p[:-1])
            #grad[-2] = dt*0.5*u[-2] + dt*p[-2]
        return grad
        

        

    
    
    pproblem = simpleMpiVectorOCP(y0,yT,T,a,J,grad_J,parallel_J=parallel_J,parallel_grad_J=parallel_grad_J)
    problem = Problem1(y0,yT,T,a,J,grad_J)

    return problem,pproblem


def local_u_size(n,m,i):

    q = n/m
    r = n%m

    if i == 0:
        if r>0:
            return q+1
        else:
            return q
    else:
        if r-i>0:
            return q+2
        else:
            return q+1

    

def test_mpi_solvers():
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    

    N = 100
    m = comm.Get_size()
    rank = comm.Get_rank()

    u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
    #print local_u_size(N+1,m,rank),rank

    print
    if rank==1:
        u[-1]=0
    u2 = np.zeros(N+m)
    u2[-1]=0
    y = mpi_problem.parallel_ODE_penalty_solver(u,N,m)
    y2,Y = non_mpi.ODE_penalty_solver(u2,N,m)

    if rank ==0:
        print y-y2[0],rank
    if rank==1:
        print y -y2[1],rank
    mu =1
    print
    #comm.Barrier()
    p = mpi_problem.parallel_adjoint_penalty_solver(u,N,m,1)
    p2, P = non_mpi.adjoint_penalty_solver(u2,N,m,1)
    if rank ==0:
        print p-p2[0],rank
    if rank ==1:
        print p-p2[1],rank
    print
    print mpi_problem.parallel_penalty_functional(u,N,1),non_mpi.Penalty_Functional(u2,N,m,1)
    
    grad = mpi_problem.penalty_grad(u,N,m,mu)
    print

    Js,grad_Js=non_mpi.generate_reduced_penalty(float(T)/N,N,m,mu)

    grads = grad_Js(u2)
    if m==2:
        if rank == 0:
            print grad-grads[:len(grad)],rank,len(grad)
        if rank == 1:
            print grad-grads[len(grad):],rank,len(grad)


def test_solve():
    
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    

    N = 1000
    m = comm.Get_size()
    rank = comm.Get_rank()

    res2 = non_mpi.penalty_solve(N,m,[1])
    res = mpi_problem.parallel_SD_penalty_solve(N,m,[1],options={'jtol':1e-3})#parallel_penalty_solve(N,m,[1])
    
    print res[0].niter
    print
    print res[0].x
    print
    u=res[0].x.gather_control()
    if rank==0:
        print max(abs(u-res2.x))

def time_measure_test():
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    
    try:
        N = int(sys.argv[1])
    except:
        N = 100000
    m = comm.Get_size()
    rank = comm.Get_rank()
    t0 = time.time()
    res2 = non_mpi.penalty_solve(N,m,[1])
    t1=time.time()
    comm.Barrier()
    t3=time.time()
    res = mpi_problem.parallel_penalty_solve(N,m,[1])
    t4 = time.time()

    print t1-t0,t4-t3,(t1-t0)/(t4-t3)

def test_pppc():
    import matplotlib.pyplot as plt
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    

    N = 1000
    m = comm.Get_size()
    rank = comm.Get_rank()
    opt = {'jtol':1e-4}
    tol_list = [1e-6]
    res = mpi_problem.parallel_PPCLBFGSsolve(N,m,[3*N**2],tol_list=tol_list,options=opt)
    res2 = mpi_problem.PPCLBFGSsolve(N,m,[N**2],options=opt)#[-1]
    res3 =mpi_problem.solve(N,Lbfgs_options={'jtol':1e-6})
    x = res[-1].x.gather_control()
    if rank==0:
        print res[0].niter,res[-1].niter, res2.niter,res3.niter
        print x-res2.x
        t = np.linspace(0,T,N+1)
        plt.plot(t,x[:N+1])
        plt.plot(t,res2.x[:N+1],'b--')
        plt.plot(t,res3.x,'r--')
        plt.show()

def find_error():
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    m = comm.Get_size()
    N = 11
    mu = [1]
    rank = comm.Get_rank()
    u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
    #y = mpi_problem.parallel_ODE_penalty_solver(u,N,m)
    print u,rank
    
if __name__ =='__main__':
    #test_mpi_solvers()
    test_solve()
    #time_measure_test()
    #test_pppc()
    #find_error()
