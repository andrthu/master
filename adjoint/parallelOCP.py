from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1



def partition_func(n,m):

    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(np.zeros(N+1))
    else:
        partition.append(np.zeros(N))

    for i in range(1,m):
        if rest-i>0:
            partition.append(np.zeros(N+2))
        else:
            partition.append(np.zeros(N+1))

    return partition

def v_comm_numbers(n,m):
    """
    This function helps with mpi communication, i.e,
    it creates the tuples required for scatterv and gatherv 
    built-in communication functions.

    For both theese functions you need a 'start' and 'length'
    tuple.

    For Gatherv, the start tuple contains integres that explains
    how many elements away from the start of the recieving array 
    to start to append the recieved arrays.

    The length-tuple gives the number of recived elements from 
    each each process.


    Example:
    Let n= 11 and m = 3
    We have: 
    u0 = [0,1,2,3]
    u1 = [3,4,5,6,7]
    u2 = [7,8,9,10]

    remove first elements for u1 and u2:    
    u0 = [0,1,2,3]
    u1 = [4,5,6,7]
    u2 = [8,9,10]

    and we want to gather them into:
    
    u = [0,1,2,3,4,5,6,7,8,9,10]

    To achive this our function would return the following tuples:

    start:  (0,4,8)
    length: (4,4,3)    
    
    """

    N = n/m
    r = n%m

    scatter_s = [0]
    scatter_l = []
    gather_s = [0]
    gather_l = []

    if r>0:
        scatter_l.append(N + 1)
        gather_l.append(N + 1)
    else:
        scatter_l.append(N)
        gather_l.append(N)
    
    for  i in range(1,m):
        if r -i >0:
            scatter_l.append(N+2)
            gather_l.append(N+1)
        else:
            scatter_l.append(N+1)
            gather_l.append(N)

        scatter_s.append(scatter_s[i-1] + scatter_l[i-1]-1)
        gather_s.append(scatter_s[i]+1)
        
    return tuple(scatter_s),tuple(scatter_l),tuple(gather_s),tuple(gather_l)

def u_part(n,m,i):
    """
    Given a vector u of length n, partitioned into m parts with one 
    integer overlap, this function returns the first integer place of u that
    belongs to i-th partition.

    Example:
    n=11, m=3
    u = [0,1,2,3,4,5,6,7,8,9,10]

    We partition u:
    u0 = [0,1,2,3]
    u1 = [3,4,5,6,7]
    u2 = [7,8,9,10]
    
    Now we want to place in u does ui start with?

    i=0: 0
    i=1: 3
    i=2: 7
    """

    
    N = n/m
    rest = n%m
    if i==0:
        return 0

    if rest>0:
        start = N
    else:
        start = N-1
    for j in range(i-1):
        if rest-(j+1)>0:
            start += N+1
        else:
            start += N

    return start

def interval_partition(n,m,i):
    """
    We want to create m overlapping partitions of an empty vector of 
    length n. This functions returns a numpy.zeros with the correct
    length for thi i-th partition.

    Example:
    n = 11, m=3

    i=0: [0,0,0,0]
    i=1: [0,0,0,0,0]
    i=2: [0,0,0,0]
    """
    
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = np.zeros(N+1)
        else:
            state = np.zeros(N)
    else:
        if rest - i >0:
            state = np.zeros(N+2)
        else:
            state = np.zeros(N+1)
    return state

class POCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,
                 Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J

        self.comm = MPI.COMM_WORLD
    


    def parallel_ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        
        comm = self.comm
        
        rank = comm.Get_rank()
        
        T = self.T
        
        ss,sl,gs,gl = v_comm_numbers(N+1,m)
        
        dt = float(T)/N
        y = interval_partition(N+1,m,rank)
        
        
        if rank == 0:
            y[0]=self.y0
            Y = np.zeros(N+1)
            Y_list = []
        else:
            y[0] = u[N+rank]
            Y = None
            Y_list =None

        start=u_part(N+1,m,rank)
                
        for j in range(len(y)-1):
            
            y[j+1] = self.ODE_update(y,u,j,start+j,dt)
        if rank ==0:
            y_send = y.copy()
        else:
            y_send = y[1:].copy()
        comm.Barrier()
        y_list=comm.gather(y,root=0)
        comm.Gatherv(y_send,[Y,gl,gs,MPI.DOUBLE])
        

                     
        return y,Y,y_list


    def parallel_penalty_functional(self,u,N,mu):

        comm = self.comm

        m = comm.Get_size()
        rank = comm.Get_rank()
        y ,Y,y_list = self.parallel_ODE_penalty_solver(u,N,m)

        J_val = self.parallel_J(u,y,self.yT,self.T,mu,comm)
        return J_val
        last_y = None
        if rank!=0:
            Y = np.zeros(N+1)
        
        comm.Bcast(Y,root=0)
        
        J_val = self.J(u[:N+1],Y[-1],self.yT,self.T)
        
        penalty = np.zeros(1)
        if rank == 0:
            
            for k in range(m-1):
                penalty[0] = penalty[0]+mu*((y_list[k][-1]-u[N+1+k])**2)

        comm.Bcast(penalty,root=0)
        
        return J_val + 0.5*penalty[0]

    def initial_penalty(self,y,u,mu,N,i):
        #rank = self.comm.Get_rank()
        return mu*(y[-1]-u[N+i+1])

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
        
        rank = comm.Get_rank()
        
        l = interval_partition(N+1,m,rank)#partition_func(N+1,m)
        y,Y,y_list = self.parallel_ODE_penalty_solver(u,N,m)

        if rank == m-1:
            l[-1] = self.initial_adjoint(y[-1])
        else:
            l[-1]=self.initial_penalty(y,u,mu,N,rank) #my*(y[i][-1]-u[N+1+i])
            
        
        for j in range(len(l)-1):
            l[-(j+2)] = self.adjoint_update(l,y,j,dt)


        if rank == 0:
            
            L = np.zeros(N+1)
            l_list = []
            
        else:
            
            L = None
            l_list =None
            
            
        if rank == m-1:
            l_send = l.copy()
        else:
            l_send = l[:-1].copy()


        ss,sl,gs,gl = v_comm_numbers(N+1,m)
        new_gl = [i for i in gl]
        new_gl[0] = gl[0]-1
        new_gl[-1] = gl[-1]+1
        new_gl = tuple(new_gl)
        comm.Barrier()
        l_list=comm.gather(l,root=0)
        comm.Gatherv(l_send,[L,new_gl,ss,MPI.DOUBLE])
        
        return l,L,l_list
    def penalty_grad(self,u,N,m,mu):

        comm = self.comm
        rank = comm.Get_rank()
        p,P,p_list = self.parallel_adjoint_penalty_solver(u,N,m,mu)
        

        if rank!=0:
            P = np.zeros(N+1)
        comm.Bcast(P,root=0)
        #print P[:10],rank
        grad = np.zeros(N+m)

        grad[:N+1] = self.grad_J(u[:N+1],P,float(self.T)/N)
        
            
        lam = np.zeros(m)
        
        
        if rank == 0:
            my_lam = np.array([0])
            comm.send(p[-1],dest=1)
        elif rank!= m-1:
            p_n = comm.recv(source=rank-1)
            comm.send(p[-1],dest=rank+1)

            my_lam = np.array([p[0]-p_n])
        else:
            p_n = comm.recv(source=rank-1)
            my_lam = np.array([p[0]-p_n])
        """
        if rank == m-1:
            my_lam = np.array([0])
        else:
            
            my_lam = np.array([p[0]-u[N+1+rank]])
        """
        start = tuple(np.linspace(0,m-1,m))
        length = tuple(np.zeros(m) +1)
        comm.Barrier()
        comm.Allgatherv(my_lam,[lam,length,start,MPI.DOUBLE])
        #comm.bcast(lam,root=0)
        grad[N+1:] = lam[1:]
        
        return grad

    def parallel_penalty_one_iteration_solve(self,N,m,my_list,x0=None,
                                             Lbfgs_options=None,
                                             algorithm='my_lbfgs',scale=False):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        x = None
        if algorithm=='my_lbfgs':
            x0 = self.Vec(x0)
        Result = []
        rank = self.comm.Get_rank()
        for i in range(len(my_list)):
            
            def J(u):                
                return self.parallel_penalty_functional(u,N,my_list[i])

            def grad_J(u):

                return self.penalty_grad(u,N,m,my_list[i])
            
            
            if algorithm=='my_lbfgs':
                self.update_Lbfgs_options(Lbfgs_options)

                if scale:
                    scaler={'m':m,'factor':self.Lbfgs_options['scale_factor']}
                    
                    solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options,
                                   scale=scaler)
                else:
                    solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options)
                    
                
                res = solver.one_iteration(self.comm)
                Result.append(res)
                #print res.array(),rank
                #x0 = res['control']
                #print J(x0.array())
                if rank ==0:
                    Result.append(res)
                else:
                    Result.append(0)
            
            return Result
            if len(Result)==1:
                return res
            else:
                return Result

    def parallel_penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None,
                               algorithm='my_lbfgs',scale=False):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        x = None
        if algorithm=='my_lbfgs':
            x0 = self.Vec(x0)
        Result = []
        rank = self.comm.Get_rank()
        for i in range(len(my_list)):
            
            def J(u):                
                return self.parallel_penalty_functional(u,N,my_list[i])

            def grad_J(u):

                return self.penalty_grad(u,N,m,my_list[i])
            
            
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
                #print J(x0.array())
                if rank ==0:
                    Result.append(res)
                else:
                    Result.append(0)
            
            return Result
            if len(Result)==1:
                return res
            else:
                return Result







class PProblem1(POCP):
    """
    optimal control with ODE y=ay'+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,parallel_J,options=None):

        POCP.__init__(self,y0,yT,T,J,grad_J,parallel_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]/ 
        return l[-(i+1)]/(1.-dt*a) 

def generate_problem(y0,yT,T,a):
    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)

    def parallel_J(u,y,yT,T,mu,comm):
        m = comm.Get_size()
        rank = comm.Get_rank()
        n = len(u) - m+1

        t = np.linspace(0,T,n)

        start = u_part(n,m,rank)
        end = u_part(n,m,rank+1)
        s = np.zeros(1)
        if rank == m-1:
            s[0] = 0.5*((y[-1]-yT)**2 + trapz(u[start:end+1]**2,t[start:end+1]))
        else:
            s[0] = 0.5*trapz(u[start:end+1]**2,t[start:end+1])
            s[0] += 0.5*mu*(y[-1]-u[n+rank])**2
        """
        if rank==0:
            S=np.zeros(1)
        else:
            S =None
        """
        S =np.zeros(1)
        comm.Barrier()
        comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

        return S

    
    
    pproblem = PProblem1(y0,yT,T,a,J,grad_J,parallel_J=parallel_J)
    problem = Problem1(y0,yT,T,a,J,grad_J)

    return problem,pproblem

def test_parallel_gradient():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    non_parallel,problem = generate_problem(y0,yT,T,a)
    comm = problem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    N = 200000
    
    u = np.zeros(N+m)
    u[:N+1] = 100*np.linspace(0,T,N+1)
    
    t0= time.time()
    p = non_parallel.adjoint_solver(u[:N+1],N)
    non_p_grad = non_parallel.grad_J(u[:N+1],p,1./N)
    t1 = time.time()
    mu=1
    comm.Barrier()
    t2 = time.time()
    grad = problem.penalty_grad(u,N,m,mu)
    t3=time.time()

    print 'seq time: ',t1-t0,'parallel time: ',t3-t2, 'rank:',rank
    print 'seq time/parallel time: ', (t1-t0)/(t3-t2), 'rank:',rank
def test_one_itt_solve():
    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    non_parallel,problem = generate_problem(y0,yT,T,a)

    comm = problem.comm

    m = comm.Get_size()
    N = 500

    

    problem.parallel_penalty_one_iteration_solve(N,m,[1])


def test_parallel_solve():
    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    non_parallel,problem = generate_problem(y0,yT,T,a)

    comm = problem.comm

    m = comm.Get_size()
    rank = comm.Get_rank()
    N = 10000
    t0 = time.time()
    res1=non_parallel.penalty_solve(N,m,[1])
    t1=time.time()
    comm.Barrier()
    t2=time.time()
    res2=problem.parallel_penalty_solve(N,m,[1])
    t3=time.time()
    
    
    
    print 'seq time: ',t1-t0,'parallel time: ',t3-t2, 'rank:',rank
    print 'seq time/parallel time: ', (t1-t0)/(t3-t2), 'rank:',rank
    
    
    if rank==0:
        import matplotlib.pyplot as plt

        t = np.linspace(0,T,N+1)

        plt.plot(t,res1['control'].array()[:N+1],'r--')
        plt.plot(t,res2[0]['control'].array()[:N+1])
        plt.show()
if __name__ == "__main__":
    #test_parallel_solve()
    #test_one_itt_solve()
    test_parallel_gradient()
    """
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    
    non_parallel,problem = generate_problem(y0,yT,T,a)#PProblem1(y0,yT,T,a,J,grad_J,parallel_J=parallel_J)
    

    comm = problem.comm

    m = comm.Get_size()
    rank = comm.Get_rank()
    N = 100000
    
    u = np.zeros(N+m)
    u[:N+1] = 100*np.linspace(0,T,N+1) 
    
    #non_parallel = Problem1(y0,yT,T,a,J,grad_J)
    
    t0 = time.time()
    a = non_parallel.Penalty_Functional(u,N,m,1)
    t1 = time.time()
    #print 'time: ',t1-t0,'val: ',a,rank

    t2 = time.time()
   
    #print L
    b=problem.parallel_penalty_functional(u,N,1)
    t3 = time.time()
    #print 'time: ',t3-t2,'val: ', b,rank
    
    print (t1-t0)/(t3-t2),rank

    
    y,Y,y_list=problem.parallel_ODE_penalty_solver(u,N,m)
    
    
    l,L,l_list = problem.parallel_adjoint_penalty_solver(u,N,m,1)
    g = problem.penalty_grad(u,N,m,1)
    if rank==0:
        #print g
        print Y[-1]
    """
    """
    t = np.linspace(1,2,100)
    l=[]
    for i in range(3):
        l.append(u_part(100,3,i))
    print trapz(t**2,t)
    print trapz(t[:l[1]+1]**2,t[:l[1]+1]) + trapz(t[l[1]:l[2]+1]**2,t[l[1]:l[2]+1]) +trapz(t[l[2]:]**2,t[l[2]:])
    """
