from mpiVectorOCP import simpleMpiVectorOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
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

class GeneralPowerEndTermMPIOCP(simpleMpiVectorOCP):

    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/p*(y(T)-yT)**p
    with y' = ay + u
    """

    def __init__(self,y0,yT,T,a,power,J,grad_J,parallel_J=None,parallel_grad_J=None,options=None):
        simpleMpiVectorOCP.__init__(self,y0,yT,T,a,J,grad_J,parallel_J=parallel_J,
                                    parallel_grad_J=parallel_grad_J,options=options)
        self.power = power
    
        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**(p-1)

def interval_length_and_start(N,m,i):

    q = N/m
    r = N%m
    
    start = 0
    end = 0
    j = 0
    while j <=i:
        start = end

        if r-j>0:
            end +=q+1
        else:
            end += q
        j+=1

    return start,end

def non_lin_problem(y0,yT,T,a,p,c=0,func=None):
    
    if func==None:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-c)**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-c+p[:-1])
            grad[0] = 0.5*dt*(u[0]-c)
            grad[-1] = 0.5*dt*(u[-1]-c) + dt*p[-2]
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
                s[0] = ((y[-1]-yT)**p)/ float(p)
            
                I = dt*np.sum((u[:-2]-c)**2)
                I+= dt*0.5*(u[-2]-c)**2
                s[0] += 0.5*I 
            elif rank!=0:
                s[0] = 0.5*dt*np.sum((u[:-1]-c)**2)
                s[0] += 0.5*mu*(y[-1]-lam)**2

            else:
                s[0] = 0.5*dt*np.sum((u[1:]-c)**2)            
                s[0] += 0.5*0.5*dt*(u[0]-c)**2
                s[0] += 0.5*mu*(y[-1]-lam)**2
        
            S =np.zeros(1)
            comm.Barrier()
            comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

            return S[0]


        def parallel_grad_J(u,p,dt,comm):

            rank = comm.Get_rank()
            m = comm.Get_size()
            if rank ==0:

                grad = np.zeros(len(u))
                grad[1:] = dt*(u[1:] + p[:-1]-c)
                grad[0] = 0.5*dt*(u[0]-c)

            elif rank!=m-1:
                grad = np.zeros(len(u))

                grad[:-1]=dt*(u[:-1]+p[:-1]-c)
            else:

                grad = np.zeros(len(u))
                grad[:-2] = dt*(u[:-2]+p[:-2]-c)
                grad[-2] = dt*0.5*(u[-2]-c) + dt*p[-2]
            return grad

    else:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-func(t))**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            t = np.linspace(0,T,len(u))
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-func(t[1:])+p[:-1])
            grad[0] = 0.5*dt*(u[0]-func(t[0]))
            grad[-1] = 0.5*dt*(u[-1]-func(t[-1])) + dt*p[-2]
            return grad


        def parallel_J(u,y,yT,T,N,mu,comm):
            m = comm.Get_size()
            rank = comm.Get_rank()
        
        
            dt = float(T)/N
            
            t_start,t_end = interval_length_and_start(N+1,m,rank)

            t = dt*np.linspace(t_start,t_end-1,t_end-t_start)

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
                s[0] = ((y[-1]-yT)**p)/ float(p)
            
                I = dt*np.sum((u[:-2]-func(t[:-1]))**2)
                I+= dt*0.5*(u[-2]-func(t[-1]))**2
                s[0] += 0.5*I 
            elif rank!=0:
                s[0] = 0.5*dt*np.sum((u[:-1]-func(t[:]))**2)
                s[0] += 0.5*mu*(y[-1]-lam)**2

            else:
                s[0] = 0.5*dt*np.sum((u[1:]-func(t[1:]))**2)            
                s[0] += 0.5*0.5*dt*(u[0]-func(t[0]))**2
                s[0] += 0.5*mu*(y[-1]-lam)**2
        
            S =np.zeros(1)
            comm.Barrier()
            comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

            return S[0]


        def parallel_grad_J(u,p,dt,comm):

            rank = comm.Get_rank()
            m = comm.Get_size()
            N = u.global_length()
            t_start,t_end = interval_length_and_start(N+1-m,m,rank)

            t = dt*np.linspace(t_start,t_end-1,t_end-t_start)
            if rank ==0:

                grad = np.zeros(len(u))
                grad[1:] = dt*(u[1:] + p[:-1]-func(t[1:]))
                grad[0] = 0.5*dt*(u[0]-func(t[0]))

            elif rank!=m-1:
                grad = np.zeros(len(u))

                grad[:-1]=dt*(u[:-1]+p[:-1]-func(t))
            else:

                grad = np.zeros(len(u))
                grad[:-2] = dt*(u[:-2]+p[:-2]-func(t[:-1]))
                grad[-2] = dt*0.5*(u[-2]-func(t[-1])) + dt*p[-2]
            return grad



    problem = GeneralPowerEndTermMPIOCP(y0,yT,T,a,p,J,grad_J,parallel_J=parallel_J,parallel_grad_J=parallel_grad_J)

    return problem


def get_speedup(task='both',name='speedup'):
    
    #import matplotlib.pyplot as plt

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    c = 0.5
    f = lambda x : 100*np.cos(5*np.pi*x)
    PROBLEM_NUMBER = 1
    
    try:
        N = int(sys.argv[1])
    except:
        N = 10000
        
    if PROBLEM_NUMBER==1:
        problem = non_lin_problem(y0,yT,T,a,p,c=c)#,func=lambda x:x**2)
        mu_val = [N,N**2]
    elif PROBLEM_NUMBER == 2:
        mu_val = [np.sqrt(N)]
        y0_2 = 24.6
        yT_2 = 170.9
        T_2 = 1.4
        a_2 = 2.5
        f = lambda x : 100*np.cos(5*np.pi*x)
        problem = non_lin_problem(y0_2,yT_2,T_2,a_2,p,func=f)
        name = name + str(PROBLEM_NUMBER)
    elif PROBLEM_NUMBER == 3:
        p = 2
        mu_val = [1]
        problem = non_lin_problem(y0,yT,T,a,p,c=c,func=f)
        name = name + str(PROBLEM_NUMBER)

        
    comm = problem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    if task == 'both':
        t0 = time.time()
        seq_res=problem.solve(N,Lbfgs_options={'jtol':1e-10,'maxiter':100})
        t1 = time.time()
        comm.Barrier()
        t2 = time.time()
        #par_res=problem.parallel_penalty_solve(N,m,mu_val,Lbfgs_options={'jtol':0,'maxiter':50,'ignore xtol':True})
        par_res=problem.parallel_PPCLBFGSsolve(N,m,mu_val,options={'jtol':1e-10,'maxiter':100,'ignore xtol':True})
        t3 = time.time()
    
        print 
        print t1-t0,t3-t2,(t1-t0)/(t3-t2), seq_res.niter,par_res[-1].niter,seq_res.lsiter,par_res[-1].lsiter

        x = par_res[-1].x.gather_control()
        if rank == 0:
            #plt.plot(x[:N+1])
            #plt.plot(seq_res.x,'r--')
            print max(abs(x[:N+1]-seq_res.x))/max(abs(seq_res.x))
        #plt.show()


    elif task =='seq':
        t0 = time.time()
        seq_res=problem.solve(N,Lbfgs_options={'jtol':1e-10,'maxiter':0})
        t1 = time.time()
        #print t1-t0,seq_res.niter,seq_res.lsiter
        if rank == 0:
            out = open('outputDir/'+name+'_'+str(N)+'.txt','w')
            out.write("seq: %f %d %d \n"%(t1-t0,seq_res.niter,seq_res.lsiter))


            out.close()
    elif task == 'par':
        t2 = time.time()
        #par_res=problem.parallel_penalty_solve(N,m,mu_val,Lbfgs_options={'jtol':1e-5,'maxiter':100,'ignore xtol':True})
        par_res=problem.parallel_PPCLBFGSsolve(N,m,mu_val,options={'jtol':1e-10,'maxiter':0,'ignore xtol':True})
        t3 = time.time()
        if rank == 0:
            if name[0] =='w':
                outputname = 'outputDir/'+name+'_'+str(int(N/m))+'.txt'
            else:
                outputname = 'outputDir/'+name+'_'+str(N)+'.txt'
            out = open(outputname,'a')
            out.write("par %d: %f %d %d \n"%(m,t3-t2,par_res[-1].counter()[0],par_res[-1].counter()[1]))
            out.close()

        #print t3-t2,par_res[-1].niter,par_res[-1].lsiter


def main():
    try:
        
        sys.argv[3]
        name = 'weak_speedup'
    except:
        name = 'speedup'
    try:
        
        val = int(sys.argv[2])

        if val == 0:
            get_speedup(task='seq',name=name)
        
        else:
            get_speedup(task='par',name=name)
    except:
        get_speedup()
    """
    out=open('outputDir/hei_test.txt','w')
    out.write('hei \n hallo \n')
    out.close()
    """
def compare_seq_to_seq():
    
    #import matplotlib.pyplot as plt

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    c = 0.5

    
    problem = non_lin_problem(y0,yT,T,a,p,c=c,func=lambda x:10*np.sin(3*np.pi*x))
    comm = problem.comm
    rank = comm.Get_rank()
    N = 1000

    I = [35,36,37,38,39,50]
    iter_res = []
    legend = []
    Res=[]
    for i in I:
        
        seq_res=problem.solve(N,Lbfgs_options={'jtol':0,'maxiter':i-5})
        Res.append(seq_res)
        if rank ==0:
            legend.append(i)
            #plt.plot(seq_res.x)
            iter_res.append(seq_res.niter)

    
    if rank ==0:
        print iter_res
        #plt.legend(legend)
        for i in range(len(Res)-1):
            print max(abs(Res[i].x-Res[-1].x))
        #plt.show()

        
        
    
    
if __name__ == '__main__':
    

    main()
    #get_speedup()
    #compare_seq_to_seq()
