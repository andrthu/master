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
from mpiVectorOCP import MpiVectorOCP,simpleMpiVectorOCP,generate_problem,local_u_size


def test_solve(N,problem,pproblem,name='solveSpeed'):
    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    
    ue,t,_ = pproblem.simple_problem_exact_solution(N)

    if m == 1:
        opt = {'jtol':1e-10}
        t0 = time.time()
        res = pproblem.solve(N,Lbfgs_options=opt)
        t1=time.time()
        val = t1-t0
        temp1 = open('temp_time.txt','a')
        temp1.write('%f ' %val)
        temp1.close()
        
        fu,gr=res.counter()
        norm = np.sqrt(np.sum((ue[1:-1]-res.x[1:-1])**2)/len(ue))
        temp2 = open('temp_info.txt','w')
        temp2.write('%d %d %d %d %e'%(int(fu),int(gr),res.niter,res.lsiter))
        temp2.close()

    else:
        """
        seq: 30.018747 24 25 4 24
        par 2: 31.869821 44 44 13 39 
        par 3: 25.423143 48 48 12 41 
        par 4: 33.682735 78 78 10 39 
        par 5: 31.024316 84 84 15 66 
        par 6: 33.121180 104 104 17 55 
        """
        try:
            
            itr_list = pre_chosen_itr[N]
            opt = {'jtol':1e-7,'maxiter':itr_list[m-2]}
        except:
            opt = {'jtol':1e-5}
        mu_list = [0.01*N]
        if name=='solveSpeed':
            comm.Barrier()
            t0 = time.time()
            res = pproblem.parallel_PPCLBFGSsolve(N,m,mu_list,tol_list=[1e-5,1e-10],options=opt)
            t1 = time.time()
            comm.Barrier()
            res=res[-1]
        elif name=='solveSpeed_ord':
            print 'lel'
        loc_time = np.zeros(1)
        loc_time[0] = t1-t0
        grad_norm = res.mpi_grad_norm()
        if rank == 0:
            time_vec = np.zeros(m)
        else:
            time_vec = None
        loc_size = tuple(np.zeros(m)+1)
        loc_start = tuple(np.linspace(0,m-1,m))
        comm.Gatherv(loc_time,[time_vec,loc_size,loc_start,MPI.DOUBLE])
        np.save('outputDir/vectorOut/par_sol_'+str(rank),res.x.local_vec)
        comm.Barrier()
        if rank==0:

            min_time = min(time_vec)
            
            time_saver = open('temp_time.txt','a')
            time_saver.write('%f ' %min_time)
            time_saver.close()

            fu,gr=res.counter()
            norm,fu_val = read_vector(N,m,problem)
            info_file = open('temp_info.txt','w')
            info_file.write('%d %d %d %d %e %e %e'%(int(fu),int(gr),res.niter,res.lsiter,norm,grad_norm,fu_val))
            info_file.close()
