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

def weak(problem,N,m):

    t0 = time.time()
    problem.parallel_penalty_solve(m*N,m,[(m*N)**2],Lbfgs_options={'jtol':0,'maxiter':100})
    t1 = time.time()

    return t1-t0
