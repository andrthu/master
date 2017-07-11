import cProfile,pstats
from my_bfgs.mpiVector import MPIVector
from mpiVectorOCP import generate_problem,local_u_size
import sys
import numpy as np
def main():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c = 0.5
    _,problem = generate_problem(y0,yT,T,a)
    N = 50000
    
    rank=problem.comm.Get_rank()
    comm=problem.comm
    m = comm.Get_size()
    opt = {'jtol':0,'maxiter':5,'ignore xtol':True}
    #problem.parallel_penalty_solve(N,m,[1],Lbfgs_options=opt)
    res=problem.parallel_PPCLBFGSsolve(N,m,[1],options=opt)
    print res[-1].niter
    #u = MPIVector(np.zeros(local_u_size(N+1,m,rank))+1,comm)
    #problem.penalty_grad(u,N,m,1)
    #problem.parallel_penalty_functional(u,N,1)
    #problem.Gradient(u,N)
    #problem.Functional(u,N)
    #problem.Penalty_Functional(u,N,m,1)
    #problem.PPCLBFGSsolve(N,m,[N,N**2])
    #problem.solve(N,Lbfgs_options=opt)
    #problem.penalty_solve(N,m,[N**2],Lbfgs_options=opt)
    
def look():
    stats = pstats.Stats("profmpi.prof")
    stats.sort_stats("cumtime")
    stats.print_stats(30)
def find():
    pr = cProfile.Profile()
    res = pr.run("main()")
    pr.dump_stats("profmpi.prof")
    
if __name__ == '__main__':
    
    try:
        sys.argv[1] 
        A = True
    except:
        print 'main'
        main()
        A = False

    if A:
        look()

    
    #"""
    #pr = cProfile.Profile()
    #foo = main()
    #N = 100000
    #cProfile.run("foo(N)")
    #pr.dump_stats("lbfgsppcProfile.prof")
    #"""
    
   
"""    
mpiexec -n 4 python -m cProfile -o profmpi.prof mpiProfileOCP.py
python mpiProfileOCP.py 0
"""
