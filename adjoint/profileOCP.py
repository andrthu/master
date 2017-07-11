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
    c = 0
    problem = generate_problem(y0,yT,T,a)[1]
    N = 125000
    m = 1
    opt = {'jtol':0,'maxiter':5,'ignore xtol':True}
    u = np.zeros(N+m)+1
    #problem.Penalty_Gradient(u,N,m,1)
    #problem.Gradient(u,N)
    #problem.Functional(u,N)
    #problem.Penalty_Functional(u,N,m,1)
    #problem.PPCLBFGSsolve(N,m,[N,N**2])
    problem.solve(N,Lbfgs_options=opt)
    #problem.penalty_solve(N,m,[N**2],Lbfgs_options=opt)
    
def look():
    stats = pstats.Stats("lbfgsppcProfile.prof")
    stats.sort_stats("cumtime")
    stats.print_stats(30)
def find():
    pr = cProfile.Profile()
    res = pr.run("main()")
    pr.dump_stats("lbfgsppcProfile.prof")
    
if __name__ == '__main__':
    try:
        sys.argv[1] 
        look()
    except:
        main()
    
    #"""
    #pr = cProfile.Profile()
    #foo = main()
    #N = 100000
    #cProfile.run("foo(N)")
    #pr.dump_stats("lbfgsppcProfile.prof")
    #"""
    
   
"""    
python -m cProfile -o lbfgsppcProfile.prof profileOCP.py
python profileOCP.py 0
"""
