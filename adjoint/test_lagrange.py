import numpy as np
from optimalContolProblem import *
from test_functionals import make_coef_J
import matplotlib.pyplot as plt
from concistencyTest import lin_problem
def test_LagAndPen():
    
    

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N  = 700   
    m  = 20

    C = [1]
    D = [0]

    my = [1,10,50,70,100]#0.1*N,0.7*N,2*N,10*N,100*N]

    
            
            
    problem = lin_problem(y0,yT,T,a)

    #res
    opt = {"mem_lim" :10,'jtol':0,'maxiter':60}
            
    seq_res= problem.solve(N)
    res1 = problem.penalty_solve(N,m,my,Lbfgs_options=opt)
            
            
    res2 = problem.lagrange_penalty_solve(N,m,my,Lbfgs_options=opt)
           
    print 
    #print res1[0]['iteration'],res1[1]['iteration'],res1[2]['iteration']
    #print res2[0]['iteration'],res2[1]['iteration'],res2[2]['iteration']
            
    plt.plot(seq_res.x,'--r')
    plt.plot(res1[-1].x[:N+1],'-r')
    plt.plot(res2[-1].x[:N+1])
    plt.show()

            




if __name__ == "__main__":
    test_LagAndPen()
    
