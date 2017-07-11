from numpy import *
from crank_nicolson_OCP import create_simple_CN_problem
import matplotlib.pyplot as plt

def test():
    
    y0 = 3.2
    yT = 10.5
    T = 5
    a = -2

    problem = create_simple_CN_problem(y0,yT,T,a,c=0)
    
    N = 1000
    m = 10
    res3 = problem.solve(N,algorithm='my_steepest_decent')
    print res3.counter()
    """
    res2 = problem.penalty_solve(N,m,[1,10],algorithm='my_steepest_decent')[-1]

    print res2.counter()
    plt.plot(res2.x[:N+1])
    plt.plot(res3.x)
    plt.show()
    """
    pc=None
    pc = problem.PC_maker4(N,m)
    res = problem.alternate_direction_penalty_solve(N,m,[1,10,100],ppc=pc)
    
    x = res[2]

    print res[0].counter(),res[1].counter()

    plt.plot(x[:N+1])
    plt.plot(res3.x,'--r')
    plt.show()

    plt.plot(x[N+1:])
    plt.show()
   
    return

if __name__ =='__main__':
    test()
