import numpy as np
from crank_nicolson_OCP import create_simple_CN_problem
import matplotlib.pyplot as plt

def test_con():

    problem = create_simple_CN_problem(1,1,1,1)

    N = [100,1000,10000,100000,1000000]

    u = lambda x :x

    for n in N:
        t = np.linspace(0,1,n+1)
        grad = problem.solve(n,Lbfgs_options={'jtol':1e-7})

        print grad.counter()
        plt.plot(t,grad.x)
    plt.show()


    return 0


if __name__=='__main__':
    test_con()
