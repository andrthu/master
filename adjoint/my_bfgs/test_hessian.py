import numpy as np
from my_vector import *

from LmemoryHessian import MuLMIH,LimMemoryHessian


def test_matvec():


    I = np.identity(20)

    t = np.linspace(0,1,20)
    
    mu = 2

    s1 = SimpleVector(t)
    y1 = mu*SimpleVector(np.exp(t))
        
    s2 = MuVector([SimpleVector(t),SimpleVector(np.zeros(20))])
    y2 = MuVector([SimpleVector(np.zeros(20)),SimpleVector(np.exp(t))])
    
    H = LimMemoryHessian(I)
    MuH = MuLMIH(I,mu)

    H.update(y1,s1)
    MuH.update(y2,s2)

    x = SimpleVector(np.zeros(20)+1)
    x1 = SimpleVector(np.zeros(20)+1)
    
    print H.matvec(x).array()
    print
    print MuH.matvec(x1).array()
    print
    print (H.matvec(x)-MuH.matvec(x1)).array()

if __name__ == '__main__':
    
    test_matvec()
