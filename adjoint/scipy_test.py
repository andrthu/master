from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
from test_functionals import make_coef_J
def end_term_test(): 
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    power = 4
    coef = [1,0.5]
    
    J, grad_J = make_coef_J(coef[0],coef[1],power=power)
        
    def Jfunc(u,y,yT,T,power):
        return J(u,y,yT,T)
            
    problem = GeneralPowerY(y0,yT,T,a,power,Jfunc,grad_J)

    problem.scipy_simple_test(N,make_plot=False)


if __name__ == "__main__":
    end_term_test()
