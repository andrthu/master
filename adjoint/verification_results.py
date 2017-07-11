import numpy as np
from taylorTest import general_taylor_test
from test_LbfgsPPC import jump_difference
from test_exact import euler_con,crank_con
from crank_nicolson_OCP import create_simple_CN_problem


def main():

    y0 = 3.2
    yT = 11.5
    T  = 1
    a  = -3.9


    problem = create_simple_CN_problem(y0,yT,T,a,c=0)
    

    general_taylor_test(problem)
    euler_con(y0,yT,T,a)
    crank_con(y0,yT,T,a)
    jump_difference(problem)
    return 0

if __name__ =='__main__':
    main()
