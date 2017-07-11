import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from crank_nicolson_OCP import create_simple_CN_problem


def main():

    y0 = 3.2
    yT=11.5
    a = -0.097
    T=100
    problem = create_simple_CN_problem(y0,yT,T,a)

    N = 1000
    ue,t,_ = problem.simple_problem_exact_solution(N)
    seq_res = problem.solve(N,Lbfgs_options={'jtol':1e-7})
    MU = [0.01*N,0.1*N,N,10*N,100*N,100000*N,10000000*N]

    Ls = seq_res.counter()[0]+seq_res.counter()[1]
    
    m = 16
    table = {'err':[],'L':[],'S':[],'err2':[],'L2':[],'S2':[]}

    for i in range(len(MU)):

        res1 = problem.PPCLBFGSsolve(N,m,[MU[i]],tol_list=[1e-5,1e-5],options = {'jtol':1e-5})
        #res2 = problem.penalty_solve(N,m,[MU[i]],Lbfgs_options={'jtol':1e-5})
        res2=res1
        err1=np.sqrt(trapz((res1.x[1:N]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))
        err2=np.sqrt(trapz((res2.x[1:N]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))

        L1 = res1.counter()[1]+res1.counter()[0]
        L2 = res2.counter()[1]+res2.counter()[0]

        table['err'].append(err1)
        table['err2'].append(err2)
        table['L'].append(L1)
        table['L2'].append(L2)
        table['S'].append(m*Ls/float(L1))
        table['S2'].append(m*Ls/float(L2))

    print Ls, np.sqrt(trapz((seq_res.x[1:-1]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))
    data = pd.DataFrame(table,index=MU)
    res3 = problem.PPCLBFGSsolve(N,m,[100*N,1000000*N,10000000*N],tol_list=[1e-5,1e-5])
    print res3[-1].counter(),res3[0].counter(),np.sqrt(trapz((res3[-1].x[1:N]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))


    #data.to_latex('report/draft/parareal/mu_test2.tex')
    print data


    m = 16 

    res2mu = problem.PPCLBFGSsolve(N,m,[100*N,100000*N],tol_list=[1e-5,1e-5],options = {'jtol':1e-5})

    mu2err1 = table['err'][4]
    mu2err2 = np.sqrt(trapz((res2mu[-1].x[1:N]-ue[1:-1])**2,t[1:-1]))/np.sqrt(trapz(ue**2,t))

    Lmu2 = res2mu[-1].counter()[0]+res2mu[-1].counter()[1]

    print mu2err1,mu2err2,Lmu2,table['L'][4],m*Ls/float(Lmu2)

if __name__=='__main__':

    main()
