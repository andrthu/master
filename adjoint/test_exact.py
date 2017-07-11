from crank_nicolson_OCP import create_simple_CN_problem
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from test_LbfgsPPC import non_lin_problem
import pandas as pd
from taylorTest import lin_problem
def check_exact():

    y0 = 3.2
    yT = 10.5
    T = 10
    a = -2

    problem = create_simple_CN_problem(y0,yT,T,a,c=20)
    problem.c =20
    #problem = non_lin_problem(y0,yT,T,a,2,func=lambda x : 0*x)

    N = 100*T

    ue,t,_ = problem.simple_problem_exact_solution(N)

    res = problem.solve(N,Lbfgs_options={'jtol':1e-10})
    res2 = problem.PPCLBFGSsolve(N,80,[10000,200000],tol_list=[1e-5,1e-8,1e-8],options={'jtol':1e-5})[-1]
    val1=problem.Functional(ue,N)
    val2 = problem.Functional(res.x,N)
    print val1,val2,val1-val2
    print res.counter(),res2.counter()
    u0 = res.x[-1]
    y = problem.ODE_solver(res.x,N)
    u_test = lambda x : u0*np.exp(a*(T-t))
    print max(abs(res.x[1:-1]-ue[1:-1])),np.sqrt(trapz((res.x[1:-1]-ue[1:-1])**2,t[1:-1])),max(abs(res2.x[1:N]-ue[1:-1]))
    plt.plot(t,ue+20,'--')
    plt.plot(t,res.x)
    plt.plot(t,res2.x[:N+1])
    #plt.plot(t,u_test(t),'.')
    plt.show()


def crank_con(y0,yT,T,a):


    
    problem = create_simple_CN_problem(y0,yT,T,a,c=0)
    gen_con(problem,name='CN_exact_convergence')

def euler_con(y0,yT,T,a):

    problem,_ = lin_problem(y0,yT,T,a)#non_lin_problem(y0,yT,T,a,2,func=lambda x : 0*x)
    gen_con(problem,name='euler_exact_convergence')


def gen_con(problem,name='exact_convergence'):

    N_val = [50,100,1000,10000]#,600000]#,1000000]

    table = {'norm':[],'val':[],'norm r':['--'],'val r':['--']}
    for i in range(len(N_val)):
        ue,t,_ = problem.simple_problem_exact_solution(N_val[i])
        res = problem.solve(N_val[i],Lbfgs_options={'jtol':1e-10})
        val1=problem.Functional(ue,N_val[i])
        val2 = problem.Functional(res.x,N_val[i])
        exact_norm = 1#max(abs(ue[1:-1]))
        #table['norm'].append(max(abs(res.x[1:-1]-ue[1:-1]))/exact_norm)
        table['norm'].append(np.sqrt(np.sum((res.x[1:-1]-ue[1:-1])**2)/len(ue))/exact_norm)
        print val1,val2
        table['val'].append((val1-val2)/val2)
        plt.plot(t,res.x)
        #plt.show()
        if i>0:
            table['norm r'].append(np.log(table['norm'][i]/table['norm'][i-1])/np.log(N_val[i]/N_val[i-1]))
            table['val r'].append(np.log(table['val'][i]/table['val'][i-1])/np.log(N_val[i]/N_val[i-1]))

    dt_val = float(problem.T)/np.array(N_val)
    data = pd.DataFrame(table,index = dt_val)
    data =data.ix[:,['norm','val','norm r','val r']]
    
    #data.to_latex('report/whyNotEqual/'+name+'.tex')

    print data
    plt.plot(t,ue)
    plt.show()
if __name__ =='__main__':
    check_exact()
    """
    y0 = 3.2
    yT = 1.5
    T = 10
    a = -2
    """
    y0 = 3.2
    yT = 11.5
    T = 100
    a = -0.097
    #crank_con(y0,yT,T,a)
    #euler_con(y0,yT,T,a)
