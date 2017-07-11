import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ODE_pararealOCP import SimplePpcProblem,PararealOCP,SimpleExplicitPpcProblem
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
from optimalContolProblem import OptimalControlProblem,Problem1,Explicit_problem1
from scipy import linalg
from non_linear import Explicit_quadratic,Explicit_sine,ExplicitNonLinear

from mpi4py import MPI
from mpiVectorOCP import MpiVectorOCP,simpleMpiVectorOCP,generate_problem,local_u_size
from my_bfgs.mpiVector import MPIVector
from test_LbfgsPPC import GeneralPowerEndTermPCP,non_lin_problem
from runge_kutta_OCP import RungeKuttaProblem
from order_one_integration import exp_int,imp_int

def lin_problem(y0,yT,T,a,c=0,implicit=True):
    
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = imp_int((u-c)**2,t)#trapz((u)**2,t)

        return 0.5*I + (1./2)*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]-c+p[:-1])
        grad[0] = 0#0.5*dt*(u[0]) 
        grad[-1] =  dt*p[-2]+dt*(u[-1]-c)#0.5*dt*(u[-1])
        return grad

    def runge_grad(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))

        factor = dt*(3.+a*dt+(a*dt*0.5)**2)/6.
        factor2 = dt*(3+2*a*dt+0.75*(a*dt)**2+0.25*(a*dt)**3)/6.

        factor22 = 1+dt*(6*a+dt*a+2*dt*a**2+0.5*(dt*a)**2+0.5*a*(dt*a)**2+0.25*(dt*a)**3)/6.

        grad[:-1] =dt*(u[:-1]+p[:-1])
        grad[0] = 0.5*dt*(u[0]) +factor2*factor22*p[1]
        grad[-1] = 0.5*dt*(u[-1]) +factor*p[-1]
        return grad


    problem1 = RungeKuttaProblem(y0,yT,T,a,J,runge_grad)
    if implicit:
        problem2 = SimplePpcProblem(y0,yT,T,a,J,grad_J)
        #problem2 = Problem1(y0,yT,T,a,J,grad_J)
    else:
        def grad_J(u,p,dt):
            t = np.linspace(0,T,len(u))
            grad = np.zeros(len(u))
            grad[:-1] = dt*(u[:-1]-c+p[1:])
            
            return grad
        
        problem2 =SimpleExplicitPpcProblem(y0,yT,T,a,J,grad_J)#Explicit_problem1(y0,yT,T,a,J,grad_J)
    return problem2,problem1





def quadratic_state(y0,yT,T,a):
    
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = exp_int((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[:-1] = dt*(u[:-1]+p[1:])
        
        return grad
        

    problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)
    #problem=Explicit_sine(y0,yT,T,a,J,grad_J)

    return problem
def non_lin_state(y0,yT,T,F,DF):
    
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[:-1] = dt*(u[:-1]+p[1:])
        grad[0] = 0.5*dt*(u[0])+dt*p[1] 
        grad[-1] = 0.5*dt*(u[-1]) 
        return grad
        

    problem = ExplicitNonLinear(y0,yT,T,F,DF,J,grad_J)
    

    return problem


def taylor_test_non_penalty():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = -0.9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = float(T)/(N)
    
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(8):

        print J(u+h/(10**i))-J(u)


    def grad_J(x):
        l = problem.adjoint_solver(x,N)
        return problem.grad_J(x,l,dt)
    def grad_J2(x):
        l = problem2.adjoint_solver(x,N)
        return problem.grad_J(x,l,dt)
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(8):
        eps = 1./(10**i)
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)
        eps_list.append(eps)
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        #print max(abs(grad_fd[:]-grad[:]))
    
    data2 = pd.DataFrame(table,index=eps_list)
    #data2.to_latex('report/draft/discertizedProblem/taylorTest1.tex')
    
    print data2
    
    Q = np.vstack([np.log(np.array(eps_list)),np.ones(len(eps_list))]).T
    LS_F=linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)'])))[0]
    LS_grad = linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)-dJ(u)v'])))[0]
    print LS_F[0],np.exp(LS_F[1])
    print LS_grad[0],np.exp(LS_grad[1])
    import matplotlib.pyplot as plt


    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(grad)
    ax1.set_title('numerical gradient')
    ax2= fig.add_subplot(222)
    ax2.set_title('start of gradient')
    ax2.plot(np.linspace(0,4,5),grad[:5])
    ax3 = fig.add_subplot(223)
    ax3.set_title('end of gradient')
    ax3.plot(np.linspace(96,100,5),grad[-5:])
    ax4 =  fig.add_subplot(224)
    ax4.plot(grad)
    ax4.plot(grad_fd,'r--')
    ax4.set_title('finite difference gradient')
    ax4.legend(['num grad','fd grad'])
    plt.show()
    fig.savefig('report/draft/draft2/num_grad.png')
    
    """
    #grad2 = grad_J2(u)
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.legend(['num grad','finite diff grad'])
    plt.xlabel('gradient index')
    plt.ylabel('gradient value')
    plt.show()
    """
def taylor_penalty_test():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = .9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    m = 10
    my = 1
    

    J,grad_J = problem.generate_reduced_penalty(dt,N,m,my)
    h = 100*np.random.random(N+m)
    u = np.zeros(N+m) +1
    """
    for i in range(10):

        print abs(J(u+h/(10**i))-J(u))


    print

    for i in range(10):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print
    """
    for i in range(10):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        #print max(abs(grad_fd[:]-grad[:]))
    
    table = {'J(v+w)-J(v)':[],'J(v+w)-J(v)-dJ(v)w':[],'rate1':['--'],
             'rate2':['--'],'e w':[]}
    eps_list = []
    for i in range(9):
        eps = 1./(10**i)
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)
        eps_list.append(eps)
        table['J(v+w)-J(v)'].append(func_val)
        table['J(v+w)-J(v)-dJ(v)w'].append(grad_val)
        table['e w'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(v+w)-J(v)'][i-1]/table['J(v+w)-J(v)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(v+w)-J(v)-dJ(v)w'][i-1]/table['J(v+w)-J(v)-dJ(v)w'][i]))/np.log(10))
    print
    data2 = pd.DataFrame(table,index=eps_list)
    data2.to_latex('report/draft/discertizedProblem/penalty_taylorTest.tex')
    
    print data2
    
    import matplotlib.pyplot as plt
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(grad[:N+1])
    ax1.plot(grad_fd[:N+1],'r--')
    ax1.set_title('Control part of gradient')
    ax1.legend(['num grad','fd grad'])
    ax1.set_ylabel('Gradient value')

    ax4 =  fig.add_subplot(212)
    ax4.plot(grad[N+1:])
    ax4.plot(grad_fd[N+1:],'r--')
    ax4.set_title('Lambda part of gradient')
    ax4.set_ylabel('Gradient value')
    ax4.set_xlabel('index')
    ax4.legend(['num grad','fd grad'],loc=4)
    plt.show()
    fig.savefig('report/draft/draft2/pen_num_grad.png')
    """
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    
    plt.show()
    """
def quad_end():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 1.9
    p = 2
    c =0
    
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    my =10
    m=6
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(10):

        print J(u+h/(10**i))-J(u)


    def grad_J(x):
        l = problem.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)

    print

    for i in range(9):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(9):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    #grad2 = grad_J2(u)
    #plt.plot(grad)
    #plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    #plt.show()
    
    print
    print 'Penalty:'
    J,grad_J = problem.generate_reduced_penalty(dt,N,m,my)
    h = 100*np.random.random(N+m)
    u = np.zeros(N+m) 
    for i in range(10):

        print abs(J(u+h/(10**i))-J(u))


    print

    for i in range(10):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(10):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    
    plt.show()

    y,Y = problem.ODE_penalty_solver(u,N,m)
    plt.plot(Y)
    plt.show()
    
    p,P = problem.adjoint_penalty_solver(u,N,m,my)
    plt.plot(P)
    plt.show()
def finite_diff(J,u,epsilon):
    grad = np.zeros(len(u))

    for i in range(len(u)):
        Eps = np.zeros(len(u))
        Eps[i] = epsilon
        
        grad[i] = (J(u+Eps)-J(u))/epsilon

    return grad


def augemted_test():


    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    
    h = 100*np.random.random(N+1)


    
def runge_kutta_test():

    
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c =0.5

    problem2,problem = lin_problem(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = float(T)/(N)
    
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(8):

        print J(u+h/(10**i))-J(u)


    def grad_J(x):
        l = problem.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)
    def grad_J2(x):
        l = problem2.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(9):
        eps = 1./(10**i)
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)
        eps_list.append(eps)
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i])/np.log(10))
            table['rate2'].append(np.log(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i])/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        #print max(abs(grad_fd[:]-grad[:]))
    
    data2 = pd.DataFrame(table,index=eps_list)
    
    
    print data2
    
    Q = np.vstack([np.log(np.array(eps_list)),np.ones(len(eps_list))]).T
    LS_F=linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)'])))[0]
    LS_grad = linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)-dJ(u)v'])))[0]
    print LS_F[0],np.exp(LS_F[1])
    print LS_grad[0],np.exp(LS_grad[1])
    import matplotlib.pyplot as plt


    #grad2 = grad_J2(u)
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.show()
def taylor_test_mpi():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c =0.5


    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    rank = comm.Get_rank()
    m = comm.Get_size()
    N = 100
    mu =1
    dt = float(T)/(N)
    
    h = MPIVector(100*np.random.random(local_u_size(N+1,m,rank)),comm)
    
    

    J = lambda x: mpi_problem.parallel_penalty_functional(x,N,mu)
    u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
    for i in range(8):
        eps = 1./(10**i)
        #print J(u+h*eps)-J(u),rank


    def grad_J(x):
        l = mpi_problem.parallel_adjoint_penalty_solver(x,N,m,mu)
        return mpi_problem.penalty_grad(x,N,m,mu)

    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(9):
        eps = 1./(10**i)
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)
        eps_list.append(eps)
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    
    data = pd.DataFrame(table,index=eps_list)
    if rank==0:
        print data


def taylor_quadratic_state():

    y0 = 1.2
    yT = 2
    T  = 1
    a  = 10.9
    p = 2
    c =0.5

    problem = quadratic_state(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    m = 1
    dt = float(T)/(N)
    
    h = 100*np.random.random(N+m)
    
    mu = 1

    J2 = lambda u: problem.Functional(u,N)
    J3 = lambda u: problem.Penalty_Functional(u,N,m,mu)
    u = np.zeros(N+m) +1
    for i in range(8):

        print J2(u+h/(10**i))-J2(u)


    def grad_J2(x):
        
        return problem.Gradient(x,N)
    def grad_J3(x):
        #return
        return problem.Penalty_Gradient(x,N,m,mu)
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(8):
        eps = 1./(10**i)
        
        grad_val = abs(J2(u+h*eps) - J2(u) - eps*h.dot(grad_J2(u)))
        func_val = J2(u+h*(eps))-J2(u)
        eps_list.append(eps)
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J2,u,eps)
        grad = grad_J2(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    data2 = pd.DataFrame(table,index=eps_list)
    #data2.to_latex('report/draft/discertizedProblem/taylorTest1.tex')
    
    print data2
    """
    Q = np.vstack([np.log(np.array(eps_list)),np.ones(len(eps_list))]).T
    LS_F=linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)'])))[0]
    LS_grad = linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)-dJ(u)v'])))[0]
    print LS_F[0],np.exp(LS_F[1])
    print LS_grad[0],np.exp(LS_grad[1])
    """
    import matplotlib.pyplot as plt


    

    #grad2 = grad_J2(u)
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.legend(['num grad','finite diff grad'])
    plt.xlabel('gradient index')
    plt.ylabel('gradient value')
    plt.show()
def general_non_linear_test():


    y0 = 1.2
    yT = 2
    T  = 1
    a  = 10.9
    p = 2
    c =0.5

    F = lambda x : a*x**2
    DF = lambda x :2*x*a
    
    F = lambda x : a*np.cos(np.pi*2*x)
    DF = lambda x : -a*np.pi*2*np.sin(np.pi*2*x)

    F = lambda x : a*np.log(x-1)
    DF = lambda x: a*1./(x-1)
    
    #F = lambda x : np.exp(x)
    #DF = lambda x : np.exp(x)

    
    
    problem = non_lin_state(y0,yT,T,F,DF)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    m = 10
    
    dt = float(T)/(N)
    
    h = np.random.random(N+1)
    h2 = np.random.random(N+m)
    
    mu = 1

    J = lambda u: problem.Functional(u,N)
    J2 = lambda u: problem.Penalty_Functional(u,N,m,mu)
    u = np.zeros(N+1) +1
    u2 = np.zeros(N+m) +1.1
    for i in range(8):

        print J2(u2+h2/(10**i))-J2(u2),J(u+h/(10**i)),J(u)


    def grad_J(x):
        
        return problem.Gradient(x,N)
    def grad_J2(x):
        #return
        return problem.Penalty_Gradient(x,N,m,mu)
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    table2 = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(8):
        eps = 1./(10**i)
        
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)

        grad_val2 = abs(J2(u2+h2*eps) - J2(u2) - eps*h2.dot(grad_J2(u2)))
        func_val2 = J2(u2+h2*(eps))-J2(u2)
        
        eps_list.append(eps)
        table2['J(u+v)-J(u)'].append(func_val2)
        table2['J(u+v)-J(u)-dJ(u)v'].append(grad_val2)
        table2['e v'].append(eps*max(h2))
        
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))

            table2['rate1'].append(np.log(abs(table2['J(u+v)-J(u)'][i-1]/table2['J(u+v)-J(u)'][i]))/np.log(10))
            table2['rate2'].append(np.log(abs(table2['J(u+v)-J(u)-dJ(u)v'][i-1]/table2['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    data2 = pd.DataFrame(table,index=eps_list)
    #data2.to_latex('report/draft/discertizedProblem/taylorTest1.tex')
    data3 = pd.DataFrame(table2,index=eps_list)
    print data2
    print
    print data3
    """
    Q = np.vstack([np.log(np.array(eps_list)),np.ones(len(eps_list))]).T
    LS_F=linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)'])))[0]
    LS_grad = linalg.lstsq(Q, np.log(np.array(table['J(u+v)-J(u)-dJ(u)v'])))[0]
    print LS_F[0],np.exp(LS_F[1])
    print LS_grad[0],np.exp(LS_grad[1])
    """
    import matplotlib.pyplot as plt


    

    #grad2 = grad_J2(u)
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.legend(['num grad','finite diff grad'],loc=4)
    plt.xlabel('gradient index')
    plt.ylabel('gradient value')
    plt.show()

def general_taylor_test(problem,N=100,m=10):

    T = problem.T
    problem.t,problem.T_z = problem.decompose_time(N,m)
    dt = float(T)/(N)
    
    h = 100*np.random.random(N+1)
    h2 = 100*np.random.random(N+m)
    
    mu = 1

    J = lambda u: problem.Functional(u,N)
    J2 = lambda u: problem.Penalty_Functional(u,N,m,mu)
    u = np.zeros(N+1) +1
    u2 = np.zeros(N+m) +1.1
    for i in range(8):

        print J2(u2+h2/(10**i))-J2(u2),J(u+h/(10**i)),J(u)


    def grad_J(x):
        
        return problem.Gradient(x,N)
    def grad_J2(x):
        #return
        return problem.Penalty_Gradient(x,N,m,mu)
    print
    table = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    table2 = {'J(u+v)-J(u)':[],'J(u+v)-J(u)-dJ(u)v':[],'rate1':['--'],
             'rate2':['--'],'e v':[]}
    eps_list = []
    for i in range(9):
        eps = 1./(10**i)
        
        grad_val = abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
        func_val = J(u+h*(eps))-J(u)

        grad_val2 = abs(J2(u2+h2*eps) - J2(u2) - eps*h2.dot(grad_J2(u2)))
        func_val2 = J2(u2+h2*(eps))-J2(u2)
        
        eps_list.append(eps)
        table2['J(u+v)-J(u)'].append(func_val2)
        table2['J(u+v)-J(u)-dJ(u)v'].append(grad_val2)
        table2['e v'].append(eps*max(h2))
        
        table['J(u+v)-J(u)'].append(func_val)
        table['J(u+v)-J(u)-dJ(u)v'].append(grad_val)
        table['e v'].append(eps*max(h))
        if i!=0:
            table['rate1'].append(np.log(abs(table['J(u+v)-J(u)'][i-1]/table['J(u+v)-J(u)'][i]))/np.log(10))
            table['rate2'].append(np.log(abs(table['J(u+v)-J(u)-dJ(u)v'][i-1]/table['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))

            table2['rate1'].append(np.log(abs(table2['J(u+v)-J(u)'][i-1]/table2['J(u+v)-J(u)'][i]))/np.log(10))
            table2['rate2'].append(np.log(abs(table2['J(u+v)-J(u)-dJ(u)v'][i-1]/table2['J(u+v)-J(u)-dJ(u)v'][i]))/np.log(10))
    print
    
    for i in range(10):
        eps = 1./(2**i)
        grad_fd = finite_diff(J2,u2,eps)
        grad = grad_J2(u2)
        print max(abs(grad_fd[:]-grad[:]))

    grad_fd2 = finite_diff(J,u,eps)

    data2 = pd.DataFrame(table,index=eps_list)
    #data2.to_latex('report/draft/discertizedProblem/taylorTest1.tex')
    data3 = pd.DataFrame(table2,index=eps_list)
    print data2
    print
    print data3
    
    import matplotlib.pyplot as plt


    t = np.linspace(0,problem.T,N+1)


    #grad2 = grad_J2(u)
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(211)
    ax1.plot(t,grad[:N+1])
    plt.plot(t,grad_J(u))
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\nabla\hat J_{v}(v,\Lambda)$',fontsize=20)
    ax2 = plt.subplot(212)
    ax2.plot(grad[N+1:])
    ax2.set_xlabel('i')
    ax2.set_ylabel(r'$\nabla\hat J_{\lambda_i}(v,\Lambda)$',fontsize=20)
    #plt.savefig('report/draft/draft2/pen_num_grad.png')

    plt.figure(figsize=(12,6))

    plt.plot(t,grad_J(u))
    plt.plot(t,grad_fd2,'r--')
    plt.xlabel('t')
    plt.ylabel(r'$\nabla\hat J$',fontsize=20)
    plt.legend([r'$\nabla\hat J(v)$',r'$\frac{J(v+\epsilon w)-J(v)}{\epsilon}$'])
    #plt.savefig('report/draft/draft2/num_grad.png')
    """
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.legend(['num grad','finite diff grad'],loc=4)
    plt.xlabel('gradient index')
    plt.ylabel('gradient value')
    """
    plt.show()


if __name__ == '__main__':
    #taylor_test_non_penalty()
    #taylor_penalty_test()
    #quad_end()
    #runge_kutta_test()
    #taylor_test_mpi()
    taylor_quadratic_state()
    #general_non_linear_test()
"""
J(u+eh) = J(u) + O(e)

J(u+eh) = J(u) + eh*grad_J(u) + O(e**2)

"""
