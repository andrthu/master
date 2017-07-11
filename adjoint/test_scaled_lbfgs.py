from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
import matplotlib.pyplot as plt
import pandas as pd


def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def test1():
    
    T=1
    y0=1
    a=1
    alpha=0.2
    yT=10

    N = 800
    m = 20
    mu=1

    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    


    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)

    res1=problem.penalty_solve(N,m,[mu])
    res2=problem.penalty_solve(N,m,[mu],scale=True)
    print res1['iteration'],res2['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array()[:N+1])
    plt.plot(t,res2['control'].array()[:N+1],'r--')
    plt.show()
    plt.plot(res1['control'].array()[N+1:])
    plt.plot(res2['control'].array()[N+1:])
    plt.show()


def test2():
    y0 = 3
    yT = 0
    T  = 1.
    a  = 1.3
    P  = 4
    N=700
    m = 10
    mu = 10
    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return (0.5*I + (1./power)*(y-yT)**power)

    def J2(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)
    
    problem  = GeneralPowerY(y0,yT,T,a,P,J,grad_J)
    res=problem.penalty_solve(N,m,[mu],scale=True)
    res2=problem.penalty_solve(N,m,[mu],scale=False)
    print(res['iteration'],res2['iteration'])

    t = np.linspace(0,T,N+1)
    plt.plot(t,res['control'].array()[:N+1],'r--')
    plt.plot(t,res2['control'].array()[:N+1])
    plt.show()
def test3():

    T=1
    y0=1
    a=1
    alpha=0.2
    yT=0

    N = 800
    m = 5
    mu=10

    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    
    mem=[0,1,5]
    table = {'unscaled'         : [],
             'scaled'           : [],
             'scaled hessian'   : [],
             'steepest descent' : []}
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    res3=problem.penalty_solve(N,m,[mu],algorithm='my_steepest_decent',scale=True)
    for i in range(len(mem)):
        problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
        opt1 = {'mem_lim':mem[i],'maxiter':200,'scale_hessian':False}
        opt2 = {'mem_lim':mem[i],'maxiter':200,'scale_hessian':True}
        res1=problem.penalty_solve(N,m,[mu],Lbfgs_options=opt1)
        res2=problem.penalty_solve(N,m,[mu],scale=True,Lbfgs_options=opt2)
        #res3=problem.penalty_solve(N,m,[mu],algorithm='my_steepest_decent',scale=True)
        res4=problem.penalty_solve(N,m,[mu],scale=True,Lbfgs_options=opt1)
        print res1['iteration'],res2['iteration'],res3.niter
        
        table['unscaled'].append(res1['iteration'])
        table['scaled'].append(res4['iteration'])
        table['scaled hessian'].append(res2['iteration'])
        table['steepest descent'].append(res3.niter)

        t = np.linspace(0,T,N+1)
        plt.plot(t,res1['control'].array()[:N+1])
        plt.plot(t,res2['control'].array()[:N+1],'r--')
        plt.plot(t,res3.x[:N+1])
        plt.show()
        plt.plot(res1['control'].array()[N+1:])
        plt.plot(res2['control'].array()[N+1:])
        plt.show()
        
    iter_data = pd.DataFrame(table,index=['mem_lim=0','mem_lim=1','mem_lim=5'])
    print iter_data
    #iter_data.to_latex('iter_data.tex')

def scaled_and_memory_lim():
    
    y0 = 1.2
    a = 0.9
    T = 1.
    yT = 5
    alpha = 0.5
    N = 500
    m = 5
    mu = 1
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)

    
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    res = problem.penalty_solve(N,m,[mu],algorithm='my_steepest_decent',
                                scale=True)

    non_penalty_res = problem.solve(N)
    cont0 = non_penalty_res['control'].array()
    table = {'scaled itr'          : [],
             'unscaled itr'        : [],
             'scaled err'       : [],
             'unscaled err'     : [],
             'steepest descent': [res.niter,'--','--','--',]}

    mem = [0,1,5,10]
    
    t = np.linspace(0,T,N+1)


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    plot_list=[ax1,ax2,ax3,ax4]
    for i in range(len(mem)):
        if mem[i]==0:
            opt1 = {'mem_lim':mem[i],'maxiter':500,}
            opt2 = {'mem_lim':mem[i],'maxiter':500,}#'scale_hessian':True}
        else:
            opt1 = {'mem_lim':mem[i],'maxiter':500,}#'scale_factor':2000}
            opt2 = {'mem_lim':mem[i],'maxiter':500,}#'scale_hessian':True}
        unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt1)
        scaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt2,scale=True)

        cont1=unscaled_res['control'].array()[:N+1]
        cont2=scaled_res['control'].array()[:N+1]

        err1 = l2_diff_norm(cont0,cont1,t)
        err2 = l2_diff_norm(cont0,cont2,t)
        
        #plot_list[i].plot(t,cont0)
        plot_list[i].plot(t,cont1)
        plot_list[i].plot(t,cont2,'r--')
        plot_list[i].legend(['unscaled','scaled'])
        
        table['scaled itr'].append(scaled_res['iteration'])
        table['unscaled itr'].append(unscaled_res['iteration'])
        table['scaled err'].append(err2)
        table['unscaled err'].append(err1)

    data = pd.DataFrame(table,index=mem)
    print data
    #data.to_latex('report/draft/optimization/scaled_and_memory_lim.tex')
    
    plt.show()
    

def change_gamma():
    
    y0 = 1.2
    a = 0.9
    T = 1.
    yT = 5
    alpha = 0.5
    N = 500
    m = 5
    mu = 1
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)

    
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)


    gamma = [1,10,20,100,200,500,1000,2000,4000,10000,20000,50000,100000]
    #gamma = [100,200,500,1000,2000,4000]
    table = {'scaled iter':[],
             'unscaled iter':[],
             'scaled gamma':[],}
             
    unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options={'maxiter':500,'mem_lim':1})
    t = np.linspace(0,T,N+1)
    for i in range(len(gamma)):
        opt = {'maxiter':100,'scale_factor':gamma[i],'mem_lim':1}


        #unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt)
        scaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt,scale=True)

        iter1 = scaled_res['iteration']
        iter2 = unscaled_res['iteration']

        table['scaled iter'].append(iter1)
        table['unscaled iter'].append(iter2)
        table['scaled gamma'].append(scaled_res['scaler'].gamma)

    


    data = pd.DataFrame(table,index=gamma)
    print data
    #plt.show()
    #data.to_latex('report/draft/optimization/change_gamma.tex')
    plt.plot(t,scaled_res['control'].array()[:N+1],'r--')
    plt.plot(t,unscaled_res['control'].array()[:N+1])
    plt.show()
    
    plt.plot(scaled_res['control'].array()[N+1:],'r--')
    plt.plot(unscaled_res['control'].array()[N+1:])
    plt.show()


def scaled_initial_hessian():


    y0 = 1.2
    a = 0.9
    T = 1.
    yT = 5
    alpha = 0.5
    N = 500
    m = 10
    mu = 10
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)

    
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    non_penalty_res = problem.solve(N)

    print 'reeeeeeeeeeeeeeeeeeeeeee',non_penalty_res['iteration']
    cont0 = non_penalty_res['control'].array()
    table1 = {'scaled itr'          : [],
             'unscaled itr'        : [],
             'scaled err'       : [],
             'unscaled err'     : [],}

    mem = [1,2,5,10]
    
    t = np.linspace(0,T,N+1)

    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    plot_list=[ax1,ax2,ax3,ax4]
    for i in range(len(mem)):
        if mem[i]==0:
            opt1 = {'mem_lim':mem[i],'maxiter':500,}
            opt2 = {'mem_lim':mem[i],'maxiter':500,}#'scale_hessian':True}
        else:
            opt1 = {'mem_lim':mem[i],'maxiter':500,}#'scale_factor':2000}
            opt2 = {'mem_lim':mem[i],'maxiter':500,'scale_hessian':True}
        unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt1)
        scaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt2,scale=True)

        cont1=unscaled_res['control'].array()[:N+1]
        cont2=scaled_res['control'].array()[:N+1]

        err1 = l2_diff_norm(cont0,cont1,t)
        err2 = l2_diff_norm(cont0,cont2,t)
        
        #plot_list[i].plot(t,cont0)
        plot_list[i].plot(t,cont1)
        plot_list[i].plot(t,cont2,'r--')
        plot_list[i].legend(['unscaled','scaled'])
        
        table1['scaled itr'].append(scaled_res['iteration'])
        table1['unscaled itr'].append(unscaled_res['iteration'])
        table1['scaled err'].append(err2)
        table1['unscaled err'].append(err1)

    data1 = pd.DataFrame(table1,index=mem)
    print data1
    #data1.to_latex('report/draft/optimization/hessian_scaled_and_memory_lim.tex')
    
    plt.show()

    gamma = [0.25,0.5,0.75,1,10,20,100,200,500,1000,2000,20000,50000]
    #gamma = [100,200,500,1000,2000,4000]
    table2 = {'scaled iter':[],
             'unscaled iter':[],
             'scaled gamma':[],}
    #mu = 100
    unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options={'maxiter':500,'mem_lim':10})
    t = np.linspace(0,T,N+1)
    for i in range(len(gamma)):
        opt = {'maxiter':100,'scale_factor':gamma[i],'mem_lim':10,'scale_hessian':True}


        #unscaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt)
        try:
            scaled_res =problem.penalty_solve(N,m,[mu],Lbfgs_options=opt,scale=True)

            iter1 = scaled_res['iteration']
            iter2 = unscaled_res['iteration']
        except:
            iter1 = 'fail'
            iter2 = unscaled_res['iteration']

        table2['scaled iter'].append(iter1)
        table2['unscaled iter'].append(iter2)
        table2['scaled gamma'].append(scaled_res['scaler'].gamma)

    


    data2 = pd.DataFrame(table2,index=gamma)
    print data2
    #plt.show()
    #data2.to_latex('report/draft/optimization/hessian_change_gamma.tex')
    plt.plot(t,scaled_res['control'].array()[:N+1],'r--')
    plt.plot(t,unscaled_res['control'].array()[:N+1])
    plt.show()
    
    plt.plot(scaled_res['control'].array()[N+1:],'r--')
    plt.plot(unscaled_res['control'].array()[N+1:])
    plt.show()

if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    #scaled_and_memory_lim()
    #change_gamma()
    scaled_initial_hessian()
