import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
from parallelOCP import v_comm_numbers
from test_LbfgsPPC import non_lin_problem,l2_norm,l2_diff_norm


def compare_pc_and_nonpc_for_different_m():
    import sys
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = -1.9
    p = 2


    
    problem = non_lin_problem(y0,yT,T,a,p,c=0)
    try:
        N = int(sys.argv[1])
        mu = N
        tol = 1e-4
        tol1 = 1e-4
        tol2 = 1e-4
    except:
        N = 1000
        mu = N
        tol1 = 1e-5
        tol2 = 1e-3
    M = [1,2,3,4,5,6,8,16,32]
    M = [1,64,128]
    #M = [1,2,3]
    res1 = problem.solve(N,algorithm='my_steepest_decent',Lbfgs_options={'jtol':tol1})
    t = np.linspace(0,T,N+1)
    res1_norm = l2_norm(res1.x,t)
    
    """
    table = {'pc itr'          : ['--'],
             'non-pc itr'      : ['--'],
             'pc err'          : ['--'],
             'non-pc err'      : ['--'],
             'non-penalty itr' : [res1['iteration']],}

    table2 = {'pc itr'            : ['--'],
              'non-pc itr'        : ['--'],
              'scaled pc itr'     : ['--'],
              'scaled non-pc itr' : ['--'],
              'non-penalty itr'   : [res1['iteration']],
              'scaled pc lsitr'   : ['--'],
              'pc fugr'          : [res1.lsiter],}
    """
    table = {'pc fu'          : ['--'],
             'non-pc fu'      : ['--'],
             'pc err'          : ['--'],
             'non-pc err'      : ['--'],
             'non-penalty itr' : [res1.niter],}

    table2 = {#'pc fu'          : [res1.counter()[0]],
              #'non-pc fu'      : [res1.counter()[0]],
              'pc fugr'        : [res1.counter()[1]+res1.counter()[0]],
              'npc fugr'       : [res1.counter()[1]+res1.counter()[0]],
              'pc err'         : ['--'],
              'non-pc err'     : ['--'],
              'ideal pc-S'     : [1],
              'ideal non-pc-S' : [1],}

   

    res2 = []
    res3 = []
    opt = {'mem_lim':10,'jtol':tol2,'beta_scale':True,'maxiter':1000}
    fu_gr_sum = res1.counter()[0]+res1.counter()[1]
    for m in M[1:]:

        #scaled_pc_res = problem.PPCLBFGSsolve(N,m,[m*mu],options=opt,scale=True)
        #scaled_nonpc_res = problem.penalty_solve(N,m,[m*mu],Lbfgs_options=opt,scale=True)
        #nonpc_res = problem.penalty_solve(N,m,[10,1000],algorithm='my_steepest_decent',Lbfgs_options=opt)
        pc_res = problem.scaled_PPCSDsolve(N,m,[10,1000],tol_list=[tol2,tol2/10,tol2/100,tol2/500],options=opt)
        nonpc_res=pc_res
        if type(pc_res)==list:
            pc_res=pc_res[-1]
            nonpc_res=nonpc_res[-1]

        pc_fugr = pc_res.counter()
        npc_fugr = nonpc_res.counter()
        print pc_res.val(),nonpc_res.val(),res1.val()
        res2.append(pc_res)
        res3.append(nonpc_res)
        
        S1 = float(fu_gr_sum)/((pc_fugr[0]+pc_fugr[1])/float(m))
        S2 = float(fu_gr_sum)/((npc_fugr[0]+npc_fugr[1])/float(m))
        err1 = l2_diff_norm(res1.x,pc_res.x[:N+1],t)/res1_norm
        err2 = l2_diff_norm(res1.x,nonpc_res.x[:N+1],t)/res1_norm

        table['pc fu'].append(pc_res.niter)
        table['non-pc fu'].append(nonpc_res.niter)
        table['pc err'].append(err1)
        table['non-pc err'].append(err2)
        table['non-penalty itr'].append('--')
        
        #table2['pc fu'].append(pc_fugr[0])
        #table2['non-pc fu'].append(npc_fugr[0])
        #table2['scaled pc itr'].append(scaled_pc_res.niter)
        #table2['scaled non-pc itr'].append(scaled_nonpc_res['iteration'])
        #table2['non-penalty itr'].append('--')
        #table2['scaled pc lsitr'].append(pc_res.lsiter)
        table2['pc fugr'].append(pc_fugr[1]+pc_fugr[0])
        table2['npc fugr'].append(npc_fugr[1]+npc_fugr[0])
        table2['pc err'].append(err1)
        table2['non-pc err'].append(err2)
        table2['ideal pc-S'].append(S1)
        table2['ideal non-pc-S'].append(S2)
    data = pd.DataFrame(table,index=M)
    Order1 = ['non-penalty itr','non-pc fu','non-pc err','pc fu','pc err']
    data11 = data.reindex_axis(Order1, axis=1)
    print data11
    #data11.to_latex('report/draft/parareal/pc_itr_err.tex')
    
    data2 = pd.DataFrame(table2,index=M)
    #Order = ['non-penalty itr','non-pc fu','scaled non-pc fu','pc fu','scaled pc itr','pc fugr','scaled pc lsitr']
    Order = ['pc fugr','npc fugr','pc err','non-pc err','ideal pc-S','ideal non-pc-S']
    data3 = data2.reindex_axis(Order, axis=1)
    print data3
    #data3.to_latex('report/draft/parareal/scaled_nonScaled_iterations_'+str(N)+'.tex')


    plt.figure()
    plt.plot(t,res1.x,'r--')
    for i in range(len(res2)):
        plt.plot(t,res2[i].x[:N+1])
    plt.legend(M,loc=4)
    plt.title('pc control')
    plt.show()
    
    plt.figure()
    plt.plot(t,res1.x,'r--')
    for i in range(len(res2)):
        plt.plot(t,res3[i].x[:N+1])
    plt.legend(M,loc=4)
    plt.title('non-pc control')
    plt.show()
if __name__=='__main__':
    compare_pc_and_nonpc_for_different_m()
