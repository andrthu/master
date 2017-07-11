from heatControl import HeatControl
import numpy as np
from dolfin import *

def test_problem(N,Tn,T,m):


    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh,"CG",1)

    test1 = HeatControl(V,mesh)
    test2 = HeatControl(V,mesh)
    
    r = Expression('sin(2*pi*x[0])')
    ic_expression = Expression('10*x[0]*(1-x[0])')
    ic = project(ic_expression,V)

    RHS1 = []
    RHS2 = []
    
    
    for i in range(Tn+1):
        RHS1.append(project(Constant(0.0),V))
        RHS2.append(project(Constant(0.0),V))


    ut = project(r,V)
    
    opt1 = {'c' : 0.01,'rhs':RHS1,'uT':ut,'T':T,'alpha':10}
    opt2 ={'c' : 0.01,'rhs':RHS2,'uT':ut,'T':T,'alpha':10}


    start = 0
    end = T

    res1 = test1.solver(opt1,ic,start,end,Tn,algorithm='my_steepest_decent')
    #res2 = test2.penalty_solver(opt2,ic,start,end,Tn,m,[1],algorithm='my_steepest_decent')
    res2=test2.PPCSD_solver(opt2,ic,start,end,Tn,m,[10])


    print res1.val(),res1.niter,res2.niter,res2.val()
    
    print res1.x
    test1.PDE_solver(ic,opt1,start,end,Tn,show_plot=True)

def test_equation_solvers():
    

    N = 20
    Tn = 40
    T = 1

    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh,"CG",1)

    test1 = HeatControl(V,mesh)
    
    r = Expression('sin(2*pi*x[0])')
    ic_expression = Expression('10*x[0]*(1-x[0])')
    ic = project(ic_expression,V)

    RHS = []

    for i in range(Tn+1):
        RHS.append(project(Constant(0.0),V))


    ut = project(r,V)
    opt = {'c' : 1,'rhs':RHS,'uT':ut,'T':T}

    #test1.PDE_solver(ic,opt,0,T,Tn,show_plot=True)
    test1.adjoint_solver(ic,opt,0,T,Tn,show_plot=True)
if __name__ == '__main__':
    set_log_level(ERROR)
    test_problem(N=10,Tn=20,T=0.5,m=3)
    #test_equation_solvers()
