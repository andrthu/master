from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *


def make_coef_J(c,d,power=2):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((c*u-d)**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        

        return dt*(c*(c*u-d)+p)
        g = dt*(c*(c*u-d)+p)
        i = -1
        j = 0
        g[i] = 0.5*dt*c*(c*u[i]-d)
        g[j] = dt*(0.5*c*(c*u[j]-d)+p[j])

        return g


    return J, grad_J



def test_coef():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 700   

    C = [0.1,2,100]
    D = [1,100,1000]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])
            
            problem = Problem1(y0,yT,T,a,J,grad_J)
            opt = {"mem_lim":10}
            try:
                res = problem.solve(N,Lbfgs_options=opt)
            except:
                res = {'iteration':-1}


            print
            print "c=%.2f, d=%d" %(C[i],D[j])
            print res['iteration']
            print

def make_alpha(alpha):
    def J(u,y,yT,T,alpha):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*(I + alpha*(y-yT)**2)

    def grad_J(u,p,dt,alpha):

        return dt*(u+p)


    return J, grad_J

def test_alpha():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 700

    A = [0.1,10,100]

    for i in range(len(A)):
        
        J,grad_J = make_alpha(A[i])
        problem = Problem3(y0,yT,T,a,A[i],J,grad_J)
            
        res = problem.plot_solve(N)

        print
        print "alpha = %.1f" %A[i]
        print res['iteration']
        print


def test_adjoint_coef():
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    m=15

    C = [0.1,2,10]
    D = [-1,1,100]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])

            problem = Problem1(y0,yT,T,a,J,grad_J)
            opt = {"mem_lim":40}
            try:
                res1,res2 = problem.penalty_and_normal_solve(N,m,[0.5*N],Lbfgs_options=opt,show_plot=True)
            except Warning:
                res1 = {'iteration':-1}

            #res2 = problem.solve(N)
            print
            print "c=%.1f, d=%d" %(C[i],D[j])
            print "m=%d gives iter=%d" %(m,res1['iteration'])
            print "m=1 gives iter=%d" %(res2['iteration'])
            print


def test_variable_a():
    y0 = 1
    yT = 1
    T  = 1
    a  = lambda t : 10*np.sin(2.*np.pi*t/T) + 1
    N = 700

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):

        return dt*(u+p)
    
    
    problem = Problem2(y0,yT,T,a,J,grad_J)

    res = problem.plot_solve(N)

def memory_test():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    my = [500]
    mul = [1,3]
    
    M = [2,4,8,16,32]

    coef=[[2,1000],[2,-1],[0.1,100],[10,100]]
    
    for k in range(len(coef)):
        J,grad_J = make_coef_J(coef[k][0],coef[k][1])
    
        problem = Problem1(y0,yT,T,a,J,grad_J)

        L = []
    
        for i in range(len(M)):
    
            L.append(problem.lbfgs_memory_solve(N,M[i],my,mul=mul))

        res1 = problem.solve(N)
        print "*******************************"
        print "J(u) = Integral( (%.1fu - %d)**2) " % (coef[k][0],coef[k][1])
        print "--------------m=1--------------" 
        print "|lbfgs memory=10| #iterations=%d| #iterations/m=%.2f"%(res1['iteration'],res1['iteration']) 
        for i in range(len(M)):
            print "--------------m=%d--------------" %(M[i])
            for j in range(len(mul)):
                print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(mul[j]*max(M[i],10),L[i][j]['iteration'],L[i][j]['iteration']/float(M[i]))

        import matplotlib.pyplot as plt
        
        plt.plot(np.linspace(1,2,2),np.zeros(2))
        plt.show()

def test_scipy_solve():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    
    J,grad_J = make_coef_J(0.1,100)
    
    problem = Problem1(y0,yT,T,a,J,grad_J)

    res = problem.scipy_solver(N)
    print res.nit

def non_linear_and_coef():


    y0 = 1
    yT = 10
    T  = 1
    a  = 1
    N1 = 8000
    


    C = [0.1,2,100]
    D = [-1,0.01,1000]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])
            opt = {"mem_lim" : 10}
            try:
                problem = Explicit_sine(y0,yT,T,a,J,grad_J)

                res = problem.plot_solve(N1,opt=opt,state=True)
                print res['iteration']
                print C[i],D[j]
            except:
                print 'lol'

def test_quadratic():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    
    M = [2,4,8,16,32]

    coef=[[2,0],[1,-1],[1,0.5]]

    power = 4

    for i in range(len(coef)):

        J, grad_J = make_coef_J(coef[i][0],coef[i][1],power=power)
        
        def Jfunc(u,y,yT,T,power):
            return J(u,y,yT,T)
            
        problem = GeneralPowerY(y0,yT,T,a,power,Jfunc,grad_J)
        

        problem.simple_test(N)


def test_cubic():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    
    M = [2,4,8,16,32]

    coef=[[2,0],[1,-1],[1,0.5]]

    for i in range(len(coef)):

        J, grad_J = make_coef_J(coef[i][0],coef[i][1],power=3)

        

        problem = CubicY(y0,yT,T,a,J,grad_J)

        problem.simple_test(N)

def test_quadratic_result():
    
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

    problem.simple_test(N,make_plot=True)

def test_finite_diff():
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 100

    
    J,grad_J=make_coef_J(1,10)
    
    problem = Problem1(y0,yT,T,a,J,grad_J)
    
    u = 10*np.sin(4*np.pi*np.linspace(0,T,N+1))

    problem.finite_diff(u,N)
    

if __name__ == "__main__":

    #test_coef()
    #test_alpha()
    #test_adjoint_coef()
    #test_variable_a()
    #memory_test()
    #test_scipy_solve()
    #non_linear_and_coef()
    #test_cubic()
    #test_quadratic()
    test_quadratic_result()
    #test_finite_diff()
    

"""
*******************************
J(u) = Integral( (2.0u - 1000)**2) 
--------------m=1--------------
|lbfgs memory=10| #iterations=5| #iterations/m=5.00
--------------m=2--------------
|lbfgs memory=10| #iterations=6| #iterations/m=3.00
|lbfgs memory=30| #iterations=6| #iterations/m=3.00
--------------m=4--------------
|lbfgs memory=10| #iterations=8| #iterations/m=2.00
|lbfgs memory=30| #iterations=8| #iterations/m=2.00
--------------m=8--------------
|lbfgs memory=10| #iterations=-1| #iterations/m=-0.12
|lbfgs memory=30| #iterations=12| #iterations/m=1.50
--------------m=16--------------
|lbfgs memory=16| #iterations=-1| #iterations/m=-0.06
|lbfgs memory=48| #iterations=21| #iterations/m=1.31
--------------m=32--------------
|lbfgs memory=32| #iterations=-1| #iterations/m=-0.03
|lbfgs memory=96| #iterations=39| #iterations/m=1.22

    
*******************************
J(u) = Integral( (2.0u - -1)**2) 
--------------m=1--------------
|lbfgs memory=10| #iterations=4| #iterations/m=4.00
--------------m=2--------------
|lbfgs memory=10| #iterations=5| #iterations/m=2.50
|lbfgs memory=30| #iterations=5| #iterations/m=2.50
--------------m=4--------------
|lbfgs memory=10| #iterations=17| #iterations/m=4.25
|lbfgs memory=30| #iterations=17| #iterations/m=4.25
--------------m=8--------------
|lbfgs memory=10| #iterations=22| #iterations/m=2.75
|lbfgs memory=30| #iterations=20| #iterations/m=2.50
--------------m=16--------------
|lbfgs memory=16| #iterations=51| #iterations/m=3.19
|lbfgs memory=48| #iterations=28| #iterations/m=1.75
--------------m=32--------------
|lbfgs memory=32| #iterations=100| #iterations/m=3.12
|lbfgs memory=96| #iterations=43| #iterations/m=1.34


*******************************
J(u) = Integral( (0.1u - 100)**2) 
--------------m=1--------------
|lbfgs memory=10| #iterations=11| #iterations/m=11.00
--------------m=2--------------
|lbfgs memory=10| #iterations=-1| #iterations/m=-0.50
|lbfgs memory=30| #iterations=12| #iterations/m=6.00
--------------m=4--------------
|lbfgs memory=10| #iterations=-1| #iterations/m=-0.25
|lbfgs memory=30| #iterations=-1| #iterations/m=-0.25
--------------m=8--------------
|lbfgs memory=10| #iterations=-1| #iterations/m=-0.12
|lbfgs memory=30| #iterations=28| #iterations/m=3.50
--------------m=16--------------
|lbfgs memory=16| #iterations=-1| #iterations/m=-0.06
|lbfgs memory=48| #iterations=35| #iterations/m=2.19
--------------m=32--------------
|lbfgs memory=32| #iterations=-1| #iterations/m=-0.03
|lbfgs memory=96| #iterations=51| #iterations/m=1.59


*******************************
J(u) = Integral( (10.0u -100)**2) 
--------------m=1--------------
|lbfgs memory=10| #iterations=4| #iterations/m=4.00
--------------m=2--------------
|lbfgs memory=10| #iterations=5| #iterations/m=2.50
|lbfgs memory=30| #iterations=5| #iterations/m=2.50
--------------m=4--------------
|lbfgs memory=10| #iterations=7| #iterations/m=1.75
|lbfgs memory=30| #iterations=7| #iterations/m=1.75
--------------m=8--------------
|lbfgs memory=10| #iterations=11| #iterations/m=1.38
|lbfgs memory=30| #iterations=11| #iterations/m=1.38
--------------m=16--------------
|lbfgs memory=16| #iterations=33| #iterations/m=2.06
|lbfgs memory=48| #iterations=19| #iterations/m=1.19
--------------m=32--------------
|lbfgs memory=32| #iterations=-1| #iterations/m=-0.03
|lbfgs memory=96| #iterations=35| #iterations/m=1.09

"""
