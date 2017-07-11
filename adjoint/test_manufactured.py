from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
from test_functionals import make_coef_J

def make_sin_functional(Time,power=2):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u-np.sin(np.pi*t))**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        t = np.linspace(0,Time,len(u))

        return dt*((u-np.sin(np.pi*t))+p)
        


    return J, grad_J

def test_manufactured_solution():
    
    y0 = 1
    yT = 2*np.exp(1) - 1
    T  = 1
    a  = 1
    import matplotlib.pyplot as plt

    J,grad_J = make_coef_J(1,1)

    M = [2,4,8]
    h_val = []
    error = []
    error2 = [[],[],[]]

    fig,ax = plt.subplots(2, 2)
    teller = -1
    for N in [500,1000,1500,2000,5000,8000]:
        
        h_val.append(1./N) 
        
        problem = Problem1(y0,yT,T,a,J,grad_J)
        

        res = problem.scipy_solver(N,disp=False)
        
        err = max(abs(res.x-1))
        error.append(err)
        print "max(|u-1|)=%f for N=%d and iter=%d"%(err,N,res.nit)
        t = np.linspace(0,T,N+1)
        if N!= 1000 and N!=5000:
            teller += 1
            ax[teller/2,teller%2].plot(t,res.x)
        for i in range(len(M)):
            res2 = problem.scipy_penalty_solve(N,M[i],[10])

            err2 = max(abs(res2.x[:N+1]-1))
            print "m=%d: err=%f for N=%d and iter=%d"%(M[i],err2,N,res2.nit)
            error2[i].append(err2)
            if N!= 1000 and N!=5000:
                ax[teller/2,teller%2].plot(t,res2.x[:N+1])
        if N!= 1000 and N!=5000:
            ax[teller/2,teller%2].legend(['m=1','m=2','m=4','m=8'],loc=4)
            ax[teller/2,teller%2].set_title('N='+str(N))
            
            
    plt.show()

            
    Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
    LS=linalg.lstsq(Q, np.log(np.array(error)))[0]
    print LS[0],np.exp(LS[1])

    

    for i in range(len(M)):
        Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
        LS=linalg.lstsq(Q, np.log(np.array(error2[i])))[0]
        print LS[0],np.exp(LS[1])
        plt.plot(np.log(np.array(h_val)),np.log(np.array(error2[i])))
        plt.show()

    plt.plot(np.log(np.array(h_val)),np.log(np.array(error)))
    plt.xlabel('log(1/N)')
    plt.ylabel('log(error)')
    plt.show()


def test_quadratic_manufactured_solution():
    
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    
    solution = 3.1

    J,grad_J = make_coef_J(1,solution)

    
    problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)
    N = 200000
    u = np.zeros(N+1) + solution
    yT = problem.ODE_solver(u,N)[-1]
    print "yT=%f"%yT
    
    h_val = []
    error = []
    error2 = [[],[],[]]
    M = [2,4,8]
    
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(2, 2)
    teller = -1
    for N in [50,100,150,200,500,1000]:
        
        h_val.append(1./N) 
        
        problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)
        res = problem.scipy_solver(N,disp=False)

        #u = np.zeros(N+1)
        #problem.finite_diff(u,N)
        t = np.linspace(0,T,N+1)
        
        if N!= 100 and N!=500:
            teller += 1
            ax[teller/2,teller%2].plot(t,res.x)
            
        
        err = max(abs(res.x-solution))
        error.append(err)
        print "max(|u-%.1f|)=%f for N=%d and iter=%d"%(solution,err,N,res.nit)
        
        for i in range(len(M)):
            res2 = problem.scipy_penalty_solve(N,M[i],[100])

            err2 = max(abs(res2.x[:N+1]-solution))
            print "m=%d: err=%f for N=%d and iter=%d"%(M[i],err2,N,res2.nit)
            error2[i].append(err2)

            
            if N!= 100 and N!=500:
                ax[teller/2,teller%2].plot(t,res2.x[:N+1])
        if N!= 100 and N!=500:
            ax[teller/2,teller%2].legend(['m=1','m=2','m=4','m=8'],loc=4)
            ax[teller/2,teller%2].set_title('N='+str(N))
        
    plt.show()
            

    Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
    LS=linalg.lstsq(Q, np.log(np.array(error)))[0]
    print LS[0],np.exp(LS[1])

    

    plt.plot(np.log(np.array(h_val)),np.log(np.array(error)))
    plt.xlabel('log(1/N)')
    plt.ylabel('log(error)')
    plt.show()

    for i in range(len(M)):
        Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
        LS=linalg.lstsq(Q, np.log(np.array(error2[i])))[0]
        print LS[0],np.exp(LS[1])
        plt.plot(np.log(np.array(h_val)),np.log(np.array(error2[i])))
        plt.show()


def test_manufactured_resultChange_solution():
    
    solution = 20
    T = 1
    y0 = 1
    yT = (solution+1)*np.exp(T) - solution    
    a  = 1
    power = 4

    J,grad_J = make_coef_J(1,solution,power=power)
    def Jfunc(u,y,yT,T,power):
        return J(u,y,yT,T)

    import matplotlib.pyplot as plt
    h_val = []
    error = []
    error2=[[],[],[]]
    M=[2,4,8]
    

    opt={'gtol': 1e-6, 'disp': False,'maxcor':10}

    fig,ax = plt.subplots(2, 2)
    teller = -1
    for N in [50,100,150,200,500,1000]:
        
        h_val.append(1./N) 
        t = np.linspace(0,T,N+1)
        problem = GeneralPowerY(y0,yT,T,a,power,Jfunc,grad_J)
        #u = np.zeros(N+1)
        #problem.finite_diff(u,N)

        res = problem.scipy_solver(N,disp=False,options=opt)
        
        err = max(abs(res.x-solution))
        error.append(err)
        print "max(|u-%.1f|)=%f for N=%d and iter=%d"%(solution,err,N,res.nit)
        
        
        #plt.plot(t,2*t+19)
        if N!= 100 and N!=500:
            teller += 1
            ax[teller/2,teller%2].plot(t,res.x)

        
        for i in range(len(M)):
            res2 = problem.scipy_penalty_solve(N,M[i],[1],options=opt)

            err2 = max(abs(res2.x[:N+1]-solution))
            print "m=%d: err=%f for N=%d and iter=%d"%(M[i],err2,N,res2.nit)
            error2[i].append(err2)

            if N!= 100 and N!=500:
                ax[teller/2,teller%2].plot(t,res2.x[:N+1])
        if N!= 100 and N!=500:
            ax[teller/2,teller%2].legend(['m=1','m=2','m=4','m=8'],loc=4)
            ax[teller/2,teller%2].set_title('N='+str(N))
        
    plt.show()
    Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
    LS=linalg.lstsq(Q, np.log(np.array(error)))[0]
    print LS[0],np.exp(LS[1])

    

    plt.plot(np.log(np.array(h_val)),np.log(np.array(error)))
    plt.xlabel('log(1/N)')
    plt.ylabel('log(error)')
    plt.show()

    for i in range(len(M)):
        Q = np.vstack([np.log(np.array(h_val)),np.ones(len(h_val))]).T
        LS=linalg.lstsq(Q, np.log(np.array(error2[i])))[0]
        print LS[0],np.exp(LS[1])
        plt.plot(np.log(np.array(h_val)),np.log(np.array(error2[i])))
        plt.show()


def test_sin_solution():

    y0 = 1    
    T  = 1
    a  = 1
    K = 1./(a*(1+(np.pi/a)**2))
    K2 = np.pi/a
    #yT = K*(-K2*np.cos(np.pi*T)-np.sin(np.pi*T)) + (y0+K2*K)*np.exp(a*T)
    yT = 1
    J,grad_J = make_sin_functional(T)
    
    N = 500
    problem = Problem1(y0,yT,T,a,J,grad_J)

    res = problem.scipy_solver(N,disp=False)
    res2 = problem.scipy_penalty_solve(N,10,[1000])
    import matplotlib.pyplot as plt
    t = np.linspace(0,T,N+1)
    
    plt.plot(t,res.x)#-np.sin(np.pi*t))
    plt.plot(t,res2.x[:N+1])#-np.sin(np.pi*t))
    #plt.plot(t,np.sin(np.pi*t))
    plt.show()

if __name__ == "__main__":

    #test_sin_solution()
    #test_manufactured_solution()
    #test_quadratic_manufactured_solution()
    test_manufactured_resultChange_solution()

    
