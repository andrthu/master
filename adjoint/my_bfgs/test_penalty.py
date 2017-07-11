from my_vector import *
from lbfgs import Lbfgs,MuLbfgs
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import trapz
from scipy.optimize import minimize
import test_serial as ser

#n number of points, m number of intervalls
def partition_func(n,m):

    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(np.zeros(N+1))
    else:
        partition.append(np.zeros(N))

    for i in range(1,m):
        if rest-i>0:
            partition.append(np.zeros(N+2))
        else:
            partition.append(np.zeros(N+1))

    return partition


#solve state equation
def solver(y0,a,n,m,u,lam,T):
    
    dt = float(T)/n
    y = partition_func(n+1,m)

    y[0][0]=y0
    for i in range(1,m):
        y[i][0]=lam[i-1]

    start=1
    for i in range(m):        
        for j in range(len(y[i])-1):
            y[i][j+1]=(y[i][j]+dt*u[start+j])/(1.-dt*a)
        start=start+len(y[i])-1

    Y=np.zeros(n+1)
    start=0
    for i in range(m):
        Y[start:start+len(y[i])-1]=y[i][:-1]
        start = start+len(y[i])-1
    Y[-1]=y[-1][-1]
    return y,Y

#solving the adjoint equation for each interval
def adjoint_solver(y0,a,n,m,u,lam,T,yT,my,get_y=False):

    dt = float(T)/n

    l = partition_func(n+1,m)
    y,Y = solver(y0,a,n,m,u,lam,T)

    #"initial" values
    l[-1][-1] = y[-1][-1] - yT
    for i in range(m-1):
        l[i][-1] = my*(y[i][-1]-lam[i])

    for i in range(m):
        for j in range(len(l[i])-1):
            l[i][-(j+2)] = (1+dt*a)*l[i][-(j+1)]

    L=np.zeros(n+1)

    start=0
    for i in range(m):
        L[start:start+len(l[i])-1] = l[i][:-1]
        start = start + len(l[i])-1
    L[-1]=l[-1][-1]

    
    if get_y==True:
        return l,L,y,Y
    
    return l,L

def Functional2(y,u,lam,yT,T,my):
    t = np.linspace(0,T,len(u))

    #the normal functional
    F = trapz(u**2,t) + (y[-1][-1]-yT)**2
    #the peenalty terms
    penalty = 0
    for i in range(len(lam)):
        penalty = penalty + my*((y[i][-1]-lam[i])**2)
        
    return 0.5*(F+penalty)

def L2error(u,v,t):
    x = (u-v)**2
    return trapz(x,t)

def mini_solver(y0,a,T,yT,n,m,my_list,show_output=False,mem_limit=15):
    
    t=np.linspace(0,T,n+1)
    #initial guess for control and penalty control is set to be 0
    x0 = SimpleVector(np.zeros(n+m))
    
    dt = t[1]-t[0]

    def ser_J(u):
        return ser.Func(u,a,y0,yT,T)

    def ser_grad_J(u):
        l = ser.adjoint_solver(y0,a,len(u)-1,u,T,yT)
        return ser.L2_grad(u,l,dt)

    res1 = minimize(ser_J,np.zeros(n+1),method='L-BFGS-B', jac=ser_grad_J,
                   options={'gtol': 1e-6, 'disp': False})
    res2 = []
    res3 = []
    
    H = None
    
    #solve problem for increasing mu
    for k in range(len(my_list)):
        #define reduced functional dependent only on u
        def J(u):
            y,Y=solver(y0,a,n,m,u[:n+1],u[n+1:],T)
            return Functional2(y,u[:n+1],u[n+1:],yT,T,my_list[k])
        
        #define our gradient using by solving adjoint equation
        def grad_J(u):
            #adjoint_solver(y0,a,n,m,u,lam,T,yT,my)
            l,L = adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,my_list[k])
            g = np.zeros(len(u))
            
            g[:n+1]=dt*(u[:n+1]+L)

            for i in range(m-1):
                g[n+1+i]=l[i+1][0]-l[i][-1]
                
            return g
    
        def Mud_J(u):
            
            l,L,y,Y = adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,my_list[k],get_y=True)
            
            u1   = u[:n+1]
            l1   = u[n+1:]
            du1  = float(T)*(u[:n+1]+L)/n
            ADJ1 = np.zeros(m-1)
            STA1 = np.zeros(m-1)
            for i in range(m-1):
                ADJ1[i] = l[i+1][0]
                STA1[i] = y[i][-1]

            y1 = np.zeros(len(u))
            y2 = np.zeros(len(u))
            
            y1[:n+1] = du1
            y1[n+1:] = ADJ1
            y2[n+1:] = l1 - STA1
            
            Y1 = SimpleVector(y1)
            Y2 = SimpleVector(y2)
            
            S1 = SimpleVector(u)
            S2 = SimpleVector(np.zeros(len(u)))
            return MuVector([Y1,Y2]),MuVector([S1,S2])

            
            
            
            

        #minimize J using initial guess x, and the gradient/functional above
        """
        default = {"jtol"                   : 1e-4,
                   "rjtol"                  : 1e-6,
                   "gtol"                   : 1e-4,
                   "rgtol"                  : 1e-5,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,
                   "mem_lim"                : 5,
                   "Hinit"                  : "default",
                   "beta"                   : 1, 
                   "mu_val"                 : 1,
                   "old_hessian"            : None,
                   "penaly_number"          : 1,
                   "return_data"            : False, }
        """
        #mem_limit = 10
        options = {"mu_val": my_list[k], "old_hessian": H, 
                   "return_data": True,"mem_lim":mem_limit, "beta":1,
                   "save_number":-1,"jtol" : 1e-4,}
        
        options2={"mem_lim" : mem_limit,"return_data": True,"jtol" : 1e-4,}
        
        S1 = MuLbfgs(J,grad_J,x0,Mud_J,Hinit=None,options=options)
        S2 = Lbfgs(J,grad_J,x0,options=options2)
        try:
            data1 = S1.solve()
        except Warning:
            data1  = {'control'   : x0, 'iteration' : -1, 'lbfgs': H }
            
        except RuntimeError:
            data1  = {'control'   : x0, 'iteration' : -1, 'lbfgs': H }
        try:
            data2 = S2.solve()
        except:
            data2 = {'control'   : x0, 'iteration' : -1, 'lbfgs': H }

        res2.append(data1)
        res3.append(data2)
        
        
        H = data1['lbfgs']
        
        if show_output==True:
            
            x1 = data1['control']
            plot(t,x1.array()[:n+1])
            plot(t,res1.x)
            legend(["mu","normal"])
            print data1['iteration']
            show()
    
    return res1,res2,res3


def test_mu_values():


    y0 = 1
    a = 1
    T = 1
    yT = 1

    M = [2, 5, 10]
    N = [100, 300, 500, 900, 1000, 2000]

    #MY = [0.05, 0.1, 0.2, 0.5, 0.8, 1]
    A =0
    B =0

    iteration_number1=[[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
    iteration_number2 = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
    iteration_number3 =[[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]

    Error1 = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
    Error2 = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]

    for i in range(len(M)):
        for j in range(len(N)):
            n = N[j]
            t = np.linspace(0,1,n+1)
            MY = [n*0.05, n*0.1, n*0.2, n*0.5, n*0.8]
            
            res1,res2,res3 = mini_solver(y0,a,T,yT,N[j],M[i],MY)
            print M[i],N[j],"sucsess"
            A=A+1

                
                
            iteration_number1[i][j].append(res1.nit)
                
            
            
            print
            print "-------------------------"
            for k in range(len(MY)):
                iteration_number2[i][j].append(res2[k]['iteration'])
                iteration_number3[i][j].append(res3[k]['iteration'])
                
                Le1 = L2error(res1.x,res2[k]['control'][:n+1],t)
                Le2 = L2error(res1.x,res3[k]['control'][:n+1],t)
                print Le1
                print Le2

                Error1[i][j].append(Le1)
                Error2[i][j].append(Le2)
                
                    
            print "-------------------------"
            print
            
                
                
    for i in range(len(M)):
        print "----------------------------"
        for j in range(len(N)):
            print "error for M=%d and N=%d" % (M[i],N[j]) 
            print "| %.1e| %.1e| %.1e| %.1e| %.1e" %(Error1[i][j][0],Error1[i][j][1],Error1[i][j][2],Error1[i][j][3],Error1[i][j][4])
        print "----------------------------"
        for j in range(len(N)):
            print "error for M=%d and N=%d" % (M[i],N[j]) 
            print "| %.1e| %.1e| %.1e| %.1e| %.1e" %(Error2[i][j][0],Error2[i][j][1],Error2[i][j][2],Error2[i][j][3],Error2[i][j][4])


    for i in range(len(M)):
        print "----------------------------"
        for j in range(len(N)):
            print "error for M=%d and N=%d" % (M[i],N[j]) 
            print "| %.1e| %.1e| %.1e| %.1e|" %(Error1[i][j][1]/Error1[i][j][0],Error1[i][j][2]/Error1[i][j][1],Error1[i][j][3]/Error1[i][j][2],Error1[i][j][4]/Error1[i][j][3])
        print "----------------------------"
        for j in range(len(N)):
            print "error for M=%d and N=%d" % (M[i],N[j]) 
            print "| %.1e| %.1e| %.1e| %.1e|" %(Error2[i][j][1]/Error2[i][j][0],Error2[i][j][2]/Error2[i][j][1],Error2[i][j][3]/Error2[i][j][2],Error2[i][j][4]/Error2[i][j][3])
                                                     
    
    for i in range(len(M)):
        print 
        print iteration_number1[i]
        print 
        print iteration_number2[i]
        print 
        print iteration_number3[i]

        
        for k in range(len(N)):
            
            figure()
            plot(np.array(iteration_number2[i][k]))
            plot(np.array(iteration_number3[i][k]),'--')
            title("m="+str(M[i]) + " " + "n="+str(N[k]))
            show()

    
def test_MuLbfgs():

    import time as time
    
    y0 = 1
    T = 1
    yT = 1
    a = 1

    N = [2000,3000]
    m = 10

    for i in range(len(N)):
        t0 = time.time()
        MY = [0.1*N[i],0.5*N[i]]
        t = np.linspace(0,T,N[i]+1)
        
        res1,res2,res3 = mini_solver(y0,a,T,yT,N[i],m,MY)
        t1 = time.time()
        print
        print t1-t0
        figure()
        plot(t,res1.x)
        for j in range(len(MY)):
            plot(t,res2[j]['control'][:N[i]+1])
            #print res2[j]['iteration'],res3[j]['iteration']
            print
            print "result for %d. iteration for N=%d " %(j+1,N[i])
            print "number of iteration for MuLbfgs : %d" % res2[j]['iteration']
            print "number of iteration for lbfgs : %d" % res3[j]['iteration']
            print
        legend(['serial','1. iter','2. iter'],loc=2)
        title('N='+str(N[i]))
        xlabel('time')
        ylabel('control')
        show()
    
if __name__ == "__main__":

    
    y0 = 1
    a = 1
    T = 1
    yT = 1
    n = 2000
    m = 10

    my_list = [0.1,10,20,100,500]
    #my_list = [500,800,900,1000]
    
    #mini_solver(y0,a,T,yT,n,m,my_list,show_output=True)

    test_mu_values()
    #test_MuLbfgs()

"""
terminal>> python test_mu_values()
----------------------------
error for M=2 and N=100
| 7.1e-03| 5.4e-01| 5.3e-04| 5.4e-01| 5.4e-01
error for M=2 and N=300
| 9.2e-04| 2.5e-04| 5.4e-01| 1.0e-05| 5.4e-01
error for M=2 and N=500
| 3.4e-04| 8.8e-04| 2.3e-05| 5.4e-01| 2.7e-06
error for M=2 and N=900
| 1.1e-04| 7.6e-04| 7.4e-06| 1.1e-06| 4.2e-07
error for M=2 and N=1000
| 8.8e-05| 7.3e-04| 2.6e-03| 9.9e-07| 3.8e-07
error for M=2 and N=2000
| 2.2e-05| 5.8e-04| 2.7e-04| 4.0e-04| 8.1e-04
----------------------------
error for M=2 and N=100
| 7.1e-03| 2.0e-03| 5.4e-04| 1.1e-04| 4.2e-05
error for M=2 and N=300
| 9.2e-04| 2.4e-04| 6.3e-05| 1.2e-05| 4.7e-06
error for M=2 and N=500
| 3.4e-04| 8.8e-05| 2.3e-05| 4.5e-06| 1.7e-06
error for M=2 and N=900
| 1.1e-04| 2.8e-05| 7.0e-06| 1.4e-06| 5.2e-07
error for M=2 and N=1000
| 8.8e-05| 2.2e-05| 5.7e-06| 1.1e-06| 4.2e-07
error for M=2 and N=2000
| 2.2e-05| 5.6e-06| 1.6e-06| 2.8e-07| 1.0e-07
----------------------------
error for M=5 and N=100
| 7.2e-02| 5.4e-01| 5.4e-01| 5.4e-01| 5.4e-01
error for M=5 and N=300
| 1.4e-02| 4.1e-03| 5.4e-01| 5.4e-01| 5.4e-01
error for M=5 and N=500
| 5.7e-03| 5.4e-01| 4.7e-04| 9.4e-05| 5.4e-01
error for M=5 and N=900
| 1.9e-03| 5.4e-01| 5.4e-01| 5.4e-01| 5.4e-01
error for M=5 and N=1000
| 1.5e-03| 5.4e-01| 1.3e-03| 4.0e-05| 5.4e-01
error for M=5 and N=2000
| 4.0e-04| 5.4e-01| 1.2e-03| 2.5e-05| 5.4e-01
----------------------------
error for M=5 and N=100
| 7.2e-02| 2.7e-02| 8.5e-03| 5.4e-01| 6.9e-04
error for M=5 and N=300
| 1.4e-02| 4.1e-03| 1.1e-03| 1.9e-04| 7.1e-05
error for M=5 and N=500
| 5.7e-03| 1.6e-03| 4.0e-04| 5.7e-05| 2.9e-05
error for M=5 and N=900
| 1.9e-03| 4.9e-04| 1.3e-04| 2.2e-05| 9.5e-06
error for M=5 and N=1000
| 1.5e-03| 3.9e-04| 1.0e-04| 1.8e-05| 7.7e-06
error for M=5 and N=2000
| 4.0e-04| 9.9e-05| 2.6e-05| 4.6e-06| 1.9e-06
----------------------------
error for M=10 and N=100
| 1.8e-01| 5.4e-01| 5.4e-01| 5.4e-01| 5.4e-01
error for M=10 and N=300
| 5.1e-02| 1.8e-02| 5.3e-03| 5.4e-01| 5.4e-01
error for M=10 and N=500
| 2.4e-02| 5.4e-01| 5.4e-01| 5.4e-01| 5.4e-01
error for M=10 and N=900
| 8.9e-03| 2.8e-03| 1.0e-03| 1.7e-03| 5.0e-04
error for M=10 and N=1000
| 7.4e-03| 5.4e-01| 5.4e-01| 5.4e-01| 5.4e-01
error for M=10 and N=2000
| 2.1e-03| 2.4e-03| 9.4e-04| 5.1e-04| 3.0e-04
----------------------------
error for M=10 and N=100
| 1.8e-01| 5.4e-01| 5.4e-01| 7.4e-03| 5.4e-01
error for M=10 and N=300
| 5.1e-02| 1.8e-02| 5.4e-03| 1.0e-03| 4.3e-04
error for M=10 and N=500
| 2.4e-02| 7.4e-03| 2.1e-03| 3.7e-04| 1.6e-04
error for M=10 and N=900
| 8.9e-03| 2.5e-03| 6.8e-04| 1.2e-04| 4.9e-05
error for M=10 and N=1000
| 7.4e-03| 2.1e-03| 5.6e-04| 9.6e-05| 4.0e-05
error for M=10 and N=2000
| 2.1e-03| 5.5e-04| 1.4e-04| 2.4e-05| 1.0e-05
----------------------------
error for M=2 and N=100
| 7.6e+01| 9.9e-04| 1.0e+03| 1.0e+00|
error for M=2 and N=300
| 2.7e-01| 2.2e+03| 1.9e-05| 5.4e+04|
error for M=2 and N=500
| 2.6e+00| 2.6e-02| 2.4e+04| 5.1e-06|
error for M=2 and N=900
| 7.0e+00| 9.8e-03| 1.5e-01| 3.8e-01|
error for M=2 and N=1000
| 8.3e+00| 3.5e+00| 3.9e-04| 3.9e-01|
error for M=2 and N=2000
| 2.6e+01| 4.6e-01| 1.5e+00| 2.0e+00|
----------------------------
error for M=2 and N=100
| 2.8e-01| 2.7e-01| 2.0e-01| 3.9e-01|
error for M=2 and N=300
| 2.6e-01| 2.6e-01| 2.0e-01| 3.8e-01|
error for M=2 and N=500
| 2.6e-01| 2.6e-01| 2.0e-01| 3.8e-01|
error for M=2 and N=900
| 2.5e-01| 2.6e-01| 2.0e-01| 3.8e-01|
error for M=2 and N=1000
| 2.5e-01| 2.6e-01| 2.0e-01| 3.7e-01|
error for M=2 and N=2000
| 2.5e-01| 2.9e-01| 1.7e-01| 3.7e-01|
----------------------------
error for M=5 and N=100
| 7.6e+00| 1.0e+00| 1.0e+00| 1.0e+00|
error for M=5 and N=300
| 3.0e-01| 1.3e+02| 1.0e+00| 1.0e+00|
error for M=5 and N=500
| 9.5e+01| 8.7e-04| 2.0e-01| 5.7e+03|
error for M=5 and N=900
| 2.8e+02| 1.0e+00| 1.0e+00| 1.0e+00|
error for M=5 and N=1000
| 3.5e+02| 2.5e-03| 3.0e-02| 1.3e+04|
error for M=5 and N=2000
| 1.4e+03| 2.3e-03| 2.0e-02| 2.2e+04|
----------------------------
error for M=5 and N=100
| 3.7e-01| 3.2e-01| 6.4e+01| 1.3e-03|
error for M=5 and N=300
| 3.0e-01| 2.7e-01| 1.7e-01| 3.7e-01|
error for M=5 and N=500
| 2.8e-01| 2.5e-01| 1.5e-01| 5.0e-01|
error for M=5 and N=900
| 2.6e-01| 2.6e-01| 1.8e-01| 4.2e-01|
error for M=5 and N=1000
| 2.5e-01| 2.6e-01| 1.8e-01| 4.2e-01|
error for M=5 and N=2000
| 2.5e-01| 2.6e-01| 1.8e-01| 4.1e-01|
----------------------------
error for M=10 and N=100
| 3.1e+00| 1.0e+00| 1.0e+00| 1.0e+00|
error for M=10 and N=300
| 3.5e-01| 3.0e-01| 1.0e+02| 1.0e+00|
error for M=10 and N=500
| 2.3e+01| 1.0e+00| 1.0e+00| 1.0e+00|
error for M=10 and N=900
| 3.2e-01| 3.5e-01| 1.7e+00| 2.9e-01|
error for M=10 and N=1000
| 7.3e+01| 1.0e+00| 1.0e+00| 1.0e+00|
error for M=10 and N=2000
| 1.1e+00| 4.0e-01| 5.4e-01| 5.9e-01|
----------------------------
error for M=10 and N=100
| 3.1e+00| 1.0e+00| 1.4e-02| 7.3e+01|
error for M=10 and N=300
| 3.5e-01| 3.0e-01| 1.9e-01| 4.3e-01|
error for M=10 and N=500
| 3.1e-01| 2.8e-01| 1.8e-01| 4.2e-01|
error for M=10 and N=900
| 2.8e-01| 2.8e-01| 1.7e-01| 4.2e-01|
error for M=10 and N=1000
| 2.8e-01| 2.7e-01| 1.7e-01| 4.2e-01|
error for M=10 and N=2000
| 2.6e-01| 2.5e-01| 1.7e-01| 4.1e-01|

[[2], [2], [2], [2], [2], [2]]

[[5, -1, 8, -1, -1], [5, 5, -1, 5, -1], [4, 3, 6, -1, 8], [4, 3, 7, 5, 6], [4, 3, 3, 9, 5], [4, 3, 3, 3, 3]]

[[5, 5, 4, 3, 3], [5, 4, 4, 3, 3], [4, 4, 4, 3, 3], [4, 4, 4, 3, 3], [4, 4, 4, 3, 3], [4, 4, 3, 3, 3]]

[[2], [2], [2], [2], [2], [2]]

[[7, -1, -1, -1, -1], [8, 16, -1, -1, -1], [8, -1, 6, 21, -1], [8, -1, -1, -1, -1], [8, -1, 7, 9, -1], [9, -1, 6, 20, -1]]

[[7, 8, 8, -1, 10], [8, 8, 9, 10, 10], [8, 9, 9, 10, 14], [8, 9, 10, 15, 15], [8, 9, 10, 15, 15], [9, 10, 11, 16, 16]]

[[2], [2], [2], [2], [2], [2]]

[[20, -1, -1, -1, -1], [22, 11, 22, -1, -1], [22, -1, -1, -1, -1], [22, 11, 30, 31, 22], [22, -1, -1, -1, -1], [21, 11, 11, 11, 11]]

[[20, -1, -1, 34, -1], [22, 22, 22, 23, 22], [22, 22, 21, 22, 22], [22, 20, 22, 22, 22], [22, 21, 22, 22, 22], [21, 21, 21, 22, 22]]


"""
