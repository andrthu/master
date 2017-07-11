import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def u_part(n,m,i):
    """
    Given a vector u of length n, partitioned into m parts with one 
    integer overlap, this function returns the first integer place of u that
    belongs to i-th partition.

    Example:
    n=11, m=3
    u = [0,1,2,3,4,5,6,7,8,9,10]

    We partition u:
    u0 = [0,1,2,3]
    u1 = [3,4,5,6,7]
    u2 = [7,8,9,10]
    
    Now we want to place in u does ui start with?

    i=0: 0
    i=1: 3
    i=2: 7
    """

    
    N = n/m
    rest = n%m
    if i==0:
        return 0

    if rest>0:
        start = N
    else:
        start = N-1
    for j in range(i-1):
        if rest-(j+1)>0:
            start += N+1
        else:
            start += N

    return start



def implicit_solver(y0,a,dt,N,f=None,partition=None):
    """
    Solves the equation y'-ay = f y(0)=y0 using implicit euler
    and N steps of timestep dt 
    """
    
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(len(a))
    if partition==None:
        start = 0
        partition = [N]
    else:
        start = u_part(partition[0],partition[1],partition[2])
    if type(a)!=np.ndarray:
        aa = np.zeros(partition[0]+1)+a
        a = aa
        
    for i in range(N):
        #print start+i,len(a),len(f)
        y[i+1] = (y[i] +dt*f[start+i+1])/(dt*a[start+i+1]+1)

    return y

def second_order_solver(y0,a,dt,N,f=None,partition=None):
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(len(a))
    if partition==None:
        start = 0
        partition = [N]
    else:
        start = u_part(partition[0],partition[1],partition[2])
    if type(a)!=np.ndarray:
        aa = np.zeros(partition[0]+1)+a
        a = aa
    for i in range(N):
        #print start+i,len(a),len(f)
        y[i+1] = (y[i]*(1-0.5*dt*a[start+i]) + dt*0.5*(f[start+i+1]+f[start+i]))/(0.5*dt*a[start+i+1]+1)
    return y

def int_par_len(n,m,i):
    """
    With fine resolution n and m time decomposition intervals,
    functions find number of points in interval i
    """
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = N+1
        else:
            state = N
    else:
        if rest - i >0:
            state = N+2
        else:
            state = N+1
    return state

def propagator_iteration(y,c_y,a,y0,N,M,dt,dT,f=[None,None]):
    """
    updates the interval initialvalues and the fine solution using
    the new initial values. 
    """
    if len(a)==2:
        a1=a[0]
        a2=a[1]
    else:
        a1=a
        a2=a

    if f[0]==None:
        f[0] = np.zeros(M+1)
    S = np.zeros(M+2)
    for i in range(M):
        S[i+2] = (y[i][-1] - c_y[i+1])/dT
    #S[1:]=f[0]+S[1:]
    delta = implicit_solver(0,a1,dT,M,f=S)
    for i in range(M):
        c_y[i+1] = y[i][-1] + delta[i+1]

        
    y=[]
    for i in range(M):
        y.append(second_order_solver(c_y[i],a2,dt,int_par_len(N+1,M,i)-1,f=f[1],partition=(N+1,M,i)))
    
    return y,c_y

def gather_y(y,N):
    """
    Gathers tje y arrays into one array Y
    """
    Y = np.zeros(N+1)
    
    start = len(y[0])
    Y[:start] = y[0][:]
    
    for i in range(len(y)-1):
        
        end = start + len(y[i+1])-1
        
        Y[start:end] = y[i+1][1:]
        start = end
    return Y

def parareal_solver(y0,a,T,M,N,order=3,f=[None,None]):
    """
    implementation of the parareal scheme for our simple ODE
    using N+1 as fine resolution and M time decompositions, and
    doing k=order iterations
    """
    
    dt = float(T)/N
    dT =  float(T)/M
    coarse_y = implicit_solver(y0,a[0],dT,M,f=f[0])
    
    y=[]
    for i in range(M):
        y.append(second_order_solver(coarse_y[i],a[1],dt,int_par_len(N+1,M,i)-1,f=f[1],partition=(N+1,M,i)))    
    
    for k in range(order-1):
        y,coarse_y=propagator_iteration(y,coarse_y,a,y0,N,M,dt,dT,f=f)
        
    Y = gather_y(y,N)
    return Y

def parareal_tol_solver(y0,a,T,M,N,ye,tol,plot_itr=False,f=[None,None]):
    """
    Does parareal iteration until error between numerical and exact
    solution ye is under tolerance tol
    """
    import matplotlib.pyplot as plt
    dt = float(T)/N
    dT =  float(T)/M
    
    coarse_y = implicit_solver(y0,a[0],dT,M,f=f[0])
    
    y=[]
    for i in range(M):
        y.append(implicit_solver(coarse_y[i],a[1],dt,int_par_len(N+1,M,i)-1,f=f[1],partition=(N+1,M,i)))
    
    Y = gather_y(y,N)
    k = 1
    if plot_itr:
        plt.plot(ye,'r--')
        plt.plot(Y)
    while np.max(abs(ye-Y))/np.max(abs(ye))>tol and k<20:
        y,coarse_y=propagator_iteration(y,coarse_y,a,y0,N,M,dt,dT,f=f)
        Y = gather_y(y,N)
        if plot_itr:
            plt.plot(Y)
        k+=1
    if plot_itr:
        plt.show()
    return Y,k

def test_order():

    a = 1.3
    T = 1
    y0 = 3.52

    N = 10000
    M = 10

    t = np.linspace(0,T,N+1)
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N)
    plt.plot(t,ye,'r--')
    plt.plot(t,yn,'b--')
    leg = ['exact','euler']
    error = [np.max(abs(ye-yn))]
    stop = int(np.ceil(np.log(1./N)/np.log(1./M))+1) +10
    
    for k in range(1,stop):
        y = parareal_solver(y0,a,T,M,N,order=k)
        plt.plot(t,y)
        leg.append('k='+str(k))
        error.append(np.max(abs(ye-y)))
    plt.legend(leg)
    plt.show()
    print error
    print
    #print np.log(1./N)/np.log(1./M)

    
def test_convergence():
    
    a = 1.3
    T = 4
    y0 = 3.52

    N = 10000
    tol = 1./1000
    t = np.linspace(0,T,N+1)
    A = lambda x: np.sin(2*np.pi*x)
    a = A(t)
    c= 10
    #C = np.zeros(N+1) +c
    C = c*t
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N,f=C)
    error = np.max(abs(ye-yn)/np.max(abs(ye)))

    M = [2,5,10,25,100]
    K = []
    K2 = []
    E = []
    for m in M:
        f = [c*np.linspace(0,T,m+1),C]
        aa = [A(np.linspace(0,T,m+1)),a]
        Y,k=parareal_tol_solver(y0,aa,T,m,N,yn,tol,plot_itr=True,f=f)
        K.append(k)
        K2.append(np.log(1./N)/np.log(1./m))
        E.append(np.max(abs(yn-Y))/np.max(abs(yn)))

    print K
    print
    print K2
    print 
    print E


def constant_iteration(k=3,M=[3,6,50,20,80,100,200,500,1000,2000],N=10000,plot_=True):

    a = 1.3
    T = 4
    y0 = 3.52

    #N = 10000
    tol = 1./1000
    t = np.linspace(0,T,N+1)
    A = lambda x: np.cos(2*np.pi*x)
    A= lambda x: np.cos(np.pi*x)
    a = A(t)
    c= 10
    #C = np.zeros(N+1) +c
    C = c*t
    ye = y0*np.exp(-np.sin(2*np.pi*t)/(2*np.pi))
    yn = second_order_solver(y0,a,float(T)/N,N,f=None)
    #print max(abs(ye-yn))
    import matplotlib.pyplot as plt
    #M = [3,6,12,24,80,100,200,500,1000,2000]
    #plt.figure(figsize=(20,10))
    #plt.plot(t,ye,'--')
    Y_list = []
    error_list = []
    coarse_dts = []
    
    for m in M:
        f = [None,None]
        #f=[c*np.linspace(0,T,m+1),C]
        aa = [A(np.linspace(0,T,m+1)),a]
        Y = parareal_solver(y0,aa,T,m,N,order=k,f=f)
        Y_list.append(Y)
        error = max(abs(Y-ye))
        #print error
        error_list.append(error)
        coarse_dts.append(float(T)/m)

        #plt.plot(t,Y,'.')
    #plt.show()
    table ={"err":None,"dT":None,"rate":None}
    table["err"] = error_list
    table["dT"] = coarse_dts
    error_list = np.array(error_list)
    coarse_dts = np.array(coarse_dts)
    
    rate=np.log(error_list[1:]/error_list[:-1])/np.log(coarse_dts[1:]/coarse_dts[:-1])
    #print rate
    rate2 = ['--']
    for i in range(len(rate)):
        rate2.append(rate[i])
    table["rate"]=rate2
    
    
    data = pd.DataFrame(table,index=M)
    #print data
    plot_ = False
    plot2_ = True
    if plot_:
        plt.figure(figsize=(10,16))
        ax1 = plt.subplot(311)
        ax1.plot(t,ye,'--')
        ax1.plot(t,Y_list[1],'ro',markersize=1)
        ax1.legend(['exact','Parareal,N='+str(M[1])],loc=2)
        #ax1.set_title('dT='+str(coarse_dts[1]))

        ax2 = plt.subplot(312)
        ax2.plot(t,ye,'--')
        ax2.plot(t,Y_list[2],'o',markersize=1)
        ax2.legend(['exact','Parareal,N='+str(M[2])],loc=2)
        #ax2.set_title('dT='+str(coarse_dts[2]))

        ax3 = plt.subplot(313)
        ax3.plot(t,ye,'--')
        ax3.plot(t,Y_list[3],'o',markersize=1)
        ax3.legend(['exact','Parareal,N='+str(M[3])],loc=2)
        ax3.yaxis.set_ticks(np.linspace(3,5,6))
        #plt.savefig('report/draft/draft2/parareal_img.png')
        #ax3.set_title('dT='+str(coarse_dts[5]))
        """
        ax4 = plt.subplot(414)
        ax4.plot(t,yn)
        ax4.plot(t,Y_list[6],'.')
        """
        plt.show()
    if plot2_:
        lam = implicit_solver(y0,A(np.linspace(0,T,21)),float(T)/20,20)

        plt.figure(figsize=(10,16))
        plt.plot(np.linspace(0,T,21),lam,'rs',markersize=10)
        plt.plot(t,Y_list[3],'o',markersize=1)
        plt.xlabel(r'$t$', fontsize=40)
        #plt.ylabel('y(t)', fontsize=40)
        plt.title('Decomposed equation on 20 subintervals',fontsize=25)
        plt.legend([r'$\lambda_i$',r'$y(t)$'],loc=2,fontsize=30)
        plt.show()
    return data
    

def create_con_table():

    
    N = 1000000
    M = [40,50,100,200,500,1000,2000]

    Data = []
    
    for k in range(1,5):

        Data.append(constant_iteration(k=k,N=N,M=M,plot_=False))
        print Data[k-1]
        Data[k-1].to_latex('report/draft/draft2/tables/parareal_convergence'+str(k)+'.tex')

    return 0


def check_stab():

    a = 1.3
    T = 4
    y0 = 3.52

    A = lambda x: -np.cos(2*np.pi*x)
    
    N = 6

    t = np.linspace(0,T,N+1)
    dt = T/float(N)
    y = implicit_solver(y0,A(t),dt,N)
    ye = y0*np.exp(-np.sin(2*np.pi*t)/(2*np.pi))

    plt.plot(t,y)
    plt.plot(t,ye)
    plt.show()

if __name__ == "__main__":
    a  = 1
    T  = 1
    y0 = 1

    N = 100000
    M = 2
    #test_order()
    #test_convergence()
    #parareal_solver(y0,a,T,M,N,order=1,show_plot=True)
    print constant_iteration(k=1)
    #create_con_table()
    #check_stab()
