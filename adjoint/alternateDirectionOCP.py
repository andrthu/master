from optimalContolProblem import OptimalControlProblem, Problem1
from parallelOCP import interval_partition,v_comm_numbers,u_part
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
import matplotlib.pyplot as plt
class AlternateDirectionOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,decopled_J,decopled_grad,Lbfgs_options=None,options=None,implicit=True):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,Lbfgs_options,options,implicit)


        self.decopled_J = decopled_J
        self.decopled_grad = decopled_grad

    
    def ODE_interval_solver(self,v,y0,dt):
        y = np.zeros(len(v))
        y[0]=y0

        for i in range(len(v)-1):
            y[i+1] = self.ODE_update(y,v,i,i,dt)
        
        return y
        
    def adjoint_interval_solver(self,v,y0,lam,dt,mu,m,i):
        
        y = self.ODE_interval_solver(v,y0,dt)
        p = np.zeros(len(v))

        if i == m-1:
            p[-1] = y[-1]-self.yT
        else:
            p[-1] = mu*(y[-1]-lam)
        for j in range(len(v)-1):
            p[-(j+2)]=self.adjoint_update(p,y,j,dt)
        
        return p


    def partition_control(self,u,N,m):

        new_u = []
        ss,sl,_,_ = v_comm_numbers(N+1,m)
        for i in range(m):

            ui=interval_partition(N+1,m,i)
            ui[:] = u.copy()[ss[i]:ss[i]+sl[i]]
            new_u.append(ui)

        return new_u
        
    def alternate_Penalty_Functional(self,u,lam0,lam1,N,m,mu,i):
        
        dt = self.T/float(N)
        y = self.ODE_interval_solver(u,lam0,dt)
        J_val = self.decopled_J(u,y[-1],lam1,mu,dt)

       
        return J_val 

    def alternate_Penalty_Gradient(self,u,lam0,lam1,N,m,mu,i):
        dt = float(self.T)/N
        p = self.adjoint_interval_solver(u,lam0,lam1,dt,mu,m,i)
        
        return self.decopled_grad(u,p,dt)



    def alternate_direction_penalty_solve(self,N,m,my_list,tol_list=None,x0=None,Lbfgs_options=None,algorithm='my_lbfgs',ppc=None):
        self.t,self.T_z = self.decompose_time(N,m)
        dt=float(self.T)/N
        if x0==None:
            x0 = self.initial_control(N,m=m)#np.zeros(N+m)
        x = None
        #if algorithm=='my_lbfgs':
            #x0 = self.Vec(x0)
        Result = []

        initial_counter = self.counter.copy()
        
        for i in range(len(my_list)):
            def J(u):   
                self.counter[0]+=1
                return self.Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):
                self.counter[1]+=1
                return self.Penalty_Gradient(u,N,m,my_list[i])
                
            
            J_lam = lambda u2: J(np.hstack((x0[:N+1],u2)))
            if ppc==None:
                grad_lam = lambda u2: grad_J(np.hstack((x0[:N+1],u2)))[N+1:]
            else:
                grad_lam = lambda u2: ppc(grad_J(np.hstack((x0[:N+1],u2)))[N+1:])

            self.update_SD_options(Lbfgs_options)
            SDopt = self.SD_options
            
            Solver = SteepestDecent(J_lam,grad_lam,x0.copy()[N+1:],options=SDopt)
            lam_res = Solver.solve()
            x0[N+1:]=lam_res.x[:]
            x2=self.partition_control(x0[:N+1],N,m)
            v_res=[]
            v_x = []
            ##########i=0################
            
            Ji = lambda v: self.alternate_Penalty_Functional(v,self.y0,x0[N+1],N,m,my_list[i],0)
            grad_Ji = lambda v: self.alternate_Penalty_Gradient(v,self.y0,x0[N+1],N,m,my_list[i],0)
                
            Solver = SteepestDecent(Ji,grad_Ji,x2[0].copy(),options=SDopt)
            #print Ji(x2[0])
            print 0
            v_res.append(Solver.solve())
            v_x.append(Solver.solve().x)
            ############i=2,..,m-1######################
            for j in range(1,m-1):
                Ji = lambda v: self.alternate_Penalty_Functional(v,x0[N+1],x0[N+1+j],N,m,my_list[i],j)
                grad_Ji = lambda v: self.alternate_Penalty_Gradient(v,x0[N+1],x0[N+1+j],N,m,my_list[i],j)
                print j
                Solver = SteepestDecent(Ji,grad_Ji,x2[j].copy(),options=SDopt)

                v_res.append(Solver.solve())
                v_x.append(Solver.solve().x)
            ################i=m################
            Ji = lambda v: self.alternate_Penalty_Functional(v,x0[-1],self.yT,N,m,1,m-1)
            grad_Ji = lambda v: self.alternate_Penalty_Gradient(v,x0[-1],self.yT,N,m,1,m-1)
            print m-1
            Solver = SteepestDecent(Ji,grad_Ji,x2[-1].copy(),options=SDopt)

            v_res.append(Solver.solve())
            v_x.append(Solver.solve().x)
            #################end################
            v_gather = self.explicit_gather(v_x,N,m)
            
            
            x0[:N+1]= v_gather[:]
            #x0[N+1:]=lam_res.x[:]
            plt.plot(x0[N+1:])
        plt.show()

        #v_res.add_FuncGradCounter(self.counter-initial_counter)
        lam_res.add_FuncGradCounter(self.counter-initial_counter)
        res = [lam_res,v_res,x0]
        return res



class SimpleADOCP(AlternateDirectionOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,decopled_J,decopled_grad,options=None):

        AlternateDirectionOCP.__init__(self,y0,yT,T,J,grad_J,decopled_J,decopled_grad,options)

        self.a = a

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i]+dt*u[j+1])/(1.-dt*a)


    

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return l[-(i+1)]/(1.-dt*a)

def create_simple_ADOCP_problem(y0,yT,T,a):


    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))
        dt = t[1]-t[0]
        I = dt*np.sum(u[1:]**2)#trapz((u)**2,t)

        return 0.5*I + (1./2)*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0
        #grad[-1] =  dt*p[-2]+dt*(u[-1])
        return grad


    def decomp_J(u,y,lam,mu,dt):

        val = dt*np.sum(u[1:]**2)
        
        pen = mu*(y-lam)**2
        return 0.5*(val+pen)

    def decomp_grad(u,p,dt):

        grad = np.zeros(len(p))

        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0
        return grad

    
    problem = SimpleADOCP(y0,yT,T,a,J,grad_J,decomp_J,decomp_grad)

    return problem
    


    
def test_stuff():

    y0=1
    a = 1
    yT=1
    T=1

    problem = create_simple_ADOCP_problem(y0,yT,T,a)
    
    N = 1000
    m = 10
    u = np.zeros(N+m)+1

    dt = T/float(N)
    mu = 10
    u2=problem.partition_control(u[:N+1],N,m)

    y,Y = problem.ODE_penalty_solver(u,N,m)
    
    s=0
    val = problem.alternate_Penalty_Functional(u2[0],y0,u[N+1],N,m,mu,0)#decopled_J(u2[i],y[i][-1],u[N+1+i],mu,dt)
    s+=val
    print val
    for i in range(1,m-1):
        
        val = problem.alternate_Penalty_Functional(u2[i],u[N+i],u[N+1+i],N,m,mu,i)#decopled_J(u2[i],y[i][-1],u[N+1+i],mu,dt)
        s+=val
        print val
    val = problem.alternate_Penalty_Functional(u2[-1],u[-2],u[-1],N,m,1,m-1)#decopled_J(u2[-1],y[-1][-1],yT,1,dt)
    
    print val

    s+=val
    s2 = problem.Penalty_Functional(u,N,m,mu)
    print s,s2,s-s2
    
    p,P = problem.adjoint_penalty_solver(u,N,m,mu)
    
    grads=[]
    lam_val = np.zeros(m+1)
    lam_val[1:-1] = u[N+1:]
    lam_val[0]=y0
    lam_val[-1]=yT
    for i in range(m):
        if i ==m-1:
            muv=mu
        else:
            muv=mu
        grads.append(problem.alternate_Penalty_Gradient(u2[i],lam_val[i],lam_val[i+1],N,m,muv,i))#decopled_grad(u2[i],p[i],dt))
        
    #print grads
    grad_gather = problem.explicit_gather(grads,N,m)
    grad = problem.Penalty_Gradient(u,N,m,mu)
    
    print max(grad_gather-grad[:N+1]),'lel'

def test_solve():

    y0=1
    a = 1
    yT=1
    T=1

    problem = create_simple_ADOCP_problem(y0,yT,T,a)
    
    N = 100
    m = 3

    res = problem.alternate_direction_penalty_solve(N,m,[10000,100000])

    print res[-1]
    plt.plot(res[-1][:N+1])
    plt.show()
if __name__=='__main__':
    test_stuff()
    #test_solve()
