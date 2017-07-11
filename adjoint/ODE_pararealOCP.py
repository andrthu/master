from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem
import time


class PararealOCP(OptimalControlProblem):


    def __init__(self,y0,yT,T,J,grad_J,options=None,implicit=False):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options=options,implicit=implicit)
        
        self.end_diff = None
        
    def initial_adjoint(self,y):
        """
        Initial condition for adjoint equation. Depends on the Functional
        """
        self.end_diff = y - self.yT
        return y - self.yT

    def adjoint_step(self,omega,dT,step=1,lam=np.zeros(2)):
        
        step_dt = dT/float(step)

        v = np.zeros(step+1)
        v[-1] = omega
        for i in range(step):
            v[-(i+2)] = self.adjoint_ppc_update(v,lam,i,step_dt)

        return v[0]

    def ODE_step(self,omega,dT,step=1,lam=np.zeros(2)):
        step_dt = dT/float(step)

        v = np.zeros(step+1)
        v[0] = omega
        for i in range(step):
            v[i+1] = self.ODE_ppc_update(v,lam,i,0,step_dt)
        return v[-1]
        
        
    def adjoint_ppc_update(self,l,y,i,dt):
        return self.adjoint_update(l,y,i,dt)
    def ODE_ppc_update(self,y,u,i,j,dt):
        return self.ODE_update(y,u,i,j,dt)
        
    def PC_maker2(self,N,m,step=1,mu=1.):
        #"""
        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x.copy()[:]
            
            dT = float(self.T)/m
            dt = float(self.T)/N
            
            
            for i in range(1,m):
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-i],dT,step=step)

            
            for i in range(1,m):
                S[i] = S[i] + self.ODE_step(S[i-1],dT,step=step)
            #print S
            #time.sleep(1)
            S = S
            x[:]=S.copy()[1:-1]
            return x
        #"""
        #pc = lambda x:x
        return pc

    def PC_maker3(self,N,m,step=1,mu=1.):      

        def pc(x):
            Nc = len(x)-m
            lam_pc = self.PC_maker2(N,m,step,mu)
            lam = x.copy()[Nc+1:]
            lam2 = lam_pc(lam)
            x[Nc+1:]= lam2.copy()[:]
            return x
        return pc

    def PC_maker4(self,N,m,step=1,mu=1.,con=None):
        #"""
        
        def pc(x):
            S = np.zeros(m+1)
            #S[0]  = self.end_end_adjoint +self.y0
            #S[-1] = -self.end_start_adjoint      
            
            S[1:-1] = x.copy()[:]
            #S[-2] += -self.end_start_adjoint
            dT = float(self.T)/m
            dt = float(self.T)/N
            
            #print 'step 0: ',S
            for i in range(1,m):
                #S[-(i+1)] = S[-(i+1)] + S[-i]/(1-self.a*dT)
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-i],dT,step=step)
                #S[-(i+1)] = S[-(i+1)] + S[-i]*(1+self.a*dT)
            #S = S/(1-self.a*dt)
            #print 'step 1: ',S
            for i in range(1,m):
                S[i] = S[i] + self.ODE_step(S[i-1],dT,step=step)
                #S[i] = S[i] + (S[i-1])/(1-self.a*dT)
                #S[i] = S[i] + (S[i-1])*(1+self.a*dT)
                #print dT*con[i]
            #S = S/(1-self.a*dt)
            #print S
            #time.sleep(1)
            #print 'step 2: ',S
            #S[-2]+=self.end_start_adjoint
            #S = S/(1-self.a*dt)
            #print x,S
            x[:]=S.copy()[1:-1]
            return x
        #"""
        #pc = lambda x:x
        return pc
    def PC_maker5(self,N,m,step=1,mu=1.):      
        
        coarse_I = np.linspace(0,N,m+1)
        coarse_v = np.zeros(m+1)
        def pc(x):
            
            for i in range(m+1):
                coarse_v[i] = x[int(coarse_I[i])]
            Nc = len(x)-m
            lam_pc = self.PC_maker4(N,m,step,mu,con=coarse_v)
            lam = x.copy()[Nc+1:]
            lam2 = lam_pc(lam)
            x[Nc+1:]= lam2.copy()[:]
            return x
        return pc


    def adjoint_propogator(self,m,delta0,S):

        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[-1] = delta0

        for i in range(m):

            delta[-(i+2)]=self.adjoint_propogator_update(delta,S,i,dT)

        return delta

    

    def ODE_propogator(self,m,delta0,S):
        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[0] = delta0

        for i in range(m):
            delta[i+1] = self.ODE_ppc_update(delta,S/dT,i,i-1,dT)

        return delta

    def PC_maker(self,N,m):

        
        def pc(x):
            Nc = len(x)-m
            S = np.zeros(m+1)
            S[1:-1] = x[Nc+1:].copy()
            #S[-1] = self.end_diff
            delta =self.adjoint_propogator(m,0,S)

            for i in range(1,len(S)-1):
                S[i] = S[i]+delta[i]

            #S2 = np.zeros(m)
            #S2[1:]=S[:-1].copy()

            delta2 = self.ODE_propogator(m,0,S)

            for i in range(len(S)-1):
                S[i+1] = S[i+1]+delta2[i+1]
            
            x[Nc+1:]=S[1:-1]
            
            return x
        
        #def pc(x):
            #return x
        return pc

    def PPCLBFGSsolve(self,N,m,my_list,tol_list=None,x0=None,options=None,scale=False):
        dt=float(self.T)/N
        if x0==None:
            x0 = self.initial_control(N,m=m)
        
        result = []
        PPC = self.PC_creator(N,m,step=1)
        if scale:
            scaler = {'m':m,'factor':1}
        else:
            scaler = None
        initial_counter = self.counter.copy()
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_Lbfgs_options(options)
            Lbfgsopt = self.Lbfgs_options
            if tol_list!=None:
                try:
                    opt = {'jtol':tol_list[i]}
                    self.update_Lbfgs_options(opt)
                    Lbfgsopt = self.Lbfgs_options
                except:
                    print 'no good tol_list'
            PPC = self.PC_creator(N,m,step=1,mu=my_list[i])
            Solver = SplitLbfgs(J,grad_J,x0,m=m,Hinit=None,
                                options=Lbfgsopt,ppc=PPC,scale=scaler)
            res = Solver.normal_solve()
            
            if scale:
                res.rescale()
            x0=res.x
            
            res.add_FuncGradCounter(self.counter-initial_counter)
            result.append(res)

        if len(result)==1:
            return res
        else:
            return result

    def PPCLBFGSadaptive_solve(self,N,m,mu0=1,x0=None,options=None,scale=False,
                               mu_stop_codition=None,mu_updater=None,
                               tol_update=None):
        dt=float(self.T)/N
        if x0==None:
            x0 = self.initial_control(N,m=m)
        
        result = []
        PPC = self.PC_creator(N,m,step=1)
        if scale:
            scaler = {'m':m,'factor':1}
        else:
            scaler = None
        mu = mu0
        
        if mu_stop_codition==None:
            mu_stop_codition = self.adaptive_stop_condition
        if mu_updater==None:
            mu_updater = self.adaptive_mu_update
        if tol_update == None:
            tol_update = lambda x,dt,mu: x

        while mu_stop_codition(mu0,dt,m): ###### OBS!!! ######
            

            #if  not self.adaptive_stop_condition(10*mu0,dt,m):
                #mu = 10*mu0
            J,grad_J = self.generate_reduced_penalty(dt,N,m,mu)

            self.update_Lbfgs_options(options)
            Lbfgsopt = self.Lbfgs_options
            options.update({'jtol':tol_update(Lbfgsopt['jtol'],dt,mu)})
            Solver = SplitLbfgs(J,grad_J,x0,m=m,Hinit=None,
                                options=Lbfgsopt,ppc=PPC,
                                scale=scaler)
            try:
                res = Solver.normal_solve()
            except Warning:
                try:
                    print 'ai ai ai'
                    mu = 2*mu0
                    J,grad_J = self.generate_reduced_penalty(dt,N,m,mu)
                    Solver = SplitLbfgs(J,grad_J,x0,m=m,Hinit=None,
                                        options=Lbfgsopt,ppc=PPC,scale=scaler)
                    res = Solver.normal_solve()
                except Warning:
                    return result
            if scale:
                res.rescale()
            x0=res.x
            res.add_mu(mu)
                
            result.append(res)
            mu0=mu
            mu = mu_updater(mu,dt,m,res.niter) ###### OBS!!! ######
        
        return result

    PC_creator = PC_maker5

    def PPCLBFGSsolve2(self,N,m,my_list,x0=None,options=None,scale=False):
        dt=float(self.T)/N
        if x0==None:
            x0 = SimpleVector(x0 = self.initial_control(N,m=m))
        
        result = []
        PPC = self.PC_creator(N,m,step=1)
        if scale:
            scaler = {'m':m,'factor':1}
        else:
            scaler = None
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_Lbfgs_options(options)
            Lbfgsopt = self.Lbfgs_options

            Solver = Lbfgs(J,grad_J,x0,Hinit=None,
                           options=Lbfgsopt,pc=PPC,scale=scaler)
            res = Solver.solve()
            x0=res['control']
            result.append(res)
        if len(result)==1:
            return res
        else:
            return result

    def PPCSDsolve(self,N,m,my_list,x0=None,tol_list=None,options=None,split=False):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        
        result = []
        PPC = self.PC_maker2(N,m,step=1)
        initial_counter = self.counter.copy()
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_SD_options(options)
            
            if tol_list!=None:
                try:
                    opt = {'jtol':tol_list[i]}
                    self.update_SD_options(opt)
                except:
                    print 'no good tol_list'

            SDopt = self.SD_options
            Solver = PPCSteepestDecent(J,grad_J,x0.copy(),PPC,
                                       decomp=m,options=SDopt)
            if split:
                res = Solver.split_solve(m)
            else:
                res = Solver.solve(m)
            x0=res.x
            result.append(res)
        res.add_FuncGradCounter(self.counter-initial_counter)
        if len(result)==1:
            #y,Y = self.ODE_penalty_solver(x0,N,m)
            #import matplotlib.pyplot as plt
            #plt.plot(Y)
            #plt.show()
            return res
        else:
            return result

    def scaled_PPCSDsolve(self,N,m,my_list,tol_list=None,x0=None,options=None):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        
        result = []
        PPC = self.PC_maker2(N,m,step=1)
        initial_counter = self.counter.copy()
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_SD_options(options)
            SDopt = self.SD_options

            Solver = SteepestDecent(J,grad_J,x0,scale={'m':m},
                                    options=SDopt)
            
            res = Solver.PPC_solve(PPC)
            print res.x[N+1:]
            res.rescale()
            print res.x[N+1:]
            
            
            x0=res.x
            result.append(res)
        res.add_FuncGradCounter(self.counter-initial_counter)
        if len(result)==1:
            #y,Y = self.ODE_penalty_solver(x0,N,m)
            #import matplotlib.pyplot as plt
            #plt.plot(Y)
            #plt.show()
            return res
        else:
            return result

    
    def adjoint_propogator_update(self,l,rhs,i,dt):
        raise NotImplementedError,'not implemented'

class SimplePpcProblem(PararealOCP):
    """
    optimal control with ODE y'=ay+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a

    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)

        
    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a)
    """
    def adjoint_propogator_update(self,l,rhs,i,dt):
        a = self.a
        return (l[-(i+1)]+rhs[-(i+1)])/(1-dt*a)

    """
class SimpleExplicitPpcProblem(PararealOCP):
    """
    optimal control with ODE y'=ay+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a

    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (1.+dt*a)*y[i] +dt*u[j]

        
    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]*(1.+dt*a)

    def adjoint_propogator_update(self,l,rhs,i,dt):
        a = self.a
        return (l[-(i+1)](1.+dt*a)+rhs[-(i+1)])



def find_gradient2():
    import matplotlib.pyplot as plt
    y0 = 1
    yT = 10
    T  = 1
    a  = 1
    N = 1000
    m = 100
    dt =float(T)/N

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        #grad =np.zeros(len(u))
        #grad[0] = 0.5*u[0]
        #grad[1:-1] = u[1:-1]+p[1:-1]
        #grad[-1] = 0.5*u[-1]+p[-1]
        #return dt*grad
        return dt*(u+p)

    Problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    red_J,red_grad_J = Problem.generate_reduced_penalty(dt,N,m,10)

    pc = Problem.PC_maker2(N,m)
    
    t = np.linspace(0,T,N+1)
    x = np.zeros(N+m)
    x[:N+1] = 5-4*t
    x[N+1:] = 9 * np.linspace(0,T,m+1)[1:-1] + 1

    grad = red_grad_J(x)

    plt.plot(grad[N+1:],'r--')
    plt.plot(pc(grad[N+1:]))
    plt.show()
    
def find_gradient():

    import matplotlib.pyplot as plt
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    m = 10
    dt =float(T)/N

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        #grad =np.zeros(len(u))
        #grad[0] = 0.5*u[0]
        #grad[1:-1] = u[1:-1]+p[1:-1]
        #grad[-1] = 0.5*u[-1]+p[-1]
        #return dt*grad
        return dt*(u+p)

    Problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    red_J,red_grad_J = Problem.generate_reduced_penalty(dt,N,m,10)

    x = np.zeros(N+m)

    grad = red_grad_J(x)

    print grad

    lam = np.zeros(m+1)
    

    lam[1:-1] = grad[N+1:]
    plt.plot(np.linspace(0,T,m+1),lam)
    plt.show()

    delta = Problem.adjoint_propogator(m,0,lam)

    for i in range(1,len(lam)-1):
        lam[i] = lam[i]+delta[i]

    plot(np.linspace(0,T,m+1),lam,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()


    delta = Problem.ODE_propogator(m,0,lam)
    
    for i in range(len(lam)-1):
        lam[i+1] = lam[i+1]+delta[i+1]

    
    plot(np.linspace(0,T,m+1),lam,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()



    
    

if __name__ == "__main__":
    from matplotlib.pyplot import *
    #find_gradient2()
    
    y0 = 1
    yT = 30
    T  = 1
    a  = 4

    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        """
        grad =np.zeros(len(u))
        grad[0] = 0.5*u[0]
        grad[1:-1] = u[1:-1]+p[1:-1]
        grad[-1] = 0.5*u[-1]+p[-1]
        return dt*grad
        #"""
        return dt*(u+p)


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    
    N = 1000
    m = 100
    solver_time =[]

    t2 = time.time()
    res = problem.PPCSDsolve(N,m,[1,10,50,100])[-1]
    t1 = time.time()
    solver_time.append((t1-t2)/m)
    
    t2 = time.time()
    res2 = problem.scipy_solver(N,disp=True)
    t1 = time.time()
    solver_time.append((t1-t2))

    t2 = time.time()
    res3 = problem.solve(N,algorithm='my_steepest_decent')
    t1 = time.time()
    solver_time.append((t1-t2))

    print 'normal: ',res3.niter
    
    res4 = problem.penalty_solve(N,m,[500],algorithm='my_lbfgs')
    t2 = time.time()
    res5 = problem.scipy_penalty_solve(N,m,[500],disp=True)
    t1 = time.time()
    solver_time.append((t1-t2)/m)

    print solver_time
    plot(np.linspace(0,T,N+1),res3.x,'--r')
    plot(np.linspace(0,T,N+1),res.x[:N+1],'--b')
    plot(np.linspace(0,T,N+1),res5.x[:N+1])
    plot(np.linspace(0,T,N+1),res4['control'].array()[:N+1])
    plot(np.linspace(0,T,N+1),res2.x)
    legend(['sd','sdp','scip','myp','sci'])
    show()
    
"""
-(l[-(i+1)]-l[(i+2)])/dT = a*l[-(i+2)] + S[-(i+1)]/dT

l[-(i+2)](1-a*dT) = l[-(i+1)] + S[-(i+1)]

l[-(i+2)] = (l[-(i+1)]+S[-(i+1)])/(1-dT*a) 
"""
