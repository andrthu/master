from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
import time

class OptimalControlProblem():
    """
    class for solving problem on the form

    minimize J(y(u),u) = A(u) + B(y(T),yT)
    E(y(t),u(t)) = 0
    """

    def __init__(self,y0,yT,T,J,grad_J,Lbfgs_options=None,options=None,implicit=True):
        """
        Initialize OptimalControlProblem

        Valid options are:
        * y0: initial condition of the ODE
        * yT: 'End goal' of the ODE in the functional
        * T: End time 
        * J: Functional depending on u,y,T and yT.
        * grad_J : gradient for above Functional, depends on u, p and dt 
        * Lbfgs_options: options for the L-bfgs algorithm for solving
          - jtol: Tolorance for end criteria in L-BFGS
          - maxiter: maximal number of iterations
          - mem_lim: number of iterations remembered by inverted hessian
          - Vector: Vector type
          - Hinit: initial approximation to inverted hessian
          - beta: scalar multiplied with Hinit
          - return_data: output is just control, or more information
        * options: More options
          - Vector: What type of vector shall be used. 
          - Lbfgs: which L-bfgs implementation shall be used
        """

        
        self.T      = T
        self.y0     = y0
        self.yT     = yT
        self.J      = J
        self.grad_J = grad_J
        
        self.T_z = NoneVec()
        self.t   = None

        self.set_options(options,Lbfgs_options)
        
        self.Vec = self.options['Vector']

        self.counter=np.zeros(2)

        self.jump_diff = 0
        self.c = 0

        self.end_end_adjoint = None
        self.end_start_adjoint = None
        if implicit:
            self.serial_gather = self.implicit_gather
        else:
            self.serial_gather = self.explicit_gather

    def set_options(self,user_options,user_Lbfgs):
        """
        Method for setting options
        """
        options = self.default_options()
        Lbfgs_options = self.default_Lbfgs_options()
        
        if user_options!=None:
            for key, val in user_options.iteritems():
                options[key]=val

        if user_Lbfgs!=None:
            for key, val in user_Lbfgs.iteritems():
                Lbfgs_options[key]=val

        
        self.options = options
        self.Lbfgs_options = Lbfgs_options
        self.SD_options = self.default_SD_options()

        

    def default_Lbfgs_options(self):
        """
        default options for LGFGS
        """
        default = {"jtol"         : 1e-4,
                   "maxiter"      : 500,
                   "mem_lim"      : 10,
                   "Vector"       : SimpleVector,
                   "Hinit"        : "default",
                   "beta"         : 1,
                   "return_data"  : True,
                   "scale_factor" : 20,}

        return default

    def update_Lbfgs_options(self,options):
        if options!=None:            
            for key, val in options.iteritems():
                self.Lbfgs_options[key]=val

    def default_options(self):
        """
        Default options
        """

        default = {"Vector" : SimpleVector,
                   "Lbfgs"  : Lbfgs,
                   "state_int"       : False}
        return default

    def default_SD_options(self):
        """
        defaalt options for steepest decent
        """
        default = {}
        default.update({"jtol"         : 1e-4,
                        "maxiter"      : 500,
                        "scale"        : False,
                        "scale_factor" : 20})
        return default

    def update_SD_options(self,options):
        """
        Method for updating steepest decent options
        
        Arguments:
        -options: Dictionary withe options
        """
        if options!=None:
            for key, val in options.iteritems():
                self.SD_options[key]=val
        
        
    def ODE_update(self,y,u,i,j,dt):
        """
        Finite diffrence scheme for state equation
        """
        raise NotImplementedError,'ODE_update not implemented'

    def adjoint_update(self,l,y,i,dt):
        """
        Finite diffrence scheme for adjoint equation
        """
        raise NotImplementedError,'adjoint_update not implemented'

    def initial_adjoint(self,y):
        """
        Initial condition for adjoint equation. Depends on the Functional
        """
        return y - self.yT

    def initial_penalty(self,y,u,my,N,i,m=1):
        """
        initial conditian for the adjoint equations, when partitioning in time
        """
        return my*(y[i][-1]-u[-m+i+1])

    def initial_lagrange(self,y,u,my,N,i,G):
        """
        Same as above, when we use augmented Lagrange
        """
        
        return self.initial_penalty(y,u,my,N,i,m) - G[i]

    def initial_control(self,N,m=1):
        """
        Returns zero vector of control dimension.
        
        Default is control dimension equal to state equation 
        """

        return np.zeros(N+m)


    def adaptive_mu_update(self,mu,dt,m,last_iter):
        #return float(m)*100*mu/(last_iter)
        return 100*mu
        return 10*mu*np.sqrt(m)

    def adaptive_stop_condition(self,mu,dt,m):

        return mu<self.adaptive_stop_formula(mu,dt,m)

    def adaptive_stop_formula(self,mu,dt,m):
        return m/(16.*dt)
    
    def ODE_solver(self,u,N,y0=None):
        """
        Solving the state equation without partitoning

        Arguments:
        * u: the control
        * N: Number of discritization points
        * y0: initial condition
        """
        T = self.T
        if y0==None:
            y0 = self.y0
        dt = float(T)/N        
        
        y = np.zeros(N+1)
        y[0]=y0

        for i in range(N):
            y[i+1] = self.ODE_update(y,u,i,i,dt)

    
        return y

    def ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        y = partition_func(N+1,m)

        y[0][0]=y0
        for i in range(1,m):
            y[i][0] = u[-m+i]
        
        Nc = len(u)-m
        start=0
        u_send = u[:Nc+1]
        for i in range(m):        
            for j in range(len(y[i])-1):
                y[i][j+1] = self.ODE_update(y[i],u_send,j,start+j,dt)
            start = start + len(y[i]) - 1
            

        Y = np.zeros(N+1)
        """
        start = 0
        for i in range(m):
            Y[start:start+len(y[i])-1] = y[i][:-1]
            start = start + len(y[i]) - 1
        Y[-1] = y[-1][-1]
        """
        Y[0] = y[0][0]
        start = 1
        for i in range(m):
            Y[start:start+len(y[i])-1] = y[i][1:]
            start = start + len(y[i]) - 1
        #"""

        diff_vals = np.zeros(m-1)
        for i in range(m-1):
            diff_vals = (y[i+1][0]-y[i][-1])

        self.jump_diff = np.max(abs(diff_vals))
        return y,Y

    def adjoint_solver(self,u,N,l0=None):
        """
        Solving the adjoint equation using finite difference

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
              

        y = self.ODE_solver(u,N)

        l=np.zeros(N+1)
        if l0==None:
            l[-1] = self.initial_adjoint(y[-1])
        else:
            l[-1]=l0
        
        for i in range(N):
            l[-(i+2)] = self.adjoint_update(l,y,i,dt)
             
        return l
    
    def adjoint_penalty_solver(self,u,N,m,my,init=initial_penalty):
        """
        Solving the adjoint equation using finite difference, and partitioning
        the time interval.

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """

        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        
        l = partition_func(N+1,m)
        y,Y = self.ODE_penalty_solver(u,N,m)

            
        l[-1][-1] = self.initial_adjoint(y[-1][-1])
        for i in range(m-1):
            l[i][-1]=init(self,y,u,my,N,i,m) #my*(y[i][-1]-u[N+1+i])
            
        for i in range(m):
            self.t = self.T_z[i]
            for j in range(len(l[i])-1):
                l[i][-(j+2)] = self.adjoint_update(l[i],y[i],j,dt)
        self.t=np.linspace(0,T,N+1)
        L=self.serial_gather(l,N,m)#np.zeros(N+1)
        """
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
        ""
        L[0] = l[0][0]
        start = 1
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][1:]
            start += len(l[i])-1
        
        #"""
        return l,L

    def implicit_gather(self,l,N,m):
        L=np.zeros(N+1)
        #"""
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
        return L

    def explicit_gather(self,l,N,m):
        L=np.zeros(N+1)
        L[0] = l[0][0]
        start = 1
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][1:]
            start += len(l[i])-1
        return L

    
    def Lagrange_Penalty_Functional(self,u,N,m,my,G):
        """
        Add penalty terms and lagrangian terms to the functional, for
        augmented lagrange method

        Arguments:

        * u: control
        * N: discretization points
        * m: number of parttition intervalls
        * my: penalty variable
        * G: lagrange multiplier
        
        """
        y,Y = self.ODE_penalty_solver(u,N,m)

        J_val = self.J(u[:N+1],y[-1][-1],self.yT,self.T)

        pen = 0

        for i in range(m-1):
            pen = pen + my*(y[i][-1]-u[N+1+i])**2-2*G[i]*(y[i][-1]-u[N+1+i])
    
        return J_val + 0.5*pen

    def func_state_part(self,u,N):
        return self.ODE_solver(u,N)[-1]
    def penalty_func_state_part(self,y,Y):
        return y[-1][-1]
    def Functional(self,u,N):
        """
        Reduced functional, that only depend on control u 
        """

        return self.J(u,self.func_state_part(u,N),self.yT,self.T)  
        
        

    def Penalty_Functional(self,u,N,m,my):
        """
        Reduced functional, that only depend on control u. Also adds
        penalty terms
        """
        y,Y = self.ODE_penalty_solver(u,N,m)
        Nc = len(u) -m
        J_val = self.J(u[:Nc+1],self.penalty_func_state_part(y,Y),self.yT,self.T)

        penalty = 0

        for i in range(m-1):
            penalty = penalty + my*((y[i][-1]-u[-m+1+i])**2)
        #print penalty
        return J_val + 0.5*penalty

    def Gradient(self,u,N):
        l = self.adjoint_solver(u,N)
        dt = float(self.T)/N
        return self.grad_J(u,l,dt)

    def Penalty_Gradient(self,u,N,m,mu):

        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
            
        g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

        for j in range(m-1):
            g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
        return g

    def Penalty_Gradient2(self,u,N,m,mu):

        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
        
        self.end_end_adjoint = l[0][0]
        self.end_start_adjoint = l[-1][0]

        g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

        for j in range(m-1):
            g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
        return g

    def generate_reduced_penalty(self,dt,N,m,my):

        def J(u):      
            self.counter[0]+=1
            return self.Penalty_Functional(u,N,m,my)

        def grad_J(u):
            """
            l,L = self.adjoint_penalty_solver(u,N,m,my)
            
            Nc = len(u) - m
            g = np.zeros(len(u))
            
            g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

            for j in range(m-1):
                g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
            return g
            """
            self.counter[1]+=1
            return self.Penalty_Gradient(u,N,m,my)
        return J,grad_J


    def generate_reduced_penalty2(self,dt,N,m,my):

        def J(u):      
            self.counter[0]+=1
            return self.Penalty_Functional(u,N,m,my)

        def grad_J(u):
            """
            l,L = self.adjoint_penalty_solver(u,N,m,my)
            
            Nc = len(u) - m
            g = np.zeros(len(u))
            
            g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

            for j in range(m-1):
                g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
            return g
            """
            self.counter[1]+=1
            return self.Penalty_Gradient2(u,N,m,my)
        return J,grad_J
    
    def solve(self,N,x0=None,Lbfgs_options=None,algorithm='my_lbfgs'):
        """
        Solve the optimazation problem without penalty

        Arguments:
        * N: number of discritization points
        * x0: initial guess for control
        * Lbfgs_options: same as for class initialisation
        """
        self.t = np.linspace(0,self.T,N+1)
        dt=float(self.T)/N
        if x0==None:
            x0 = self.initial_control(N)# np.zeros(N+1)
        if algorithm=='my_lbfgs':
            x0 = self.Vec(x0)
        

        initial_counter = self.counter.copy()
        def J(u):
            self.counter[0]+=1
            return self.Functional(u,N)

        def grad_J(u):
            self.counter[1]+=1
            #l = self.adjoint_solver(u,N)
            return self.Gradient(u,N)#grad_J(u,l,dt)
       
        if algorithm=='my_lbfgs':
            self.update_Lbfgs_options(Lbfgs_options)
            
            #solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options)
            solver=SplitLbfgs(J,grad_J,x0.array(),
                              options=self.Lbfgs_options)
            #res = solver.solve()
            res = solver.normal_solve()
        elif algorithm=='my_steepest_decent':
            self.update_SD_options(Lbfgs_options)
            SDopt = self.SD_options

            Solver = SteepestDecent(J,grad_J,x0.copy(),
                                    options=SDopt)
            res = Solver.solve()

            import matplotlib.pyplot as plt
            #Y = self.ODE_solver(res.x,N)
            #plt.plot(Y)
            #plt.show()
        res.add_FuncGradCounter(self.counter-initial_counter)
        return res

    def decompose_time(self,N,m):
        return np.linspace(0,self.T,N+1),NoneVec()
        
    def penalty_solve(self,N,m,my_list,tol_list=None,x0=None,Lbfgs_options=None,algorithm='my_lbfgs',scale=False):
        """
        Solve the optimazation problem with penalty

        Arguments:
        * N: number of discritization points
        * m: number ot processes
        * my_list: list of penalty variables, that we want to solve the problem
                   for.
        * x0: initial guess for control
        * options: same as for class initialisation
        """
        self.t,self.T_z = self.decompose_time(N,m)
        dt=float(self.T)/N
        if x0==None:
            x0 = self.initial_control(N,m=m)#np.zeros(N+m)
        x = None
        if algorithm=='my_lbfgs':
            x0 = self.Vec(x0)
        Result = []

        initial_counter = self.counter.copy()

        for i in range(len(my_list)):
            #"""
            def J(u):   
                self.counter[0]+=1
                return self.Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):
                self.counter[1]+=1
                return self.Penalty_Gradient(u,N,m,my_list[i])
                #"""
                
            #J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])
            if algorithm=='my_lbfgs':
                self.update_Lbfgs_options(Lbfgs_options)
                if tol_list!=None:
                    try:
                        opt = {'jtol':tol_list[i]}
                        self.update_Lbfgs_options(opt)
                    except:
                        print 'no good tol_list'
                if scale:
                    scaler={'m':m,'factor':self.Lbfgs_options['scale_factor']}
                    
                    #solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options,scale=scaler)
                    solver = SplitLbfgs(J,grad_J,x0.array(),options=self.Lbfgs_options,scale=scaler)
                else:
                    #solver = Lbfgs(J,grad_J,x0,options=self.Lbfgs_options)
                    solver = SplitLbfgs(J,grad_J,x0.array(),m=m,options=self.Lbfgs_options)
                #res = solver.solve()
                res = solver.normal_solve()
                
                x0 = res['control']
                #print J(x0.array())
            elif algorithm=='my_steepest_decent':

                self.update_SD_options(Lbfgs_options)
                SDopt = self.SD_options
                if scale:
                    
                    scale = {'m':m,'factor':SDopt['scale_factor']}
                    Solver = SteepestDecent(J,grad_J,x0.copy(),
                                             options=SDopt,scale=scale)
                    res = Solver.solve()
                    res.rescale()
                else:
                    Solver = PPCSteepestDecent(J,grad_J,x0.copy(),
                                               lambda x: x,options=SDopt)
                    res = Solver.split_solve(m)
                x0 = res.x.copy()
                
            elif algorithm=='slow_steepest_decent':
                self.update_SD_options(Lbfgs_options)
                SDopt = self.SD_options
                Solver = SteepestDecent(J,grad_J,x0.copy(),
                                        options=SDopt)
                res = Solver.solve()
                x0 = res.x.copy()
                
                
            elif algorithm == 'split_lbfgs':
                self.update_Lbfgs_options(Lbfgs_options)
                Solver = SplitLbfgs(J,grad_J,x0,m,options=self.Lbfgs_options)

                res = Solver.solve()
                x0 = res.x.copy()
            res.jump_diff=self.jump_diff
            Result.append(res)
            print 'jump diff:',self.jump_diff
        res.add_FuncGradCounter(self.counter-initial_counter)
        if len(Result)==1:
            return res
        else:
            return Result
    

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
        import matplotlib.pyplot as plt
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
            
            Solver =SteepestDecent(J_lam,grad_lam,x0.copy()[N+1:],options=SDopt)
            lam_res = Solver.solve()
            
            #x0[N+1:]=lam_res.x[:]

            J_v = lambda u3 : J(np.hstack((u3,x0[N+1:])))
            grad_v= lambda u3 : grad_J(np.hstack((u3,x0[N+1:])))[:N+1]
            
            Solver = SteepestDecent(J_v,grad_v,x0.copy()[:N+1],options=SDopt)

            v_res = Solver.solve()
            x0[N+1:]=lam_res.x[:]
            x0[:N+1]= v_res.x[:]
            
            plt.plot(x0[N+1:])
        plt.show()

        v_res.add_FuncGradCounter(self.counter-initial_counter)
        lam_res.add_FuncGradCounter(self.counter-initial_counter)
        res = [lam_res,v_res,x0]
        return res

            
        
    def lagrange_penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None):
        """
        Solve the optimazation problem with augmented lagrange

        Arguments:
        * N: number of discritization points
        * m: number ot processes
        * my_list: list of penalty variables, that we want to solve the problem
                   for.
        * x0: initial guess for control
        * Lbfgs_options: same as for class initialisation
        """

        dt=float(self.T)/N
        if x0==None:
            x0 = self.Vec(self.initial_control(N,m=m))
        x = None
        Result = []

        G = np.zeros(m-1)
        for i in range(len(my_list)):
            print
            print my_list[i],G
            print 
            def init_pen(self,y,u,my,N,k):
                return self.initial_lagrange(y,u,my,N,k,G)
            
            def J(u):                
                return self.Lagrange_Penalty_Functional(u,N,m,my_list[i],G)

            def grad_J(u):

                l,L = self.adjoint_penalty_solver(u,N,m,my_list[i],init=init_pen )

                g = np.zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)

                for j in range(m-1):
                    g[N+1+j]=l[j+1][0]-l[j][-1] + G[j]
                    
                return g
            self.update_Lbfgs_options(Lbfgs_options)
            #solver = Lbfgs(J,grad_J,x0,options=Loptions)
            solver = SplitLbfgs(J,grad_J,x0.array(),options=self.Lbfgs_options)
            
            #res = solver.solve()
            res=solver.normal_solve()
            Result.append(res)
            x0 = res['control']
            print 
            y,Y = self.ODE_penalty_solver(res['control'].array(),N,m)
            for j in range(m-1):
                G[j]=G[j]-my_list[i]*(y[j][-1]-y[j+1][0])

        if len(Result)==1:
            return res
        else:
            return Result


        
    def plot_solve(self,N,x0=None,opt=None,state=False):
        """
        Solve the problem and plot the control

        Arguments:

        * N: # discrite points
        * x0: inital gues
        * opt: optians for Lbfgs
        * state: if not false, method also plots the state
        """
        res = self.solve(N,x0=x0,Lbfgs_options=opt)
        t = np.linspace(0,self.T,N+1)
        u = res['control'].array()

        import matplotlib.pyplot as plt

        if state==False:
            plt.plot(t,u)
            plt.ylabel('control')
            plt.xlabel('time')
        else:
            y = self.ODE_solver(u,N)
            plt.plot(t,u)
            plt.plot(t,y)
            plt.xlabel('time')
            plt.legend(["con","state"])
            
        plt.show()

        return res
        
    def penalty_and_normal_solve(self,N,m,my_list,x0=None,Lbfgs_options=None,
                                 show_plot=False):
        """
        solves problem using both normal and penalty approach 
        """
        res1 = self.penalty_solve(N,m,my_list,x0,Lbfgs_options)
        res2 = self.solve(N,x0,Lbfgs_options)
        if show_plot==True:
            import matplotlib.pyplot as plt
            t = np.linspace(0,self.T,N+1)
            u1 = res1['control'].array()[:N+1]
            u2 = res2['control'].array()
            plt.plot(t,u1)
            plt.plot(t,u2)
            plt.ylabel('control')
            plt.xlabel('time')
            plt.show()
            
        return res1,res2

    
    def lbfgs_memory_solve(self,N,m,my_list,mul=[1,2]):
        """
        Method used in testing
        """
        
        results = []
        for i in range(len(mul)):
            opt = {"mem_lim" : mul[i]*max(m,10)}
            try:
                res = self.penalty_solve(N,m,my_list,Lbfgs_options=opt)
            except Warning:
                res = {'iteration' : -1}
            results.append(res)
        return results

    def simple_test(self,N,make_plot=False):

        if make_plot:
            teller=0
            import matplotlib.pyplot as plt
            t = np.linspace(0,self.T,N+1)

        M = [2,4,8,16,32]
        L=[]
        mul=[1,3]
        for i in range(len(M)):

            L.append(self.lbfgs_memory_solve(N,M[i],[0.5*N],mul=mul))
        try:
            res1 = self.solve(N)
        except Warning:
            res1 = {'iteration' : -1}


        print "--------------m=1--------------" 
        print "|lbfgs memory=10| #iterations=%d| #iterations/m=%.2f"%(res1['iteration'],res1['iteration']) 
        if make_plot:
            if res1['iteration']!=-1:
                plt.plot(t,res1['control'].array(),'r--')
        for i in range(len(M)):
            print "--------------m=%d--------------" %(M[i])
            for j in range(len(mul)):
                print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(mul[j]*max(M[i],10),L[i][j]['iteration'],L[i][j]['iteration']/float(M[i]))

                if make_plot:
                    
                    if j == len(mul)-1:
                        if L[i][j]['iteration']!=-1:
                            plt.plot(t,L[i][j]['control'].array()[:N+1])
                            teller =teller+1
        if make_plot:
            
            if teller==len(M):
                plt.legend(['m=1','m=2','m=4','m=8','m=16','m=32'],loc=4)
            plt.title('Controlls for non linear in y term problem')
            plt.xlabel('time')
            plt.ylabel('controls')
            plt.show()
    def scipy_solver(self,N,disp=False,options=None):
        """
        solve the problem using scipy LBFGS instead of self made LBFGS
        """
        dt=float(self.T)/N
        
            
        def J(u):
            return self.Functional(u,N)

        def grad_J(u):
            l = self.adjoint_solver(u,N)
            return self.grad_J(u,l,dt)


        opt={'gtol': 1e-6, 'disp': disp,'maxcor':10}
        
        if options!=None:
            for key, val in options.iteritems():
                opt[key] = val

        
        res = minimize(J,np.zeros(N+1),method='L-BFGS-B', jac=grad_J,
                       options=opt)
        
        return res

    def scipy_penalty_solve(self,N,m,my_list,disp=False,x0=None,options=None):
        
        dt=float(self.T)/N
        if x0==None:

            x0 = self.initial_control(N,m=m)

        Result = []

        for i in range(len(my_list)):
        
            def J(u):                
                return self.Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):

                l,L = self.adjoint_penalty_solver(u,N,m,my_list[i])

                g = np.zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)
                for j in range(m-1):
                    g[N+1+j]= l[j+1][0] - l[j][-1]
                    
                return g
            opt={'gtol':1e-4, 'disp': disp,'maxcor':10}
        
            if options!=None:
                for key, val in options.iteritems():
                    opt[key] = val

            res = minimize(J,x0,method='L-BFGS-B', jac=grad_J,
                           options=opt)

            Result.append(res)

            x0 = res.x.copy()

        if len(Result)==1:
            return res
        else:
            return Result

    def finite_diff(self,u,N):
        
        import matplotlib.pyplot as plt
        eps = 1./10000

        finite_grad_J = np.zeros(len(u))

        for i in range(len(u)):
            e = np.zeros(len(u))
            e[i]=eps
            J1 = self.Functional(u,N)
            J2 = self.Functional(u+e,N)
            finite_grad_J[i] = (J2-J1)/eps

        l = self.adjoint_solver(u,N)
        
        grad_J = self.grad_J(u,l,self.T/float(N))
        t = np.linspace(0,self.T,len(u))
        plt.plot(t,grad_J)
        plt.plot(t,finite_grad_J,'r--')
        plt.legend(['adjoint','finite diff'])
        plt.show()


    def mu_to_N_relation(self,C,gamma):

        M = [2,4,8,16]
        import matplotlib.pyplot as plt

        fig,axs = plt.subplots(2, 2)
        for i in range(len(M)):

            N = C*M[i]
            mu = gamma*N
            
            t = np.linspace(0,self.T,N+1)

            res1 = self.scipy_solver(N)
            res2 = self.scipy_penalty_solve(N,M[i],[mu])

            e = np.sqrt(trapz((res1.x-res2.x[:N+1])**2,t))

            print
            print "number of procesess: %d" % M[i]
            print "number of time intervals: %d" % N
            print "number of iterations needed: %d and %d" % (res1.nit,res2.nit)
            print "L2 diffrence between normal and penalty approach: %.2e"% e
            print
        

            x1 = i/2
            x2= i%2
            axs[x1,x2].plot(t,res1.x)
            axs[x1,x2].plot(t,res2.x[:N+1])
            axs[x1,x2].legend(['serial','penalty'])
            axs[x1,x2].set_title('m='+str(M[i])+' N='+str(N)+' my='+str(mu))
            axs[x1,x2].set_xlabel("t")
            axs[x1,x2].set_ylabel("control")
        plt.show()


    def scipy_simple_test(self,N,make_plot=False):

        if make_plot:
            teller=0
            import matplotlib.pyplot as plt
            t = np.linspace(0,self.T,N+1)

        M = [2,4,8,16,32]
        L=[]
        mul=[1,3]
        for i in range(len(M)):

            L.append(self.scipy_lbfgs_memory_solve(N,M[i],[0.5*N],mul=mul))
        try:
            res1 = self.scipy_solver(N)
        except Warning:
            res1 = {'iteration' : -1}


        print "--------------m=1--------------" 
        print "|lbfgs memory=10| #iterations=%d| #iterations/m=%.2f"%(res1.nit,res1.nit) 
        if make_plot:
            if res1['iteration']!=-1:
                plt.plot(t,res1['control'].array(),'r--')
        for i in range(len(M)):
            print "--------------m=%d--------------" %(M[i])
            for j in range(len(mul)):
                print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(mul[j]*max(M[i],10),L[i][j].nit,L[i][j].nit/float(M[i]))

                if make_plot:
                    
                    if j == len(mul)-1:
                        if L[i][j].nit!=-1:
                            plt.plot(t,L[i][j]['control'].array()[:N+1])
                            teller =teller+1
        if make_plot:
            
            if teller==len(M):
                plt.legend(['m=1','m=2','m=4','m=8','m=16','m=32'],loc=4)
            plt.title('Controlls for non linear in y term problem')
            plt.xlabel('time')
            plt.ylabel('controls')
            plt.show()



    def scipy_lbfgs_memory_solve(self,N,m,my_list,mul=[1,2]):
        """
        Method used in testing
        """
        
        results = []
        for i in range(len(mul)):
            opt = {'gtol': 1e-4,'maxcor' : mul[i]*max(m,10)}
            try:
                res = self.scipy_penalty_solve(N,m,my_list,options=opt)
            except Warning:
                res = {'iteration' : -1}
            results.append(res)
        return results

    def simple_problem_exact_solution(self,N):
        
        T = self.T
        yT = self.yT
        a = self.a
        y0 = self.y0
        
        D = 1. + (np.exp(a*T)**2-1)/(2.*a)
        
        CC = np.exp(a*T)*self.c*(np.exp(a*T)-1)/a
        C = (yT*np.exp(a*T)-y0*np.exp(a*T)**2 -CC)/D

        u_f = lambda x : C*np.exp(a*(-t)) +self.c
        

        t = np.linspace(0,T,N+1)
        u = u_f(t)

        return u,t,u_f

    def sin_prop_exact_solution(self,N,b):

        T = self.T
        yT = self.yT
        a = self.a
        y0 = self.y0
        t = np.linspace(0,T,N+1)

        
        D = 1. + (np.exp(a*T)**2-1)/(2.*a)
        
        CC = np.exp(a*T)*b/(a**2+1.) +b*(np.sin(T)+np.cos(T)/a)/(a+1./(a))

        C = (yT*np.exp(a*T)-y0*np.exp(a*T)**2 +np.exp(a*T)*CC)/D

        u_f = lambda x : C*np.exp(a*(-x)) +b*np.sin(x)
        

        
        u = u_f(t)
        return u,t,u_f
        
class Problem1(OptimalControlProblem):
    """
    optimal control with ODE y'=ay+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a

    #"""
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i]+dt*u[j+1])/(1.-dt*a)


    

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return l[-(i+1)]*(1.+dt*a)
        return l[-(i+1)]/(1.-dt*a)
    """
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        return y[i] +dt*(u[j]+a*y[i])

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return l[-(i+1)]/(1.-dt*a)
        return l[-(i+1)] + dt*a*l[-(i+1)]
    #"""

class Explicit_problem1(OptimalControlProblem):
    

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=True)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        return y[i] +dt*(u[j]+a*y[i])

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)] + dt*a*l[-(i+1)]




class Problem2(OptimalControlProblem):
    """
    optimal control with ODE y'=a(t)y + u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a(dt*(j+1)))


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return (1+dt*a(self.T-dt*i))*l[-(i+1)] 


class Problem3(OptimalControlProblem):
    """
    optimal control with ODE y=ay'+u
    and J(u)=||u||**2 + alpha*(y(T)-yT)**2
    """

    def __init__(self,y0,yT,T,a,alpha,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a
        self.alpha = alpha
        
        def JJ(u,y,yT,T):
            return J(u,y,yT,T,self.alpha)
        def grad_JJ(u,p,dt):
            return grad_J(u,p,dt,self.alpha)
        self.J=JJ
        self.grad_J = grad_JJ


    def initial_adjoint(self,y):
        return self.alpha*(y - self.yT)

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)] 


class NoneVec():

    def __getitem__(self,i):
        return None

if __name__ == "__main__":

    from matplotlib.pyplot import *

    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = Problem1(y0,yT,T,a,J,grad_J)

    N = 1000
    m = 10
    
    res1 = problem.solve(N,algorithm='my_lbfgs')

    t=np.linspace(0,T,N+1)
    try:
        u1 = res1['control'].array()[:N+1]
    except:
        u1 = res1.x[:N+1]
    figure()
    plot(t,u1)
    #show()


    opt =None# {"mem_lim":20}
    res2 = problem.penalty_solve(N,m,[100**2,1000**2],Lbfgs_options=opt,algorithm='my_lbfgs')[-1]
    print res2.jump_diff
    try:
        u2 = res2['control'].array()[:N+1]
    except:
        u2 = res2.x[:N+1]

    plot(t,u2,'r--')
    show()
    print res1.counter(),res1.niter
    print res2.counter(),res2.niter
    

    
