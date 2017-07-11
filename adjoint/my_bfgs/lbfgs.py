import numpy as np
from linesearch.strong_wolfe import *
from scaler import PenaltyScaler
from diagonalMatrix import DiagonalMatrix

#from matplotlib.pyplot import *

from my_vector import SimpleVector, MuVector,MuVectors
from LmemoryHessian import LimMemoryHessian, MuLMIH


class LbfgsParent():
    """
    Parent class for L-BFGS optimization algorithm.
    """

    def __init__(self,J,d_J,x0,Hinit=None,options=None,scale=None):
        """
        Initials for LbfgsParent

        Valid options are:

        * J : Functional you want minimized
        * d_J : Gradient of the functional
        * x0 : Initial guess
        * Hinit : Initial approximation for the inverted hessian
        * options : Options are spesific to the sunder classes 
        """
        
        self.J   = J
        self.d_J = d_J
        self.x0  = x0
         

        self.set_options(options)   
        if Hinit==None:
            #self.Hinit = np.identity(len(x0))
            self.Hinit = DiagonalMatrix(len(x0))
        self.scaler = self.scale_problem(scale)


        
            
        

    
    def set_options(self,user_options):
        """
        Method for setting options
        """
        options = self.default_options()
        
        if user_options!=None:
            for key, val in user_options.iteritems():
                options[key]=val

        #options["line_search_options"]['ftol'] = options['jtol']
        #options["line_search_options"]['gtol'] = max([1-options['jtol'],0.9])
        self.options = options
        


    def check_convergance(self,df0,k):
        """
        Stopping criterion for the algorithm based on L2norm of gardient.
        """

        if self.scaler==None:
            grad_norm = np.sqrt(np.sum((df0.array())**2)/len(df0))
        else:
            N = self.scaler.N
            gamma = self.scaler.gamma
            y = df0.array()
            grad_norm = np.sum((y[:N+1])**2)/len(df0)
            grad_norm+=np.sum((y[N+1:])**2)/(len(df0)*gamma**2)
            grad_norm = np.sqrt(grad_norm)
        if grad_norm<self.options['jtol']:
            return 1
            
        if k>self.options['maxiter']:
            return 1
        return 0


    def scale_problem(self,scale):
        
        if scale==None:
            return None

        J      = self.J
        grad_J = self.d_J

        try:
            x0 = self.x0.array()
            my_vec = True
        except AttributeError:
            my_vec = False
            x0 = self.x0

        m = scale['m']
        if scale.has_key('factor'):
            scaler = PenaltyScaler(J,grad_J,x0,m,
                                   factor=scale['factor'])
        else:
            scaler = PenaltyScaler(J,grad_J,x0,m)

        
        N = len(x0)-m
        
        y0 = scaler.var(x0)
        
        J_ = lambda x: J(scaler.func_var(x))
        grad_J_ = lambda x : scaler.grad(grad_J)(scaler.func_var(x))
        
        self.J = J_
        if my_vec:
            self.x0 = self.options['Vector'](y0)
        else:
            self.x0 = y0
        self.d_J = grad_J_
        self.scale = True
        if self.options['scale_hessian']==True:
            #self.Hinit[range(N+1,N+m),range(N+1,N+m)] = 1./scaler.gamma**2
            self.Hinit.diag[N+1:] = 1./scaler.gamma**2
        return scaler

    def rescale(self,x):
        if self.scaler==None:
            return x
        N = self.scaler.N
        gamma=self.scaler.gamma
        y = x.array()
        y[N+1:] = y[N+1:].copy()*gamma
        return self.options['Vector'](y)
        
    def default_options(self):
        """
        Class spesific default options
        """
        raise NotImplementedError, 'Lbfgs.default_options() not implemented' 
        

    def do_linesearch(self,J,d_J,x,p):
        """
        Method that does a linesearch using the strong Wolfie condition

        Arguments:

        * J : The functional
        * d_J : The gradient
        * x : The starting point of the linesearch
        * p : The direction of the search, i.e. -d_J(x)

        Return value:
        
        * x_new : The ending point of the linesearch
        * alpha : The step length
        """
        x_new = x.copy()
    
        Vec = self.options['Vector']

        def phi(alpha):
            """
            Convert functional to a one variable functon dependent
            on step size alpha
            """
            
            x_new=x.copy()
            x_new.axpy(alpha,p)
            return J(x_new.array())
    
        def phi_dphi(alpha):
            """
            Derivative of above function
            """
            x_new = x.copy()
        
            x_new.axpy(alpha,p)
        
            f = J(x_new.array())
            djs = p.dot(Vec(d_J(x_new.array())))
        
            return f,float(djs)
            
        
        phi_dphi0 = J(x.array()),float(p.dot(Vec(d_J(x.array()))))
        
        
        if self.options["line_search"]=="strong_wolfe":
            ls_parm = self.options["line_search_options"]
            
            ftol      = ls_parm["ftol"]
            gtol      = ls_parm["gtol"]
            xtol      = ls_parm["xtol"]
            start_stp = ls_parm["start_stp"]
            
            ls = StrongWolfeLineSearch(ftol,gtol,xtol,start_stp,
                                        ignore_warnings=False)

        alpha = ls.search(phi, phi_dphi, phi_dphi0)

        
        x_new=x.copy()
        x_new.axpy(alpha,p)
    
        return x_new, float(alpha)

    def solve(self):
        """
        Method that does optimization
        """
        raise NotImplementedError, 'Lbfgs.default_solve() not implemented'



########################################
##########################
############################
#########################     LBFGS
############################
##############################################

class Lbfgs(LbfgsParent):
    """
    Straight foreward L_BFGS implementation
    """
    def __init__(self,J,d_J,x0,pc=None,Hinit=None,options=None,scale=None):
        """
        Initials for LbfgsParent

        Valid options are:

        * J : Functional you want minimized
        * d_J : Gradient of the functional
        * x0 : Initial guess
        * Hinit : Initial approximation for the inverted hessian
        * options : Options are as follows:
          - jtol : Stopping tolerance
          - maxiter : maximal amount of allowed iteration before exiting solver
          - line_search_options: options for the linesearch
          - mem_lim : Number of iterations the inverted hessian remembers
          - Hinit : Initial inverted Hessian
          - beta : scaling variable for inverted hessian
          - return_data : boolean return the data instance or control
        """

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=Hinit,options=options,scale=scale)
        
        mem_lim = self.options['mem_lim']
        
        beta = self.options["beta"]
        self.pc = pc
            
        if pc==None:
            self.p_direction = self.direction
        else:
            self.p_direction = self.pc_direction
        Hessian = LimMemoryHessian(self.Hinit,mem_lim,beta=beta)
        self.data = {'control'   : self.x0,
                     'iteration' : 0,
                     'lbfgs'     : Hessian ,
                     'scaler'    : self.scaler,}

    def default_options(self):
        """
        Method that gives sets the default options
        """
        
        ls = {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1}
        
        default = {"jtol"                   : 1e-4,
                   "rjtol"                  : 1e-6,
                   "gtol"                   : 1e-4,
                   "rgtol"                  : 1e-5,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,                   
                   "mem_lim"                : 5,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : False,
                   "scale_hessian"          : False,}
        
        return default
        
    def direction(self,grad,H):
        return H.matvec(-grad)

    def pc_direction(self,grad,H):
        Vec = self.options['Vector']
        return -Vec(self.pc(H.matvec(grad).array()))
    
    def solve(self):
        """
        Method that solves the opttmizaton problem

        Return value:
        * x : The optimal control
        or : 
        * data
         - control : The optimal control
         - iterations : number of iterations reqiered
         - lbfgs : The class of the limited memory inverted hessian
        """

        Vec = self.options['Vector']     # Choose vector type
        x0 = self.x0                     # set initial guess
        n = x0.size()                    # find number of variables
        x = Vec(np.zeros(n))             # convert to vector class    
        Hk = self.data['lbfgs']          # get inverted hessian


        df0 = Vec(self.d_J(x0.array()))  # initial gradient
        df1 = Vec(np.zeros(n))           # space for gradient  

        iter_k = self.data['iteration']           
    
        p = Vec(np.zeros(n))

        tol = self.options["jtol"]
        max_iter = self.options['maxiter']

        #the iterations
        while self.check_convergance(df0,iter_k)==0:
            
            p = self.p_direction(df0,Hk) #Hk.matvec(-df0)

            
            x,alfa = self.do_linesearch(self.J,self.d_J,x0,p)

            df1.set(self.d_J(x.array()))
            
            s = x-x0
            """
            if self.scaler!=None:
                s = self.rescale(s)
            """
            y = df1-df0
            #s =self.rescale(s)
            #y = self.rescale(y)
            Hk.update(y,s)
             
            x0=x.copy()
            df0=df1.copy()

            iter_k=iter_k+1
            self.data['iteration'] = iter_k
            self.data['control'] = x
        x = self.rescale(x)
        self.data['control'] = x
        if self.options["return_data"] == True:

            return self.data

        return x


    def one_iteration(self,comm):
        """
        Method that does one iteration of lbfgs.

        The point of this method is to check if parallel actually works.
        """
        
        rank = comm.Get_rank()

        x0 = self.x0                     # set initial guess
        
        
        Vec = self.options['Vector']     # Choose vector type
        
        n = x0.size()                    # find number of variables
        x = Vec(np.zeros(n))             # convert to vector class    
        Hk = self.data['lbfgs']          # get inverted hessian


        df0 = Vec(self.d_J(x0.array()))  # initial gradient
        df1 = Vec(np.zeros(n))           # space for gradient  
        
        iter_k = self.data['iteration']          
        

        p = self.p_direction(df0,Hk) #Hk.matvec(-df0)
        
            
        x,alfa = self.do_linesearch(self.J,self.d_J,x0,p)
        df1.set(self.d_J(x.array()))
        s = x-x0
        y = df1-df0
        #print y.array(),rank, 'hei'
        Hk.update(y,s)

        x0=x.copy()
        df0=df1.copy()
    
        

        return x0
        
########################################
##########################
############################
#########################     MU LBFGS
############################
##############################################


class MuLbfgs(LbfgsParent):
    """
    L-BFGS class made s.t. it can save and take in previous invertad hessians
    and modify them by updating a mu variable. Usful in a penalty setting.
    """

    def __init__(self,J,d_J,x0,Mud_J,Hinit=None,options=None):
        """
        Initials for LbfgsParent

        Valid options are:

        * J : Functional you want minimized
        * d_J : Gradient of the functional
        * x0 : Initial guess
        * Mud_J : Helps with the mu stuff
        * Hinit : Initial approximation for the inverted hessian
        * options : Options are as follows:
          - jtol : Stopping tolerance
          - maxiter : maximal amount of allowed iteration before exiting solver
          - line_search_options: options for the linesearch
          - mem_lim : Number of iterations the inverted hessian remembers
          - Hinit : Initial inverted Hessian
          - beta : scaling variable for inverted hessian
          - mu_val : The current mu
          - old_hessian : memory of previous inverted Hessian
          - save_number : Size of memory taken from old hessian
          - return_data : boolean return the data instance or control
        """

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=Hinit,options=options)

        self.Mud_J = Mud_J
        
        mem_lim  = self.options['mem_lim']
        beta     = self.options["beta"]
        mu       = self.options["mu_val"]
        H        = self.options["old_hessian"]
        save_num = self.options["save_number"]

        Hessian = MuLMIH(self.Hinit,mu=mu,H=H,mem_lim=mem_lim,beta=beta,
                         save_number=save_num)

        self.data = {'control'   : x0,
                     'iteration' : 0,
                     'lbfgs'     : Hessian }

    
    def default_options(self):
        """
        Method that gives sets the default options
        """

        ls = {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1}
        
        default = {"jtol"                   : 1e-4,
                   "rjtol"                  : 1e-6,
                   "gtol"                   : 1e-4,
                   "rgtol"                  : 1e-5,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,
                   "mem_lim"                : 5,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1, 
                   "mu_val"                 : 1,
                   "old_hessian"            : None,
                   "penaly_number"          : 1,
                   "save_number"            :-1,          
                   "return_data"            : False, }
        
        return default


    def solve(self):

        """
        Method that solves the opttmizaton problem

        Return value:
        * x : The optimal control
        or : 
        * data
         - control : The optimal control
         - iterations : number of iterations reqiered
         - lbfgs : The class of the limited memory inverted hessian
        """

        Vec = self.options['Vector']
        x0=self.x0
        n=x0.size()
        m=self.options["penaly_number"]
        
        x = Vec(np.zeros(n))
        
        Hk = self.data['lbfgs']
        
        

        mu = self.options["mu_val"]
        
        mu_df0, mu_x0 = self.Mud_J(x0)
        mu_df1 = None
        mu_x1  = None
        

        iter_k = self.data['iteration']


        df0 = Vec(self.d_J(x0.array()))
        df1 = Vec(np.zeros(n))

        p = Vec(np.zeros(n))

        tol = self.options["jtol"]
        max_iter = self.options['maxiter']



        while self.check_convergance(df0,iter_k)==0:
            
            
            p = Hk.matvec(-df0)
            
            
            x,alfa = self.do_linesearch(self.J,self.d_J,x0,p)

            df1.set(self.d_J(x.array()))
            mu_df1,mu_x1 = self.Mud_J(x)


            Hk.update(mu_df1-mu_df0,mu_x1-mu_x0)



            mu_df0 = mu_df1.copy()
            mu_x0  = mu_x1.copy()
            
            
            
            x0=x.copy()
            df0=df1.copy()
            

            iter_k=iter_k+1
            self.data['iteration'] = iter_k
            self.data['control']   = x
        
        if self.options["return_data"]:
            return self.data

        return x



    
if __name__== "__main__":

    
    def J(x):

        s=0
        for i in range(len(x)):
            s = s + (x[i]-1)**2
        return s


    def d_J(x):

        return 2*(x-1)

    x0=SimpleVector(np.linspace(1,30,30))
    

    solver = Lbfgs(J,d_J,x0)
    
    print solver.solve().array()
