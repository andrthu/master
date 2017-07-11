from dolfin import *
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize as Mini
import numpy as np
from fenicsOptimalControlProblem import FenicsOptimalControlProblem


class HeatControl(FenicsOptimalControlProblem):
    
    #opt = {'rhs':f, 'c':c.'uT':uT,alpha:1}

    def __self__(self,V,mesh,options={}):
        super.__init__(self,V,mesh,options)

        self.end_state_diff = None

    def adjoint_ic(self,opt,U):
        try:
            alpha = Constant(opt[alpha])
        except:
            alpha = Constant(1.0)
        
        self.end_state_diff = project((U[-1]-opt['uT']),self.V)
        return project(alpha*((U[-1]-opt['uT'])),self.V)

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

        

    def PDE_form(self,ic,opt,u,u_,v,rhs,timestep):

        c = Constant(opt['c'])
        

        F = (self.Dt(u,u_,timestep)*v + c*u.dx(0)*v.dx(0)-rhs*v)*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")

        return F,bc
        
        

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        c = Constant(opt['c'])

        F = -(-self.Dt(p,p_,timestep)*v + c*p.dx(0)*v.dx(0))*dx

        bc = DirichletBC(self.V,0.0,"on_boundary")

        return F,bc


    def get_control(self,opt,ic,m):
        
        N = len(ic.vector().array())
        f = opt['rhs']
        Nt = len(f)
        x = np.zeros(N*Nt+(m-1)*N)
        
        for i in range(Nt):
            x[i*N:(i+1)*N] = f[i].copy().vector().array()[:]
        return x

    def get_opt(self,control,opt,ic,m):
        
        f = opt['rhs']
        Nt = len(f)
        N = len(ic.vector().array())

        for i in range(Nt):
            f[i].vector()[:] = control[i*N:(i+1)*N]
            

        return opt,ic
        

    def grad_J(self,P,opt,ic,h):
        
        f = opt['rhs']
        N = len(ic.vector().array())
        Nt = len(f)
        x = np.zeros(N*Nt)
        timestep = opt['T']/float(len(f))
        #x[0:N] = f[0].copy().vector().array()[:]
        for i in range(0,Nt):
            x[i*N:(i+1)*N]=timestep*(P[-(i+1)].copy().vector().array()[:] + f[i].copy().vector().array()[:])

        return h*x
        
    
    def J(self,opt,ic,U,start,end):
        n = len(U)
        timestep = (end-start)/float(n)
        s = 0
        
        try:
            alpha = opt['alpha']
        except:
            alpha = 1

        f = opt['rhs']
        s += 0.5*assemble(f[0]**2*dx)
        for i in range(n-2):
            s += assemble(f[i+1]**2*dx)
        s += 0.5*assemble(f[-1]**2*dx)

        s2 = alpha*0.5*assemble((U[-1]-opt['uT'])**2*dx)
        return timestep*s +s2

    def penalty_grad_J(self,P,opt,ic,m,h):
        f = opt['rhs']
        N = len(ic.vector().array())
        Nt = len(f)
        x = np.zeros(N*Nt +(m-1)*N )
        timestep = opt['T']/float(len(f))
        #x[0:N] = f[0].copy().vector().array()[:]
        p = self.gather_penalty_funcs(P)
        for i in range(0,Nt):
            x[i*N:(i+1)*N]=timestep*(p[-(i+1)].copy().vector().array()[:] + f[i].copy().vector().array()[:])
        
        for i in range(m-1):
            x[Nt*N+ i*N:Nt*N+ (i+1)*N] = project((P[i+1][-1]-P[i][0]),
                                              self.V).vector().array().copy()[:]
        return h*x
        
    def rhs_finder(self,Rhs,rhs,i):
        rhs.assign(Rhs[i])


    def pde_propogator(self,ic,opt,S,bc,dT,N):

        delta = []

        u_ = project(Constant(0.0),self.V)

        delta.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)
        r = project(Constant(1./dT)*S[0],self.V)

        F,_ = self.PDE_form(ic,opt,u,u_,v,r,dT)
        
        for i in range(N-1):
            r.assign(project(Constant(1./dT)*S[i],self.V))
            
            solve(F==0,u,bc)
            #plot(u)
            u_.assign(u)
            delta.append(u_.copy())
            
        
        return delta

        

    def adjoint_propogator(self,opt,S,bc,dT,N):

        delta = []

        p_ = project(Constant(0.0),self.V)

        delta.append(p_.copy())

        p = Function(self.V)
        v = TestFunction(self.V)
        r = project(Constant(1./dT)*S[-1],self.V)
        u = S[-1]
        
        L,_ = self.adjoint_form(opt,u,p,p_,v,dT)
        
        for i in range(N-1):
            r.assign(project(Constant(1./dT)*S[-(i+1)],self.V))
            u.assign(S[-(1+i)])
            solve(L==r,p,bc)
            p_.assign(p)
            delta.append(p_.copy())
            
        
        return delta
        
        

    def PC_maker(self,opt,ic,start,end,Tn,m):

        xN = self.xN
        bc = DirichletBC(self.V,0.0,"on_boundary")
        dT = float(end-start)/m
        def pc(x):
            start = len(x) - (m-1)*xN
            S = []
            
            for i in range(m-1):
                f = Function(self.V)
                f.vector()[:] = x[start +i*xN:start+(i+1)*xN]
                S.append(f.copy())
                #plot(f)
            
            trivial = project(Constant(0.0),self.V)
            #S.append(trivial.copy())
            adj_prop=list(reversed(self.adjoint_propogator(opt,S,bc,dT,m)))
            
            for i in range(len(S)):
                S[i] = project(S[i] + adj_prop[i],self.V)
            #+adj_prop[-1]
            #S = [project(trivial+adj_prop[-1],self.V)] + S[:-1] 

            pde_prop = self.pde_propogator(ic,opt,S,bc,dT,m)

            for i in range(len(S)-1):
                S[i] = project(S[i] + pde_prop[i],self.V)
                
            
            for i in range(m-1):
                x[start +i*xN:start+(i+1)*xN] = S[i].vector().array()[:]

            return x

        return pc

    def PPCSD_solver(self,opt,ic,start,end,Tn,m,mu_list,options=None):

        h = self.mesh.hmax()
        X = Function(self.V)
        xN = len(ic.vector().array())
        control0 = self.get_control(opt,ic,m)
        
        self.update_SD_options(options)
        
        res =[]
        PPC = self.PC_maker(opt,ic,start,end,Tn,m)
        for k in range(len(mu_list)):
            mu = mu_list[k]
            J,grad_J=self.create_reduced_penalty_j(opt,ic,start,end,Tn,m,mu)
            

            solver = PPCSteepestDecent(J,grad_J,control0.copy(),PPC,
                                       options=self.SD_options)
            
            result = solver.solve()
            res.append(result)
            control0 = result.x.copy()
        if len(res)==1:
            return res[0]
        else:
            return res

        


if __name__== '__main__':
    import time
    set_log_level(ERROR)
    mesh = UnitIntervalMesh(10)

    V = FunctionSpace(mesh,"CG",1)

    test1 = HeatControl(V,mesh)
    test2 = HeatControl(V,mesh)
    ic = project(Constant(0.0),V)
    start = 0
    end = 0.5
    Tn = 10
    RHS = []
    RHS2=[]
    m = 3
    r = Expression('sin(pi*x[0])')
    for i in range(Tn+1):
        RHS.append(project(r,V))
        RHS2.append(project(r,V))
    ut = Constant(0.0)
    
    opt1 = {'c' : 0.1,'rhs':RHS,'uT':project(ut,V),'T':end-start}
    opt2 ={'c' : 0.1,'rhs':RHS2,'uT':project(ut,V),'T':end-start}
    res1 = test1.solver(opt1,ic,start,end,Tn,algorithm='my_steepest_decent')
    #res = test1.penalty_solver(opt,ic,start,end,Tn,m,[1],algorithm = 'my_steepest_decent')
    res=test2.PPCSD_solver(opt2,ic,start,end,Tn,m,[1])
    print res.val(),res.niter,res1.niter,res1.val()

    #test1.PDE_solver(ic,opt,start,end,Tn,show_plot=True)
    """
    solver_type = 'my_steepest_decent'
    res = test1.solver(opt,ic,start,end,Tn,algorithm=solver_type,
                       options={'jtol':1e-6})
    try:
        print np.max(res.x), np.sqrt(np.sum(res.dJ**2)/len(res.x)),res.val()
    except:
        print res['control'].array(),np.max(res['control'].array())
    #res = test1.penalty_solver(opt,ic,start,end,Tn,m,[1])
    N = len(ic.vector().array())
    #print res
    i = 0
    """
    """
    while i <Tn:
        f = Function(V)
        
        f.vector()[:] = res.x[i*N:(i+1)*N]
        plot(f)
        time.sleep(1)
        i+=1
    
    """
