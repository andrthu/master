from dolfin import *
import numpy as np
import time
from partition_funcs import int_par_len

def rhs_part(N,M,i):

    f = N/M
    r = N%M

    s = 0

    for j in range(i):
        if r - j >0:
            s += f +1
        else:
            s += f
    return s

def int_par_len2(n,m,i):
    """
    With fine resolution n and m time decomposition intervals,
    functions find number of points in interval i
    """
    N=n/m
    rest=n%m

   
    if rest - i >0:
        state = N+1
    else:
        state = N
    return state



def gather_u(u):
    """
    Gathers the u lists into one list U
    """
    U = []
    #print len(u[0])
    for j in range(len(u[0])):
        U.append(u[0][j])
        
    for i in range(1,len(u)):
        #print len(u[i])
        for j in range(1,len(u[i])):
            U.append(u[i][j])
        
    return U

class PararealPdeSolver():

    def __init__(self,V,mesh):
        self.V = V
        self.mesh = mesh

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

        
    def PDE_form(self,u,u_,v,r,dt):
        
        return (self.Dt(u,u_,dt)*v + inner(grad(u),grad(v))-r*v)*dx 
        
        

    def implicit_solver(self,ic,rhs,bc,dt,N):

        U = []

        u_ = ic.copy()

        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)
        r = rhs[0]
        F = self.PDE_form(u,u_,v,r,dt)
        
        for i in range(N):
            r.assign(rhs[i])
            
            solve(F==0,u,bc)
            u_.assign(u)
            U.append(u_.copy())
            
        interactive()
        return U

    def propagator_iteration(self,u,c_u,rhs,bc,N,M,dt,dT):
        
        S = []
        S.append(project(Constant(0.0),self.V))
        for i in range(M):
            
            
            S.append(project((u[i][-1].copy()-c_u[i+1].copy())*Constant(1./dT),self.V))

        delta = self.implicit_solver(project(Constant(0.0),self.V),S,bc,dT,M)
        for i in range(M):
            c_u[i+1] = project(u[i][-1].copy() + delta[i+1].copy(),self.V)

        u = []
        
       
        for i in range(M):
            u.append(self.implicit_solver(c_u[i],rhs[rhs_part(N,M,i):-1],bc,dt,
                                          int_par_len2(N,M,i)))
            #print "prop;",len(u[i])
        return u,c_u

    def parareal_order_solver(self,ic,rhs,bc,dt,dT,N,M,order=3):
        
        coarse_rhs = []
        for i in range(M):
            coarse_rhs.append(rhs[rhs_part(N,M,i)])
        coarse_u = self.implicit_solver(ic,coarse_rhs,bc,dT,M)

        u = []
        for i in range(M):
            u.append(self.implicit_solver(coarse_u[i],rhs[rhs_part(N,M,i):],
                                          bc,dt,int_par_len2(N,M,i)))

        for k in range(order-1):
            u,coarse_u = self.propagator_iteration(u,coarse_u,rhs,
                                                   bc,N,M,dt,dT)
        #print len(u[0]),len(u[1])
        return gather_u(u)


if __name__ == "__main__":
    set_log_level(ERROR)
    mesh = UnitIntervalMesh(20)

    V = FunctionSpace(mesh,"CG",2)

    test = PararealPdeSolver(V,mesh)

    N = 99
    M = 10
    rhs = []
    for i in range(N+1):
        rhs.append(project(Constant(0.0),V))

    bc = DirichletBC(V,0.0,"on_boundary")

    dt = 1./N
    dT = 1./M
    ic = project(Expression('x[0]*(1-x[0])'),V)

    
    
    U2 = test.implicit_solver(ic,rhs,bc,dt,N)
    U1 = test.parareal_order_solver(ic,rhs,bc,dt,dT,N,M,order = 3)
    
    
    for i in range(len(U1)):
        print assemble( (U1[i]-U2[i])**2*dx)#/assemble(U2[i]**2*dx)
        #plot(U1[i]-U2[i])
    
    

    
