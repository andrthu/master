from dolfin import *
from scipy.optimize import minimize as Mini
import numpy as np
#from dolfin_adjoint import *


def ODE_solver(y0,alpha,N,U,show_plot=False):

    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)

    y = TrialFunction(V)
    v = TestFunction(V)

    u = Function(V)
    u.vector()[:]=U.copy()[:]
    #u = interpolate(U,V)
    alpha = Constant(alpha)
    
    a = (y.dx(0)*v - alpha*v*y)*dx
    L = u*v*dx

    ic = y0
    bc = DirichletBC(V, Constant(ic),"on_boundary && near(x[0],0.0)")

    Y = Function(V)
    #A,b=assemble_system(a,L,bc)
    solve(a==L,Y,bc)
    if show_plot:
        plot(Y)
        interactive()
    return Y

def adjoint_ODE_solver(y0,alpha,yT,N,u,show_plot=False):
    
    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)
    y = ODE_solver(y0,alpha,N,u.copy())

    l = TrialFunction(V)
    v = TestFunction(V)
    alpha = Constant(alpha)
    
    a = (-l.dx(0)*v - alpha*v*l)*dx
    L = Constant(0.0)*v*dx

    ic = y.vector().array()[0]-yT

    bc = DirichletBC(V, Constant(ic),"on_boundary && near(x[0],1.0)")
    lam = Function(V)

    solve(a==L,lam,bc)
    if show_plot:
        plot(lam)
        interactive()
    return lam
    

def J_Func(u,y0,alpha,yT,N):
    
    y = ODE_solver(y0,alpha,N,u,show_plot=False)

    
    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)
    U = Function(V)
    U.vector()[:]=u.copy()[:]
    A = assemble(U**2*dx)
    return 0.5*A + 0.5*(y.vector().array()[0]-yT)**2

def opti(y0,alpha,yT,N):

    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh,'CG',1)
    dt = mesh.hmax()
    print dt,'loool'
    def J(u):
        return J_Func(u,y0,alpha,yT,N)

    def grad_J(u):

        lam = adjoint_ODE_solver(y0,alpha,yT,N,u)

        gr = u.copy() + lam.vector().array().copy()
        
        return gr*dt


    res = Mini(J,np.zeros(N+1),method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': True})

    
    y = ODE_solver(y0,alpha,N,res.x,show_plot=True)
    print res.x
    x = Function(V)
    x.vector()[:] = res.x[:]
    plot(x)
    interactive()

def finite_diff(u,y0,yT,N,J):

    eps = 1./100

    grad_J = np.zeros(len(u))

    for i in range(len(u)):
        e = np.zeros(len(u))
        e[i]=eps
        J1 = J(u,y0,yT,N)
        J2 = J(u+e,y0,yT,N)        
        grad_J[i] = (J2-J1)/eps
        print J1,J2

    return grad_J

def dol_solve(y0,alpha,yT,N,U):

    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh,'CG',1)
    dt = mesh.hmax()


    y = TrialFunction(V)
    v = TestFunction(V)

    u = Function(V)
    u.vector()[:]=U.copy()[:]
    #u = interpolate(U,V)
    alpha = Constant(alpha)
    
    a = (y.dx(0)*v - alpha*v*y)*dx
    L = u*v*dx

    ic = y0
    bc = DirichletBC(V, Constant(ic),"on_boundary && near(x[0],0.0)")

    Y = Function(V)
    #A,b=assemble_system(a,L,bc)
    solve(a==L,Y,bc)

    J = Functional(inner(u,u)*dx + (Y(1)-yT)**2)

    
    dJ = compute_gradient(J, Control(u))

    
if __name__ == "__main__":
    
    y0=1
    yT=10
    N=100
    a=[1,2,5]
    u = np.zeros(N+1)
    u[N/2]=1
    #dol_solve(y0,1,yT,N,u)
    #u= Constant(0.0)
    for i in range(len(a)):
        #y = ODE_solver(y0,a[i],N,u,show_plot=True)

        #adjoint_ODE_solver(y0,a[i],yT,N,u,show_plot=True)

        opti(y0,a[i],yT,N)

        """
        a = finite_diff(u,y0,yT,N,J_Func)

        mesh = UnitIntervalMesh(N)

        V = FunctionSpace(mesh,'CG',1)

        A = Function(V)
        A.vector()[:]=a
        print a 
        plot(A)
        interactive()
        """
