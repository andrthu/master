from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps,trapz
from EBsolver import *
from scipy.optimize import minimize
from non_lin_ex import *

def L2_grad(u,l,dt):
    return dt*(l+u)

def mini_solver(y0,a,T,yT,n,F,sol,adj,gr):
    t=linspace(0,T,n+1)
    dt=float(T)/n
    def J(u):
        return F(sol(y0,a,len(u)-1,u,T),u,yT,T)

    def grad_J(u):
        l = adj(y0,a,len(u)-1,u,T,yT)
        return gr(u,l,dt)


    res = minimize(J,zeros(n+1)+1,method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': True})

    u=res.x
    print res.x
    print res.message
    val = J(u)
    y = solver(y0,a,n,u,T)
    plot(t,u)
    plot(t,y)
    legend(['control','state'])
    title('J(u)= ' + str(val))
    print J(u)
    show()
 
   
if __name__ == '__main__':
    #mini_solver(y0,a,T,yT,n)
    mini_solver(1,1,1,1,100,Functional2,solver,adjoint_solver,L2_grad)
    #mini_solver(1,2,1,1,1000,quad_F,solver,adjoint_solver,quad_grad)
    #mini_solver(1,10,1,1,1000,exp_Func,solver,adjoint_solver,exp_grad)
    #mini_solver(1,10,1,-20,1000,add_lin_F,solver,adjoint_solver,add_lin_grad)
