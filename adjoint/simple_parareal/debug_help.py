"""
def pde_propogator(self,opt,S,bc,dT,N):

        delta = []

        

        delta.append('0')

        for i in range(N):
            
            
            delta.append('S[%d]'%i)
            
        
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
        
        for i in range(N):
            r.assign(project(Constant(1./dT)*S[-(i+1)],self.V))
            u.assign(S[-(1+i)])
            solve(L==r,p,bc)
            p_.assign(p)
            delta.append(p_.copy())
            
        
        return delta
        
        

    def PC_maker(self,opt,ic,start,end,Tn,m):

        xN = self.xN
        bc = DirichletBC(self.V,0.0,"on_boundary")
        dT = 1./m
        def pc(x):
            start = len(x) - (m-1)*xN
            S = []
            
            for i in range(m-1):
                f = Function(self.V)
                f.vector()[:] = x[start +i*xN:start+(i+1)*xN]
                S.append(f.copy())

            S.append(project(Constant(0.0),self.V))
            adj_prop=list(reversed(self.adjoint_propogator(opt,S,bc,dT,m)[1:]))
            for i in range(len(S)):
                S[i] = project(S[i] + adj_prop[i],self.V)
            
            S = [project(Constant(0.0),self.V)] + S[:-1] 

            pde_prop = self.pde_propogator(opt,S,bc,dT,m)[1:]

            for i in range(len(S)):
                S[i] = project(S[i] + pde_prop[i],self.V)
                
            
            for i in range(m-1):
                x[start +i*xN:start+(i+1)*xN] = S[i].vector().array()[:]

            return x

        return pc
"""

N = 5
S = [str(i+1) for i in range(N-1)]
S = S #+ ['0']
delta = []       

print 'initial S'
print S
delta.append('0')
 
for i in range(N-1):     
    delta.append('S[%d]'%-(i+1))
print
print 'adjoint prop, delta'
print delta
print
delta2 = list(reversed(delta[:-1]))
print 'reversed adjoint prop, delta2'
print delta2
for i in range(len(S)):
    S[i] = S[i] + '+'+delta2[i]
print
print 'after adjoint prop'
print S
print

#S = ['0'] + S[:-1]
#print 'before pde prop'
#print S
#print 

delta = []
delta.append('0')
for i in range(N-1):
    delta.append('Sp[%d]'%i)
print 'pde prop, delta'
print delta
print

for i in range(len(S)):
    S[i] = S[i] + '+'+delta[i]
print 'result'
print S
