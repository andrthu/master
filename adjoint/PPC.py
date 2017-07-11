import numpy as np

class PararealPC():

    def __init__(self,state_step,adjoint_step):

        self.state_step = state_step
        self.adjoint_step = adjoint_step



    def create_PC1(self,N,m):

        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x.copy()[:]
            
            dT = float(self.T)/m
            
            
            
            for i in range(1,m):
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-i],dT,step=step)

            
            for i in range(1,m):
                S[i] = S[i] + self.state_step(S[i-1],dT,step=step)
       
            x[:]=S.copy()[1:-1]
            return x
            
            

        return pc

    def create_PC2(self,N,m):

        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x.copy()[N+1:]
            
            dT = float(self.T)/m
            
            
            
            for i in range(1,m):
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-i],dT,step=step)

            
            for i in range(1,m):
                S[i] = S[i] + self.state_step(S[i-1],dT,step=step)
       
            x[N+1:]=S.copy()[1:-1]
            return x
            
            

        return pc
