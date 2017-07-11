from linalg.vector import Vector
import numpy as np

class SimpleVector(Vector):

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[index]=value
        
    def local_size(self):
        return len(self.data)
    
    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data

    def set(self, array, local=True):
        ''' Sets the values of the vector to the values in the numpy.array. 
        If local=False, the global array must be returned in a distributed environment. '''
        self.data = array.copy()

    def scale(self, s):
        ''' Scales the vector by s. '''
        self.data = s*self.data

    def axpy(self, a, x):
        ''' Adds a*x to the vector. '''
        
        y=a*x.data.copy()
        self.data = self.data + y 

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data)

    def dot(self,y):
        
        return np.sum(self.data*y.data)
    
    def matDot(self,A):
        """
        n=len(self.data)
        y = np.zeros(n)
        A = np.matrix(A)
        for i in range(n):
            y[i]=np.sum(np.array(A[i,:][0])*self.data)
        """
        return A(self.data)
    
    def matApy(self,A):
        y = self.matDot(A)
        self.set(y)

    def vecVecMat(self,x):
        a=np.matrix(x.array().copy())
        b=np.matrix(self.array().copy())

        

        return b.T*a

    def copy(self):
        ''' Returns a deep-copy of the vector. '''
        """
        d = self.data.copy()
        """
        d = np.zeros(len(self.data))
        
        d[:]=self.data[:]
        """
        for i in range(len(d)):
            d[i] = self.data[i]
        """
        return SimpleVector(d)


class MuVector(Vector):

    #data =[x,x_mu]

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        
        def J (mu):
            return self.data[0][index]+ mu*self.data[1][index]
        return J

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[0][index]=value[0]
        self.data[1][index]=value[1]

    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data[0].array()

    def copy(self):
        ''' Returns a deep-copy of the vector. '''
        d = [self.data[0].copy(),self.data[1].copy()]
        return MuVector(d)

    def axpy(self,a,y):

        self.data[0].axpy(a,y.data[0])
        self.data[1].axpy(a,y.data[1])

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data[0])

    def scale(self,a):
        self.data[0].scale(a)
        self.data[1].scale(a)

    def lin_func(self,mu):

        
        a = self.data[0] + mu*self.data[1]
        
        return a.copy().array()
        

    def muVecVec(self,x,mu):
        
        
        d = x.dot(self.data[0])
        d_mu = x.dot(self.data[1])
        return d + mu*d_mu

        
        

class MuRho():

    def __init__(self,sk,yk):

        self.sk = sk
        self.yk = yk

    def func(self,mu):
        
        Irho = self.yk.muVecVec(self.sk.data[0],mu)
        Irho = Irho + self.yk.muVecVec(mu*self.sk.data[1],mu)
        return 1./Irho
        
class MuVectors():

    def __init__(self,sk_u,sk_l,yk_u,ADJk,STAk,mu,Vec=SimpleVector):


        self.sk_u = sk_u
        self.sk_l = sk_l
        self.yk_u = yk_u
        self.ADJk = ADJk
        self.STAk = STAk
        self.mu   = mu
        self.Vec  = Vec
        

        self.sk = self.create_sk()
        self.yk = self.create_yk()

        self.Rho = self.create_Mu_Rho()


    def __len__(self):

        return len(self.sk_u) + len(self.sk_l)


    
    def create_Mu_Rho(self):

        sk_u = self.sk_u
        sk_l = self.sk_l
        yk_u = self.yk_u
        ADJk = self.ADJk
        STAk = self.STAk

        
        def F(mu):

            Irho = sk_u.dot(yk_u) + sk_l.dot(ADJk) - mu*sk_l.dot(STAk-sk_l)

            return 1./Irho

    def create_sk(self):

        n = len(self)
        n_u = len(self.sk_u)
        n_l = n-n_u
        
        s = np.zeros(n)
        s1 = np.zeros(n)
        s[:n_u] = self.sk_u.array()[:]
        s[n_u:] = self.sk_l.array()[:]

        return MuVector([self.Vec(s),self.Vec(s1)])


    def create_yk(self):

        n = len(self)
        n_u = len(self.sk_u)
        n_l = n-n_u

        y  = np.zeros(n)
        y1 = np.zeros(n)

        y[:n_u]  = self.yk_u.array()[:]
        y[n_u:]  = self.ADJk.array()[:]
        y1[n_u:] = (self.sk_l-self.STAk).array()[:]

        return MuVector([self.Vec(y),self.Vec(y1)])
        
if __name__ == "__main__":

    x = SimmpleVector(np.zeros(10))
    y = SimmpleVector(np.linspace(1,10,10))
    
    print x.array()

    x.__setitem__(2,1)
    
    print x.array()

    x.axpy(2,y)
    x.scale(0.5)
    print x.array(),x.size()

    x.set(np.zeros(10) +1)
    print y.array()
    print x.dot(y)


    I = np.identity(10)
    I[2,2]=0
    
    print x.matDot(I)
    x.matApy(I)
    print x.array()
    
