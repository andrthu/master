import numpy as np
from mpiVector import MPIVector

class DiagonalMatrix():

    def __init__(self,n,diag=None,mpi=None):
        
        self.n = n
        if diag==None:
            self.diag = np.zeros(n) +1
        else:
            self.diag = diag

        if mpi!=None:
            self.diag = MPIVector(self.diag,mpi)
    def __call__(self,x):
        
        return self.diag[:]*x[:]

class DiagonalMpiMatrix():

    def __init__(self,n,comm,diag=None):
        
        self.n = n
        if diag==None:
            self.diag = MPIVector(np.zeros(n) +1,comm)
        else:
            self.diag = MPIVector(diag,comm)

        
    def __call__(self,x):
        
        y=self.diag*x.local_vec
        #print type(y.local_vec.local_vec)
        return y

if __name__ =='__main__':
    n =10
    d = np.linspace(0,5,n)
    D = DiagonalMatrix(n,diag=d)

    x = np.zeros(n)+1
    print D(x)
