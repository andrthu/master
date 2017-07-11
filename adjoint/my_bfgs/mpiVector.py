import numpy as np
from mpi4py import MPI

class MPIVector():

    def __init__(self,vec,comm):
        self.local_vec = vec
        self.comm = comm
        self.length = None
        
    def __add__(self,other):
        #print type(self.local_vec),type(other.local_vec)
        #return MPIVector(self.local_vec + other.local_vec,self.comm)
        return MPIVector(self[:]+other[:],self.comm)  

    def __sub__(self,other):
        #return MPIVector(self.local_vec - other.local_vec,self.comm)
        return MPIVector(self[:] - other[:],self.comm)  
    def __neg__(self):
        return -1*self

    def __mul__(self,a):
        #v = self.local_vec.copy()*a
        return MPIVector(a*self.local_vec,self.comm)
    __rmul__=__mul__

    def dot(self,other):
        comm = self.comm       
        local_res = np.array([np.sum(self.local_vec*other.local_vec)])
        #print local_res
        global_res = np.zeros(1)
        comm.Allreduce(local_res,global_res,op=MPI.SUM)
        return global_res[0]
        
    def __str__(self):
        return str(self.local_vec)
    def __len__(self):
        return len(self.local_vec)
    def __getitem__(self,i):
        return self.local_vec[i]
    def __setitem__(self,i,val):
        self.local_vec[i] = val
    def __abs__(self):
        return abs(self.local_vec)

    def copy(self):
        return MPIVector(self.local_vec.copy(),self.comm)
    
    def global_length(self):
        
        if self.length!=None:
            return self.length
        
        local_length = np.zeros(1)
        local_length[0] = len(self)
        global_length = np.zeros(1)

        self.comm.Allreduce(local_length,global_length,op=MPI.SUM)
        self.length = int(global_length[0])
        return int(global_length[0])

    def l2_norm(self):

        l = self.global_length()
        local_norm = np.zeros(1)
        local_norm[0] = np.sum(self.local_vec**2)/l

        global_norm = np.zeros(1)

        self.comm.Allreduce(local_norm,global_norm,op=MPI.SUM)

        return np.sqrt(global_norm[0])
    def linf_norm(self):

        local_norm = np.zeros(1)
        local_norm[0] = np.max(abs(self))
        global_norm = np.zeros(1)

        self.comm.Allreduce(local_norm,global_norm,op=MPI.MAX)
        
        return global_norm[0]

        

    def gather_control(self):
        
        comm = self.comm
        m = comm.Get_size()
        rank = comm.Get_rank()
        N = self.global_length()

        lam = np.zeros(m)
        control_length = np.zeros(m)
        if rank==0:
            local_lam = np.zeros(1)
            local_length = np.zeros(1)
            local_length[0] = len(self)
            loc_control = self[:]
        else:
            local_lam = np.zeros(1)
            local_lam[0] = self[-1]
            local_length = np.zeros(1)
            local_length[0] = len(self)-1
            loc_control=self[:-1]
        
        length = tuple(np.zeros(m)+1)
        start = tuple(np.linspace(0,m-1,m))
        comm.Allgatherv(local_lam,[lam,length,start,MPI.DOUBLE])
        comm.Allgatherv(local_length,[control_length,length,start,MPI.DOUBLE])
        
        
        global_control = np.zeros(N+1-m)
        start = np.zeros(m)
        for i in range(m-1):
            start[i+1]= start[i]+control_length[i]

        comm.Allgatherv(loc_control,[global_control,tuple(control_length),tuple(start),MPI.DOUBLE])
        
        control = np.zeros(N)
        control[:N+1-m] = global_control[:]
        control[N+1-m:] = lam[1:]

        return control


def test_mpivec():

    comm = MPI.COMM_WORLD
    n = 10
    rank = comm.Get_rank()

    a = (rank+1)*(np.zeros(n)+1)
    b = rank*(np.zeros(n)+1)
    v = MPIVector(a,comm)
    u = MPIVector(b,comm)
    
    if rank==0:
        print 'sum'
    print u+v
    comm.Barrier()
    print
    if rank==0:
        print 'scalar multiplication:'
    print 2*v
    comm.Barrier()
    print
    if rank==0:
        print 'dot prod:'
    print v.dot(v)
    
    comm.Barrier()
    print
    if rank==0:
        print 'length:'
    print len(v)


if __name__=='__main__':
    test_mpivec()
    
