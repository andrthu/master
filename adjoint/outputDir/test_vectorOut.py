import numpy as np
from mpi4py import MPI
import sys

def write_Vector():

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    name = 'vectorOut/test_vec'+str(rank)+'.npy'

    x = np.zeros(10)+rank+1

    np.save(name,x)
    
def read_Vector(m=3):

    for i in range(m):
        name = 'vectorOut/test_vec'+str(i)+'.npy'
        v = np.load(name)
        print v
    
    
def main():

    try:
        a=sys.argv[1]
        if a=='0':
            write_Vector()
        else:
            read_Vector()
            
    except:
        write_Vector()

if __name__=='__main__':
    main()
