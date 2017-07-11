#!/bin/bash
COUNTER=2;
N=$1;
mpiexec -n 1 -mca btl ^openib python speedup_mpi.py $N 0;
echo "First";
while [  $COUNTER -lt 13 ]; do
    mpiexec -n $COUNTER -mca btl ^openib python speedup_mpi.py $N 1;
    let COUNTER=COUNTER+1;
    echo "new $COUNTER"
done

