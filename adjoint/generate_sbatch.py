import sys

val = sys.argv[1]
name = sys.argv[2]

task = sys.argv[3]
node = sys.argv[4]
m = int(task)*int(node)
seq_par = sys.argv[5]
string = '''
print %s

name = '%s'
print name
''' %(val,name)

string2 = """#!/bin/bash
# Job name:
#SBATCH --job-name=testSpeed
#
# Project:
#SBATCH --account=uio
#
# Wall Clock limit:
#SBATCH --time=00:30:00
#
# Max memory usage (MB):
#SBATCH --mem-per-cpu=1000M
#
# Number of tasks (cores):
#SBATCH --ntasks-per-node=%s
#SBATCH --nodes=%s

## Set up job environment:
source /cluster/bin/jobsetup

mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;

mpiexec python func_grad_speedup_test.py %s 1 0 0;
rm temp_time%s_%s.txt;
rm temp_info%s_%s.txt;
"""%(task,node,val,val,val,val,val,val,val,val,val,val,val,m,val,m,val)

string3 = """#!/bin/bash
# Job name:
#SBATCH --job-name=testSpeed
#
# Project:
#SBATCH --account=uio
#
# Wall Clock limit:
#SBATCH --time=00:09:00
#
# Max memory usage (MB):
#SBATCH --mem-per-cpu=1000M
#
# Number of tasks (cores):
#SBATCH --ntasks-per-node=%s
#SBATCH --nodes=%s

## Set up job environment:
source /cluster/bin/jobsetup

mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;
mpiexec python func_grad_speedup_test.py %s 1 1 0;

mpiexec python func_grad_speedup_test.py %s 1 0 0;
mpiexec python func_grad_speedup_test.py %s 1 2 0;


rm temp_time%s_%s.txt;
rm temp_info%s_%s.txt;
"""%(task,node,val,val,val,val,val,val,val,val,val,val,val,val,m,val,m,val)


lel="""
mpiexec python func_grad_speedup_test.py %s 1 0 0;
mpiexec python func_grad_speedup_test.py %s 1 2 0;

rm temp_time.txt;
""" #%(task,node,val,val,val)

new_script = open(name,'w')

if seq_par == '0':
    print string2
    new_script.write(string2)

else:
    print string3
    new_script.write(string3)
new_script.close()
