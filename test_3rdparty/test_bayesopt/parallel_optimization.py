import scipy.optimize as so
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10 # for testing
step = N//size # say that N is divisible by size
# MPI wrapper for cost function
def parallel_function_caller(x, stop_all):
    stop_all[0] = comm.bcast(stop_all[0], root=0)
    summ=0
    if stop_all[0]==0:
        print('hello im a process:', rank)
        # cost function here
        x = comm.bcast(x, root=0)
        print('x is: ', x)
        array = np.arange(x[0]-N/2.+rank*step-42, x[0]-N/2.+(rank+1)*step-42, 1.)
        print('array is:', array)
        summl = np.sum(np.square(array))
        summ = comm.reduce(summl,op=MPI.SUM, root=0)
        if rank==0:
            print("cost = ", str(summ))
    return summ

if rank==0:
    stop = [0]
    x = np.zeros(1)
    x[0]=20
    xs = so.minimize(parallel_function_caller, x, args=(stop))
    print("argmin cost = \n ", str(xs))
    stop = [1]
    parallel_function_caller(x, stop)
    print('------ Finished ------')
else: 
    stop=[0]
    x=np.zeros(1)
    while stop[0]==0:
        parallel_function_caller(x, stop)


