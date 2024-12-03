# # Bayesian Optimization 
# My own application of SMT-EGO

import numpy as np 
import scipy.optimize as so

x_data = []
y_data = []

def fun_npt(x, fun, **kwargs):
    """Evaluate function fun on n points at once
    x must be a 2d array (rows = nr of points, cols = dimension)
    """
    npt, dim = x.shape
    out = np.zeros((npt, 1))
    for i in range(npt):
        out[i, :] = fun(x[i,:], **kwargs) 
    return out


def costfun(x, verbose=True):
    '''Evaluate cost function on one point x'''
    #f = sum((x-1)**2 + 4*np.cos(x) + 2*np.cos(2*x) + 10*np.cos(10*x))
    f = (x[0]-1)**2 + (x[1]-2)**2
    #xlim = 20
    #if x<=xlim and x>=-xlim:
    #    f = (x[0]-1)**2 + (x[1]-2)**2
    #else:
    #    f = xlim
    if verbose:
        print('costfun: evaluation ', x, f)
    global x_data, y_data
    x_data += [x]
    y_data += [f]
    return f 
    

def parallel_function_wrapper(x, stop_all, fun):
    stop_all[0] = comm.bcast(stop_all[0], root=0)
    f = 0
    if stop_all[0]==0:
        print('from parallel process:', rank)
        # cost function here
        x = comm.bcast(x, root=0)
        f = comm.reduce(fun(x) , op=mpi.SUM, root=0)
        if rank==0:
            print("from rank 0:  ", x, f)
    return f


#def callback(x):
#    print('in callback: ', x)


##################################################################
from mpi4py import MPI as mpi 
comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # flu.MpiUtils.get_rank()


print()
print('Beginning Optimization ..........................')

costfun_parallel_scp = lambda x: parallel_function_wrapper(x, [0], costfun)

x = [0.9, 0.3]
initial_simplex = [[0.3, 0.2],[0.3, 0.1],[0.2, 0.15]]
tol = 1e-2
return_all = True
disp = True
maxfev = 100
adaptive = True

## semi-naive run
print('**** Naive run')
costfun_parallel_scp = lambda x: costfun(x)
res = so.minimize(costfun_parallel_scp,
    x0=x, method='Nelder-Mead', tol=tol, 
    options=dict(maxfev=100, adaptive=adaptive, initial_simplex=initial_simplex,
    disp=disp, return_all=return_all),)
#    callback=callback)

# store retained values of cost fun
# remove unique values from allvec
uidx = np.unique(res.allvecs, axis=0, return_index=True)[1]
alv = [res.allvecs[index] for index in sorted(uidx)]

x_good = alv
y_good = [0]*len(x_good)
# loop on allvecs
for ii, el in enumerate(x_good):
    # retrieve value of cost function
    for jj in range(len(x_data)):
        if np.all(x_data[jj]==el):
            print('Allvecs from NM: elt', jj)
            y_good[ii] = y_data[jj]


##################################################
#print('**** Smart run')
#if rank==0:
#    stop = [0]
#    res = so.minimize(costfun_parallel_scp,
#        x0=x, method='Nelder-Mead', tol=1e-2, 
#        options=dict(maxfev=maxfev, adaptive=adaptive, initial_simplex=initial_simplex,
#        disp=disp, return_all=return_all))
#    x_opt, y_opt = res.x, res.fun
#    print("argmin = \n ", str(x_opt))
#    stop = [1]
#    parallel_function_wrapper(x, stop, costfun)
#    print('------ Finished ------')
#else: 
#    stop=[0]
#    x=np.zeros(1)
#    while stop[0]==0:
#        parallel_function_wrapper(x, stop, costfun)
#    


