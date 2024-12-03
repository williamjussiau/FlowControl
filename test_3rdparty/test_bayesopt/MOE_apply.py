# # Bayesian Optimization 
# My own application of SMT-EGO

import sys
import numpy as np 
import scipy.optimize as so

## Use the EGO from SMT 
from smt.applications.ego import EGO
from smt.sampling_methods import LHS
import smt.surrogate_models
import smt.applications

# * Choose your criterion to perform the optimization: EI, SBO or LCB
# * Choose the size of the initial DOE
# * Choose the number of EGO iterations

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
    xlim = 1
    if x<=xlim and x>=-xlim:
        f = x**2
    else:
        f = xlim+np.abs(x)
    if verbose:
        print('costfun: evaluation %2.10f +++ %2.10f' %(x, f) )
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



##################################################################
from mpi4py import MPI as mpi 
comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # flu.MpiUtils.get_rank()


print()
print('Beginning Bayesian Optimization ..........................')
ndim = 1
xlimits = np.array([[-5, 5]])
xlimits = np.repeat(xlimits, ndim, axis=0)

#number of points in the initial DOE
ndoe = 25 #(at least ndim+1)

#number of iterations with EGO 
n_iter = 10


x = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 1000)).T
y = fun_npt(x, costfun)

#Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
sampling = LHS(xlimits=xlimits, random_state=1)
xdoe = sampling(ndoe)
ydoe = fun_npt(xdoe, costfun) # or load already computed points 

# MOE 1
from smt.applications import MOE
moe1 = MOE(n_clusters=1)
print('MOE1 enabled experts: ', moe1.enabled_experts)
moe1.set_training_values(xdoe, ydoe)
moe1.train()
y_moe1 = moe1.predict_values(x)

# MOE 2: several clusters
moe2 = MOE(smooth_recombination=False, n_clusters=3, allow=['KRG', 'LS', 'IDW'])
print('MOE2 enabled experts: ', moe2.enabled_experts)
moe2.set_training_values(xdoe, ydoe)
moe2.train()
y_moe2 = moe2.predict_values(x)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax.plot(x, y, '--', color='black')
ax.plot(xdoe, ydoe, '.')
ax.plot(x, y_moe1)
ax.plot(x, y_moe2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(['True', 'DOE', 'MOE1', 'MOE2'])
plt.savefig('MOE.png')



sys.exit()






#EGO calcriterion='EI' #'EI' or 'SBO' or 'LCB'
import warnings
warnings.filterwarnings('ignore')

criterion = 'SBO'
#surrogate = smt.surrogate_models.KRG(print_global=False)
ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, ydoe=ydoe, xlimits=xlimits, verbose=False)
#x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=lambda x: parallel_function_wrapper(x, [0], costfun_npt))
#x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_npt)



use_smt = True# use smt or scipy

if use_smt:
    def costfun_npt(x, **kwargs):
        '''Wrapper around costfun to fit the format of SMT.EGO
        This function is used as is in SMT.EGO'''
        return fun_npt(x, costfun, **kwargs)
    costfun_parallel_smt = lambda x: parallel_function_wrapper(x, [0], costfun_npt)
    
    # for naive run
    #costfun_parallel_smt = lambda x: costfun_npt(x, verbose=True)
    #x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_parallel_smt)

    if rank==0:
        stop = [0]
        x = np.zeros(1)
        x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_parallel_smt)
        print("argmin cost = \n ", str(x_opt))
        stop = [1]
        parallel_function_wrapper(x, stop, costfun_npt)
        print('------ Finished ------')
    else: 
        stop=[0]
        x=np.zeros(1)
        while stop[0]==0:
            parallel_function_wrapper(x, stop, costfun_npt)
else: 
    costfun_parallel_scp = lambda x: parallel_function_wrapper(x, [0], costfun)

    # for naive run
    #costfun_parallel_scp = lambda x: costfun(x)
    #x = np.zeros(1)
    #x[0] = 0.9
    #res = so.minimize(costfun_parallel_scp, x0=x)
    
    if rank==0:
        stop = [0]
        x = np.zeros(1)
        x[0] = 0.9
        res = so.minimize(costfun_parallel_scp, x0=x)
        x_opt, y_opt = res.x, res.fun
        print("argmin cost = \n ", str(x_opt))
        stop = [1]
        parallel_function_wrapper(x, stop, costfun)
        print('------ Finished ------')
    else: 
        stop=[0]
        x=np.zeros(1)
        while stop[0]==0:
            parallel_function_wrapper(x, stop, costfun)
    
if rank==0:
    print('\n')
    print('---------------------------')
    print('Xopt for myfun: ', x_opt,y_opt )
    print('---------------------------')
   
    if use_smt:
        pwd = '/stck/wjussiau/fenics-python/test/test_bayesopt/'
        import pandas as pd
        columns = ['y'] + ['x'+str(ii+1) for ii in range(ndim)]
        df = pd.DataFrame(columns=columns, data=np.hstack((y_data, x_data)))
        fname = pwd + 'ego_data_r%d.csv' %(rank)
        print('Exporting file to: ', fname)
        df.to_csv(fname, sep=',', index=False)
        
        
        x_true = np.atleast_2d(np.linspace(xlimits[0,0], xlimits[0,1], 2000)).T
        y_true = costfun_npt(x_true, verbose=False)
        df_true = pd.DataFrame(columns=columns, data=np.hstack((y_true, x_true)))
        df_true.to_csv(pwd + 'true_data.csv', sep=',', index=False)












