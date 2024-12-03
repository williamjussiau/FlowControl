# # Bayesian Optimization 
# My own application of SMT-EGO

import numpy as np 
import scipy.optimize as so

## Use the EGO from SMT 
from smt.applications.ego import EGO
from smt.sampling_methods import LHS
import smt.surrogate_models

import matplotlib.pyplot as plt

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
        f = xlim
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
print('Beginning Optimization ..........................')
ndim = 1
use_smt = True# use smt or scipy

if use_smt:
    import optim_utils as ou
    def costfun_npt(x, **kwargs):
        '''Wrapper around costfun to fit the format of SMT.EGO
        This function is used as is in SMT.EGO'''
        return fun_npt(x, costfun, **kwargs)
    costfun_parallel_smt = lambda x: parallel_function_wrapper(x, [0], costfun_npt)

    xlimits = np.array([[-5, 5]])
    xlimits = np.repeat(xlimits, ndim, axis=0)
    
    #number of points in the initial DOE
    ndoe = 10 #(at least ndim+1)
    
    #number of iterations with EGO 
    n_iter = 12
    
    #Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
    sampling = LHS(xlimits=xlimits, random_state=1)
    xdoe = sampling(ndoe)
    ydoe = ou.fun_array(xdoe, fun=costfun) # or load already computed points 
    
    xdoe_additional = np.atleast_2d(np.array([-1, 0.443, 1, xdoe[0][0]])).T 
    xdoe_additional = np.array([[x for x in xdoe_additional[0] if x not in xdoe[0]]]) 
    ydoe_additional = ou.fun_array(xdoe_additional, costfun)
    
    xdoe = np.vstack((xdoe, xdoe_additional))
    ydoe = np.vstack((ydoe, ydoe_additional))
    
    #EGO calcriterion='EI' #'EI' or 'SBO' or 'LCB'
    import warnings
    warnings.filterwarnings('ignore')
    
    criterion = 'EI'
    surrogate = smt.surrogate_models.KRG(print_global=False, theta0=[0.01], n_start=20,
        #corr='matern52', 
        theta_bounds=[0.01, 20], poly='constant')
    ego = EGO(n_iter=n_iter, 
        criterion=criterion, xdoe=xdoe, ydoe=ydoe, xlimits=xlimits, 
        verbose=False, n_start=20, surrogate=surrogate)
        
        #x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=lambda x: parallel_function_wrapper(x, [0], costfun_npt))
        #x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_npt)

    
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
        res = so.minimize(costfun_parallel_scp,
            x0=x, method='Nelder-Mead', tol=1e-3, options=dict(maxiter=100))
        #res = so.minimize(costfun_parallel_scp, x0=x)
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
    
if rank==0 and use_smt:
    print('\n')
    print('---------------------------')
    print('Xopt for myfun: ', x_opt,y_opt )
    print('---------------------------')
   
    #if use_smt:
    #    pwd = '/stck/wjussiau/fenics-python/test/test_bayesopt/'
    #    import pandas as pd
    #    columns = ['y'] + ['x'+str(ii+1) for ii in range(ndim)]
    #    df = pd.DataFrame(columns=columns, data=np.hstack((y_data, x_data)))
    #    fname = pwd + 'ego_data_r%d.csv' %(rank)
    #    print('Exporting file to: ', fname)
    #    df.to_csv(fname, sep=',', index=False)
    #    
    #    x_true = np.atleast_2d(np.linspace(xlimits[0,0], xlimits[0,1], 2000)).T
    #    y_true = costfun_npt(x_true, verbose=False)
    #    df_true = pd.DataFrame(columns=columns, data=np.hstack((y_true, x_true)))
    #    df_true.to_csv(pwd + 'true_data.csv', sep=',', index=False)

    # Show GPR
    x = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 200)).T
    
    # KRG
    krg = surrogate

    ypred = krg.predict_values(x)
    yvar = krg.predict_variances(x)
    
    # EI
    crit = ego.EI(points=x, y_data=y_data, x_data=x_data)

    # Plot
    fig, ax = plt.subplots(1)
    # DOE
    ax.plot(xdoe, ydoe, 'o', color='black')
    ax.plot(x_data, y_data, '.', color='blue')
    # Prediction
    ax.plot(x, ypred)
    # Variance
    nsig = 3
    var_plus = ypred + nsig * np.sqrt(yvar)
    var_minus = ypred - nsig * np.sqrt(yvar)
    ax.fill_between(x.T[0], var_plus.T[0], var_minus.T[0], alpha=0.3, color="g")
    # Acquisition function (EI...)
    ax2 = ax.twinx()
    ax2.plot(x, crit, color='red')
    # Utils
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['DOE', 'Optim', 'KRG', 'Var'], loc='best')
    plt.savefig('show_KRG_after_EGO.png')
    plt.close('all') 
    




