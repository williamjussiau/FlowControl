'''
Utilitary functions for optimization
'''

import numpy as np 
import pandas as pd
from smt.sampling_methods import LHS
from smt.applications.ego import EGO
import smt.surrogate_models as smod
import sobol_seq as qmc 
import scipy.optimize as so
from blackbox_opt.bb_optimize import bb_optimize 
from blackbox_opt.DFO_src.dfo_tr import params

from mpi4py import MPI as mpi 
import time

comm = mpi.COMM_WORLD
rank = comm.Get_rank() # flu.MpiUtils.get_rank()
sz = comm.Get_size()

def fun_array(x, fun, **kwargs):
    """Evaluate function fun on n points at once
    x must be a 2d array (rows = nr of points, cols = dimension)
    """
    # pretty much every operation in here might be blocking
    # x = np.atleast_2d(x)
    # npt, dim = x.shape # fails in dim 1
    #if type(x) is int:
    #    dim = 1
    #    npt = 1
    #else:
    #    dim = len(x.shape)
    #    npt = x.shape[0]
    #print('fun_array::x = ', x, type(x))
    npt = x.shape[0]
    out = np.zeros((npt, 1))
    for i in range(npt):
        #if dim==1:
        #    J = fun(x[i], **kwargs)
        #else:
        J = fun(x[i, :], **kwargs) 
        #if type(J) is tuple:
        #    J = J[0]
        out[i, :] = J
    return out


def costfun(x, verbose=True, allout=False):
    '''Evaluate cost function on one point x'''
    #f = sum((x-1)**2 + 4*np.cos(x) + 2*np.cos(2*x) + 10*np.cos(10*x))
    xlim = 1
    if x<=xlim and x>=-xlim:
        f = x**2
    else:
        f = xlim
    if verbose:
        print('costfun: evaluation %2.10f +++ %2.10f' %(x, f) )
    if allout:
        return f, 777
    return f 
    

def parallel_function_wrapper(x, stop_all, fun):#, **kwargs):
    '''Allows for the evaluation of fun in parallel in an outer process
    Use case is displayed below'''
    stop_all[0] = comm.bcast(stop_all[0], root=0)
    f = 0
    x = comm.bcast(x, root=0)
    #print('parallel wrapper::x=', x, type(x))
    #print('parallel wrapper::stop=', stop_all, type(stop_all))
    if stop_all[0]==0:
        # cost function here
        fe = fun(x)#, **kwargs)
        f = comm.reduce(fe/sz, op=mpi.SUM, root=0) # op = MPI.MAX? 
        #print('type of f is:', type(f))
        if rank==0:
            print("from rank 0: arg=", x, ' >>> cost=', f)
    else:
        print('##### Stopping function evaluation on process: ', rank)
    return f


def construct_simplex(x0, rectangular=True, edgelen=1):
    '''Construct simplex around x0 to initialize Nelder-Mead algorithm
    A rectangular simplex has x0 as a vertex and edgelen[i]*e_i as edges
    x0 should be numpy.ndarray
    scipy and Matlab do as follows: rectangular simplex with edge k
    vertex being x0[k]*1.05, except if x0 is 0, then 0.00025'''
    x0 = x0.ravel()
    n = x0.shape[0]

    # transform edgelen to list if necessary
    if type(edgelen) is float:
        edgelen = [edgelen]*n
    
    # rectangular simplex, x0+e_i
    if rectangular:
        simplex = np.zeros((n+1, n))
        # fill first vertex with x0
        simplex[0] = x0
        # fill (2,n+1) vertices with x0 + edgelen*e_(i-1)
        for ii in range(1, n+1):
            e_i = np.zeros((n,))
            e_i[ii-1] = 1
            simplex[ii] = x0 + e_i * edgelen[ii-1]
    # nonrectangular simplex, centered around x0
    # still rectangular though, but x0 is not on an edge
    else:
        simplex = np.vstack((np.zeros((1,n)), np.diag(edgelen)))
        a = 1/(n+1)
        # this simplex has center a=1/(n+1) repeated on each axis
        # shift it so that its center is x0
        simplex = simplex - a + x0  

    return simplex


def nm_select_evaluated_points(x_best, x_all, y_all, verbose=False):
    '''For NM algorithm: from best-so-far points x_best
     retrieve corresponding value of cost function
     For x in x_best: find x in x_all, get index, associate y_all[index]'''
    # remove unique values from allvec
    uidx = np.unique(x_best, axis=0, return_index=True)[1]
    alv = [x_best[index] for index in sorted(uidx)]
    
    x_good = alv
    y_good = [0]*len(x_good)
    # loop on allvecs
    for ii, el in enumerate(x_good):
        # retrieve value of cost function
        for jj in range(len(x_all)):
            if np.all(x_all[jj]==el):
                if verbose:
                    print('Best-so-far: idx=', jj, ' - value=', y_all[jj])
                y_good[ii] = y_all[jj]

    return x_good, y_good


def cummin(y, return_index=True):
    '''Return cumulative minimum of 1D array y, along with indices of cummin'''
    # get cummin of slice
    y_cummin = np.minimum.accumulate(y)
    if return_index:
        where_cummin = np.isclose(y_cummin, y.T).astype(int)
        idx = where_cummin.argmax(1)
        return y_cummin, idx
    return y_cummin


def minimize(costfun, x0, alg, options, verbose=True):
    '''Wrapper for launching optimization algorithm
    Supported algorithms:
        Scipy: NM, COBYLA, BFGS, SLSQP
        DFO
        BO
    TODO: return all?
    '''
    # Start timing
    tstart = time.time()
    # Get algorithm in lower case and default options
    alg = alg.lower()
    default_options = optimizer_default_options(alg=alg)
    options['disp'] = verbose
    options = optimizer_check_options(default_options, options)
    #options = {**default_options, **options} # replaced
    # Perform optim with given algo and options
    if alg=='nm':
        res = so.minimize(fun=costfun, x0=x0, method='Nelder-Mead', 
            options=options)
    if alg=='cobyla':
        res = so.minimize(fun=costfun, x0=x0, method='COBYLA', 
            options=options)
    if alg=='bfgs':
        # warning: niter does not count gradient finite diff 
        res = so.minimize(fun=costfun, x0=x0, method='BFGS', 
            options=options)
    if alg=='slsqp':
        # warning: niter does not count gradient finite diff 
        res = so.minimize(fun=costfun, x0=x0, method='SLSQP', 
            options=options)
    if alg=='dfo':
        # keep keys listed in class bb_optimization.DFO_src.dfo_tr.params
        #dfo_params = list(params().__dict__.keys())
        #options = {key: options[key] for key in dfo_params if key in options} # replaced
        res = bb_optimize(func=costfun, x_0=x0, alg='DFO', options=options)
        res.nfev = res.func_eval
    if alg=='bo':
        #Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
        sampling = LHS(xlimits=options['xlimits'], random_state=options['random_state'])
        xdoe = sampling(options['n_doe'])
        ydoe = fun_array(xdoe, costfun)
        
        # Create GP objects
        criterion = options['criterion']
        surrogate = smod.KRG(print_global=False, 
            theta0=options['theta0'], n_start=options['n_start'], corr=options['corr'], 
            theta_bounds=options['theta_bounds'], poly=options['poly'])
        ego = EGO(n_iter=options['n_iter'], 
            criterion=options['criterion'], xdoe=xdoe, ydoe=ydoe,
            xlimits=options['xlimits'], verbose=options['verbose'], 
            n_start=options['n_start'], surrogate=surrogate)
            
        # Wrap costfun
        costfun_npt = lambda x: fun_array(x, costfun)
        costfun_parallel_smt = lambda x: parallel_function_wrapper(x, [0], costfun_npt)
        
        if rank==0:
            stop=[0]
            x=np.zeros(1)
            x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_parallel_smt)
            #print("argmin cost = \n ", str(x_opt))
            stop = [1]
            parallel_function_wrapper(x, stop, costfun_npt)
            #print('------ Finished ------')
        else: 
            stop=[0]
            x=np.zeros(1)
            while stop[0]==0:
                parallel_function_wrapper(x, stop, costfun_npt)
        #res = dict(nfev=options['n_doe']+options['n_iter'])
        # inline class definition
        res = type('obj', (object,), {'nfev': options['n_doe']+options['n_iter']})
        #dict(x_data=x_data, y_data=y_data, x_opt=x_opt, y_opt=y_opt,ind_best=ind_best)
        # but variables are mostly local ---> broadcast or write?
    # End timing
    tend = time.time()
    if verbose:
        print("Total time is {} seconds with ".format(tend - tstart) + alg + (
            " method."))
    return res


def optimizer_default_options(alg):
    '''Define default algorithm parameters
    Display help to show parameters of each algorithm
    Use is: "I want to use algorithm alg, so I build 
    this default options dictionary and potentially modify
    some of the parameters"
    ----------------------------------------------------------------
    if alg=='nm':
        options = {'maxiter': None, 'maxfev': maxfev,
            'disp': False, 'return_all': True, 'initial_simplex': None,
            'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
    if alg=='cobyla':
        # niter is nfev
        options = {'rhobeg': 1.0, 'maxiter': maxfev, 'disp': False, 'catol': 0.0002}
    if alg=='bfgs':
        # niter is not nfev
        options = {'gtol': 1e-05, 'norm': inf, 'eps': 1.4901161193847656e-08,
        'maxiter': maxfev, 'disp': True, 'return_all': True, 'finite_diff_rel_step': None}
    if alg=='slsqp':
        # niter is not nfev
        options = {'maxiter': maxfev, 'ftol': 1e-06, 'iprint': 1,
        'disp': True, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
    if alg=='dfo':
        tol = 1e-6
        options = dict(maxfev=maxfev, init_delta=1, tol_delta=tol, 
        tol_f=tol, tol_norm_g=tol, sample_gen='auto', verbosity=1)
    '''
    alg = alg.lower()
    maxfev=100
    if alg=='nm':
        options = {'maxiter': None, 'maxfev': maxfev,
            'disp': False, 'return_all': True, 'initial_simplex': None,
            'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
    if alg=='cobyla':
        # niter is nfev
        options = {'rhobeg': 1.0, 'maxiter': maxfev, 'disp': False, 'catol': 0.0002}
    if alg=='bfgs':
        # niter is not nfev
        options = {'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08,
            'maxiter': maxfev, 'disp': True, 'return_all': True, 
            'finite_diff_rel_step': None}
    if alg=='slsqp':
        # niter is not nfev
        options = {'maxiter': maxfev, 'ftol': 1e-06, 'iprint': 1,
        'disp': True, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
    if alg=='dfo':
        tol = 1e-6
        options = dict(maxfev=maxfev, init_delta=0.5, tol_delta=tol, 
        tol_f=1e-4, tol_norm_g=tol, sample_gen='auto', verbosity=1)
        # true defaults are:
        #init_delta = 1.0  # initial delta (i.e. trust region radius)                                         
        #tol_delta = 1e-10  # smallest delta to stop                                                          
        #max_delta = 100.0  # max possible delta                                                              
        #gamma1 = 0.8  # radius shrink factor                                                                 
        #gamma2 = 1.5  # radius expansion factor                                                              
        #eta0 = 0.0   # step acceptance test (pred/ared) threshold                                 
        #eta1 = 0.25  # (pred/ared) level to shrink radius                                                    
        #eta2 = 0.75  # (pred/ared) level to expand radius                                                    
        #tol_f = 1e-15  # threshold for abs(fprev - fnew)- used to stop                            
        #tol_norm_g = 1e-15  # threshold for norm of gradient- used to stop                                   
        #tol_norm_H = 1e-10  # threshold for the (frobenius) norm of H                                        
        #maxfev = 1000  # maximum number of iterations                                                        
        #min_del_crit = 1e-8  # minimum delta for the criticality step                             
        #min_s_crit = 0.1  # minimum step for the criticality step    
    if alg=='bo':
        options = dict(theta0=[0.01], n_start=20,
            corr='squar_exp', theta_bounds=[0.01, 20], poly='constant',
            #corr='matern52'
            n_iter=10, criterion='EI', xlimits=[], verbose=False,
            random_state=1, n_doe=10)
    return options 


def optimizer_check_options(default_options, options):
    '''For given optimizer, keep entries in options dictionary {options}
    contained in dictionary {default_options}
    This is used to prevent scipy errors when an option is incompatible
    with an algorithm (e.g. initial simplex for BFGS)'''
    # Loop over keys in opts, and keep only keys that are also in default_opts
    new_options = default_options
    for key in options.keys():
        if key in new_options.keys():
            new_options[key] = options[key]
    return new_options


def get_results(func, x_0, alg, options):
    """
    This function calls the main blackbox optimization
    code with the specified algorithm, options and starting point and
    prints the best point found along with its optimal value.
    input: func: an imported blackbox function object
           x_0: starting point: numpy array with shape (n,1)
           alg: selected algorithm: string
           options : a dictionary of options customized for each algorithm
    """
    res = optimize(func, x_0, alg, options)

    print("Printing result for function " + func.__name__ + ":")
    print("best point: {}, with obj: {:.6f}".format(
        np.around(res.x.T, decimals=5), float(res.fun)))
    if alg.lower()=="dfo":
        nf = res.func_eval
    else:
        nf = res.nfev
    print('nr f evaluations: ', nf)
    print("------------- " + alg + " Finished ----------------------\n")
    return res


def write_results(x_data, y_data, optim_path, colnames, 
        x_cummin_all=None, y_cummin_all=None, idx_current_slice=0, nfev=0,
        verbose=True):
    '''Write optimization results to csv file
    Write all iterations and compute cummin'''
    # Format data from list to np.array
    x_data_wr = np.array(x_data)
    y_data_wr = np.atleast_2d(np.array(y_data)).T

    # Make DataFrame
    df = pd.DataFrame(data=np.hstack((y_data_wr, x_data_wr)), columns=colnames)
    if verbose:
        print('Logging J file to: ', optim_path)
    # Write file
    df.to_csv(optim_path + 'J_costfun.csv', sep=',', index=False, mode='w', header=True)

    if x_cummin_all is not None:
        # Compute cumulative minimum
        # Find cumulative minimum of y_data and corresponding parameters x_data
        y_data_i = y_data_wr[idx_current_slice:idx_current_slice+nfev]
        # Get cummin of slice and idx
        y_cummin, idx_cummin = cummin(y_data_i, return_index=True)
        # Find corresponding param 
        x_cummin = x_data_wr[idx_cummin+idx_current_slice, :]
        # Increment slice 
        idx_current_slice = idx_current_slice + nfev
        # append to old cummin
        x_cummin_all = np.vstack((x_cummin_all, x_cummin))
        y_cummin_all = np.vstack((y_cummin_all, y_cummin))
        # write dataframe
        df_cummin = pd.DataFrame(data=np.hstack((y_cummin_all, x_cummin_all)), columns=colnames)
        df_cummin.to_csv(optim_path + 'J_costfun_cummin.csv', sep=',', index=False, mode='w', header=True)
        return x_cummin_all, y_cummin_all, idx_current_slice


def run_parallel(rank, costfun_parallel):
    pass
    # launch function in parallel parallel
    #if rank==0:
        #    stop = [0]
        #    res = so.minimize(costfun_parallel_scp,
        #        x0=x, method='Nelder-Mead', tol=1e-2, 
        #        options=dict(maxfev=300, adaptive=True, initial_simplex=initial_simplex))
        #    x_opt, y_opt = res.x, res.fun
        #    print("argmin = \n ", str(x_opt))
        #    stop = [1]
        #    optim_utils.parallel_function_wrapper(x, stop_all=stop, fun=costfun)
        #    print('------ Finished ------')
        #else: 
        #    stop=[0]
        #    x=np.zeros(1)
        #    while stop[0]==0:
        #        optim_utils.parallel_function_wrapper(x, stop_all=stop, fun=costfun)


def sobol_sample(ndim, npt, xlimits=None, skip=1e3, shuffle=False):
    '''Generate samples from Sobol set
    ndim: size of points
    npt: number of points
    xlimits: limits of generated points (initially in [0,1]) as array 
    skip: skip first elements of Sobol sequence
    shuffle: (int) modify skip variable to mimic shuffling'''
    # Modify skip variable to get another set of Sobol points
    if shuffle: # False or 0
        np.random.seed(shuffle)
        skip += np.random.randint(1e4)
    # Generate points in [0, 1]
    X = qmc.i4_sobol_generate(dim_num=ndim, n=npt, skip=skip) 
    # Scale points to xlimits
    if xlimits is not None:
        xlimits = np.array(xlimits)
        # Check size
        if xlimits.shape==(2, ndim):
            xlimits = xlimits.T
        try:
            # Scale
            X *= (xlimits[:, 1] - xlimits[:, 0])
            X += xlimits[:, 0]
        except:
            print("xlimits has wrong size: ", xlimits.shape, " - should be (ndim, 2).")
    return X



if __name__=='__main__':
    print('Hello I am process -------------------------------------------------- ', rank)

    use_smt = True# use smt or scipy

    if use_smt:
        print('Beginning Bayesian Optimization ..........................')
        ndim = 1
        xlimits = np.array([[-5, 5]])
        xlimits = np.repeat(xlimits, ndim, axis=0)
        #number of points in the initial DOE
        ndoe = 6 #(at least ndim+1)
        #number of iterations with EGO 
        n_iter = 3
        #Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
        sampling = LHS(xlimits=xlimits, random_state=1)
        xdoe = sampling(ndoe)
        
        import warnings
        warnings.filterwarnings('ignore') # remove warnings in 1D
        
        criterion = 'SBO'
        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits, verbose=False)
        
        def costfun_array(x, **kwargs):
            '''Wrapper around costfun to fit the format of SMT.EGO
            This function is used as is in SMT.EGO'''
            return fun_array(x, costfun, **kwargs)
        costfun_parallel_smt = lambda x: parallel_function_wrapper(x, [0], costfun_array)
        if rank==0:
            stop = [0]
            x = np.zeros(1)
            x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=costfun_parallel_smt)
            print("argmin cost = \n ", str(x_opt))
            stop = [1]
            parallel_function_wrapper(x, stop, costfun_array)
            print('------ Finished ------')
        else: 
            stop=[0]
            x=np.zeros(1)
            while stop[0]==0:
                parallel_function_wrapper(x, stop, costfun_array)

        warnings.filterwarnings('default')
    else: 
        print('Beginning scipy.optimize..........................')
        import scipy.optimize as so
        costfun_parallel_scp = lambda x: parallel_function_wrapper(x, [0], costfun, allout=False)
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
       



