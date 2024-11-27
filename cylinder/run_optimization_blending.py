"""
----------------------------------------------------------------------
Run optimization
Youla
Q = Blending(multiK)
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import main_flowsolver as flo
import utils_flowsolver as flu
import importlib
importlib.reload(flu)
importlib.reload(flo)
print = flu.print0

from scipy import signal as ss
import scipy.io as sio 
import scipy.optimize as so

import youla_utils
import optim_utils

flo.set_log_level(flo.LogLevel.ERROR) # DEBUG TRACE PROGRESS INFO WARNING CRITICAL ERROR

import pdb
import sys
import warnings
import os
import getopt

x_data = []
y_data = []

######################################################################
def make_controller(G, K0, x, scaling=None, Qi=None):
    #rtol = 1e-8

    #x = x / np.linalg.norm(x, ord=2)

    # Make Q = lincomb(x, Qi)
    QQ = sum([x[i] * Qi[i] for i in range(len(x))]) 
    #QQ = youla_utils.control.minreal(QQ, tol=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        QQ = youla_utils.balreal(QQ)

    # Insert in Youla parametrization
    #K = youla_utils.youla_left_coprime(G, K0, QQ)
    K = youla_utils.youla(G, K0, QQ)
    #K = youla_utils.control.minreal(K, tol=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K = youla_utils.balreal(K)

    print('controller K has order: ', K.A.shape[0])
    
    # Check stability (for numerical issues)
    print('feedback is stable: ', youla_utils.isstablecl(G, K, 1)) 
    return K 


def eval_controller(G, K0, x, criterion,
                    params_flow, params_time, params_save, params_solver, params_mesh,
                    verbose=False, Kss=None, write_csv=False, scaling=None, Qi=None):
    # Ensure x is 1D
    x = x.reshape(-1,)

    # Build controller
    Kss = make_controller(G, K0, x=x, scaling=scaling, Qi=Qi)

    # Initialize FlowSolver
    fs = flo.FlowSolver(params_flow=params_flow, params_time=params_time, params_save=params_save,
                        params_solver=params_solver, params_mesh=params_mesh, verbose=verbose)
    fs.load_steady_state(assign=True)
    fs.init_time_stepping()

    # Initialize time loop
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    x_ctrl = np.zeros((Kss.nstates,))
    diverged = False
    for i in range(fs.num_steps):
        if fs.t>=fs.Tc:
            # measurement
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            y_meas_err = np.array([y_steady - y_meas])
            # step controller
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            # saturation
            u_ctrl = flu.saturate(u_ctrl, -2, 2)
        # step fluid
        if fs.perturbations:
            ret = fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            ret = fs.step(u_ctrl)
        # or step perturbation!!! big mistake here
        if ret==-1:
            print('Problem in solver -- exiting loop...')
            diverged = True
            break
          
    ## TODO
    # x**2
    J = flu.compute_cost(fs=fs, criterion=criterion, u_penalty=0.1 * 5e-2, fullstate=True, verbose=True,
        diverged=diverged, diverged_penalty=50)
    J *= 100 

    # Write CSV
    flu.write_optim_csv(fs, x, J, diverged=diverged, write=write_csv)

    global x_data, y_data
    x_data += [x.copy()]
    y_data += [J]

    return J, fs, Kss


def main(argv):
    flu.MpiUtils.check_process_rank()

    # Process argv
    ######################################################################
    k_arg = None
    opts, args = getopt.getopt(argv, "k:") # controller nr
    for opt, arg in opts:
        if opt=='-k':
            k_arg = arg

    # Flow parameters
    ######################################################################
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0, 
                 'sensor_location': np.array([[3.0, 0.0]]), 
                 'sensor_type': ['v'], 
                 'actuator_angular_size': 10,}
    params_time={'dt': 0.005, 
                 'Tstart': 0, 
                 'num_steps': 40000,
                 'Tc': 0.0} 
    params_save={'save_every': 2000,
                 'save_every_old': 2000,
                 'savedir0': '',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 0}
    params_mesh = flu.make_mesh_param('o1')


    # Control parameters
    ### BLENDING ###
    ######################################################################
    sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/multiK/'
    G = flu.read_ss(sspath + 'G.mat')
    K0 = flu.read_ss(sspath + 'K_0.mat')
    Gstab = youla_utils.control.feedback(G, K0, +1)

    # read all ki
    Ki_files = [f for f in os.listdir(sspath) if os.path.isfile(os.path.join(sspath, f))]
    Ki_files = [f for f in Ki_files if f[-4:]=='.mat' and f[1]=='_']
    Ki_files.sort()
    # select some controllers
    #Ki_select = [1, 2, 3, 4]#, 5, 6, 7, 8]
    Ki_select = [5, 6, 7, 8]
    #Ki_select = list(range(8))
    Ki_files = [Ki_files[i] for i in Ki_select]
    Ki = [flu.read_ss(sspath+f) for f in Ki_files]
    # make all qi
    Qi = [youla_utils.youla_Qab(K0, ki, Gstab) for ki in Ki]


    # Cost function definition
    ######################################################################
    def costfun(x, allout=False, giveK=None, verbose=False, scaling=None, Qi=Qi):
        '''Evaluate cost function on one point x'''
        print('Params seen: ', x)
        J, fs, Kss = eval_controller(x=x, G=G, K0=K0, criterion='integral',
            params_flow=params_flow, params_time=params_time, params_solver=params_solver,
            params_mesh=params_mesh, params_save=params_save, verbose=verbose, Kss=giveK,
            write_csv=True, scaling=scaling, Qi=Qi)
        print('Cost functional is: ', J)
        print('###########################################################')
        if allout:
            return J, fs, Kss
        return J
    
    def costfun_array(x, **kwargs):
        return optim_utils.fun_array(x, costfun, **kwargs)

    def costfun_parallel(x):
        return optim_utils.parallel_function_wrapper(x, [0], costfun)


    # Simulation parameters
    ######################################################################
    print('FlowSolver parameters common to all controller evaluations...')
    tbegin = 500
    NUM_STEPS_OPTIM = 40000 # TODO
    params_time['dt'] = 0.005
    params_time['dt_old'] = 0.005
    params_time['Tstart'] = tbegin
    params_time['Trestartfrom'] = 0
    params_time['restart_order'] = 2
    params_time['num_steps'] = NUM_STEPS_OPTIM 
    params_time['Tc'] = tbegin
    params_save['save_every'] = 0
    params_save['save_every_old'] = 2000
    params_solver['throw_error'] = False


    # Optimization
    ######################################################################
    optim_path = '/scratchm/wjussiau/fenics-results/cylinder_o1_opt_blend_K' + k_arg + '/' # TODO
    params_save['savedir0'] = optim_path
        
    ndim = len(Qi) 

    colnames = ['J'] + ['x'+str(i+1) for i in range(ndim)] # [y, x1, x2...]


    ### -----------------------------------------------------------------------
    # Limits
    xmin = -0.5#TODO
    xmax = 2
    xlimits = np.array([[xmin, xmax]])
    xlimits = np.repeat(xlimits, ndim, axis=0)

    ## Multi-start
    #n_doe = 1 # TODO
    #sampling = optim_utils.LHS(xlimits=xlimits, random_state=5)
    #xlist = sampling(n_doe)
    ## add custom multi start
    #xadd = np.vstack((np.zeros(ndim,), np.eye(ndim)))
    #xlist = np.vstack((xadd, xlist)) 
    # Single start @ Ki
    ei = np.zeros((ndim, ))
    ei[int(k_arg)-1] = 0.5
    xlist = np.vstack((ei, -0.5*ei))
    # start optim at ei & 0.5*ei (mostly for syntax purposes, will not run in under 1day)

    # TODO
    #scaling_factor = 1 / youla_utils.norm(youla_utils.control.feedback(G, K0, 1))
    #scaling = lambda x: np.hstack((10**x[0], scaling_factor*x[1:]))
    scaling = lambda x: x

    init_delta = 0.5 # TODO

    maxfev = 150 # TODO
    costfun_parallel = lambda x: costfun(x, scaling=scaling)
    ## -----------------------------------------------------------------------

    idx_current_slice = 0
    y_cummin_all = np.empty((0, 1))
    x_cummin_all = np.empty((0, ndim))
    for x in xlist:
        print('***************** Multi-start ++++ New point')
        # if nm, construct initial simplex around x
        initial_simplex = optim_utils.construct_simplex(x, rectangular=True, edgelen=0.25)

        # run algorithm
        res = optim_utils.minimize(costfun=costfun_parallel, x0=x, alg='dfo',
            options=dict(maxfev=maxfev, maxiter=maxfev,
            adaptive=True, initial_simplex=initial_simplex,
            init_delta=init_delta,
            # for BO
            xlimits=xlimits, n_doe=50, random_state=1, n_iter=maxfev,
            corr='matern52', criterion='SBO')) # TODO

        # write optimization result
        x_cummin_all, y_cummin_all, idx_current_slice = optim_utils.write_results(
            x_data=x_data, y_data=y_data, optim_path=optim_path, colnames=colnames,
            x_cummin_all=x_cummin_all, y_cummin_all=y_cummin_all, idx_current_slice=idx_current_slice,
            nfev=res.nfev, verbose=True)

    return 1


if __name__=='__main__':
    main(sys.argv[1:])





