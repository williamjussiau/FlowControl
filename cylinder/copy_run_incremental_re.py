"""
----------------------------------------------------------------------
Run incremental Reynolds procedure
Objective: find LTI controllers for the cylinder at moderate-high Re
Procedure: 
    - start with Re0<Rec with a stable system, extract base flow and 
    linearized model, produce controller and extract its parameters
    as theta_0
    - increasing Re incrementally (or dichotomy), use theta_0 as the 
    initial parameter of an optimization process at Re+
    (essentially, the optimization process will run at each Re+ with
    a different initial point, and hopefully with continuity of the 
    equations, it should converge pretty quickly)
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
import os
import warnings




### Predefinitions -------
def extract_parameters(Kss):
    '''Extract parameter x from controller K'''
    Ktf = flu.control.ss2tf(Kss)
    num = Ktf.num[0][0]
    den = Ktf.den[0][0][1:]
    nnum = len(num)
    nden = len(den)+1 # account for normalized 1st coeff
    x = np.hstack((num, den)) 
    return x, nnum, nden


def make_controller(x, nnum, scaling=None):
    '''Create controller K from parameter x'''
    if scaling is None:
        scaling = lambda x: x
    print('grep before scaling: (seen by optim) ', x)
    x = scaling(x)
    print('grep after scaling: (seen by K) ', x)
    Ktf = flu.control.tf(x[:nnum], np.hstack((1, x[nnum:]))) # num, den
    Kss = flu.control.tf2ss(Ktf)
    return Kss


def eval_controller(x, criterion, nnum,
                    params_flow, params_time, params_save, params_solver, params_mesh,
                    verbose=False, write_csv=False, scaling=None):
    # Ensure x is 1D
    x = x.reshape(-1,)

    # Build controller
    Kss = make_controller(x=x, nnum=nnum, scaling=scaling)

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
          
    # x**2
    J = flu.compute_cost(fs=fs, criterion=criterion, u_penalty=0.1 * 5e-2, fullstate=True, verbose=True,
        diverged=diverged, diverged_penalty=50)
    J *= 100 

    # Write CSV
    flu.write_optim_csv(fs, x, J, diverged=diverged, write=write_csv)

    # Sort results
    global x_data, y_data, E_best, J_best
    x_data += [x.copy()]
    y_data += [J]
    if J<=J_best:
        J_best = J
        E_best = fs.timeseries['dE']

    # Output
    return J, fs, Kss


if __name__=='__main__':
    ### -----------------------------------------------------------------------
    # Subcritical flow
    params_flow={'Re': 45.0, 
                 'uinf': 1.0, 
                 'd': 1.0, 
                 'sensor_location': np.array([[3.0, 0.0]]), 
                 'sensor_type': ['v'], 
                 'actuator_angular_size': 10,}
    params_time={'dt': 0.005, 
                 'dt_old': 0.005,
                 'restart_order': 1,
                 'Tstart': 0, 
                 'Trestartfrom': 0,
                 'num_steps': 99, # unused
                 'Tc': 0.0} 
    params_save={'save_every': 2000,
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_incr_re/Re45,0/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 0}
    params_mesh = flu.make_mesh_param('o1')

    ### -----------------------------------------------------------------------
    # Compute base flow
    verbose=True
    fs = flo.FlowSolver(params_flow=params_flow, params_time=params_time, params_save=params_save,
                        params_solver=params_solver, params_mesh=params_mesh, verbose=verbose)
    u_ctrl_steady=0.0
    #fs.compute_steady_state(method='picard', max_iter=3, tol=1e-9, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
    fs.load_steady_state(assign=True)
    #fs.init_time_stepping()

    ### -----------------------------------------------------------------------
    # Compute frequency response
    #Hw, Hw_timings = flu.get_Hw(fs, logwmin=-2, logwmax=1, nw=200, save_dir=fs.savedir0)
    # Matlab:
    # - Make ROM
    # - Make controller
    # - Extract controller params

    #syspath = fs.savedir0 + 'data/sysid_o8.mat'
    regpath = fs.savedir0 + 'data/K0_o4_ReCL01.mat'
    #regpath = fs.savedir0 + 'data/K0_o4_Gs2.mat'
    #regpath = fs.savedir0 + 'data/K0_o4_Gs10.mat'
    #regpath = fs.savedir0 + 'data/K0_o4_Gs100.mat'
    Kss = flu.read_ss(regpath)
    x0, nnum, nden = extract_parameters(Kss)
    # here: x0 dependent on controller parametrization
    # i.e. could write Ktf = sum(ci / s-pi) 
    # or even Ktf = Youla(some G but idk which one, Qi)

    ### -----------------------------------------------------------------------
    # Prepare loop
    basepath = '/scratchm/wjussiau/fenics-results/cylinder_incr_re/' # TODO

    hdRe = 2 # multiplier for increasing/decreasing step size
    nRe = 50 # total number of Re to try
    Re = 45 # first Re to try is Re0 + dRe
    dRe = 5 # step between Re tries
    dRemax = 5
    Re_next = Re + dRe
    Re_old = Re
    Re_tried = [] # Re tried
    Re_bool = [] # Re success (if 1, else fail) 
    was_success = True
    Re_last_fail = 0
    Re_last_success = 45

    success_list = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    #success_list = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    dRe_list = []

    for iRe in range(nRe):
    # at some point, replace with dichotomy
        ### -----------------------------------------------------------------------
        # Disp
        #Re_try = Re + dRe 
        Re_try = Re_next 
        Re_tried.append(Re_try)
        print('***************************************************')
        print('****      Trying Reynolds leap: ', Re_last_success, ' --> ', Re_try)
        print('***************************************************')
        
        ####### -----------------------------------------------------------------------
        ##### Manage folders
        ####refoldername = 'Re{:.1f}'.format(Re_try).replace('.',',') # 1 decimal
        ####refolderpath = basepath + refoldername + '/'
        ####for newfolder in ['steady/', 'timeseries/']:
        ####    if not os.path.exists(refolderpath + newfolder):
        ####        os.makedirs(refolderpath + newfolder, exist_ok=True) # create nested dirs in /Re%f/
        ####    
        ####### -----------------------------------------------------------------------
        ##### Instantiate FlowSolver
        ####NUM_STEPS_ATTRACTOR = 40000 # for computing attractor # TODO
        ####params_flow={'Re': Re_try, 
        ####             'uinf': 1.0, 
        ####             'd': 1.0, 
        ####             'sensor_location': np.array([[3.0, 0.0]]), 
        ####             'sensor_type': ['v'], 
        ####             'actuator_angular_size': 10,}
        ####params_time={'dt': 0.005, 
        ####             'dt_old': 0.005,
        ####             'restart_order': 1,
        ####             'Tstart': 0, 
        ####             'Trestartfrom': 0,
        ####             'num_steps': NUM_STEPS_ATTRACTOR, 
        ####             'Tc': 0.0} 
        ####params_save={'save_every': 2000,
        ####             'save_every_old': 2000,
        ####             'savedir0': refolderpath,
        ####             'compute_norms': True}
        ####params_solver={'solver_type': 'Krylov', 
        ####               'equations': 'ipcs',
        ####               'throw_error': True,
        ####               'perturbations': True, ####
        ####               'NL': True, ############## NL=False only works with perturbations=True
        ####               'init_pert': 1} # for attractor
        ####params_mesh = flu.make_mesh_param('o1')

        ####### -----------------------------------------------------------------------
        ##### Compute base flow
        ####verbose = 100
        ####fs = flo.FlowSolver(params_flow=params_flow, params_time=params_time, params_save=params_save,
        ####                    params_solver=params_solver, params_mesh=params_mesh, verbose=verbose)
        ####u_ctrl_steady=0.0
        ####fs.compute_steady_state(method='picard', max_iter=5, tol=1e-9, u_ctrl=u_ctrl_steady)
        ####fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
        ####fs.init_time_stepping()

        ####### -----------------------------------------------------------------------
        ##### Compute attractor and save
        ####for i in range(fs.num_steps):
        ####    u_ctrl=0.0
        ####    if fs.perturbations:
        ####        ret = fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        ####    else:
        ####        ret = fs.step(u_ctrl)

        ####### -----------------------------------------------------------------------
        ##### Optimize from attractor using theta0 as starting parameter
        ##### Redefine parameters of FlowSolver
        ####NUM_STEPS_OPTIMIZATION = 30000 # for optimization # TODO
        ####tbegin = 0 + params_time['num_steps']*params_time['dt']
        ####params_time['dt'] = 0.005
        ####params_time['Tstart'] = tbegin
        ####params_time['Trestartfrom'] = 0
        ####params_time['num_steps'] = NUM_STEPS_OPTIMIZATION 
        ####params_time['Tc'] = tbegin
        ####params_save['save_every_old'] = params_save['save_every']
        ####params_save['save_every'] = 0
        ####params_solver['throw_error'] = False

        ##### Costfun redefinition
        ####def costfun(x, scaling, verbose=verbose):
        ####    '''Evaluate cost function on one point x'''
        ####    #print('Params seen: ', x)
        ####    J, fs, Kss = eval_controller(x=x, criterion='integral', nnum=nnum,
        ####            params_flow=params_flow, params_time=params_time,
        ####            params_save=params_save, params_solver=params_solver,
        ####            params_mesh=params_mesh, verbose=verbose, write_csv=True,
        ####            scaling=scaling)
        ####    print('Cost functional is: ', J)
        ####    print('###########################################################')
        ####    return J
        ####
        #####def costfun_array(x, **kwargs):
        #####    return optim_utils.fun_array(x, costfun, **kwargs)
    
        #####def costfun_parallel(x):
        #####    return optim_utils.parallel_function_wrapper(x, [0], costfun)
 
        ##### Setup optim
        ####ndim = nnum + nden - 1 # do not count normalized denom
        ####colnames = ['J'] + ['x'+str(i+1) for i in range(ndim)] # [y, x1, x2...]
        ####
        ##### x before scaling is seen by optimization algo
        ##### x after scaling is seen by controller
        ##### decision variable=1 --> variation of eps% wrt x0
        ####delta_scaling = 50/100
        ####scaling = lambda x: x0 * (1 + delta_scaling * x)
        ####init_delta = 1
        #####scaling = lambda x: x0 + x # TODO
        #####scaling = lambda x: x
        #####init_delta = 10

        ####xZERO = 0*np.ones(x0.shape) # TODO
        #####xZERO = x0 # TODO

        ####maxfev = 30 # TODO
        ####costfun_parallel = lambda x: costfun(x, scaling=scaling)

        ####idx_current_slice = 0
        ####y_cummin_all = np.empty((0, 1))
        ####x_cummin_all = np.empty((0, ndim))

        ##### if nm, construct initial simplex around x
        ####initial_simplex = optim_utils.construct_simplex(x0, rectangular=True, edgelen=0.5)

        ####### Globals ---------------
        ####x_data = []
        ####y_data = []
        ####E_best = []
        ####J_best = np.inf

        ##### run algorithm: bfgs, nm, cobyla, dfo
        ####res = optim_utils.minimize(costfun=costfun_parallel, x0=xZERO, alg='dfo',
        ####    options=dict(maxfev=maxfev, maxiter=maxfev,
        ####    adaptive=True, initial_simplex=initial_simplex,
        ####    init_delta=init_delta)) # initial delta for DFO (param scale) 

        ##### write optimization result
        ####optim_path = refolderpath
        ####x_cummin_all, y_cummin_all, idx_current_slice = optim_utils.write_results(
        ####    x_data=x_data, y_data=y_data, optim_path=optim_path, colnames=colnames,
        ####    x_cummin_all=x_cummin_all, y_cummin_all=y_cummin_all, idx_current_slice=idx_current_slice,
        ####    nfev=res.nfev, verbose=True)



        ##### Log
        ###time_movmean = fs.dt * np.arange(dE_movmean.shape[0]) + tbegin  
        ###dE_movmean_df = flo.pd.DataFrame(np.stack((time_movmean, dE_movmean)).T,
        ###                columns=['time', 'dE_movmean'])
        ###dE_movmean_df.to_csv(refolderpath + 'dE_movmean.csv', index=False)

        #### stop stop
        ####pdb.set_trace()
        ###
        #### output Re_bool & Re_tried
        ###Re_ts = np.array((Re_tried, Re_bool)).T # tried, success
        ###Re_ts_df = flo.pd.DataFrame(Re_ts, columns=['try','success']) 
        ###Re_ts_df.to_csv(basepath + 'Re_try_success.csv', index=False)

        #### delete fs for memory management issues
        ###del fs




        ##### decide whether optimization was successful 
        ##### == whether min of moving avg of energy is the last element
        ##### i.e. energy is not increasing
        ##### false negative: energy is making bumps...
        ####movmean = lambda x, N: np.convolve(x, np.ones(N)/N, mode='valid')
        ####dE_movmean = movmean(x=E_best, N=1000) # TODO
        ####E_maxbnd = 1e-3 # E_maxbnd
        ####success = (dE_movmean[-1] == np.min(dE_movmean) and (E_best[-1:]<=E_maxbnd).bool())


        success = success_list[iRe]

        ### increment or decrement Re
        ##Re_bool.append(success)
        ##if success: # Re_try was successful --> Re=Re_try
        ##    Re = Re_try
        ##    if was_success:
        ##        dRe = flu.saturate(dRe * hdRe, 0, dRemax)
        ##    else:
        ##        dRe = flu.saturate(dRe, 0, dRemax)
        ##    # update x0 only if success, else stay on previous x0 (working on previous Re) 
        ##    x0new = res.x.reshape(-1,)
        ##    x0 = scaling(x0new) 
        ##    print('grep success! Increasing Re step to: ', dRe)
        ##    print('grep new starting point is: ', x0)
        ##else: # unsuccessful --> do not increment Re
        ##    # Re_try is redefined at the beginning of the loop as Re+dRe
        ##    # with dRe being smaller than previous
        ##    dRe = dRe / hdRe
        ##    print('grep fail! Decreasing Re step to: ', dRe)
        ##was_success = success


        ###print('Re last success is: ', Re_last_success)

        ### dRe_list.append(dRe)

        ##if was_success and not success:
        ##    dRe = -np.abs(dRe)
        ##    dRe = dRe / hdRe
        ##if not was_success and success:
        ##    dRe = np.abs(dRe)
        ##if not was_success and not success:
        ##    dRe = dRe / hdRe
        ##if was_success and success:
        ##    dRe = flu.saturate(dRe * hdRe, 0, dRemax)

        #if success:
        #    print('Success ----')
        #else:
        #    print('Fail ----')
        #Re_old = Re_try
        #was_success = success

        #Re_next = Re_try + dRe
        #dRe_list.append(dRe)






        Re_bool.append(success)

        Re_bool_arr = np.asarray(Re_bool)
        Re_tried_arr = np.asarray(Re_tried)
        Re_success_arr = Re_tried_arr[np.where(Re_bool_arr==1)[0]] 
        #Re_fail_arr = Re_tried_arr[np.where(Re_bool_arr==0)[0]] 
        Re_still_failed = [Re for Re in Re_tried_arr if Re not in Re_success_arr]

        #idx_Re_last_success = np.where(np.asarray(Re_bool, dtype=bool) == True)[0][-1] 
        #Re_last_success = Re_tried[idx_Re_last_success]
        # 
        #if not success:# or not was_success:
        #    idx_Re_last_fail = np.where(np.asarray(Re_bool, dtype=bool) == False)[0][-1] 
        #    Re_last_fail = Re_tried[idx_Re_last_fail]
        #    print('Last fail:', Re_last_fail)


        print('Last success:', Re_last_success)

        if success:
            #if 0:#Re_last_fail > Re_try: # need to implement a last^i_fail
            #    Re_next = Re_last_fail
            #else:
            if len(Re_still_failed)==0: # not len(...)
                Re_next += dRe
            else:
                Re_next = np.min(Re_still_failed)
        if not success:
            # make weighted average (not useful I think)
            avgw = 0.5
            #Re_next = np.mean([Re_try, Re_last_success])
            Re_next = avgw*Re_try + (1-avgw)*Re_last_success





            print('finishededed........................')
    sys.exit()
    
   



