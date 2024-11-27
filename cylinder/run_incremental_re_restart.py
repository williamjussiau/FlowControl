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

import youla_utils as yu
import optim_utils

flo.set_log_level(flo.LogLevel.ERROR) # DEBUG TRACE PROGRESS INFO WARNING CRITICAL ERROR

import pdb
import sys
import os
import warnings
import getopt



### Predefinitions -------
global x_data, y_data, E_best, J_best
x_data = []
y_data = []
J_best = np.inf
E_best = np.inf


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
    #print(flu.control.ss2tf(Kss))
    return Kss
# We can study other controller parametrizations 
# coeff    -- K = sum(ai*s^i)/sum(bi*s^i)
# residues -- K = sum(ci / (s-pi))
# pz       -- K = prod(s - zi)/prod(s - pi)
# basis    -- K = sum(thetai * phii(s)) 
# NOTE formulas need to be invertible


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
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas[2])
            y_meas_err = np.array([y_steady - y_meas]) 
            # feedback -
            # but there is an error in model so the controller
            # needs to be synthesized with feedback +
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
    J = flu.compute_cost(fs=fs, criterion=criterion, u_penalty=0.0*5e-3, fullstate=True, verbose=True,
        diverged=diverged, diverged_penalty=50, scaling=lambda x: np.log10(x))
        #diverged=diverged, diverged_penalty=50, scaling=lambda x: x)
    # potentially: invert scaling of x and take 10^J
    #J *= 100 

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


#if __name__=='__main__':
def main(argv):
    ### -----------------------------------------------------------------------
    # Declare globals
    global x_data, y_data, E_best, J_best

    ### -----------------------------------------------------------------------
    # Process argv
    opt_nr = 1 # opt_nr = 1-4
    is_test = 0
    opts, args = getopt.getopt(argv, "N:T") # -N integer
    for opt, arg in opts:
        if opt=='-N':
            opt_nr = int(arg)
            print('Option --- Run nr: ', opt_nr)
        if opt=='-T':
            is_test = 1
            print('Option --- TEST RUN')


    ### -----------------------------------------------------------------------
    # Subcritical flow
    params_flow={'Re': 45.0, 
                 'uinf': 1.0, 
                 'd': 1.0, 
                 #'sensor_location': np.array([[3.0, 0.0]]), 
                 'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), 
                 'sensor_type': 'v'*10, 
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
    # then back here: get ROM, make controller

    ### -----------------------------------------------------------------------
    # Prepare loop
    basepath = '/scratchm/wjussiau/fenics-results/cylinder_incr_re_{0}/'.format(opt_nr) # TODO

    #success_list = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    #success_list = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    #dRe_list = []

    ### if restarting
    # get last success
    Re_trsu = flo.pd.read_csv(basepath + 'Re_try_success.csv')
    Re_last_success = Re_trsu['try'][np.where(Re_trsu['success'])[0][-1]]

    #hdRe = 2 # multiplier for increasing/decreasing step size
    nRe = 50 # total number of Re to try
    dRe = 5 # step between Re tries
    ######Re_last_success = fs.Re # = 45
    Re_next = Re_last_success + dRe # 45 + xx # or restart + xx
    Re_tried = [Re_last_success,] # Re tried
    Re_bool = [1,] # Re success (if 1, else fail) 
    revisit = False

    ### -----------------------------------------------------------------------
    # Init controller
    if Re_last_success==45.0: # start from 45
        Qulist = [1e-2, 1e-4, 1e-2, 1e-4]
        Qvlist = [1e-2, 1e-2, 1e-4, 1e-4]
        ## first controller
        sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/incr_re/'
        G = flu.read_ss(sspath + 'sysid_o8.mat')
        Kss, _, _ = yu.lqg_regulator(G, Qx=1, Qu=Qulist[opt_nr-1], Qw=1, Qv=Qvlist[opt_nr-1]) # feedback(+)
        x0, nnum, nden = extract_parameters(Kss)
        # Write first controller
        flu.write_ss(Kss, basepath + 'Re45,0/Kbest.mat')
        np.savetxt(basepath + 'Re45,0/x0_scaled.txt', x0)
    else: # restart
        lastRepath = basepath + 'Re{0}/'.format(Re_last_success).replace('.', ',') 
        x0 = np.loadtxt(lastRepath + 'xbest_scaled.txt') 
        nnum = x0.shape[0]//2 # int division
        nden = nnum+1
        #Kss = make_controller(x0.ravel(), nnum=x0.shape[0]//2) 


    ##flu.pdb.set_trace()


    for iRe in range(nRe):
        ### -----------------------------------------------------------------------
        # Reset globals
        x_data = []
        y_data = []
        E_best = []
        J_best = np.inf

        ### -----------------------------------------------------------------------
        # Disp
        Re_try = Re_next 
        print('***************************************************')
        print('****      Trying Reynolds leap: ', Re_last_success, ' --> ', Re_try)
        print('***************************************************')
        
        ### -----------------------------------------------------------------------
        # Manage folders
        refoldername = 'Re{:.1f}'.format(Re_try).replace('.',',') # 1 decimal
        if revisit:
            refoldername += 'r' # append 'r' to folder name to differentiate
        refolderpath = basepath + refoldername + '/'
        for newfolder in ['steady/', 'timeseries/']:
            if not os.path.exists(refolderpath + newfolder):
                os.makedirs(refolderpath + newfolder, exist_ok=True) # create nested dirs in /Re%f/
            
        ### -----------------------------------------------------------------------
        # Instantiate FlowSolver
        NUM_STEPS_ATTRACTOR = 40000 # for computing attractor # TODO
        NUM_STEPS_ATTRACTOR = 4 if is_test else NUM_STEPS_ATTRACTOR
        params_flow={'Re': Re_try, 
                     'uinf': 1.0, 
                     'd': 1.0, 
                     'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), 
                     'sensor_type': 'v'*10, 
                     'actuator_angular_size': 10,}
        params_time={'dt': 0.005, 
                     'dt_old': 0.005,
                     'restart_order': 1,
                     'Tstart': 0, 
                     'Trestartfrom': 0,
                     'num_steps': NUM_STEPS_ATTRACTOR, 
                     'Tc': 0.0} 
        params_save={'save_every': 2000,
                     'save_every_old': 2000,
                     'savedir0': refolderpath,
                     'compute_norms': True}
        params_solver={'solver_type': 'Krylov', 
                       'equations': 'ipcs',
                       'throw_error': True,
                       'perturbations': True, ####
                       'NL': True, ############## NL=False only works with perturbations=True
                       'init_pert': 1} # for attractor
        params_mesh = flu.make_mesh_param('o1')

        ### -----------------------------------------------------------------------
        # Compute base flow
        verbose = 10000
        fs = flo.FlowSolver(params_flow=params_flow, params_time=params_time, params_save=params_save,
                            params_solver=params_solver, params_mesh=params_mesh, verbose=verbose)
        u_ctrl_steady=0.0
        fs.compute_steady_state(method='picard', max_iter=5, tol=1e-9, u_ctrl=u_ctrl_steady)
        fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
        fs.init_time_stepping()

        ### -----------------------------------------------------------------------
        # Compute attractor and save (auto)
        for i in range(fs.num_steps):
            u_ctrl=0.0
            if fs.perturbations:
                ret = fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
            else:
                ret = fs.step(u_ctrl)

        ### -----------------------------------------------------------------------
        # Optimize from attractor using theta0 as starting parameter
        # Redefine parameters of FlowSolver
        NUM_STEPS_OPTIMIZATION = 30000 # for optimization # TODO
        NUM_STEPS_OPTIMIZATION = 4 if is_test else NUM_STEPS_OPTIMIZATION # for optimization # TODO
        tbegin = 0 + params_time['num_steps']*params_time['dt']
        params_time['dt'] = 0.005
        params_time['Tstart'] = tbegin
        params_time['Trestartfrom'] = 0
        params_time['num_steps'] = NUM_STEPS_OPTIMIZATION 
        params_time['Tc'] = tbegin
        params_save['save_every_old'] = params_save['save_every']
        params_save['save_every'] = 0
        params_solver['throw_error'] = False

        # Costfun redefinition
        def costfun(x, scaling, verbose=verbose):
            '''Evaluate cost function on one point x'''
            #print('Params seen: ', x)
            J, fs, Kss = eval_controller(x=x, criterion='integral', nnum=nnum, # TODO 
                    params_flow=params_flow, params_time=params_time,
                    params_save=params_save, params_solver=params_solver,
                    params_mesh=params_mesh, verbose=verbose, write_csv=True,
                    scaling=scaling)
            print('Cost functional is: ', J)
            print('###########################################################')
            return J
        
        # Setup optim
        ndim = nnum + nden - 1 # do not count normalized denom
        colnames = ['J'] + ['x'+str(i+1) for i in range(ndim)] # [y, x1, x2...]
        
        # x before scaling is seen by optimization algo
        # x after scaling is seen by controller
        # decision variable=1 --> variation of eps% wrt x0
        delta_scaling = 50/100
        scaling = lambda x: x0 * (1 + delta_scaling * x) #### scaling probably useful
        xZERO = 0*np.ones(x0.shape)
        #scaling = lambda x: x # TODO
        init_delta = 1 # TODO
        #xZERO = x0 # TODO

        maxfev = 50 # TODO
        maxfev = 1 if is_test else maxfev
        costfun_parallel = lambda x: costfun(x, scaling=scaling)

        idx_current_slice = 0
        y_cummin_all = np.empty((0, 1))
        x_cummin_all = np.empty((0, ndim))

        # if nm, construct initial simplex around x
        initial_simplex = optim_utils.construct_simplex(x0, rectangular=True, edgelen=0.5)


        # run algorithm: bfgs, nm, cobyla, dfo
        res = optim_utils.minimize(costfun=costfun_parallel, x0=xZERO, alg='dfo',
            options=dict(maxfev=maxfev, maxiter=maxfev,
            adaptive=True, initial_simplex=initial_simplex,
            init_delta=init_delta)) # initial delta for DFO (param scale) 

        # write optimization result
        #flu.pdb.set_trace()

        optim_path = refolderpath
        x_cummin_all, y_cummin_all, idx_current_slice = optim_utils.write_results(
            x_data=x_data, y_data=y_data, optim_path=optim_path, colnames=colnames,
            x_cummin_all=x_cummin_all, y_cummin_all=y_cummin_all, idx_current_slice=idx_current_slice,
            nfev=res.nfev, verbose=True)


        ### -----------------------------------------------------------------------
        # Success or fail?
        # decide whether optimization was successful 
        # == whether min of moving avg of energy is the last element
        # i.e. energy is not increasing
        # false negative: energy is making bumps...
        movmean = lambda x, N: np.convolve(x, np.ones(N)/N, mode='valid')
        dE_movmean = movmean(x=E_best, N=1000) # TODO
        E_maxbnd = 1e-3 # E_maxbnd
        success = (dE_movmean[-1] == np.min(dE_movmean) and (E_best[-1:]<=E_maxbnd).bool())
        if is_test:
            success=True
        if success:
            Kbest = make_controller(res.x.ravel(), nnum, scaling=scaling) 
            K00 = make_controller(xZERO, nnum, scaling=scaling)
            x00 = scaling(xZERO.ravel())
            x0 = scaling(res.x.ravel()) # new x0 <- x0+(1+delta*xbest)
        ## ## ## for testing
        ## success = success_list[iRe]


        ### -----------------------------------------------------------------------
        # Update Re for next step
        Re_tried.append(Re_try)
        Re_bool.append(success)
        Re_bool_arr = np.asarray(Re_bool)
        Re_tried_arr = np.asarray(Re_tried)
        Re_success_arr = Re_tried_arr[np.where(Re_bool_arr==1)[0]] 
        Re_still_failed = [Re for Re in Re_tried_arr if Re not in Re_success_arr]

        revisit = False
        Re_last_success = Re_success_arr[-1]
        if success: # either increment Re or solve Re still missing
            if len(Re_still_failed)==0: # some still failed
                Re_next += dRe
            else: # all tried Re have been solved > increment
                Re_next = np.min(Re_still_failed)
                revisit = True
        if not success: # divide step
            # optional: weighted average
            Re_next = np.mean([Re_try, Re_last_success])

        print('Revisiting Re? ', revisit)


        ### -----------------------------------------------------------------------
        # Log
        time_movmean = fs.dt * np.arange(dE_movmean.shape[0]) + tbegin  
        dE_movmean_df = flo.pd.DataFrame(np.stack((time_movmean, dE_movmean)).T,
                        columns=['time', 'dE_movmean'])
        dE_movmean_df.to_csv(refolderpath + 'dE_movmean.csv', index=False)

        # Controller
        if success:
            flu.write_ss(Kbest, refolderpath + 'Kbest.mat')
            flu.write_ss(K00, refolderpath + 'K0.mat')
            np.savetxt(refolderpath + 'x0_scaled.txt', x00)
            np.savetxt(refolderpath + 'xbest_scaled.txt', x0)
            np.savetxt(refolderpath + 'xbest_raw.txt', res.x.ravel())

        # stop stop
        #pdb.set_trace()
        
        # output Re_bool & Re_tried
        Re_ts = np.array((Re_tried, Re_bool)).T # tried, success
        Re_ts_df = flo.pd.DataFrame(Re_ts, columns=['try','success']) 
        Re_ts_df.to_csv(basepath + 'Re_try_success.csv', index=False)


        # delete fs for memory management issues
        del fs


        ### -----------------------------------------------------------------------
        # End
        print('finishededed........................')



    sys.exit()
    

### -----------------------------------------------------------------------
### -----------------------------------------------------------------------
### -----------------------------------------------------------------------
if __name__=='__main__':
    main(sys.argv[1:])




