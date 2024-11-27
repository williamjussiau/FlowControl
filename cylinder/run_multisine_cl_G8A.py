"""
----------------------------------------------------------------------
Run closed-loop system with controller designed with mean resolvent
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

import sys
import getopt
import pdb

#from scipy import signal as ss
#import scipy.io as sio 

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO


def main(argv):
    flu.MpiUtils.check_process_rank()

    # Process argv
    ######################################################################
    # Argument = scenario nr (file + {system G OR controller Kss})
    k_arg = 1
    is_test = 1
    opts, args = getopt.getopt(argv, "k:t") # k numeric, t=test
    for opt, arg in opts:
        if opt=='-k':
            # range is 1-4?
            k_arg = int(arg)
            is_test = 0

    if is_test:
        print('\n ***** Default: multisine CL: test run ***** \n')
        savedir0 = '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_cl_test/'
    else:
        savedir0 = '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_cl_G8A/'

    ######################################################################
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), 
                 'sensor_type': 'v'*10,
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': savedir0
                 }
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 1,
                   'compute_norms': True}
    params_mesh = flu.make_mesh_param('o1')
    
    params_time = {'dt': 0.005, # not recommended to modify dt between runs 
                   'dt_old': 0.005,
                   'restart_order': 1, # required 1 if dt old is not dt
                   'Tstart': 1000, # start of this sim # TODO # 1st switching/stacking iter is 1000
                   'Trestartfrom': 500, # last restart file # TODO
                   'num_steps': 100000, #  
                   'Tc': 0} 
    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=100)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='picard', nip=50, tol=1e-7, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()

    print('Step several times')
    # previous controller
    ## here: fs.t = fs.Tstart
    x_ctrl0 = np.loadtxt(fs.savedir0 + 'x_ctrl_at_t={:.1f}.npy'.format(fs.t))
    Kss0 = flu.read_ss(fs.savedir0 + 'K_at_t={:.1f}.mat'.format(fs.t))

    # identified system
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter1/'
    sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter2/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter5/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter6/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter7/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter8/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/iter9/'

    # for new iter tcfd
    # it0: 0 500
    # it1: 500 1000
    # it2: 1000 1500
    # ...


    ########### MODEL
    # TODO
    #G = flu.read_ss(sspath + 'G8A0001.mat') # TODO DONOTOUCH
    G = flu.read_ss(sspath + 'Gi8.mat') # TODO DONOTOUCH



    ############ SYNTHESIS
    import youla_utils as yu
    # make new controller
    ## lqg
    # array 1-8 
    # state-feedback # stacking used: 10^3 (or sometimes 10^2 or 10^1, cool stuff wp)
    #Qulist = [1, 2, 3, 1, 2, 3, 1, 2]
    #Qvlist = [1, 1, 1, 2, 2, 2, 3, 3]
    #Qu = 10**(3) # Qu low = low penalty on u = big u  = high feedback gain
    #Qv = 10**(0) # Qv low = low noise on y = follow y = high observation gain 
    Qu = 10**(4) # Qu low = low penalty on u = big u  = high feedback gain
    Qv = 10**(4) # Qv low = low noise on y = follow y = high observation gain 
    # Qu, Qv low beneficial up to some limit (leaving linear limit)
    # lqg works with (+) feedback
    Kss1, _, _ = yu.lqg_regulator(G=G, Qx=1, Qu=Qu, Qw=1, Qv=Qv) # lqg weights 
    #Kss1 = flu.read_ss(sspath + 'Kstruct8.mat') 
    ##modelref = False
    ##if modelref:
    ##    # load mean resolvent
    ##    Gm = flu.read_ss(sspath + 'Gm14.mat')
    ##    # make lqg regulator for mean resolvent
    ##    Kss1, _, _ = yu.lqg_regulator(G=Gm, Qx=1, Qu=Qu, Qw=1, Qv=Qv) # lqg weights 
    ##    # make feedback (-)
    ##    Kss1 = -Kss1
    ##    # hinf with reference model >> warning feedback (-) in synthesis! 
    ##    ct = yu.control
    ##    # reference loop
    ##    CL_ref = ct.feedback(Gm, Kss1, -1)
    ##    # weightings
    ##    Wr = ct.tf2ss(ct.tf(0.021, 1))
    ##    We = ct.tf2ss(ct.tf(0.022, 1))
    ##    Wu = ct.tf2ss(ct.tf(0.023, 1))
    ##    Wb = ct.tf2ss(ct.tf(0.024, 1))
    ##    Wcl = ct.tf2ss(ct.tf(1, 1))
    ##    # following works with feedback (-)
    ##    Kss_i, gamma_i = yu.hinfsyn_mref(G, We, Wu, Wb, Wr, CL_ref, Wcl, syn='Hinf')
    ##    # for implementation: feedback (+)
    ##    Kss1 = -Kss_i
    # precomputed
    #Kss1 = -1 * flu.read_ss(sspath + 'Khinf8.mat') # precomputed with feedback (-), used with (+)
    print('is stable feedback? ', yu.isstablecl(G, Kss1, +1))



    ############# NEW CONTROLLER  
    ## stack
    Kss00 = Kss0
    Kss = Kss0 + Kss1 # big controller 
    ## switch
    #Kss = Kss1
    
    
    #### REDUCE
    #if yu.isstable(Kss):
    #    TT = yu.baltransform(Kss)
    hsv_threshold = [10**(-3)] # array 1-8
    Kssred, hsval, nred = yu.balred_rel(Kss, hsv_threshold) # reduced controller

    Kss0 = Kss    # switch FROM Kss0 -> in the file
    Kss1 = Kssred # switch TO Kss1
    #Kss  = Kssred # to make following op?

    ############# INITIAL STATE # warning op might not work after redux
    # size == Kss.nstates
    ## * for stacking only
    ##  xk=[xk1,0]
    #x_ctrl0_stack = np.hstack((x_ctrl0, np.zeros(Kss0.nstates,)))
    #x_ctrl1_stack = np.hstack((x_ctrl0, np.zeros(Kss1.nstates,)))
    

    #x_ctrl_Kss0 = np.hstack((x_ctrl0, np.zeros(G.nstates)))

    # here slowfast
    #Tlim = 10 # cut poles under Tlim response time
    #wlim = 2*np.pi/Tlim # cut poles under wlim pulsation
    #Kslow, Kfast = yu.slowfast(Kss0, wlim)
    #Kss0 = Kfast + Kslow # block-diagonal
    #x_ctrl_fast = np.zeros(Kfast.nstates) # reset fast modes
    #x_ctrl_slow = np.linalg.lstsq(Kslow.C, Kss00.C@x_ctrl0, rcond=None)[0] # Ax=b -> x=pinv(A)*b
    #x_ctrl_slow = np.asarray(x_ctrl_slow).reshape(-1,)
    #x_ctrl_Kss0 = np.hstack((x_ctrl_fast, x_ctrl_slow))

    x_ctrl_Kss1 = np.zeros((nred,))
    x_ctrl_Kss0 = np.hstack((x_ctrl0, np.zeros(G.nstates)))

    ##### * for stacking or switching
    ##### not time-based
    ###lstsq = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
    #### u1=u2 (lsqminnorm)
    ###Au1u2 = Kss.C
    ###bu1u2 = Kss0.C @ x_ctrl0 # this is supposedly u_ctrl_1[Tend] but not at all???
    ###x_ctrl1_u = lstsq(Au1u2, bu1u2) 
    #### du1=du2 (lsqminnorm)
    ###yend = fs.y_meas[2] # should take y_meas_previous[2] but not available
    ###Adu1du2 = np.vstack((Kss.C, Kss.C@Kss.A))
    ###bdu1du2 = np.vstack((Kss0.C, Kss0.C@Kss0.A))@x_ctrl0.reshape(-1,1) + \
    ###    np.vstack((0, Kss0.C@Kss0.B - Kss.C@Kss.B))*yend
    ###x_ctrl1_du = lstsq(Adu1du2, bdu1du2)
    #### bumpless by safonov 
    ####  --> need freqsep or slowfast decomposition

    #### backwards prediction
    make_extension = lambda T: '_restart'+str(np.round(T, decimals=3)).replace('.',',')
    ###r_backwards = 10
    ###restart_extension = make_extension(fs.Trestartfrom)
    ###filename_timeseries_prev = savedir0 + 'timeseries1D' + restart_extension + '.csv'
    ###timeseries_prev = flo.pd.read_csv(filename_timeseries_prev) 
    ###ur = np.asarray(timeseries_prev.u_ctrl[-r_backwards-1:-1]) # shift -1 or not?
    ###try:
    ###    yr = np.asarray(timeseries_prev.y_meas_3[-r_backwards:]) # shift -1 or not?
    ###except AttributeError: # for Tstart=0 because timeseries is messed up
    ###    yr = np.asarray(timeseries_pres.y_meas_1[-r_backwards:])
    ###xn, yhat, uhat = yu.condswitch(ur=ur, yr=yr, K=Kss, dt=fs.dt, w_u=1.0, w_y=1.0, w_decay=0.9)
    ###x_ctrl1_r = xn

    #### forward prediction (inf or not inf)
    #### # --> needs state of ROM

    #### state continuity in companion form 
    #### # --> control.canonical_form but need to implement every controller as companion
    #### # requires same order
    ####Kss, Ptransform = flu.control.canonical.reachable_form(Kss) 
    ####x_ctrl1 = x_ctrl0

    ##### time-based
    #### m1) u(t) = (1-a(t))u1(t) + a(t)u2(t) --> ok
    ###alpha_list = [2, 10, 20, 30]
    ####alpha = alpha_list[k_arg-1] # transition on alpha seconds: make it fast or slow?
    ###alpha = 5
    ###def alpha_t(t):
    ###    return flu.saturate((t - fs.Tstart)/alpha, 0.0, 1.0)
    #### m2) run K2 from Tsw-Ti to initialize it, then plug it at Tsw 
    T_init_K = 100
    #### m3) youla(a(t)*Q) --> difficult to implement

    ##### *** define state here
    ##### choose method
    ####x_ctrl_list = [np.zeros(Kss.nstates,), \
    ####    np.ones(Kss.nstates,), \
    ####    x_ctrl1_u, \
    ####    x_ctrl1_du, \
    ####    x_ctrl1_r]
    ####if k_arg<=5:
    ####    x_ctrl = x_ctrl_list[k_arg-1]
    ####else:
    ####    x_ctrl = np.zeros(Kss.nstates,)
    ####x_ctrl = x_ctrl_list[3] # 0, 1, u, du
    ####x_ctrl = x_ctrl1_stack 

    #x_ctrl = np.zeros(Kss.nstates,) 
    #x_ctrl = x_ctrl1_stack 
    #x_ctrl = np.asarray(x_ctrl).reshape(-1,) 

    ### log controller state 
    X_CTRL = np.zeros((fs.num_steps+1, Kssred.nstates)) # reduced state
    #X_CTRL[0, :] = x_ctrl.ravel()


    # for time-based
    #Kss0 = Kss
    #Kss1 = Kssred
    x_ctrl0 = x_ctrl_Kss0
    x_ctrl1 = x_ctrl_Kss1

    #X_CTRL0 = np.zeros((fs.num_steps+1, Kss0.nstates))
    #X_CTRL0[0, :] = x_ctrl0.ravel()

    #X_CTRL1 = np.zeros((fs.num_steps+1, Kss1.nstates))
    #X_CTRL1[0, :] = x_ctrl1.ravel()


    # loop
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0 # control amplitude at time 0
    for i in range(fs.num_steps):
        # compute control 
        #if fs.t>=fs.Tc:
        #y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas) # single sensor
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas[2]) # several sensors
        y_meas_err = +np.array([y_meas - y_steady]) #### feedback sign was +

        #### state-based switch 
        #u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
        ## log controller state
        #X_CTRL[i+1, :] = x_ctrl.ravel()

        ## time-based switch
        ## merge
        u_ctrl0, x_ctrl0 = flu.step_controller(Kss0, x_ctrl0, y_meas_err, fs.dt)
        u_ctrl1, x_ctrl1 = flu.step_controller(Kss1, x_ctrl1, y_meas_err, fs.dt)
        #u_ctrl = u_ctrl0*(1-alpha_t(fs.t)) + u_ctrl1*alpha_t(fs.t)
        # log controller state
        X_CTRL[i+1, :] = x_ctrl1.ravel() # state of reduced controller 
        #X_CTRL0[i+1, :] = x_ctrl0.ravel()
        #X_CTRL1[i+1, :] = x_ctrl1.ravel()
        # initialize K1 in open-loop, then plug 
        if 1:
            if fs.t<=fs.Tstart+T_init_K:
                u_ctrl = u_ctrl0 
            else:
                u_ctrl = u_ctrl1

        # u_ctrl = 0 # get only LCO
        u_ctrl = flu.saturate(u_ctrl, -2.0, 2.0)

        # step
        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl)

    # Save controller and controller state
    ## here: fs.t == fs.Tf+fs.Tstart
    np.savetxt(fs.savedir0 + 'x_ctrl_at_t={:.1f}.npy'.format(fs.t), X_CTRL[-1,:]) 
    flu.write_ss(path=fs.savedir0 + 'K_at_t={:.1f}.mat'.format(fs.t), sys=Kss1) 

    # Save controller states timeseries to file
    # xk controller
    x_ctrl_df = flo.pd.DataFrame(X_CTRL, index=fs.timeseries.time,\
        columns=['xk' + str(i+1) for i in range(Kss1.nstates)])
    x_ctrl_df_filename = fs.savedir0 + 'timeseries_xk' + make_extension(fs.Tstart) + '.csv'
    x_ctrl_df.to_csv(x_ctrl_df_filename, index=True)
    ## xk 0
    #x_ctrl_df = flo.pd.DataFrame(X_CTRL0, index=fs.timeseries.time,\
    #    columns=['xk' + str(i+1) for i in range(Kss0.nstates)])
    #x_ctrl_df_filename = fs.savedir0 + 'timeseries_xk0' + make_extension(fs.Tstart) + '.csv'
    #x_ctrl_df.to_csv(x_ctrl_df_filename, index=True)
    ## xk 1
    #x_ctrl_df = flo.pd.DataFrame(X_CTRL1, index=fs.timeseries.time,\
    #    columns=['xk' + str(i+1) for i in range(Kss1.nstates)])
    #x_ctrl_df_filename = fs.savedir0 + 'timeseries_xk1' + make_extension(fs.Tstart) + '.csv'
    #x_ctrl_df.to_csv(x_ctrl_df_filename, index=True)

    # End simulation
    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()


if __name__=='__main__':
    main(sys.argv[1:])

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




