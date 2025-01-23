"""
----------------------------------------------------------------------
Run openloop
Perturbation in x (div(pert)=0) or in u
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

from scipy import signal as ss
import scipy.io as sio 

import sys

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO


## ---------------------------------------------------------------------------------
if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 7500.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[1.1, 0.1]]), # sensor 
                 'sensor_type': ['v'], # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 } 
    params_time={'dt': 0.001, # in Sipp: 0.0004=4e-4 
                 'Tstart': 1000, 
                 'num_steps': 50000, # 1e6steps=1000tc 
                 'Tc': 0,
                 } 
    params_save={'save_every': 100, 
                 'save_every_old': 10000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_mov_stab/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, #######
                   'NL': True, ################# NL=False only works with perturbations=True
                   'init_pert': 0} # initial perturbation amplitude, np.inf=impulse (sequential only?)
    # cav0
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 1,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'cavity_byhand_n200.xdmf',
                   'xinf': 2.5,
                   'xinfa': -1.2,
                   'yinf': 0.5,
                   'segments': 540,
                   }

    fs = flo.FlowSolver(params_flow=params_flow,
                    params_time=params_time,
                    params_save=params_save,
                    params_solver=params_solver,
                    params_mesh=params_mesh,
                    verbose=100)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='picard', max_iter=4, tol=1e-9, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
    fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()
   
    print('Step several times')
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0

    # perturbation
    u_pert0 = 0
    tlen = 0.15
    tpeak = 1 
    # controller
    sspath = '/scratchm/wjussiau/fenics-python/cavity/data/regulator/'
    G = flu.read_ss(sspath + 'sysid_o24_ssest_QB.mat')
    Kss = flu.read_ss(sspath + 'multiK/K1.mat')

    # Youla variation
    if 1:
        import youla_utils as yu

        ## rounded values
        #p = 10**0.757
        #theta = 1e-2 * np.array([-0.035, 0.010, 0.008, -0.001])

        ## unrounded values
        #p = 10**0.7570497671651865
        #theta = 1e-2 * np.array([-0.034979965937011435,0.01048782942519186,0.007924689362618307,-0.0010975429638357453])
        #Q = yu.basis_laguerre_ss(p=p, theta=theta)
        #Kss = yu.youla(G=G, K0=Kss, Q=Q)

        ## new solution after Jcl scaling
        # J     xxx                 log10 rho           t1             t2.....
	# J 0.28179858875155867,0.3756020332314855,-1.4044038111046,-0.42648842458967473,-2.5400761437349457,-0.0376160737880351
	# J 0.2818994858878287,0.36108496355942304,-1.8588205689707427,-1.2251263221485647,-1.3663275007448843,-0.743913588359929
	# J 0.2860104856169508,0.41921996082426716,-0.72918833363516,1.1133261265849133,-4.192269632127687,1.0805177336546867
        Jcl = 1 / yu.norm(yu.control.feedback(G, Kss, 1))
        p = 10**0.376
        theta = Jcl * np.array([-1.404, -0.426, -2.540, -0.038])
        Q00 = yu.basis_laguerre_K00(G, Kss, p=p, theta=theta)
        Kss = yu.youla(G=G, K0=Kss, Q=Q00)
    
    x_ctrl = np.zeros((Kss.nstates,))
    sigma_y = 0.0
    ## legacy, should not be used
    #np.random.seed(1)
    randst = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    noise_y = sigma_y * randst.randn(fs.num_steps+1,)
    y_meas_noisy = np.zeros((fs.num_steps+1,))

    # start control at some point on attractor (not nominal x0)
    fs.Tc = 1001

    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            # mpi broadcast sensor
            # in perturbation only: 
            # measurement is done on q', so we need to add q0
            # with q(x,t)=q0(x) + q'(x,t)
            ### fs.y_meas += fs.y_meas_steady
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            # compute error relative to base flow
            # y_meas_ctrl = +(y_meas - y_steady)
            y_meas_ctrl = y_meas + noise_y[i]
            y_meas_noisy[i] = y_meas_ctrl
            # wrapper around control toolbox
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_ctrl, fs.dt)
            #print('measurement is: ',  y_meas_ctrl)
            #print('control is: ', u_ctrl)
            #print('error is: ', y_meas_ctrl)

        ## += : perturbed closed-loop
        ## = : open-loop
        #u_pert = u_pert0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        #u_ctrl += u_pert 

        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl) # step and take measurement

    fs.timeseries = fs.timeseries.join(flo.pd.DataFrame(data={'noise_y': noise_y}))
    fs.timeseries = fs.timeseries.join(flo.pd.DataFrame(data={'y_meas_noisy': y_meas_noisy}))
    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()
    print(fs.timeseries)

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------






