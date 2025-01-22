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
                 'Tstart': 0, 
                 'num_steps': 500000, # 1e6 
                 'Tc': 0,
                 } 
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_mov_attractor/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, #######
                   'NL': False, ################# NL=False only works with perturbations=True1
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
                    verbose=True)
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
    print('steady measurement is: ', fs.y_meas_steady)
    u_ctrl = 0

    # perturbation
    u_ctrl0 = 1e-2
    tlen = 0.15
    tpeak = 1 
    # controller
    sspath = '/scratchm/wjussiau/fenics-python/cavity/data/regulator/'
    G = flu.read_ss(sspath + 'sysid_o24_for_QB.mat')
    #Kss = flu.read_ss(sspath + 'K0_o10_S_KS_poles001.mat')
    Kss = flu.read_ss(sspath + 'multiK/K00_1.mat')

    import youla_utils as yu
    print('are feedbacks stable? (+) ; (-)', yu.isstablecl(G, Kss, +1), yu.isstablecl(G, Kss, -1))
    x_ctrl = np.zeros((Kss.nstates))

    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            # mpi broadcast sensor
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            # compute error relative to base flow
            y_meas_err = +(y_meas - y_steady)
            # wrapper around control toolbox
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            #print('measurement is: ',  y_meas_err)
            #print('control is: ', u_ctrl)

        # += : perturbed closed-loop
        # = : open-loop
        u_pert = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        u_ctrl += u_pert 

        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl) # step and take measurement

    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()
    print(fs.timeseries)

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------






