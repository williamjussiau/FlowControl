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

if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), ## TODO 
                 'sensor_type': 'v'*10, # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_time={'dt': 0.005, 
                 'Tstart': 0, 
                 'dt_old': 0.005,
                 'restart_order': 1, # 1 if dt has changed
                 'Trestartfrom': 0,
                 'num_steps': 100000, 
                 'Tc': 0.0} 
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_test/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ###
                   'NL': True, ### NL=False only works with perturbations=True
                   'init_pert': 0} # np.inf for impulse
    params_mesh = flu.make_mesh_param('o1')

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
    fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    #fs.load_steady_state(assign=True)
    
    print('Init time-stepping')
    fs.init_time_stepping()
 
    print('Step several times')
    u_ctrl = 0.0 # actual control amplitude (initialized)
    u_ctrl0 = 1e-2 # max amplitude of gaussian bump
    tlen = 0.15 # characteristic length of gaussian bump
    tpeak = 1 # time peak of gaussian bump
    for i in range(fs.num_steps):
        u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        # u_ctrl = u_ctrl0 * (1-np.exp(-3*fs.t)) * np.sin(5*fs.t)
        #u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)# * np.sin(5*fs.t)
        #u_ctrl = u_ctrl0 * np.sin(5*fs.t)
        #print('amplitude of control:', u_ctrl)
        # step
        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl)




    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




