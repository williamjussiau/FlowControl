"""
----------------------------------------------------------------------
Compute instant linearized models on given trajectory
for every xi in T(x0), compute A=df/dx|xi
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
                 'num_steps': 200000, # 
                 'Tc': 0,
                 } 
    params_save={'save_every': 100, 
                 'save_every_old': 10000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_cl_x0_K1opt_1/',
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


    # Param
    nsnapshots = 2001
    ny = 2
    YY = np.zeros((nsnapshots, ny))

    # Read snapshot
    fpathu = fs.paths['u_restart']
    fpathp = fs.paths['p_restart']
    u_ = flo.Function(fs.V)
    p_ = flo.Function(fs.P)
    up_ = flo.Function(fs.W)
    for ii in range(nsnapshots):
        print('Reading snapshot {0}/{1}...'.format(ii, nsnapshots))
        # Velocity 
        flu.read_xdmf(filename=fpathu, func=u_, name='u', counter=ii)
        # Pressure
        flu.read_xdmf(filename=fpathp, func=p_, name='p', counter=ii)

        # Extract measurement
        YY[ii, :] = u_(0.2, 0.01)
        

    # Export
    np.savetxt('timeseriesY.txt', YY) 



            
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------


