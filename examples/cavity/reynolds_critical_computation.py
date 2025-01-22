"""
----------------------------------------------------------------------
Computation of (A, Q) for several Reynolds numbers
Then, computation of spectrum of (A, Q) to find critical Re for which
the pair of unstable poles becomes stable
Should be Re_c=46.6
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

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

###############################################################################
###############################################################################
############################     RUN EXAMPLE      #############################
###############################################################################
###############################################################################
if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[2.5, 0.0]]), # sensor 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 } # replace by initial_state = 'div0', 'B', supply?
    params_time={'dt': 0.01, 
                 'Tstart': 0, 
                 'num_steps': 0, 
                 'Tc': 200} 
    params_save={'save_every': 25, 
                 'save_every_old': 25,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_m1_reynolds/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True}
    # nx32
    #params_mesh = {'genmesh': True,
    #               'remesh': False,
    #               'nx': 32,
    #               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
    #               'meshname': 'S53.xdmf',
    #               'xinf': 20, # 50, # 20
    #               'xinfa': -5, # -30, # -5
    #               'yinf': 8, # 30, # 8
    #               'segments': 360}
    # m1
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 32,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'M1.xdmf',
                   'xinf': 40, # 50, # 20
                   'xinfa': -25, # -30, # -5
                   'yinf': 25, # 30, # 8
                   'segments': 360}
    ## n1
    #params_mesh = {'genmesh': False,
    #               'remesh': False,
    #               'nx': 1,
    #               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
    #               'meshname': 'N1.xdmf',
    #               'xinf': 20,
    #               'xinfa': -10,
    #               'yinf': 10,
    #               'segments': 360}
    ## o1
    #params_mesh = {'genmesh': False,
    #               'remesh': False,
    #               'nx': 1,
    #               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
    #               'meshname': 'O1.xdmf',
    #               'xinf': 20,
    #               'xinfa': -10,
    #               'yinf': 10,
    #               'segments': 360}

    savedir0 = params_save['savedir0']
    #Re_list = [10, 20, 40, 45, 46, 46.5, 47, 50, 60, 80, 100, 120, 140, 160, 180]
    #Re_list = [15, 18, 25, 30, 35]
    Re_list = [10, 15, 18, 20, 25, 30, 35, 40, 45, 46, 46.5, 47, 50, 60, 80, 100, 120, 140, 160, 180]
    for rey in Re_list: 
        params_flow['Re'] = rey
        print('Reynolds number is:', params_flow['Re'])

        params_save['savedir0'] = savedir0 + 'Re_' + str(rey) + '/'

        print(params_save['savedir0'])

        fs = flo.FlowSolver(params_flow=params_flow,
                            params_time=params_time,
                            params_save=params_save,
                            params_solver=params_solver,
                            params_mesh=params_mesh,
                            verbose=True)
        print('__init__(): successful!')

        print('Compute steady state...')
        u_ctrl_steady = 0.0

        try:
            fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
        except:
            print('Failed with Newton -- Trying with Picard')
            fs.compute_steady_state(method='picard', max_iter=50, tol=1e-7, u_ctrl=u_ctrl_steady)
        #fs.load_steady_state(assign=True)
        
        # Compute matrices and export
        flu.export_flowsolver_matrices(fs, '/stck/wjussiau/fenics-python/ns/data/m1/reynolds/',
            suffix='_Re'+str(rey))

        #del fs

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




