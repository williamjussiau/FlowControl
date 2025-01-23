"""
----------------------------------------------------------------------
Frequency response computation
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
#import main_flowsolver as flo
from main_flowsolver import * 
import utils_flowsolver as flu
import importlib
importlib.reload(flu)
#importlib.reload(flo)

# FEniCS log level
set_log_level(LogLevel.INFO) # DEBUG TRACE PROGRESS INFO



###############################################################################
###############################################################################
############################     RUN EXAMPLE      #############################
###############################################################################
###############################################################################
if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 7500.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[1.1, 0.1]]), # UNUSED 
                 'sensor_type': ['v'], # UNUSED
                 'actuator_angular_size': 10, # UNUSED
                 } 
    params_time={'dt': 0.0001, # in Sipp: 0.0004 
                 'Tstart': 0, 
                 'num_steps': 10, # 1e6 
                 'Tc': 1000,
                 } 
    params_save={'save_every': 100000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_resolvent/',
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
                   'meshname': 'cavity_byhand_n200.xdmf', # cavity.xdmf, cavity_byhand_n200.xdmf
                   'xinf': 2.5,
                   'xinfa': -1.2,
                   'yinf': 0.5,
                   'segments': 540,
                   }

    fs = FlowSolver(params_flow=params_flow,
                    params_time=params_time,
                    params_save=params_save,
                    params_solver=params_solver,
                    params_mesh=params_mesh,
                    verbose=True)

    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    fs.compute_steady_state(method='picard', max_iter=5, tol=1e-9, u_ctrl=u_ctrl_steady)
    fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
    #fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()

    #############
    #print('Exiting file......')
    #sys.exit()
    #############

    pwdmat = '/scratchm/wjussiau/fenics-python/cavity/data/matrices/'

    ## Compute matrices (sequential)
    A = fs.get_A()
    B = fs.get_B()
    C = fs.get_C()
    Q = fs.get_mass_matrix()
    D = 0
    spr.save_npz(pwdmat + 'A.npz', flu.dense_to_sparse(A))
    spr.save_npz(pwdmat + 'B.npz', spr.csr_matrix(B))  
    spr.save_npz(pwdmat + 'C.npz', spr.csr_matrix(C))
    spr.save_npz(pwdmat + 'Q.npz', flu.dense_to_sparse(Q))
    # also save to mat
    for mat in ['A', 'B', 'C', 'Q']:
        flu.export_to_mat(pwdmat+mat+'.npz', pwdmat+mat+'.mat', mat)

    ## Compute frequency response
    A = spr.load_npz(pwdmat + 'A.npz')
    B = spr.load_npz(pwdmat + 'B.npz').toarray()
    C = spr.load_npz(pwdmat + 'C.npz').toarray()
    Q = spr.load_npz(pwdmat + 'Q.npz')
    D = 0

    # try altering B just to see --> yes, it works like this
    B = Q*B

    Hw, ww, hw_timings = flu.get_Hw(fs, A=A, B=B, C=C, D=D, Q=Q, 
        logwmin=-1, logwmax=3, nw=3001,
        save_dir='/scratchm/wjussiau/fenics-python/cavity/data/sysid/', verbose=True,
        save_suffix='_QB_try')
    #Hw, ww, hw_timings = flu.get_Hw_lifting(fs, A=A, C=C, Q=Q, logwmin=-1, logwmax=3, nw=1000,
    #    save_dir='/scratchm/wjussiau/fenics-python/cavity/data/sysid/', 
    #    save_suffix='_lifting_', verbose=True)
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------












