"""
----------------------------------------------------------------------
Run closed loop from LCO
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

#from scipy import signal as ss
#import scipy.io as sio 

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[3.0, 0.0]]), ## TODO 
                 'sensor_type': 'v',
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 #'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_lco_mean/'}
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_cl_lco_klqg_it1_multisin/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 1,
                   'compute_norms': True}
    params_mesh = flu.make_mesh_param('o1')
    
    params_time = {'dt': 0.002, # not recommended to modify dt between runs 
                   'dt_old': 0.005,
                   'restart_order': 1,
                   'Tstart': 200,
                   'num_steps': 400000, 
                   'Tc': 200} 
    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=True)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='picard', nip=50, tol=1e-7, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()

    print('Step several times')
    # define controller
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/m1/regulator/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/n1/regulator/'

    sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/'
    #Kss = flu.read_ss(sspath + 'Kpp1.mat')
    #Kss = flu.read_ss(sspath + 'Kpp2.mat')
    Kss = flu.read_ss(sspath + 'Klqg.mat')
    #Kss = flu.read_ss(sspath + 'Kstruct.mat')

    ##sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/regulator/'
    ##G = flu.read_ss(sspath + 'sysid_o16_d=3_ssest.mat')
    ###K0 = flu.read_ss(sspath + 'K0_o8_D0_S_KS_clpoles1.mat')
    ###K0 = flu.read_ss(sspath + 'K0_o8_D0_smallS_KS_clpoles10.mat')
    ###K0 = flu.read_ss(sspath + 'Krobust16.mat')
    ###K0 = flu.read_ss(sspath + 'Kopt_reduced13.mat')
    ##sspath_multiK = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/multiK/'
    ##K0 = flu.read_ss(sspath_multiK + 'K_0.mat')

    ##Kss = K0
    ##import youla_utils as yu
    ### Optimum for K0
    ###theta = [2.418, 1.955]
    ###p = 10**0.7810
    ### Optimum for K0, K2, K6, K7 found in optim 20/06/2022
    ##theta = [2.153, 0.211]
    ##p = 10**1.904
    ###theta = [-3.401, 6.632]
    ###p = 10**3.135
    ###theta = [-7.478, 8.765]
    ###p = 10**3.194
    ###theta = [0.008, 0.197]
    ###p = 10**0.627
    ###Kss = yu.youla_laguerre(G, K0, p=p, theta=theta) 
    ##Kss = yu.youla(G=G, K0=K0, Q=yu.basis_laguerre_ss(p=p, theta=theta))

    x_ctrl = np.zeros((Kss.nstates,))
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0 # control amplitude at time 0
    
    # loop
    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            y_meas_err = -np.array([y_steady - y_meas])
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            u_ctrl = flu.saturate(u_ctrl, -5, 5)

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




