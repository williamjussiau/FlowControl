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
    params_save={'save_every': 100, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_lco_meanflow/'}
                 #'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_cl_lco_Klqg/'}
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
                   'Tstart': 200, # start of this sim
                   'Trestartfrom': 0, # last restart file
                   'num_steps': 200000, # 1000tc 
                   'Tc': 35000} 
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
    # define controller
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/m1/regulator/'
    #sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/n1/regulator/'

    sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/'
    Kss = flu.read_ss(sspath + 'Klqg_T24.mat')
    #Kss = flu.read_ss(sspath + 'Kstruct_T24_SKSGSp.mat')
    #Kss = flu.read_ss(sspath + 'Kstruct_mean.mat')
    #Kss = Kss1+Kss2

    x_ctrl = np.zeros((Kss.nstates,))
    #x_ctrl = np.loadtxt(fs.savedir0 + 'x_ctrl_t={:.1f}.npy'.format(fs.Tstart))
    #x_ctrl = np.hstack((x_ctrl, np.zeros((Kss2.nstates,))))
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0 # control amplitude at time 0
    
    # loop
    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            y_meas_err = np.array([y_meas - y_steady]) #### feedback sign
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            u_ctrl = flu.saturate(u_ctrl, -2, 2)

        # step
        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl)

    np.savetxt(fs.savedir0 + 'x_ctrl_t={:.1f}.npy'.format(fs.Tf+fs.Tstart), x_ctrl) # TODO add time stamp
    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




