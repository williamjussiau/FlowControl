"""
----------------------------------------------------------------------
Run closed loop from LCO on a list of controllers
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
                 'sensor_location': np.array([[3.0, 0.0]]), # sensor ## TODO 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_save={'save_every': 5000, 
                 'save_every_old': 20000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_m1_cl_x0_allK_pert/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': False,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 0.0,
                   'compute_norms': True}
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 32,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'M1.xdmf',
                   'xinf': 40,
                   'xinfa': -25,
                   'yinf': 25,
                   'segments': 540}

    dtdiv = 1 # dt divider, possible values are 1, 2, 4 for now (>> Tstart is a saved step)
    params_time = {'dt': 0.0025/dtdiv, # not recommended to modify dt between runs 
                  'Tstart': 200/dtdiv,
                  'num_steps': 40000*dtdiv, 
                  'Tc': 200/dtdiv} # hasn't to be a saved step

    sspath = '/stck/wjussiau/fenics-python/ns/data/m1/regulator/'
    K_list = ['K0_LQG_2-2_d=3_Go16.mat',
             'K0_LQG_3-2_d=3_Go16.mat',
             'K0_LQG_3-3_d=3_Go16.mat',
             'K0_LQG_3-4_d=3_Go16.mat',
             'K0_LQG_4-4_d=3_Go16.mat',
             'K0_LQG_d=3_Go16.mat',
             'K0_o8_D0_d=3_Go16.mat',
             'K0_o8_D0_K00_S_GS_T.mat',
             'K0_o8_D0_K00_S_KS_GS05_Poles.mat',
             'K0_o8_D0_K00_S_KS_GS_Poles.mat',
             'K0_o8_D0_K00_S_KS_GS_T.mat',
             'K0_o8_D0_K00_S_T.mat',
             'K0_o8_D0_S_KS_GS.mat',
             'K0_o8_D0_S_KS_GS_T.mat',
             'K0_o8_D0_S_KS.mat',
             'K0_o8_D0_S_.mat',
             'K0_o8_D0_d=3_Go16.mat',
             'K0_o10_GS_d=3_Go16.mat']
    G = flu.read_ss(sspath + 'sysid_o16_d=3_ssest.mat')
   
    for i, K in enumerate(K_list):
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

        # Define output path
        fs.savedir0 = params_save['savedir0'] + 'K' + str(i) + '/'
        fs.define_paths()

        # define controller
        print('Step several times')
        print('Savepath is', fs.savedir0)
        print('Controller nr:', i, K)
        K0 = flu.read_ss(sspath + K)
        Kss = K0
        #import youla_utils as yu
        #Kss = yu.youla_laguerre(G, K0, p=10, theta=[1]*10) 
        #Kss = yu.youla(G, K0, Q=yu.basis_laguerre(p=10, theta=[1]*10))
        #Kss = yu.youla_laguerre_K00(G, K0, p=10, theta=[0.])

        x_ctrl = np.zeros((Kss.A.shape[0],))
        y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
        u_ctrl = 0 # control amplitude at time 0

        # loop
        for i in range(fs.num_steps):
            # compute control 
            if fs.t>=fs.Tc:
                y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
                y_meas_err = np.array([y_steady - y_meas])
                u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
                #u_ctrl = flu.saturate(u_ctrl, -5, 5)

            # step
            if fs.perturbations:
                fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
            else:
                #fs.step_perturbation(u_ctrl=u_ctrl, NL=True, shift=0.0)
                ret = fs.step(u_ctrl)
                if ret==-1:
                    break

        if fs.num_steps > 3:
            print('Total time is: ', time.time() - t000)
            print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
            print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
            print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
            print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
        flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall])
        
        fs.write_timeseries()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




