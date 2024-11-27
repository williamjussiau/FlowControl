"""
----------------------------------------------------------------------
Run bank of controllers
PROBABLY OBSOLETE
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
import control

flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

import pdb

######################################################################
optim_path = '/scratchm/wjussiau/fenics-results/cylinder_nx32_bank_youla_1p/'

params_flow={'Re': 100.0, 
             'uinf': 1.0, 
             'd': 1.0,
             'sensor_location': np.array([[2.5, 0.0]]), # sensor 
             'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
             'actuator_angular_size': 10, # actuator angular size
             }
params_time={'dt': 0.0025, 
             'Tstart': 0, 
             'num_steps': 40000, 
             'Tc': 0.0} 
params_save={'save_every': 40000, 
             'save_every_old': 40000,
             'savedir0': optim_path,
             'compute_norms': True}
params_solver={'solver_type': 'Krylov', 
               'equations': 'ipcs',
               'throw_error': True,
               'perturbations': False, ####
               'NL': False, ############## NL=False only works with perturbations=True
               'init_pert': 10}
params_mesh = {'genmesh': True,
               'remesh': False,
               'nx': 32,
               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
               'meshname': 'S53.xdmf',
               'xinf': 20, #50, # 20
               'xinfa': -5, #-30, # -5
               'yinf': 8, #30, # 8
               'segments': 360}


G = flu.read_ss('/stck/wjussiau/fenics-python/ns/data/nx32/regulator/sysid_nw300_o14.mat')
K0 = flu.read_ss('/stck/wjussiau/fenics-python/ns/data/nx32/regulator/K0_o3_D0_KS_S_posfeedback.mat')
import youla_utils as yu

######################################################################
def make_controller(k, path):
    # then it will be:  call systune.m(Kparam) >> return controller or matrices >> sim Kss >> compute J
    #path = '/stck/wjussiau/fenics-python/ns/nx32/data/regulator/regulator_struct_D0_KS.mat' # DO_KS
    #path = '/stck/wjussiau/fenics-python/ns/nx32/data/regulator/bank_youla/' # DO_KS
    #controller_name = 'K_youla_' + str(k1+1) + '.mat'
    #rd = sio.loadmat(path + controller_name)
    if k is None:
        rd = sio.loadmat(path)
        return ss.StateSpace(rd['A'], rd['B'], rd['C'], rd['D']) 
    #print('make_controller::loading controller from: ', path + controller_name)
    #return ss.StateSpace(k1, k2, k3, k4)
    else:
        p = 10
        return yu.youla(G, K0, Q=yu.basis_laguerre(p, k))


def eval_controller(k,
                    save_every=None, 
                    params_flow=params_flow, params_time=params_time, params_save=params_save, 
                    params_solver=params_solver, params_mesh=params_mesh, verbose=False,
                    path=None):
    
    if k is None:
        print('Path seen: %s' %(path))
        Kss = make_controller(k=None, path=path)
    else: 
        print('Params seen: %s' %(k))
        Kss = make_controller(k=k, path=None)
    Kss = control.StateSpace(Kss.A, Kss.B, Kss.C, Kss.D) 

    x_ctrl = np.zeros((Kss.A.shape[0],))
    y_ctrl = np.zeros((Kss.C.shape[0],))

    if save_every is not None:
        params_save['save_every'] = save_every

    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=verbose)
    fs.load_steady_state(assign=True)
    fs.init_time_stepping()

    dt = fs.dt
    nc = 1 # n steps of controller per fluid step
    dtc = dt/nc
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0.0 # init u 
    alpha =  3/2 # for integration of controller (AB1)
    beta =   1/2 # for integration of controller (AB1)
    bigproblem = False

    Jt = 0
    df = flo.Function(fs.V)
    for i in range(fs.num_steps):
        if fs.t>=fs.Tc:
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            y_meas_err = np.array([y_steady - y_meas])
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            #print('\t measurement (true, steady) + control: ', y_meas_err, u_ctrl)
            u_ctrl = flu.saturate(u_ctrl, -5, 5)
 
        # step
        ret = fs.step(u_ctrl)
        if ret==-1:
            bigproblem = True
            break
          
        t0n = time.time()

    Jt = sum(fs.timeseries.loc[:, 'dE']) / ((fs.t - fs.Tc)) # here fs.dt = Tf
    if bigproblem: # bigproblem: diverge (Krylov) or nan (LU)
        JJ = 1 # should be positive because return -JJ is negative >>> not a max 
    else:
        #JJ = np.mean(fs.timeseries.loc[:,'cd']) - 0.2*np.mean(fs.timeseries.loc[:,'cl'])**2
        JJ = Jt 

    print('Cost functional is: ', JJ)
    print('Next evaluation...')
    print('#############################################################')
    return JJ



if __name__=='__main__':
    # Compute LCO --- only once
    compute_lco = False
    if compute_lco:
        print('Trying to instantiate FlowSolver...')
        fs = flo.FlowSolver(params_flow=params_flow,
                            params_time=params_time,
                            params_save=params_save,
                            params_solver=params_solver,
                            params_mesh=params_mesh,
                            verbose=True)
        print('__init__(): successful!')

        print('Compute steady state...')
        u_ctrl_steady = 0.0
        fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)

        print('Init time-stepping')
        fs.init_time_stepping()
 
        print('Step several times')
        u_ctrl = 0.0 # free flow 
        for i in range(fs.num_steps):
            # step
            fs.step(u_ctrl)

        fs.write_timeseries()

    ## ---------------------------------------------------------------------------------
    t00 = time.time()

    print('FlowSolver parameters common to all controller evaluations...')
    tbegin = 100
    params_time['Tstart'] = tbegin
    params_time['num_steps'] = 30000
    params_time['Tc'] = tbegin
    params_save['save_every'] = 0
    params_solver['throw_error'] = False

    # Run controllers from bank and log values of J in file
    nk = 10
    import pandas as pd
    df_youla = pd.DataFrame(columns=['y', 'x1', 'x2'], data=np.zeros((nk, 3))) 
    #controller_path = '/stck/wjussiau/fenics-python/ns/data/nx32/regulator/'
    controller_path = '/stck/wjussiau/fenics-python/ns/data/nx32/regulator/bank_1D/'
    filename = 'costfun_eval_b1D' + '.csv'
    data_path = controller_path + filename
    data_path_scratchm = optim_path + filename
            
    for ii in range(nk):
        print('-------- Evaluating controller nr: ', ii+1, '/', nk, '-------------')
        controller_full_name = controller_path + 'K_youla_' + str(ii+1) + '.mat'
        #controller_full_name = controller_path + 'K_youla_best_t40-140.mat'
        
        # get controller
        rd = sio.loadmat(controller_full_name)
        youla_param = rd['yparam'][0] 
        
        J = eval_controller(youla_param, verbose=not ii%10, path=None)
        #J = eval_controller(youla_param, verbose=not ii%10, path=controller_full_name)

        # ensure file is dumped at every iteration
        # (in case of crash)
        df_youla.iloc[ii] = [J, youla_param[0], youla_param[1]]
        df_youla.to_csv(data_path, index=False)
        df_youla.to_csv(data_path_scratchm, index=False)











