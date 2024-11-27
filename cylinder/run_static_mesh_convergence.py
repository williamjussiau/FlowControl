"""
----------------------------------------------------------------------
Mesh convergence - Static quantities
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
                 'sensor_location': np.array([[2.5, 0.0]]), # sensor 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_time={'dt': 0.005, 
                 'Tstart': 0, 
                 'num_steps': 500, 
                 'Tc': 0.0} 
    params_save={'save_every': 10, 
                 'save_every_old': 10,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_m00/',
                 'compute_norms': False}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': False, #######
                   'NL': True, ################# NL=False only works with perturbations=True
                   'init_pert': 0}
   
    import gmsh_generate_cylinder as gm
    #param_list = [30, 40, 50, 100]
    param_list = [0]
    nparam = len(param_list)
    cd_list = []
    for ii, param in enumerate(param_list):
        print('Parameters %d/%d is: ' %(ii+1, nparam), param)
        # Make mesh here
        xinfa = -25
        xinf = 40
        yinf = 25 
        szmida = 8 # size mid zone, amont
        szmid = 20 # size mid zone, aval
        mg = gm.MeshGenerator()
        mg.set_mesh_param(default=False,
                          xinfa=xinfa,
                          xinf=xinf,
                          yinf=yinf,
                          inftol=xinf-szmid,
                          inftola=np.abs(xinfa)-szmida,
                          n1=10,
                          n2=5,
                          n3=1,
                          segments=540,
                          yint=3.0,
                          lint=1.5,
                          xplus=7.0)
        filename = '/stck/wjussiau/fenics-python/mesh/M00'
        mg.make_mesh_all(filename, verbose=True)
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 32,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                       'meshname': 'M00.xdmf',
                       'xinf': xinf, #50, # 20
                       'xinfa': xinfa, #-30, # -5
                       'yinf': yinf, #30, # 8
                       'segments': 360}
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
        fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
        #fs.load_steady_state(assign=True)
        cd_list.append(fs.cd0)
    
    # Write file of Cd
    cd_df = flo.pd.DataFrame(data=dict(param=param_list, cd=cd_list)) 
    cd_df.to_csv(fs.savedir0 + 'cd_df.csv', index=False)
    print(cd_df)
    sys.exit()

    ###print('Init time-stepping')
    ###fs.init_time_stepping()
 
    ###print('Step several times')
    #### define controller
    ###dt = fs.dt
    ###
    ###u_ctrl = 0.0 # actual control amplitude (initialized)
    ###u_ctrl0 = 1.0 # base amplitude of gaussian bump
    ###tlen = 0.5 # characteristic length of gaussian bump
    ###tpeak = 1 # peak of gaussian bump
    ###for i in range(fs.num_steps):
    ###    #u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
    ###    u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2) * np.sin(20*fs.t)
    ###    # step
    ###    if fs.perturbations:
    ###        fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
    ###    else:
    ###        fs.step(u_ctrl)

    ###if fs.num_steps > 3:
    ###    print('Total time is: ', time.time() - t000)
    ###    print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
    ###    print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
    ###    print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
    ###    print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
    ###flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall])
    ###
    ###fs.write_timeseries()
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




