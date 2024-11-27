"""
Mesh convergence - Reynolds number swipe (for computing aero coeff)
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
############################   MESH CONVERGENCE   #############################
############################    REYNOLDS SWIPE    #############################
###############################################################################
###############################################################################
# Swipe Reynolds number (should be 46.6)
# Check Strouhal number
# Check critical Reynolds number

if __name__=='__main__':
  
    # xinfa, xinf, yinf, xplus, n1, n2, n3

    C = [-30, 50, 30,  1.5,   10.,  5., 1.]

    mesh_series = 'S' # define mesh names to be looked for 
    mesh_idx = 53
    
    time_stepping = True

    reynolds = [40, 45, 46, 47, 50, 60, 80, 100, 120, 140, 160, 180]
    reynolds = [50]

    for i, re in enumerate(reynolds):
        t000 = time.time()
        mesh_name = mesh_series + str(mesh_idx)
        
        print('Trying to instantiate FlowSolver...')
        params_flow={'Re': re, 
                     'uinf': 1, 
                     'd': 1,
                     'sensor_location': np.array([[2.5, 0.0]]), # sensor 
                     'sensor_type': 'v', # u, v, p only >> reimplement make_measurement if otter
                     'actuator_angular_size': 10, # actuator angular size
                     'perturb_initial_state': True}
        params_time={'dt': 0.01, 
                     'Tstart': 0, 
                     'num_steps': 10, 
                     'Tc': 1000} 
        params_save={'save_every': 50, 
                     'save_every_old': 50,
                     'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_mesh_convergence/' +
                        mesh_name + '/' + 'Re' + str(re) + '/'}
        params_solver={'solver_type': 'Krylov', 
                       'equations': 'ipcs',
                       'throw_error': True}
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 32,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                       'meshname': mesh_name + '.xdmf',
                       'xinf': C[1], # 20
                       'xinfa': C[0], # -5
                       'yinf': C[2], # 8
                       'segments': 360}

        print('=== Reynolds is: %d (iter %d)' %(re, i))

        fs = flo.FlowSolver(params_flow=params_flow,
                            params_time=params_time,
                            params_save=params_save,
                            params_solver=params_solver,
                            params_mesh=params_mesh,
                            verbose=True)

        print('__init__(): successful!')

        print('Compute steady state...')
        u_ctrl_steady = 0.0
        fs.compute_steady_state(method='picard', nip=50, tol=1e-5, u_ctrl=u_ctrl_steady)
        #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)

        # store this
        df = flo.pd.DataFrame(columns=['cl0', 'cd0', 'ndofs', 'ncells'], index=[0])
        df['cl0'] = fs.cl0
        df['cd0'] = fs.cd0
        df['ndofs'] = fs.W.dim()
        df['ncells'] = fs.mesh.num_cells()

        if flu.MpiUtils.get_rank() == 0:
            df.to_csv(params_save['savedir0']+'data_steady.csv', sep=',', index=False)

        if time_stepping:
            print('Init time-stepping')
            fs.init_time_stepping()
       
            print('Step several times')
            u_ctrl = 0
            for i in range(fs.num_steps):
                fs.step(u_ctrl) # step and take measurement
    
            if fs.num_steps > 3:
                print('Total time is: ', time.time() - t000)
                print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
                print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
                print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
                print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
            flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall, flo.TimingType.user])
            
            fs.write_timeseries()

            



# ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------

























