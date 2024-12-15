"""
Mesh convergence
"""

from __future__ import print_function
import time
import numpy as np
import cylinder.CylinderFlowSolver as flo
import utils_flowsolver as flu
import importlib
importlib.reload(flu)
importlib.reload(flo)

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO


###############################################################################
###############################################################################
############################   MESH CONVERGENCE   #############################
###############################################################################
###############################################################################
# How to do that?
# 1. Generate mesh and store in directory
# 2. Compute flow over mesh
# 3. State whether mesh is fine enough or not
#       based on the criterion:
#       steady cl, cd (should be constant when refining)
#       strouhal number (should match litterature)
#       unstable pair of poles at p=0.139+-0.855j >> either ident or eig
if __name__=='__main__':
  
    # xinfa, xinf, yinf, xplus, n1, n2, n3

    C = [
        [-30, 50, 30,  1.5,   10.,  5., 1.],
        ]

    mesh_series = 'S' # define mesh names to be looked for 
    last_mesh_idx = 52
    # 10-20 = varying xinfa, xinf=30, yinf=20
    # 20-30 = xinfa=-30, varying xinf, yinf=20
    # 30-40 = xinfa=-30, xinf=50, varying yinf
    # 40-50 = xinfa=-30, xinf=50, yinf=30, varying xplus (default densities 10-5-1) 
    
    time_stepping = True

    for i, c in enumerate(C):
        t000 = time.time()
        mesh_name = mesh_series + str(i + last_mesh_idx + 1)
        
        print('Trying to instantiate FlowSolver...')
        params_flow={'Re': 100, 
                     'uinf': 1, 
                     'd': 1,
                     'sensor_location': np.array([[2.5, 0.0]]), # sensor 
                     'sensor_type': 'v', # u, v, p only >> reimplement make_measurement if otter
                     'actuator_angular_size': 10, # actuator angular size
                     'perturb_initial_state': True}
        params_time={'dt': 0.01, 
                     'Tstart': 00, 
                     'num_steps': 10000, 
                     'Tc': 1000} 
        params_save={'save_every': 50, 
                     'save_every_old': 50,
                     'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_mesh_convergence/' +
                        mesh_name + '/'}
        params_solver={'solver_type': 'Krylov', 
                       'equations': 'ipcs',
                       'throw_error': True}
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 32,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                       'meshname': mesh_name + '.xdmf',
                       'xinf': c[1], # 20
                       'xinfa': c[0], # -5
                       'yinf': c[2], # 8
                       'segments': 360}

        print('=== Mesh is: ', mesh_name, i, params_save['savedir0'])

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
            fs.compute_steady_state(method='picard', nip=30, tol=1e-6, u_ctrl=u_ctrl_steady)
            #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
        except:
            fs.cl0 = 0.0
            fs.cd0 = 0.0

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

























