"""
----------------------------------------------------------------------
Frequency response computation
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
                 'sensor_location': np.array([[3, 0.0]]), # sensor 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 } # replace by initial_state = 'div0', 'B', supply?
    params_time={'dt': 0.01, 
                 'Tstart': 0, 
                 'num_steps': 0, 
                 'Tc': 200} 
    params_save={'save_every': 25, 
                 'save_every_old': 25,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_resolvent/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True}
    # nx32
    # m1
    # n1
    # o1
    params_mesh = flu.make_mesh_param('o1')

    #save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/nx32/sysid/'
    #save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/m1/sysid/sensor/'
    #save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/n1/sysid/'
    save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/sysid/'
    #save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/sysid_HF/'
    #save_dir = '/scratchm/wjussiau/fenics-python/cylinder/data/m1/sysid/'

    # distance on x axis
    #DX = [1, 2, 3, 4, 5, 6, 8, 9, 10] 
    DX = [3.] 
    # distance on y axis
    #DY = [0, 0.25, 0.5, 1] 
    DY = [0.] 

    # With several sensors
    sensor_array = np.array(np.meshgrid(DX, DY)).T.reshape(-1, 2)
    ns = sensor_array.shape[0]
    #sensor_type = ['p']*ns
    sensor_type = ['v']*ns

    params_flow['sensor_location'] = sensor_array
    params_flow['sensor_type'] = sensor_type
    print('sensor is:', params_flow['sensor_location'])
    
    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=True)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    fs.load_steady_state()

    # frequency response
    freqresp = True
    if freqresp:
        ## Save matrices
        ###flu.export_flowsolver_matrices(fs, save_dir, suffix='')
        
        # Compute frequency response
        #A = fs.get_A()
        #B = fs.get_B()
        #C = fs.get_C()
        #Q = fs.get_mass_matrix()
        ##flu.get_Hw(fs, A=A, B=B, C=C, Q=Q, logwmin=-2, logwmax=2, nw=10, save_dir=save_dir)

        ## Prescribed A, B, C, D
        save_npz_path = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/matrices/'
        A = flu.spr.load_npz(save_npz_path + 'A.npz')  
        B = flu.spr.load_npz(save_npz_path + 'B.npz')  
        C = flu.spr.load_npz(save_npz_path + 'C.npz')  
        Q = flu.spr.load_npz(save_npz_path + 'E.npz')  
        D = 0 
        flu.get_Hw(fs, A=A, B=B.toarray(), C=C.toarray(), D=D, Q=Q, 
            logwmin=-1, logwmax=1, nw=10, save_dir=save_dir) 
    
    # field response at some frequency
    fieldresp = False
    if fieldresp:
        nw = 10
        wlist = np.logspace(0, 8, nw)
        A = fs.get_A()
        B = fs.get_B()
        Q = fs.get_mass_matrix()

        sz = fs.W.dim()
        xxs = np.zeros((sz, nw), dtype=np.complex)
        for ii, ww in enumerate(wlist):
            print('Computing field response at w=', ww)
            # field response
            xx = flu.get_field_response(fs, w=ww, A=A, B=B, Q=Q)
            # to complex
            xxc = xx[:sz] + 1j*xx[sz:]
            # log
            xxs[:, ii] = xxc
        # export
        flu.export_field(xxs, fs.W, fs.V, fs.P, save_dir=save_dir+'vec', time_steps=wlist)

    ## With loop
    #DDX, DDY = np.meshgrid(DX, DY)
    #for ii in range(len(DX)):
    #    for jj in range(len(DY)):
    #        dx = DDX[jj][ii]
    #        dy = DDY[jj][ii]

    #        params_flow['sensor_location'] = np.array([dx, dy])
    #        print('sensor is:', params_flow['sensor_location'])
    #        suffix = '_dx=' + str(dx) + '_dy=' + str(dy)
    #        
    #        fs = flo.FlowSolver(params_flow=params_flow,
    #                            params_time=params_time,
    #                            params_save=params_save,
    #                            params_solver=params_solver,
    #                            params_mesh=params_mesh,
    #                            verbose=True)
    #        print('__init__(): successful!')

    #        print('Compute steady state...')
    #        u_ctrl_steady = 0.0

    #        if compute_baseflow:
    #            fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    #            compute_baseflow = False
    #        else:
    #            fs.load_steady_state(assign=True)
    #        
    #        # Save matrices
    #        #flu.export_flowsolver_matrices(fs, save_dir,
    #        #    suffix=suffix)
    #        
    #        # Compute frequency response
    #        fs.get_Hw(logwmin=-2, logwmax=2, nw=300, \
    #            save_dir=save_dir, \
    #            save_suffix=suffix)

    #        del fs
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




