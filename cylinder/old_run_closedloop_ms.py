"""
----------------------------------------------------------------------
Run closed-loop with multisin excitation (in addition to K)
For identification of closed-loop (oscillating) system
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import main_flowsolver as flo
import utils_flowsolver as flu
import identification_utils as idu
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
                 'sensor_location': np.hstack((np.arange(10).reshape(-1,1), np.zeros((10,1)))), ## TODO 
                 'sensor_type': 'v',
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_cl_1/'}
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
                   'num_steps': 250000, # 400k=800tc @ dt=0.002 
                   'Tc': 200} 

    # Define controller
    sspath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/'
    #Kss = flu.read_ss(sspath + 'Kpp1.mat')
    #Kss = flu.read_ss(sspath + 'Kpp2.mat')
    Kss = flu.read_ss(sspath + 'Klqg.mat')
    #Kss = flu.read_ss(sspath + 'Kstruct.mat')

    # Define multisine
    Nper = 2 # number of periods of multisin
    N = params_time['num_steps'] // Nper # size of one multisin period
    Fs = 1/params_time['dt']
    M = 4
    ampl = 1.8

    # Loop on experiment   
    for im in range(M):
        print('*** Loop on experiment: m=', im)
        fs = flo.FlowSolver(params_flow=params_flow,
                            params_time=params_time,
                            params_save=params_save,
                            params_solver=params_solver,
                            params_mesh=params_mesh,
                            verbose=10)
        print('__init__(): successful!')

        print('Compute steady state...')
        #fs.compute_steady_state(method='picard', nip=50, tol=1e-7, u_ctrl=0.0)
        #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=0.0)
        fs.load_steady_state(assign=True)

        print('Init time-stepping')
        fs.init_time_stepping()

        print('Step several times')
        msgen = idu.multisin_generator(N=N, Fs=Fs, fmin=0.0, fmax=1.0,
                                      skip_even=False, include_fbounds=True)
        fs.timeseries['u_ms'] = np.zeros(fs.num_steps+1,)
        fs.timeseries['u_k'] = np.zeros(fs.num_steps+1,)
        #x_ctrl = np.zeros((Kss.nstates,))
        x_ctrl = np.loadtxt(fs.savedir0 + 'x_ctrl_t={:.1f}.npy'.format(fs.Tstart))
        y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
        
        # Define path for saved files
        csvname = 'data_multisine_A={:.1f}_m={:d}_P={:d}'.format(ampl, im, Nper)
        #csvname = 'timeseries_testsimple_3={:4f}_t0'.format(ampl)
        # to replace the dot by a comma:   .replace('.',',')
        fs.paths['timeseries'] = fs.savedir0 + csvname + '.csv'
        #############################################################

        # Loop
        for i in range(fs.num_steps):
            # compute control 
            if fs.t>=fs.Tc:
                y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
                y_meas_err = -np.array([y_steady - y_meas])
                u_k, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
                # add multisine
                u_ms = ampl*msgen.generate(fs.t)
                u_ctrl = u_k + u_ms
                # still saturate ?
                #u_ctrl = flu.saturate(u_ctrl, -5, 5)

            # step
            if fs.perturbations:
                fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
            else:
                fs.step(u_ctrl)
            # log
            fs.timeseries['u_ms'][i] = u_ms
            fs.timeseries['u_k'][i] = u_k

        np.savetxt(fs.savedir0 + 'x_ctrl_t={:.1f}.npy'.format(fs.Tf+fs.Tstart), x_ctrl) # TODO add time stamp
        flu.end_simulation(fs, t0=t000)
        fs.write_timeseries()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




