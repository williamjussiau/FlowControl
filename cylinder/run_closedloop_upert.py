"""
----------------------------------------------------------------------
Run closed loop from base flow with u perturbation
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import cylinder.CylinderFlowSolver as flo
import utils_flowsolver as flu
import importlib
importlib.reload(flu)
importlib.reload(flo)

from scipy import signal as ss
import scipy.io as sio 

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[3.0, 0.0]]), # sensor 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_time={'dt': 0.005, 
                 'Tstart': 0, 
                 'num_steps': 150000, 
                 'Tc': 0.0} 
    params_save={'save_every': 5000, 
                 'save_every_old': 5000,
                 'compute_norms': True,
                 #'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_cl_lin/'} # TODO
                 #'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_m1_cl_nli/'} # TODO
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_m1_cl_lin_K11/'} # TODO
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####### TODO
                   'NL': False, ################# NL=False only works with perturbations=True
                   'init_pert': 0}
    # nx32
#    params_mesh = {'genmesh': True,
#                   'remesh': False,
#                   'nx': 64,
#                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
#                   'meshname': 'S53.xdmf',
#                   'xinf': 20, #50, # 20
#                   'xinfa': -5, #-30, # -5
#                   'yinf': 8, #30, # 8
#                   'segments': 360}
    # m1
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 32,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'M1.xdmf',
                   'xinf': 40,
                   'xinfa': -25,
                   'yinf': 25,
                   'segments': 540}
    ## n1 
    #params_mesh = {'genmesh': False,
    #               'remesh': False,
    #               'nx': 1,
    #               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
    #               'meshname': 'N1.xdmf',
    #               'xinf': 20,
    #               'xinfa': -10,
    #               'yinf': 10,
    #               'segments': 540}
    ## o1 
    #params_mesh = {'genmesh': False,
    #               'remesh': False,
    #               'nx': 1,
    #               'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
    #               'meshname': 'O1.xdmf',
    #               'xinf': 20,
    #               'xinfa': -10,
    #               'yinf': 10,
    #               'segments': 540}

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
    #fs.load_steady_state(assign=True)

    print('Init time-stepping')
    #np.random.seed(2)
    #x0 = flu.Function(fs.W)
    #x0.vector()[:] += 0.1*np.random.randn(x0.vector()[:].shape[0])
    #fs.set_initial_state(x0=x0)
    fs.init_time_stepping()

    print('Step several times')
    # define controller
    sspath = '/stck/wjussiau/fenics-python/ns/data/m1/regulator/'
    #sspath = '/stck/wjussiau/fenics-python/ns/data/n1/regulator/'
    #sspath = '/stck/wjussiau/fenics-python/ns/data/o1/regulator/'
    G = flu.read_ss(sspath + 'sysid_o16_d=3_ssest.mat')
    #Kss = flu.read_ss(sspath + 'K0_o8_D0_S_KS_clpoles1.mat')
    #Kss = flu.read_ss(sspath + 'K0_o8_D0_smallS_KS_clpoles10.mat')
    Kss = flu.read_ss(sspath + 'K11.mat')
    #Kss = flu.read_ss(sspath + 'K0_o10_D0_smallS_KS_clpoles10.mat')
    #import youla_utils as yu
    #Kss = yu.youla_laguerre(G=G, K0=Kss, p=6.04, theta=[2.418, 1.955])

    x_ctrl = np.zeros((Kss.A.shape[0],))
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0.0 # actual control amplitude (initialized)
    u_ctrl0 = 1e-2 # base amplitude of gaussian bump
    tlen = 0.1 # characteristic length of gaussian bump
    tstartu = 0.5 # peak of gaussian bump
    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            y_meas_err = -np.array([y_meas - y_steady])
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)

        u_pert = u_ctrl0 * np.exp(-1/2*(fs.t-tstartu)**2/tlen**2)
        u_ctrl += u_pert

        # step
        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            #fs.step_perturbation(u_ctrl=u_ctrl, NL=True, shift=0.0)
            fs.step(u_ctrl)

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




