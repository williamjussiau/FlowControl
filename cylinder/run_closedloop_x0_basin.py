"""
----------------------------------------------------------------------
Run closed loop from LCO * (1+alpha)
Objective is to estimate the basin of attraction in the directions
spanned by the LCO
i.e. with a stabilizing controller, initialize at x0*(1+alpha) and
check convergence
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
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_cl_basin/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 0.0,
                   'compute_norms': True}
    # o1
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 1,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'O1.xdmf',
                   'xinf': 20, #50, # 20
                   'xinfa': -10, #-30, # -5
                   'yinf': 10, #30, # 8
                   'segments': 360}
    params_time = {'dt': 0.005, # not recommended to modify dt between runs 
                  'Tstart': 0,
                  'num_steps': 40000, 
                  'Tc': 0}
    
    print('Init time-stepping')
    # Read from LCO data, rebased to t0=0
    # can use counter=-2, and so on
    alpha0 = 3.6
    alphaE = 4.2
    nalpha = 5
    alpha_list = np.linspace(alpha0, alphaE, nalpha)
    J_alpha = flo.pd.DataFrame(data=dict(alpha=alpha_list, J=np.zeros(alpha_list.shape)))
    for ii, alpha in enumerate(alpha_list):
        # Make FlowSolver
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

        # Define initial state
        uu = flo.Function(fs.V)
        pp = flo.Function(fs.P)
        # this is x0+xlco
        flu.read_xdmf(fs.savedir0 + 'u_restart200_rebased.xdmf', uu, 'u', counter=-1)
        flu.read_xdmf(fs.savedir0 + 'p_restart200_rebased.xdmf', pp, 'p', counter=-1)
        uu.vector()[:] -= fs.u0.vector()[:] 
        pp.vector()[:] -= fs.p0.vector()[:]
        uu.vector()[:] *= alpha
        pp.vector()[:] *= alpha
        fa = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
        x0 = flo.Function(fs.W) 
        fa.assign(x0, [uu, pp])
        # Set & init
        fs.set_initial_state(x0=x0)
        fs.init_time_stepping()
        
        print('Step several times')
        # define controller
        #sspath = '/stck/wjussiau/fenics-python/ns/data/m1/regulator/'
        #sspath = '/stck/wjussiau/fenics-python/ns/data/n1/regulator/'
        sspath = '/stck/wjussiau/fenics-python/ns/data/o1/regulator/'
        G = flu.read_ss(sspath + 'sysid_o16_d=3_ssest.mat')
        K0 = flu.read_ss(sspath + 'K0_o8_D0_smallS_KS_clpoles10.mat')
        #K0 = flu.read_ss(sspath + 'Kopt_reduced13.mat')

        Kss = K0
        import youla_utils as yu
        theta = [2.418297, 1.954860]
        p = 10**0.787485
        Kss = yu.youla_laguerre(G, K0, p=p, theta=theta) 

        K0_cl = K0
        Kss_cl = Kss

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
                u_ctrl = flu.saturate(u_ctrl, -2, 2)

            # step
            if fs.perturbations:
                fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
            else:
                fs.step(u_ctrl)

        if fs.num_steps > 3:
            print('Total time is: ', time.time() - t000)
            print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
            print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
            print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
            print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
        flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall])
        
        fs.paths['timeseries'] = fs.savedir0 + 'timeseries1D_a=' + str(alpha) + '.csv' 
        fs.write_timeseries()

        # Terminal energy
        J = flu.compute_cost(fs, criterion='terminal', u_penalty=0*0.1*5e-3, fullstate=True)
        J_alpha.loc[ii, 'J'] = J
        print('Cost function: J=', J)

        del fs

    J_alpha.to_csv(params_save['savedir0'] + 'J_alpha.csv', index=False)


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




