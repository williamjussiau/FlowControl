"""
----------------------------------------------------------------------
Run openloop
Perturbation in x (div(pert)=0) or in u
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

## ---------------------------------------------------------------------------------
if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 7500.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[1.1, 0.1]]), # sensor 
                 'sensor_type': ['v'], # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 } 
    params_time={'dt': 0.001, # in Sipp: 0.0004=4e-4 
                 'Tstart': 1000, 
                 'num_steps': 100000, # 1000s 
                 'Tc': 0,
                 } 
    params_save={'save_every': 10000, 
                 'save_every_old': 10000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_cl_x0_K1youla_bis/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, #######
                   'NL': True, ################# NL=False only works with perturbations=True
                   'init_pert': 0} # initial perturbation amplitude, np.inf=impulse (sequential only?)
    # cav0
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 1,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'cavity_byhand_n200.xdmf',
                   'xinf': 2.5,
                   'xinfa': -1.2,
                   'yinf': 0.5,
                   'segments': 540,
                   }

    fs = flo.FlowSolver(params_flow=params_flow,
                    params_time=params_time,
                    params_save=params_save,
                    params_solver=params_solver,
                    params_mesh=params_mesh,
                    verbose=100)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='picard', max_iter=4, tol=1e-9, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
    fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()
   
    print('Step several times')
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0

    # perturbation
    u_pert0 = 0
    tlen = 0.15
    tpeak = 1 
    # controller
    sspath = '/scratchm/wjussiau/fenics-python/cavity/data/regulator/'
    G = flu.read_ss(sspath + 'sysid_o24_ssest_QB.mat')
    #Kss = flu.read_ss(sspath + 'K0_o10_S_KS_poles001.mat')
    Kss = flu.read_ss(sspath + 'multiK/K1.mat')

    # Youla variation
    if 1:
        import youla_utils as yu
        # scaling

        Jinf = yu.norm(yu.control.feedback(G, Kss, 1))

        p = 10**1.0035314401857225 
        theta = [-1.0443093055611803,1.1376193568166935,2.7110105339081074,3.4388854220713663]

        #p = 10**0.8993004378770177
        #theta = np.array([-2.4428703574553157,-1.4477079479020958,0.33646914176989645,1.1425437617517078])

        theta = np.asarray(theta)
        theta *=  1 / Jinf

        Q = yu.basis_laguerre_ss(p=p, theta=theta)
        Kss = yu.youla(G=G, K0=Kss, Q=Q)
    
    x_ctrl = np.zeros((Kss.nstates,))


    compute_energy_balance_every = 25
    energy_balance = flo.pd.DataFrame(np.zeros((int(fs.num_steps/compute_energy_balance_every), 6)), 
        columns=['time', 'K', 'P', 'D', 'F', 'W']) # energy balance
    idx_energy = 0
    for i in range(fs.num_steps):
        # compute control 
        if fs.t>=fs.Tc:
            # mpi broadcast sensor
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            # compute error relative to base flow
            y_meas_err = +(y_meas - y_steady)
            # wrapper around control toolbox
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
            #print('measurement is: ',  y_meas_err)
            #print('control is: ', u_ctrl)
            #print('error is: ', y_meas_err)

        # += : perturbed closed-loop
        # = : open-loop
        u_pert = u_pert0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        u_ctrl += u_pert 

        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl) # step and take measurement


        #####################################################################
        #####################################################################
        if not i%compute_energy_balance_every: 
            # Energy analysis???
            # Evaluate some forms with running t
            U = fs.u0 # steady state
            uu = fs.u_ # perturbation field
            inner = flu.inner
            outer = flu.outer
            dot = flu.dot
            dx = fs.dx
            grad = flu.nabla_grad

            # integral formulations
            # d(KK)/dt = PP - DD - FF + WW
            #flo.parameters['form_compiler']['optimize'] = True
            #flo.parameters['form_compiler']['cpp_optimize'] = True
            #flo.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

            C = flu.Constant(1/2)
            kk = C * (uu[0]**2 + uu[1]**2)
            KK = kk * dx
            PP = - inner(outer(uu, uu), grad(U)) * dx
            DD = flu.Constant(1/fs.Re) * inner(grad(uu), grad(uu)) * dx 
            # warning: missing factor 1/2 in the following expr
            FF = (uu[0]**2 + uu[1]**2) * dot(U+uu, fs.n) * fs.ds(fs.boundaries.idx['outlet'])
            WW = u_ctrl * dot(fs.actuator_expression, uu) * dx
            
            energy_balance.loc[idx_energy] = [fs.t] + [flo.assemble(form) for form in [KK, PP, DD, FF, WW]]
            energy_balance.at[idx_energy, 'F'] *= 1/2

            idx_energy += 1
            
            energy_balance.to_csv(fs.savedir0 + 'energy_balance.csv', index=False)
        #####################################################################
        #####################################################################


    flu.end_simulation(fs, t0=t000)
    fs.write_timeseries()
    print(fs.timeseries)

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------






