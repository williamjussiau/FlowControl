"""
----------------------------------------------------------------------
Compute mean flow (+eig and model) from saved snapshots
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import main_flowsolver as flo
import utils_flowsolver as flu
import importlib
import scipy.sparse as spr
importlib.reload(flu)
importlib.reload(flo)

if __name__=='__main__':
    # Parameters used to generate LCO snapshots
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[3.0, 0.0]]), ## TODO 
                 'sensor_type': 'v',
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 20, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_lco_mean/'}
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
                   'restart_order': 2,
                   'Tstart': 200,
                   'num_steps': 12000, 
                   'Tc': 200} 
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

    #print('Init time-stepping')
    #fs.init_time_stepping()

    # Read snapshots
    fpathu = fs.paths['u_restart']
    fpathp = fs.paths['p_restart']
    nsnapshots = int(fs.Tf / fs.dt / fs.save_every)   
    #nsnapshots = 30 

    u_mean = flo.Function(fs.V)
    u_ = flo.Function(fs.V)

    p_mean = flo.Function(fs.P)
    p_ = flo.Function(fs.P)
    for ii in range(nsnapshots):
        if not ii%20:
            print('Snapshot nr: {0}/{1}'.format(ii+1, nsnapshots))
        # Velocity 
        flu.read_xdmf(filename=fpathu, func=u_, name='u', counter=ii)
        u_mean.vector()[:] += u_.vector()[:] * 1/nsnapshots
        # Pressure # NOTE p_mean is NOT required for linearization
        flu.read_xdmf(filename=fpathp, func=p_, name='p', counter=ii)
        p_mean.vector()[:] += p_.vector()[:] * 1/nsnapshots

    # Write as file 
    flu.write_xdmf(fs.savedir0 + 'u_mean.xdmf', func=u_mean, name='u_mean')
    flu.write_xdmf(fs.savedir0 + 'p_mean.xdmf', func=p_mean, name='p_mean')

    # Assign as a single function (u,p) in W
    fa = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
    up_mean = flo.Function(fs.W)
    fa.assign(up_mean, [u_mean, p_mean])

    # Linearize operator around up_mean and save
    matpath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/matrices/'
    A_mean = fs.get_A(up_0=up_mean)
    A = fs.get_A()
    B = fs.get_B()
    C = fs.get_C()
    Q = fs.get_mass_matrix()
    D = 0
    spr.save_npz(matpath+'A_mean.npz', flu.dense_to_sparse(A_mean))
    spr.save_npz(matpath+'A.npz', flu.dense_to_sparse(A))
    spr.save_npz(matpath+'B.npz', spr.csr_matrix(B))  
    spr.save_npz(matpath+'C.npz', spr.csr_matrix(C))
    spr.save_npz(matpath+'Q.npz', flu.dense_to_sparse(Q))
    for matname in ['A_mean', 'A', 'B', 'C', 'Q']:
        flu.export_to_mat(matpath+matname+'.npz', matpath+matname+'.mat', matname)



















