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
                 'sensor_location': np.array([[3.0, 0.0]]),
                 'sensor_type': 'v',
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 #'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_meanflow_iterative/'}
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_cl_G8_save_8iter/'}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True,
                   'NL': True,
                   'init_pert': 0,
                   'compute_norms': True}
    params_mesh = flu.make_mesh_param('o1')
    

    # Loop on each iter
    #Tstart_list = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    #Tstart_list = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    Tstart_list = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

    loc_max_list = []
    val_max_list = []

    for Tstart in Tstart_list:
        NUM_STEPS = 200000 if Tstart==4000 else 100000
        print('*** Tstart=', Tstart, ' ***')
        params_time = {'dt': 0.005,
                       'dt_old': 0.005,
                       'restart_order': 2,
                       'Tstart': Tstart, # load files starting here
                       'Trestartfrom': Tstart,
                       'num_steps': NUM_STEPS,  
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
        fs.load_steady_state(assign=True)

        #print('Init time-stepping')
        #fs.init_time_stepping()

        # Read snapshots
        fpathu = fs.paths['u_restart']
        fpathp = fs.paths['p_restart']
        nsnapshots = int(fs.Tf / fs.dt / fs.save_every)   

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
        flu.write_xdmf(fs.savedir0 + 'u_mean_T{0}.xdmf'.format(Tstart), func=u_mean, name='u_mean')
        flu.write_xdmf(fs.savedir0 + 'p_mean_T{0}.xdmf'.format(Tstart), func=p_mean, name='p_mean')

        # perturbation kinetic energy field
        # pke_T = (<x>_T - x_0)^2
        du_mean = u_mean - fs.u0
        xTx = flo.dot(du_mean, du_mean)
        xTx = flu.projectm(xTx, fs.P)
        flu.write_xdmf(fs.savedir0 + 'dE_T{0}.xdmf'.format(Tstart), func=xTx, name='xTx')
        # if you assemble you get PKE
        # get max of energy field
        xTxv = xTx.vector().get_local()
        idx_max = np.argmax(xTxv)
        dofmapx = fs.P.tabulate_dof_coordinates()
        loc_max_list.append(dofmapx[idx_max])
        val_max_list.append(xTxv[idx_max])
        
        # no work in parallel?
        # Assign as a single function (u,p) in W
        fa = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
        up_mean = flo.Function(fs.W)
        fa.assign(up_mean, [u_mean, p_mean])

        # Linearize operator around up_mean and save
        matpath = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/lco/tcfd/meanflows_8iter/'
        matname = 'Amean_T=' + str(Tstart)
        A_mean = fs.get_A(up_0=up_mean)
        spr.save_npz(matpath+matname+'.npz', flu.dense_to_sparse(A_mean))
        flu.export_to_mat(matpath+matname+'.npz', matpath+matname+'.mat', 'A')

    with open(fs.savedir0 + 'maxE_val.txt', 'w') as mf:
        for item in val_max_list:
            mf.write("{}\n".format(item))
    with open(fs.savedir0 + 'maxE_loc.txt', 'w') as mf:
        for item in loc_max_list:
            mf.write("{}\n".format(item))


    if 0:
        # Save A (base flow), B, C, Q only once
        A = fs.get_A()
        B = fs.get_B()
        C = fs.get_C()
        Q = fs.get_mass_matrix()

        spr.save_npz(matpath+'A.npz', flu.dense_to_sparse(A))
        spr.save_npz(matpath+'B.npz', spr.csr_matrix(B))  
        spr.save_npz(matpath+'C.npz', spr.csr_matrix(C))
        spr.save_npz(matpath+'Q.npz', flu.dense_to_sparse(Q))

        for matname in ['A', 'B', 'C', 'Q']:
            flu.export_to_mat(matpath+matname+'.npz', matpath+matname+'.mat', matname)
















