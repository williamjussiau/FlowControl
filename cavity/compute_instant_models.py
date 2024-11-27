"""
----------------------------------------------------------------------
Compute instant linearized models on given trajectory
for every xi in T(x0), compute A=df/dx|xi
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
                 'num_steps': 200000, # 
                 'Tc': 0,
                 } 
    params_save={'save_every': 100, 
                 'save_every_old': 10000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity_cl_x0_K1opt_1/',
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


    compute_trajectory=False
    if compute_trajectory: # this saves a trajectory and associated mean flows
        ## Compute trajectory T(x0)
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
        Kss = flu.read_ss(sspath + 'multiK/K1.mat')

        # Youla variation
        if 1:
            import youla_utils as yu
            ## solutions after Jcl scaling
            # K1
            # J 0.28179858875155867,0.3756020332314855,-1.4044038111046,-0.42648842458967473,-2.5400761437349457,-0.0376160737880351
            # J 0.2818994858878287,0.36108496355942304,-1.8588205689707427,-1.2251263221485647,-1.3663275007448843,-0.743913588359929
            # J 0.2860104856169508,0.41921996082426716,-0.72918833363516,1.1133261265849133,-4.192269632127687,1.0805177336546867
            p = 10**0.376
            Jcl = 1 / yu.norm(yu.control.feedback(G, Kss, 1))
            theta = Jcl * np.array([-1.404, -0.426, -2.540, -0.038])
            Q00 = yu.basis_laguerre_K00(G, Kss, p=p, theta=theta)
            Kss = yu.youla(G=G, K0=Kss, Q=Q00)



        x_ctrl = np.zeros((Kss.nstates,))

        ### Here: perform mean calculation
        ### and also we can compute A=df/dx here but cannot export lol
        #movavg = 2000 # average over xx fields for mov mean flow
        #export_every = 500
        #u_mean_arr = np.zeros((movavg, fs.V.dim()))
        #p_mean_arr = np.zeros((movavg, fs.P.dim()))
        ##fa_vp2w = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
        #u_mean = flo.Function(fs.V) 
        #p_mean = flo.Function(fs.P)
        #appendtofile = False # create file 
        
        # WARNING
        # WRITING MAT FILES IN PARALLEL DOES NOT WORK

        for i in range(fs.num_steps):
            # compute control 
            if fs.t>=fs.Tc:
                # mpi broadcast sensor
                # in perturbation only: 
                # measurement is: y'=Cq'
                # if we want: y=Cq=C(q'+q0)=Cq'+y0
                # >>> fs.y_meas += fs.y_meas_steady
                y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
                y_meas_ctrl = y_meas 
                # wrapper around control toolbox
                u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_ctrl, fs.dt)

            # += : perturbed closed-loop
            # = : open-loop
            u_pert = u_pert0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
            u_ctrl += u_pert 

            if fs.perturbations:
                fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
            else:
                fs.step(u_ctrl) # step and take measurement


            # call to fs.u_.vector()[:] does not work in parallel ofc
            ##### Running mean flow
            #### add flow to list
            ###u_mean_arr[0, :] = fs.u_.vector()[:] 
            ###p_mean_arr[0, :] = fs.p_.vector()[:] 

            #### compute mean and export
            ###if not i%export_every and i+1>=movavg:
            ###    print('exporting mean flow at iter: {0}'.format(i))
            ###    # compute mean velocity field
            ###    u_mean.vector()[:] = 1/movavg * np.sum(u_mean_arr, axis=0)
            ###    p_mean.vector()[:] = 1/movavg * np.sum(p_mean_arr, axis=0)
            ###    
            ###    # export mean velocity field
            ###    u_mean_file = fs.savedir0 + 'mean/umean_navg{0}.xdmf'.format(movavg)
            ###    p_mean_file = fs.savedir0 + 'mean/pmean_navg{0}.xdmf'.format(movavg)
            ###    flu.write_xdmf(u_mean_file, u_mean, name='umean', time_step=i, append=appendtofile)  
            ###    flu.write_xdmf(p_mean_file, p_mean, name='pmean', time_step=i, append=appendtofile)  
            ###    appendtofile = True

            #### roll array
            ###u_mean_arr = np.roll(u_mean_arr, shift=1, axis=0)
            ###p_mean_arr = np.roll(p_mean_arr, shift=1, axis=0)

        flu.end_simulation(fs, t0=t000)
        fs.write_timeseries()
        print(fs.timeseries)


    else: # this parses fields (not mean fields) and extracts A=df/dx 
    # trajectory was saved every 100 snapshots
        ## Read fields & compute operators
        # Read snapshots
        fpathu = fs.paths['u_restart']
        fpathp = fs.paths['p_restart']

        # two options:
        # 1) while field exists, load, compute A, export 
        # 2) moving average
        # export every xx fields 
        # (warning: fields are saved at yy rate, so exporting at xx*yy rate)
        fa_vp2w = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
        fa_w2vp = flo.FunctionAssigner([fs.V, fs.P], fs.W)

        print('++++++ Instant linearized model ++++++')
        if 0: # extract instant model A=df/dx|x(t)
            # Init
            export_every = 5 
            max_iter = 500
            u_ = flo.Function(fs.V)
            p_ = flo.Function(fs.P)
            up_ = flo.Function(fs.W)

            ii = 0
            finished_parsing = False
            while (not finished_parsing) and (ii<=max_iter):
                if not ii%1:
                    print('\t >> Trying snapshot nr: {0}...'.format(ii))

                # Read fields
                try:
                    # Velocity 
                    flu.read_xdmf(filename=fpathu, func=u_, name='u', counter=ii)
                    # Pressure
                    flu.read_xdmf(filename=fpathp, func=p_, name='p', counter=ii)
                except:
                    finished_parsing = True
                    print('**** Finished parsing file, no snapshot {0}'.format(ii))
                    print('**** Exiting...')

                if not finished_parsing:
                    # Assign as a single function (u,p) in W
                    fa_vp2w.assign(up_, [u_, p_])

                    # Linearized operator around current up_ and save
                    matpath = '/scratchm/wjussiau/fenics-python/cavity/data/instant/'
                    matname = 'Ainstant_it{0}'.format(ii)
                    Aisp = flu.dense_to_sparse(fs.get_A(up_0=up_, timeit=False))
                    #flu.spr.save_npz(matpath+matname+'.npz', Aisp)
                    sio.savemat(matpath+matname+'.mat', mdict={'A': Aisp.tocsc()})

                    # Increment counter
                    ii += export_every



        print('++++++ Moving-mean linearized model ++++++')
        if 1: # extract mean flow model A=df/dx|<x(t)> 
            # Init
            movavg = 20 # average over xx SAVED fields for movmean flow
            export_every = 20
            max_iter = 1000
            u_ = flo.Function(fs.V)
            p_ = flo.Function(fs.P)
            up_ = flo.Function(fs.W)
            up_mean = flo.Function(fs.W)
            up_mean_arr = np.zeros((len(up_mean.vector()[:]), movavg))
            UP = [] # list [u(i-m+1),... u(i-1), u(i)]

            ii = 0
            finished_parsing = False
            while (not finished_parsing) and (ii<=max_iter):
                if not ii%10:
                    print('\t >> Trying snapshot nr: {0}...'.format(ii))
                # Read fields
                #try:
                #    # Velocity 
                #    flu.read_xdmf(filename=fpathu, func=u_, name='u', counter=ii)
                #    # Pressure
                #    flu.read_xdmf(filename=fpathp, func=p_, name='p', counter=ii)
                #except:
                #    finished_parsing = True
                #    print('**** Finished parsing file, no snapshot {0}'.format(ii))
                #    print('**** Exiting...')

                ## other version
                if (ii>=movavg) and (not ii%export_every): # time to compute!
                    print('Computing and exporting mean flow model @it={0}'.format(ii))
                    UP = []
                    for jj in range(movavg):
                        # Velocity 
                        flu.read_xdmf(filename=fpathu, func=u_, name='u', counter=ii-jj)
                        # Pressure
                        flu.read_xdmf(filename=fpathp, func=p_, name='p', counter=ii-jj)
                        # Assign as a single function (u,p) in W
                        #flu.show_max(u_, 'u_ just read')
                        fa_vp2w.assign(up_, [u_, p_])
                        #up_mean_arr = np.roll(up_mean_arr, shift=1, axis=1)
                        up_mean_arr[:, jj] = up_.vector()[:]
                        #UP.append(up_.copy())
                        #flu.write_xdmf(fs.savedir0+'mean/uinstant_it{0}.xdmf'.format(ii-jj), 
                        #    u_, 'umean')
                    #up_mean.vector()[:] = 1/movavg * sum([up.vector()[:] for up in UP]) 
                    up_mean.vector()[:] = np.mean(up_mean_arr, axis=1) 
                    #flu.show_max(up_mean, 'up mean')
                    #fa_w2vp = flo.FunctionAssigner([fs.V, fs.P], fs.W)
                    #fa_w2vp.assign([u_, p_], up_mean)
                    #flu.write_xdmf(fs.savedir0+'mean/umean_navg{0}_it{1}.xdmf'.format(movavg, ii), 
                    #    u_, 'umean')

                    # compute linearized model 
                    matpath = '/scratchm/wjussiau/fenics-python/cavity/data/instant/'
                    matname = 'Amean_it{1}_navg{0}'.format(movavg, ii)
                    Aisp = flu.dense_to_sparse(fs.get_A(up_0=up_mean, timeit=False))
                    sio.savemat(matpath+matname+'.mat', mdict={'A': Aisp.tocsc()})

                    # write mean flow just to check
                    fa_w2vp.assign([u_, p_], up_mean)
                    flu.write_xdmf(fs.savedir0+'mean/umean_navg{0}.xdmf'.format(movavg), 
                        u_, 'umean', ii, append=not(ii==movavg))
                    appendfile = True

                
                #if not finished_parsing:
                #    # Assign as a single function (u,p) in W
                #    fa_vp2w.assign(up_, [u_, p_])

                #    # Moving-mean flow
                #    UP.append(up_) # append to list 
                #    up_mean_arr = np.roll(up_mean_arr, shift=1, axis=1)
                #    up_mean_arr[:, 0] = up_.vector()[:]

                #    if len(UP)==movavg: # list is complete for mean computation
                #        # if export
                #        if not ii%export_every:
                #            print('Exporting moving mean flow model...')
                #            # compute mean
                #            #up_mean.vector()[:] = 1/movavg * sum([up.vector()[:] for up in UP]) 
                #            up_mean.vector()[:] = np.mean(up_mean_arr, axis=1) 
                #            flu.show_max(up_mean, 'up mean')
                #            #up_mean = flu.projectm(up_mean, fs.V) # slow
                #            #fa_vp2w.assign(up_mean, [up_mean, p_])

                #            ## compute linearized model 
                #            #matpath = '/scratchm/wjussiau/fenics-python/cavity/data/instant/'
                #            #matname = 'Amean_it{1}_navg{0}'.format(movavg, ii)
                #            #Aisp = flu.dense_to_sparse(fs.get_A(up_0=up_mean, timeit=False))
                #            #sio.savemat(matpath+matname+'.mat', mdict={'A': Aisp.tocsc()})

                #            ## write mean flow just to check
                #            #flu.write_xdmf(fs.savedir0+'mean/umean_navg{0}.xdmf'.format(movavg), 
                #            #    up_mean, 'umean', ii, append=True)
                #            
                #        # remove first elt (then append at next iter)
                #        UP.pop(0) # remove elt0 = u(i-m+1)

                # Increment counter
                if not finished_parsing:
                    ii += 1

        if 0:
            # Save A (base flow), B, C, Q only once
            A = fs.get_A(up_0=fs.up0)
            B = fs.get_B()
            C = fs.get_C()
            Q = fs.get_mass_matrix()
            flu.spr.save_npz(matpath+'A.npz', flu.dense_to_sparse(A))
            flu.spr.save_npz(matpath+'B.npz', flu.spr.csr_matrix(B))  
            flu.spr.save_npz(matpath+'C.npz', flu.spr.csr_matrix(C))
            flu.spr.save_npz(matpath+'Q.npz', flu.dense_to_sparse(Q))
            for matname in ['A', 'B', 'C', 'Q']:
                flu.export_to_mat(matpath+matname+'.npz', matpath+matname+'.mat', matname)
    


            
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------







