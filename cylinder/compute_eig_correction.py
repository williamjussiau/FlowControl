"""
----------------------------------------------------------------------
Compute eigenvalue correction of "mean flow resolvent"
to find "mean resolvent" eigenvalues
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

if __name__=='__main__':
    t000 = time.time()
    
    ################################################################################
    # Initialization
    ################################################################################
    # Instantiate flow solver for FunctionSpaces etc...
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), ## TODO 
                 'sensor_type': 'v'*10, # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_time={'dt': 0.005, 
                 'Tstart': 0, 
                 'dt_old': 0.005,
                 'restart_order': 1, # 1 if dt has changed
                 'Trestartfrom': 0,
                 'num_steps': 100000, 
                 'Tc': 0.0} 
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_eig_correction/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ###
                   'NL': True, ### NL=False only works with perturbations=True
                   'init_pert': np.inf}
    params_mesh = flu.make_mesh_param('o1')

    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=100)
    print('__init__(): successful!')
    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='picard', nip=50, tol=1e-7, u_ctrl=u_ctrl_steady)
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    fs.load_steady_state(assign=True)
    print('Init time-stepping')
    fs.init_time_stepping()

    # Dedicated folder
    resultpath = fs.savedir0

    ################################################################################
    ## FFT
    ################################################################################
    write_fft_xdmf = False # write fft field at each freq
    compute_fft = False # if false: load from file
    if compute_fft:
        n_sn = 2000 # 1 to 2000 (both included)
        dt_sn = 0.5 # time step of snapshots
        Fs_sn = 1/dt_sn

        dofmap = fs.V.dofmap()
        # list of dofs (by index, so essentialy 0:ndof)
        dof_idx_u = fs.V.dofmap().tabulate_local_to_global_dofs()
        #dof_idx_p = fs.P.dofmap().tabulate_local_to_global_dofs()
        ndofs_u = len(dof_idx_u)
        #ndofs_p = len(dof_idx_p)
        dof_coo_u = fs.V.tabulate_dof_coordinates()
        #dof_coo_p = fs.P.tabulate_dof_coordinates()

        nloc = dof_coo_u.shape[0]

        u = flo.Function(fs.V)
        # yy contains the measurement over time (columns) for several dofs (rows)
        yy = np.zeros([nloc, n_sn])
        # Loop on snapshots
        for isn in range(n_sn):
            # Read snapshot
            if not isn%10:
                print('Reading time snapshot nr: ', isn)
            flu.read_xdmf(resultpath + 'u_restart200.xdmf', u, 'u', counter=isn)
            #for iy in range(nloc):
                #yy[iy, isn] = u(dof_coo_u[iy, :])[1] 
            # if all dofs, do this instead of looping:
            yy[:, isn] = u.vector().get_local()

        # FFT of measurement
        #fftwin = np.hamming(n_sn).reshape(1, -1)
        #fftwin = np.repeat(fftwin, nloc, axis=0)
        ffty = np.fft.fft(yy) 
        ff = np.arange(0, n_sn) * Fs_sn/n_sn 
        tt = np.arange(0, n_sn) * dt_sn

        if write_fft_xdmf:
            # This will hold the FFT evaluated at some frequency
            ffty_field_mag = flo.Function(fs.V) 
            ffty_field_re = flo.Function(fs.V)
            ffty_field_im = flo.Function(fs.V)
            ffty_fi_local = np.zeros(ndofs_u,) 
            # Loop on all FFT frequencies to fill ffty 
            append = False
            for i, fi in enumerate(ff[ff<=Fs_sn/2]):
                if not i%10:
                    print('Processing FFT at frequency: ', ff[i])
                # Get value of FFT at frequency fi=ff[i]
                ffty_fi = ffty[:, i]
                # Fill ffty_field (as Function)
                #ffty_field.vector().get_local()[dof_idx_u[0:nloc]] = np.abs(ffty_fi) 
                if write_fft_xdmf:
                    for op, vec, name in zip([np.abs, np.real, np.imag], 
                                             [ffty_field_mag, ffty_field_re, ffty_field_im],
                                             ['ffty_mag', 'ffty_re', 'ffty_im']):
                        ffty_fi_local[dof_idx_u[0:nloc]] = op(ffty_fi)
                        vec.vector().set_local(ffty_fi_local) 
                        vec.vector().apply('insert')
                        flu.write_xdmf(resultpath + name + '.xdmf', \
                            vec, name, time_step=fi, append=append)
                        append=True
        
        # Plot FFT
        saveplot = True
        if saveplot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            #for iy in range(nloc):
            #    ax.plot(ff, np.abs(ffty[iy, :]))
            #    #ax.stem(ff, np.abs(ffty[iy, :]))
            ax.plot(ff, np.mean(np.abs(ffty), axis=0))
            ax.grid()
            ax.set_xlim(0, Fs_sn/2)
            ax.set_title('FFT(y) over time') 
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude (dB)')
            fig.savefig(resultpath + 'data/fig_ffty.png')
            plt.close()
             
            # Plot y timeseries
            fig, ax = plt.subplots()
            #for iy in range(nloc):
            #    ax.plot(tt, yy[iy, :])
            ax.plot(tt, np.mean(yy, axis=0)) # should be 0?
            ax.grid()
            ax.set_title('Measurement y over time') 
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Measurement')
            fig.savefig(resultpath + 'data/fig_y.png')
            plt.close()

        print('Elapsed time: ', time.time() - t000)

        # DFT back to DTFT normalization
        ffty_normz = 1/n_sn
        ffty = ffty * ffty_normz 

        # save
        np.save(fs.savedir0 + 'data/fft.npy', ffty)
        np.save(fs.savedir0 + 'data/fft_freqs.npy', ff)

    else: # do not compute = load
        ffty = np.load(fs.savedir0 + 'data/fft.npy')
        ff = np.load(fs.savedir0 + 'data/fft_freqs.npy')


    # Energy content of fft to extract harmonics
    normffty = np.linalg.norm(ffty, axis=0)

    # by hand, only fundamental
    idxmaxfft = np.argmax(normffty[1:]) # exclude mean flow (f=0)
    f0 = ff[1:][idxmaxfft] 
    w0 = f0 * 2*np.pi # equiv to idxmaxfft+1
    maxffty = ffty[:, 1:][:, idxmaxfft]

    # with scipy.signal.find_peaks -> harmonics
    allpeaks = flu.scp.signal.find_peaks(normffty, prominence=0.1) # find big peaks = harmonics 
    # find_peaks skips idx 0 = mean flow
    allpeaks_idx = allpeaks[0][allpeaks[0]<normffty.shape[0]/2] # remove max if freq>Fs/2
    f0_array = ff[allpeaks_idx] # associated freqs
    w0_array = f0_array * 2 * np.pi # associated puls
    maxffty_array = ffty[:, allpeaks_idx] # fft value
    


    import pdb
    pdb.set_trace()






    ################################################################################
    ## Linearized operators
    ################################################################################

    ################################################################################
    ## Linearize around mean flow & compute eigz
    ################################################################################
    compute_mean_flow = False
    # Get u_mean, p_mean (computed with compute_mean....)
    if compute_mean_flow: # compute
        u_mean_vec = np.mean(yy, axis=1) 
        u_mean = flo.Function(fs.V)
        u_mean.vector().set_local(u_mean_vec)
    else: # read
        u_mean = flo.Function(fs.V)
        flu.read_xdmf(fs.savedir0 + 'u_mean.xdmf', u_mean, 'u_mean')

    # dummy p_mean
    p_mean = flo.Function(fs.P)
    flu.read_xdmf(fs.savedir0 + 'p_mean.xdmf', p_mean, 'p_mean')

    # Assign as a single function (u,p) in W=[V,P]
    fa = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
    up_mean = flo.Function(fs.W)
    fa.assign(up_mean, [u_mean, p_mean])

    # Linearize operator around up_mean and save
    A_mean = fs.get_A(up_0=up_mean)
    Quv = fs.get_mass_matrix(uvp=False) # Quv0
    Quvp = fs.get_mass_matrix(uvp=True) # Quvp

    A_mean_sp = flu.dense_to_sparse(A_mean)
    Quv_sp = flu.dense_to_sparse(Quv)
    Quvp_sp = flu.dense_to_sparse(Quvp)

    # save (npz + mat)
    flu.spr.save_npz(fs.savedir0 + 'data/A_mean.npz', A_mean_sp)
    flu.spr.save_npz(fs.savedir0 + 'data/Quv.npz', Quv_sp)
    flu.spr.save_npz(fs.savedir0 + 'data/Quvp.npz', Quvp_sp)

    flu.export_to_mat(fs.savedir0 + 'data/A_mean.npz', fs.savedir0 + 'data/A_mean.mat', 'A_mean')
    flu.export_to_mat(fs.savedir0 + 'data/Quv.npz', fs.savedir0 + 'data/Quv.mat', 'Quv')
    flu.export_to_mat(fs.savedir0 + 'data/Quvp.npz', fs.savedir0 + 'data/Quvp.mat', 'Quvp')

    # Compute unstable eigenvalue mu_k of A_mean, with eigenvectors (rk, lk)
    # Amean rk = muk rk
    # Amean^H lk = muk* lk
    # has to be done with external tool (Matlab or PETSc-petsc4py)
    # PRODUCED BY EXTERNAL SCRIPT (petsc4py complex128 support)
    # PETSc produces files: data/{rk, lk, muk, mukstar}_py.npy

    ## Reload matrices
    #A_mean_sp = flu.spr.load_npz(fs.savedir0 + 'data/A_mean.npz') 
    #Quv_sp = flu.spr.load_npz(fs.savedir0 + 'data/Q.npz') 

    # Normalize eigenvectors
    herm = lambda x: x.T.conj()
    rk = np.load(resultpath + 'data/rk_py.npy')
    lk = np.load(resultpath + 'data/lk_py.npy') 
    muk = np.load(resultpath + 'data/muk_py.npy')[0]
    mukstar = np.load(resultpath + 'data/mukstar_py.npy')[0]
    # Sanity check
    erk = np.linalg.norm(A_mean_sp@rk - muk*Quv_sp@rk)
    elk = np.linalg.norm(herm(A_mean_sp)@lk - mukstar*Quv_sp@lk)
    print('Error on (rk, lk, muk): ', erk, elk)

    # Normalize wrt Q-seminorm and stuff
    Qnorm = Quv_sp # normed wrt kinetic perturbation energy seminorm > Quv
    rk_normed = rk / np.sqrt(herm(rk) @ Qnorm @ rk)
    lk_normed = lk / (herm(rk_normed) @ Qnorm @ lk)

    print('Norm of (rk, lk): ', \
        herm(rk_normed)@Qnorm@rk_normed, \
        herm(rk_normed)@Qnorm@lk_normed)


    ################################################################################
    ## Periodic jacobians
    ################################################################################
    ## recall fft computation:
    #allpeaks = flu.scp.signal.find_peaks(normffty, prominence=0.1) # find big peaks 
    #allpeaks_idx = allpeaks[0][allpeaks[0]<normffty.shape[0]/2] # remove max if freq>Fs/2
    #f0_array = ff[allpeaks_idx] # associated freqs
    #w0_array = f0_array * 2 * np.pi # associated puls
    #maxffty_array = ffty[:, allpeaks_idx] # fft value


    # Contributions of harmonics
    plist = np.array([1,2,])
    skp  = np.zeros(plist.shape, dtype=complex)
    skmp = np.zeros(plist.shape, dtype=complex)

    for ip, p in enumerate(plist):
        print('Computing contribution of harmonic nr: ', p)

        # Construct Mp = (muk +- pjw0)Q - A_mean
        Mp = (muk + p*1j*w0)*Quv_sp - A_mean_sp
        Mmp = (muk - p*1j*w0)*Quv_sp - A_mean_sp
        
        # Construct Jp = fs.get_A(up_0=up_hat_p) 
        u_hat_vec = maxffty_array[:, ip] # select harmonics nr p 
        
        u_hat_re = flo.Function(fs.V)
        u_hat_re.vector().set_local(np.real(u_hat_vec))
        u_hat_re.vector().apply('insert')
    
        u_hat_im = flo.Function(fs.V)
        u_hat_im.vector().set_local(np.imag(u_hat_vec))
        u_hat_im.vector().apply('insert')
    
        p_hat_re = flo.Function(fs.P)
        p_hat_im = flo.Function(fs.P)
        up_hat_re = flo.Function(fs.W)
        up_hat_im = flo.Function(fs.W)
    
        # Assign as a single function (u,p) in W
        # partition into Real and Imag
        # partition each Real, Imag into u, p
        fa = flo.FunctionAssigner(fs.W, [fs.V, fs.P])
        up_hat_re = flo.Function(fs.W)
        up_hat_im = flo.Function(fs.W)
        fa.assign(up_hat_re, [u_hat_re, p_hat_re])
        fa.assign(up_hat_im, [u_hat_im, p_hat_im])
    
        # compute linearized forms
        # Jac = [-Uhat_p \dot nabla ** - ** nabla Uhat_p, 0]; [0, 0]]
        # NOTE
        # this is for a value of +p
        # for the value of -p, u_hat is u_hat.conj() (donc seul Jp_im change??)
        # NOTE
        # not included in periodic Jac:
        ####    - iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
        ####    + p*div(v)*dx \
        ####    + div(u)*q*dx \
        print('Assembling jacobians...')
        Jp =    flo.PETScMatrix()
        Jp_re = flo.PETScMatrix()
        Jp_im = flo.PETScMatrix()
        v, q = flo.TestFunctions(fs.W)
        up = flo.TrialFunction(fs.W)
        u, psplit = flo.split(up)
    
        # Jp_re
        # this should be the real part of the jacobian
        # it reads (strong form):
        # -(real(u_) * nabla_grad)u - (u * nabla_grad)real(u_)
        dF0_re = -flo.dot( flo.dot(u_hat_re, flo.nabla_grad(u)), v)*fs.dx \
                 -flo.dot( flo.dot(u, flo.nabla_grad(u_hat_re)), v)*fs.dx
        flo.assemble(dF0_re, tensor=Jp_re)
        [bc.apply(Jp_re) for bc in fs.bc_p['bcu']] # None
    
        # Jp_im
        # this is the imaginary part of the jacobian
        # it reads (strong form):
        # -(imag(u_) * nabla_grad)u - (u_ * nabla_grad)imag(u_)
        dF0_im = -flo.dot( flo.dot(u_hat_im, flo.nabla_grad(u)), v)*fs.dx \
                 -flo.dot( flo.dot(u, flo.nabla_grad(u_hat_im)), v)*fs.dx
        flo.assemble(dF0_im, tensor=Jp_im)
        [bc.apply(Jp_im) for bc in fs.bc_p['bcu']] # None

        # Jmp = J(u_hat_mp) = J(u_hat_p*)
        # u_hat_p* -> same Re, opposite Im
        # -> Re(Jmp) = Re(Jp)
        # ?? Im(Jmp) = -Im(Jp)
        Jmp_im = flo.PETScMatrix()
        dF0_im_mp = -flo.dot( flo.dot(-u_hat_im, flo.nabla_grad(u)), v)*fs.dx \
                    -flo.dot( flo.dot(u, flo.nabla_grad(-u_hat_im)), v)*fs.dx
        flo.assemble(dF0_im_mp, tensor=Jmp_im)
        [bc.apply(Jmp_im) for bc in fs.bc_p['bcu']] # None
    
        # finally: Jp = Jpr + 1j*Jpi
        Jp_re_sp = flu.dense_to_sparse(Jp_re)
        Jp_im_sp = flu.dense_to_sparse(Jp_im)
        Jp_sp = Jp_re_sp + 1j*Jp_im_sp

        Jmp_re_sp = Jp_re_sp
        Jmp_im_sp = flu.dense_to_sparse(Jmp_im)
        Jmp_sp = Jmp_re_sp + 1j*Jmp_im_sp # could be -1j*Jp_im_sp? 
    
        ###########################################################################
        ## Sum
        ## Profit
        ###########################################################################
        ### sk = l_k^H @ Jp^* @ inv((muk +- jpw0) - A_mean) @ Jp @ r_k
        ### sk = sleft @ inv(muk+-....) @ sright
        ### where Q sleft = l^H J^*, Q sright = J r
        spsolve = flu.spr.linalg.spsolve
        #spsolve = lambda A, b: flu.spr.linalg.lsqr(A, b)[0] # return[0] is vec

        #dofmap = flu.get_subspace_dofs(fs.W) # u=[...], v=[...], p=[...]
        #idx_uv = np.hstack((dofmap['u'], dofmap['v']))
        #idx_p = dofmap['p']

        ### take submats on [u, v] space
        ##submat = lambda M, idx: M[idx, :][:, idx]
        ##Quv_sp_uv = submat(Quv_sp, idx_uv)
        ##rk_uv = rk_normed[idx_uv]
        ##lk_uv = lk_normed[idx_uv]
        ##Jp_uv = submat(Jp_sp, idx_uv)
        ##Jmp_uv = submat(Jmp_sp, idx_uv)

        # with mass matrix
        #Qi = Quvp_sp

        #spright = spsolve(Qi, Jp_sp @ rk_normed)  
        #spleft = herm(lk_normed) @ Qi @ Jp_sp.conj()
        #skp_ip = spleft @ spsolve(Mp, spright) 
        ##skp_ip = (herm(lk_normed) @ Jp_sp.conj()) @ spsolve(Mp, Jp_sp @ rk_normed)
    
        #smpright = spsolve(Qi, Jmp_sp @ rk_normed)  
        #smpleft = herm(lk_normed) @ Qi @ Jmp_sp.conj()
        #skmp_ip = smpleft @ spsolve(Mmp, smpright)
        ##skmp_ip = (herm(lk_normed) @ Jmp_sp.conj()) @ spsolve(Mmp, Jmp_sp @ rk_normed)

        ## same result with left inversion
        ## skp_ip = (spsolve(Mp.T, herm(Jp_sp) @ Quv_sp.T @ lk_normed.conj())).T @ Jp_sp @ rk_normed 
        ## skmp_ip = (spsolve(Mmp.T, herm(Jmp_sp) @ Quv_sp.T @ lk_normed.conj())).T @ Jmp_sp @ rk_normed 
        # without mass matrix
        # right invert
        skp_ip = herm(lk_normed) @ Jp_sp.conj() @ spsolve(Mp, Jp_sp @ rk_normed) 
        skmp_ip = herm(lk_normed) @ Jmp_sp.conj() @ spsolve(Mmp, Jmp_sp @ rk_normed) 
        # left invert (left term works with herm as well) 
        #skp_ip = (spsolve(Mp.T, herm(Jp_sp) @ lk_normed.conj())).T @ Jp_sp @ rk_normed 
        #skp_ip = (spsolve(Mmp.T, herm(Jmp_sp) @ lk_normed.conj())).T @ Jp_sp @ rk_normed) 

        print('   * Contribution of harmonic: ', skp_ip)
        print('   * Contribution of harmonic: ', skmp_ip)

        skp[ip] = skp_ip
        skmp[ip] = skmp_ip

    sk2 = np.sum(skp) + np.sum(skmp)
    print('---- Final result ----')
    print('*** +p ',  skp[:,None])
    print('*** -p: ', skmp[:,None])
    print('*** Sum: ', sk2)
    print('muk base: ', muk)
    print('muk corrected:', muk+np.sum(sk2))


    ############################################################################
    ## for external/matlab use only
    #if False:
    #    # save matrices as npz
    #    flu.spr.save_npz(resultpath + 'data/Jp.npz', Jp_sp)
    #    flu.spr.save_npz(resultpath + 'data/Jmp.npz', Jmp_sp)

    #    flu.spr.save_npz(resultpath + 'data/Mp.npz', Mp)
    #    flu.spr.save_npz(resultpath + 'data/Mmp.npz', Mmp)
    #    #flu.spr.save_npz(resultpath + 'data/A_mean.npz', A_mean_sp)
    #    #flu.spr.save_npz(resultpath + 'data/Q.npz', Quv_sp)

    #    flu.export_to_mat(resultpath + 'data/Jp.npz', resultpath + 'data/Jp.mat', 'Jp')
    #    flu.export_to_mat(resultpath + 'data/Jmp.npz', resultpath + 'data/Jmp.mat', 'Jmp')
    #    flu.export_to_mat(resultpath + 'data/Mp.npz', resultpath + 'data/Mp.mat', 'Mp')
    #    flu.export_to_mat(resultpath + 'data/Mmp.npz', resultpath + 'data/Mmp.mat', 'Mmp')
    #    #flu.export_to_mat(resultpath + 'data/A_mean.npz', resultpath + 'data/A_mean.mat', 'A_mean')
    #    #flu.export_to_mat(resultpath + 'data/Q.npz', resultpath + 'data/Q.mat', 'Q')
    ############################################################################

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------










