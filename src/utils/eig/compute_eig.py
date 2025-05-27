'''
Eigenvalues utilities
Warning: these functions should be run under a dedicated SLEPc environment
Warning: sometimes compulsory to run "mpirun -n 1 python eig_utils.py"
Warning: the function needs a few adjustments for running in parallel (~L216)
'''

import time

import numpy as np
import scipy as scp
import scipy.sparse as spr
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc

#from dolfin import *

import functools
#import matplotlib.pyplot as plt
#from matplotlib import cm

import pdb
import warnings

from eig_utils import *

#################################################################################
#################################################################################
#################################################################################
# Run
if __name__=='__main__':
    print('----- using slepc ----- begin')
    print('...............................')

    print('Start with FEM matrices')
    t0 = time.time()
    print('--- Loading matrices')
    save_npz_path = '/scratchm/wjussiau/fenics-results/cylinder_o1_eig_correction/data/'
    print('--- from sparse...')
    AA = spr.load_npz(save_npz_path + 'A_mean.npz') 
    BB = spr.load_npz(save_npz_path + 'Q.npz') 
    print('--- ... to petscmat')
    seq = True
    AA = sparse_to_petscmat(AA, sequential=seq)
    BB = sparse_to_petscmat(BB, sequential=seq)
    sz = AA.size[0]
    # targets
    #targets = np.array([[0.8+10j], [0.5+13j], [0.45+8j], [0.01+16j], [0]])
    #neiglist = np.array([[1],[1],[1],[1],[20]]) 
    targets = np.array([[0.15+1j]])
    neiglist = np.array([[1]]) 
    nt = len(targets)
    # store
    #LAMBDA = np.zeros((n*nt,), dtype=complex)
    #V = np.zeros((sz, n*nt), dtype=complex)


    import scipy.io as sio
    import scipy.sparse as spr
    def savemat(outfile, mat, matname):
        sio.savemat(outfile, mdict={matname: spr.csc_matrix(mat)})

    # solve A@x = Q*L@x
    LAMBDA = np.zeros((0,), dtype=complex)
    V = np.zeros((sz, 0), dtype=complex)
    print('--- Starting solve A x = Q L x ')
    for target, neig in zip(targets, neiglist):
        print('Current target: ', target)
        L, v, eigensolver = get_mat_vp_slepc(
            A = AA, 
            B = BB,
            target = target, 
            n=neig,
            return_eigensolver=True, 
            verbose=True,
            precond_type=PETSc.PC.Type.LU, 
            eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
            ksp_type=PETSc.KSP.Type.PREONLY,
            tol=1e-9)

        LAMBDA = np.hstack((LAMBDA, L))
        V = np.hstack((V, v)) 
    np.save(save_npz_path + 'muk_py', LAMBDA)
    np.save(save_npz_path + 'rk_py', V)
    savemat(outfile=save_npz_path + 'muk_py.mat', mat=LAMBDA, matname='muk')
    savemat(outfile=save_npz_path + 'rk_py.mat', mat=V, matname='rk')


    print('--- Starting solve A^H x = Q L* x ')
    LAMBDA = np.zeros((0,), dtype=complex)
    V = np.zeros((sz, 0), dtype=complex)
    for target, neig in zip(targets, neiglist):
        print('Current target: ', target)
        L, v, eigensolver = get_mat_vp_slepc(
            A = np.conj(AA.transpose()), 
            B = np.conj(BB.transpose()),
            target = np.conj(target), 
            n=neig,
            return_eigensolver=True, 
            verbose=True,
            precond_type=PETSc.PC.Type.LU, 
            eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
            ksp_type=PETSc.KSP.Type.PREONLY,
            tol=1e-9)

        LAMBDA = np.hstack((LAMBDA, L))
        V = np.hstack((V, v)) 
    np.save(save_npz_path + 'mukstar_py', LAMBDA)
    np.save(save_npz_path + 'lk_py', V)
    savemat(outfile=save_npz_path + 'mukstar_py.mat', mat=LAMBDA, matname='mukstar')
    savemat(outfile=save_npz_path + 'lk_py.mat', mat=V, matname='lk')

    print('Elapsed: %f' %(time.time() - t0))
    print('...............................')





