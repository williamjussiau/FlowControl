"""
Eigenvalues utilities
Warning: these functions should be run under a dedicated SLEPc environment
Warning: sometimes compulsory to run "mpirun -n 1 python eig_utils.py"
Warning: the function needs a few adjustments for running in parallel (~L216)
"""

import functools
import pdb
import time
import warnings

import numpy as np
import petsc4py
import scipy as scp
import scipy.sparse as spr
from petsc4py import PETSc
from slepc4py import SLEPc

# from dolfin import *

# import matplotlib.pyplot as plt
# from matplotlib import cm


# PETSc, scipy.sparse utility # GOTO ##########################################
def dense_to_sparse(A, eliminate_zeros=False):
    """Cast PETSc Matrix to scipy.sparse
    (Misleading name because A is not dense)"""
    if spr.issparse(A):
        return A

    if isinstance(A, np.ndarray):
        A = spr.csr_matrix(A)
        if eliminate_zeros:
            A = eliminate_zeros(A)
        return A

    Ac, As, Ar = A.getValuesCSR()
    Acsr = spr.csr_matrix((Ar, As, Ac))
    if eliminate_zeros:
        Acsr.eliminate_zeros()
    return Acsr


def sparse_to_petscmat(A, sequential=True):
    """Cast scipy.sparse matrix A to PETSc.Matrix()
    A should be scipy.sparse.xQxx and square"""
    t00 = time.time()
    A = A.tocsr()

    if sequential:
        Amat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    else:
        # Parallel
        # but much longer than sequential....!!
        # Create matrix of true size to get ownership range
        Amat = PETSc.Mat().create()
        Amat.setSizes(A.shape)
        Amat.setPreallocationNNZ(A.nnz)
        Amat.setUp()
        istart, iend = Amat.getOwnershipRange()
        # Reserve local indices
        local_csr = (
            A.indptr[istart : iend + 1] - A.indptr[istart],
            A.indices[A.indptr[istart] : A.indptr[iend]],
            A.data[A.indptr[istart] : A.indptr[iend]],
        )
        # Assign local indices and assemble
        # (Probably equiv but solution 1 is way faster)
        Amat = PETSc.Mat().createAIJ(size=A.shape, csr=local_csr)
        # Amat.assemblyBegin()
        # Amat.setValuesCSR(local_csr[0], local_csr[1], local_csr[2])
        # Amat.assemblyEnd()

    return Amat


def array_to_petscmat(A, eliminate_zeros=True):
    """Cast np array A to PETSc.Matrix()"""
    return sparse_to_petscmat(dense_to_sparse(A, eliminate_zeros))


###############################################################################


# Eig utility # GOTO ##########################################################
def get_mat_vp_slepc(
    A,
    B=None,
    n=10,
    DEBUG=False,
    target=0.0,
    return_eigensolver=False,
    verbose=False,
    eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
    precond_type=PETSc.PC.Type.NONE,
    ksp_type=PETSc.KSP.Type.GMRES,
    gmresrestart=1000,
    tol=1e-5,
    niter=1000,
    ncv=PETSc.DECIDE,
    mpd=PETSc.DECIDE,
    kspatol=None,
    ksprtol=None,
    kspdivtol=None,
    kspmaxit=None,
):
    """Get n eigenvales of matrix A
    Possibly with generalized problem (A, B).
    Direct SLEPc backend.
    A,B::PETScMatrix"""

    # EPS
    # Create problem
    eigensolver = SLEPc.EPS().create()
    eigensolver.setType(eps_type)  # default is krylovschur
    # eigensolver.setType(eigensolver.Type.POWER) # default is krylovschur
    # eigensolver.setType(eigensolver.Type.ARNOLDI) # default is krylovschur
    # Problem types:
    #   HEP = Hermitian Eigenvalue Problem
    #   NHEP = Non-Hermitian EP
    #   GHEP = Generalized HEP
    #   GHIEP = Generalized Hermitian-indefinite EP
    #   GNHEP = Generalized NHEP
    #   PGNHEP = Positive semi-definite B + GNHEP
    # Our problem is most likely not hermitian at all
    # Should be PGNHEP though
    if B is None:
        pbtype = eigensolver.ProblemType.NHEP
    else:
        pbtype = eigensolver.ProblemType.GNHEP
    eigensolver.setProblemType(pbtype)

    # Target
    # if target is not None:
    eigensolver.setTarget(target)
    eigensolver.setWhichEigenpairs(eigensolver.Which.TARGET_REAL)
    # else:
    #    eigensolver.setWhichEigenpairs(eigensolver.Which.LARGEST_REAL)

    # Tolerances
    eigensolver.setTolerances(tol, niter)
    eigensolver.setOperators(A, B)
    eigensolver.setDimensions(nev=n, mpd=mpd, ncv=ncv)

    # > ST
    # Spectral transforms:
    #   SHIFT = shift of origin: inv(B)*A + sigma*I
    #   SINVERT = shift and invert: inv(A-sigma*B)*B
    #   CAYLEY = Cayley: inv(A-sigma*B)*(A+v*B)
    #   PRECOND = preconditioner: inv(K)=approx inv(A-sigma*B)
    # see https://slepc.upv.es/handson/handson3.html
    # st = SLEPc.ST().create()
    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # if target is not None:
    #    st.setShift(target) # not being used by Johann?

    # >> KSP
    # ksp = PETSc.KSP().create()
    ksp = st.getKSP()
    ksp.setType(ksp_type)  # PETSc.KSP.Type. PREONLY MINRES GMRES CGS CGNE
    ksp.setGMRESRestart(gmresrestart)
    ksp.setTolerances(rtol=ksprtol, atol=kspatol, divtol=kspdivtol, max_it=kspmaxit)

    # >>> PC
    # pc = PETSc.PC().create()
    pc = ksp.getPC()
    pc.setType(precond_type)  # PETSc.PC.Type. LU(?) NONE works
    pc.setFactorSolverType("mumps")
    # FIELDSPLIT? need block matrix
    # pc.setFieldSplitType(pc.SchurFactType.UPPER)

    # >> KSP.PC
    # ksp.setPC(pc)
    # ksp set null space?
    if verbose == 10:

        def kspmonitor(kspstate, it, rnorm):
            # eps = SLEPc.EPS
            if not it % 100:
                print("\t --- KSP monitor --- nit: ", it, "+++ res: ", rnorm)
                # print('it: ', it)
                ##print('state: ', kspstate)
                # print('rnorm: ', rnorm)

        ksp.setMonitor(kspmonitor)

    # > ST.KSP
    # st.setKSP(ksp)

    # EPS.ST
    # eigensolver.setST(st)

    # Options?
    eigensolver.setFromOptions()

    if verbose:

        def epsmonitor(eps, it, nconv, eig, err):
            # eps = SLEPc.EPS
            print("\t --- EPS monitor --- nit: ", it, "+++ cvg ", nconv, "/", n)
            # print('eps it: ', it)
            # print('converged: ', nconv)
            ##print('eig: ', eig)
            ##print('err: ', err)

        eigensolver.setMonitor(epsmonitor)

    # Solve
    eigensolver.solve()

    nconv = eigensolver.getConverged()
    niters = eigensolver.getIterationNumber()

    if verbose:
        print("\t --- Computation terminated ---")

    sz = A.size[0]
    n = nconv
    valp = np.zeros((n,), dtype=complex)
    vecp = np.zeros((sz, n), dtype=complex)
    vecp_re = np.zeros((sz, n), dtype=float)
    vecp_im = np.zeros((sz, n), dtype=float)

    # vr = A.createVecRight()
    # vi = A.createVecRight()
    Vr, Vi = A.createVecs()

    for i in range(nconv):
        valp[i] = eigensolver.getEigenpair(i, Vr=Vr, Vi=Vi)

        ### allegedly this runs in parallel
        ## get ownership range (possibly different for real/imag???)
        # istart_r, iend_r = vr.getOwnershipRange()
        # istart_i, iend_i = vi.getOwnershipRange()
        ## fill in reserved part in vector
        # vecp_re[istart_r:iend_r, i] = np.real(V.array)
        # vecp_im[istart_r:iend_i, i] = np.imag(V.array)

        if verbose:
            print(
                "\t eig%2d = %9f + %9f*j" % (i + 1, np.real(valp[i]), np.imag(valp[i]))
            )

        # vecp = vecp_re + 1j*vecp_im
        vecp[:, i] = Vr.array + 1j * Vi.array

    LAMBDA = valp.reshape(
        -1,
    )  # make 1D
    V = vecp.reshape(sz, n)

    # if return_eigensolver or DEBUG:
    #    eigz = (eigz, eigensolver)

    return LAMBDA, V, eigensolver


#################################################################################
#################################################################################
#################################################################################
# Run
if __name__ == "__main__":
    if 0:
        print("----- using slepc ----- begin")
        print("...............................")

        print("Start with FEM matrices")
        t0 = time.time()
        print("--- Loading matrices")
        # save_npz_path = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/matrices/'
        # save_npz_path = '/scratchm/wjussiau/fenics-python/cavity/data/matrices/'
        # save_npz_path = '/scratchm/wjussiau/fenics-python/cavity/data/matrices_fine/'
        save_npz_path = "/scratchm/wjussiau/fenics-results/cylinder_o1_eig_correction/"
        print("--- from sparse...")
        AA = spr.load_npz(save_npz_path + "A_mean.npz")
        BB = spr.load_npz(save_npz_path + "Q.npz")
        print("--- ... to petscmat")
        seq = True
        AA = sparse_to_petscmat(AA, sequential=seq).transpose()
        BB = sparse_to_petscmat(BB, sequential=seq).transpose()
        sz = AA.size[0]
        # targets
        # targets = np.array([[0.8+10j], [0.5+13j], [0.45+8j], [0.01+16j], [0]])
        # neiglist = np.array([[1],[1],[1],[1],[20]])
        targets = np.array([[0.15 + 1j]])
        neiglist = np.array([[1]])
        nt = len(targets)
        # store
        # LAMBDA = np.zeros((n*nt,), dtype=complex)
        # V = np.zeros((sz, n*nt), dtype=complex)
        LAMBDA = np.zeros((0,), dtype=complex)
        V = np.zeros((sz, 0), dtype=complex)

        # solve
        print("--- Starting solve")
        for target, neig in zip(targets, neiglist):
            print("Current target: ", target)
            L, v, eigensolver = get_mat_vp_slepc(
                A=AA,
                B=BB,
                target=target,
                n=neig,
                return_eigensolver=True,
                verbose=True,
                precond_type=PETSc.PC.Type.LU,
                eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
                ksp_type=PETSc.KSP.Type.PREONLY,
                tol=1e-9,
            )

            LAMBDA = np.hstack((LAMBDA, L))
            V = np.hstack((V, v))

        np.save(save_npz_path + "eig_AmeanT", LAMBDA)
        np.save(save_npz_path + "eig_vec_AmeanT", V)
        # np.save(save_npz_path + 'eig_Amean', LAMBDA)
        # np.save(save_npz_path + 'eig_vec_Amean', V)
        print("Elapsed: %f" % (time.time() - t0))
        print("...............................")

    pass
