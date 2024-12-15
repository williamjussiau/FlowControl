'''
Utilitary functions for FlowSolver
'''

import time

import numpy as np
import scipy.signal as ss
import scipy.io as sio
import scipy.linalg as la
import scipy.sparse as spr
import scipy.sparse.linalg as spr_la
import control
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc

import os

import dolfin
from dolfin import dot, inner

import functools
from mpi4py import MPI as mpi
import matplotlib.pyplot as plt
from matplotlib import cm

#import pdb
import logging
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.debug('Importing or running: %s', __name__)


# Dolfin utility # GOTO ###################################################
def apply_fun(u, fun):
    '''Shortcut for applying numeric method to dolfin.dolfin.Function'''
    return fun(u.vector().get_local())


def show_max(u, name=''):
    '''Display max of dolfin.dolfin.Function'''
    logger.info('Max of vector "%s" is : %f' %(name, apply_fun(u, np.max)))


def write_xdmf(filename, func, name, time_step=0., append=False, write_mesh=True):
    '''Shortcut to write XDMF file with options & context manager'''
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as ff:
        ff.parameters['rewrite_function_mesh'] = write_mesh
        ff.parameters['functions_share_mesh'] = not write_mesh # does not work in FEniCS yet
        ff.write_checkpoint(func, name, time_step=time_step,
                            encoding=dolfin.XDMFFile.Encoding.HDF5,
                            append=append)


def read_xdmf(filename, func, name, counter=-1):
    '''Shortcut to read XDMF file with context manager'''
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as ff:
        ff.read_checkpoint(func, name=name, counter=counter)


def print0(*args, **kwargs):
    '''Print on process 0 only
    Could also do: print and flush stdout'''
    if MpiUtils.get_rank()==0:
        logger.info(*args, **kwargs)


# Shortcut to define projection with MUMPS
projectm = functools.partial(dolfin.project, solver_type='mumps')
###############################################################################


# MPI utility # GOTO ##########################################################
class MpiUtils():
    @staticmethod
    def get_rank():
        '''Access MPI rank in COMM WORLD'''
        return mpi.COMM_WORLD.Get_rank()

    @staticmethod
    def check_process_rank():
        '''Check process MPI rank'''
        comm = mpi.COMM_WORLD
        ip = comm.Get_rank()
        logger.info("================= Hello I am process %d", ip)

    @staticmethod
    def mpi4py_comm(comm):
        '''Get mpi4py communicator'''
        try:
            return comm.tompi4py()
        except AttributeError:
            return comm

    @staticmethod
    def peval(f, x):
        '''Parallel synced eval'''
        try:
            yloc = f(x)
        except RuntimeError:
            yloc = np.inf*np.ones(f.value_shape())
        comm = MpiUtils.mpi4py_comm(f.function_space().mesh().mpi_comm())
        yglob = np.zeros_like(yloc)
        comm.Allreduce(yloc, yglob, op=mpi.MIN)
        return yglob

    @staticmethod
    def peval2(f, x):
        '''Parallel synced eval, v2'''
        mesh = f.function_space().mesh()
        comm = mesh.mpi_comm()
        if comm.size == 1:
            return f(*x)
        # Find whether the point lies on the partition of the mesh local
        # to this process, and evaluate u(x)
        cell, distance = mesh.bounding_box_tree().compute_closest_entity(dolfin.Point(*x))
        f_eval = f(*x) if distance < dolfin.DOLFIN_EPS else None
        # Gather the results on process 0
        comm = mesh.mpi_comm()
        computed_f = comm.gather(f_eval, root=0)
        # Verify the results on process 0 to ensure we see the same value
        # on a process boundary
        if comm.rank == 0:
            global_f_evals = np.array([y for y in computed_f if y is not None], dtype=np.double)
            assert np.all(np.abs(global_f_evals[0] - global_f_evals) < 1e-9)
            computed_f = global_f_evals[0]
        else:
            computed_f = None
        # Broadcast the verified result to all processes
        computed_f = comm.bcast(computed_f, root=0)
        return computed_f

    @staticmethod
    def set_omp_num_threads():
        '''Memo for getting/setting OMP_NUM_THREADS, most likely does not work as is'''
        try:
            logger.info('nb threads was: %s', os.environ['OMP_NUM_THREADS'])
        except Exception as e:
            os.environ['OMP_NUM_THREADS'] = '1'
            raise(e)
        logger.info('nb threads is: %s', os.environ['OMP_NUM_THREADS'])

    @staticmethod
    def mpi_broadcast(x):
        '''Broadcast y to MPI (shortcut but longer)'''
        y = dolfin.MPI.comm_world.bcast(x, root=0)
        return y

###############################################################################



# PETSc, scipy.sparse utility # GOTO ##########################################
def dense_to_sparse(A, eliminate_zeros=True, eliminate_under=None):
    '''Cast PETSc or dolfin Matrix to scipy.sparse
    (Misleading name because A is not dense)'''
    def eliminate_zeros(A):
        if eliminate_under is None:
            A.eliminate_zeros()
        else:
            # assuming A is small, convert to dense, then back to sparse...
            # this is bad
            Adense = A.toarray()
            Adense[Adense<=eliminate_under] = 0
            A = spr.csr_matrix(Adense)
        return A

    if spr.issparse(A):
        #print('A is already sparse; exiting')
        if eliminate_zeros:
            A = eliminate_zeros(A)
        return A

    if isinstance(A, np.ndarray):
        A = spr.csr_matrix(A)
        if eliminate_zeros:
            A = eliminate_zeros(A)
        return A

    if not isinstance(A, PETSc.Mat):
        A = dolfin.as_backend_type(A).mat()

    Ac, As, Ar = A.getValuesCSR()
    Acsr = spr.csr_matrix((Ar, As, Ac))
    if eliminate_zeros:
        Acsr.eliminate_zeros()
    return Acsr


def sparse_to_petscmat(A):
    '''Cast scipy.sparse matrix A to PETSc.Matrix()
    A should be scipy.sparse.xQxx and square'''
    A = A.tocsr()

    Amat = PETSc.Mat().createAIJ(size=A.shape,
                                csr=(A.indptr,A.indices,A.data))

    return Amat


def array_to_petscmat(A, eliminate_zeros=True):
    '''Cast np array A to PETSc.Matrix()'''
    return sparse_to_petscmat(dense_to_sparse(A, eliminate_zeros))
###############################################################################


# Eig utility # GOTO ##########################################################
def get_mat_vp_slepc(A, B=None, n=10, DEBUG=False, target=0.0,
                     return_eigensolver=False, verbose=False,
                     eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
                     precond_type=PETSc.PC.Type.NONE,
                     ksp_type=PETSc.KSP.Type.GMRES,
                     gmresrestart=1000, tol=1e-5, niter=1000,
                     ncv=PETSc.DECIDE, mpd=PETSc.DECIDE,
                     kspatol=None, ksprtol=None, kspdivtol=None, kspmaxit=None):
    '''Get n eigenvales of matrix A
    Possibly with generalized problem (A, B).
    Direct SLEPc backend.
    A::dolfin.cpp.la.dolfin.PETScMatrix'''

    # EPS
    # Create problem
    eigensolver = SLEPc.EPS().create()
    eigensolver.setType(eps_type) # default is krylovschur
    #eigensolver.setType(eigensolver.Type.POWER) # default is krylovschur
    #eigensolver.setType(eigensolver.Type.ARNOLDI) # default is krylovschur
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
    #if target is not None:
    eigensolver.setTarget(target)
    eigensolver.setWhichEigenpairs(eigensolver.Which.TARGET_REAL)
    #else:
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
    #st = SLEPc.ST().create()
    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    #if target is not None:
    #    st.setShift(target) # not being used by Johann?

    # >> KSP
    #ksp = PETSc.KSP().create()
    ksp = st.getKSP()
    ksp.setType(ksp_type)  # PETSc.KSP.Type. PREONLY MINRES GMRES CGS CGNE
    ksp.setGMRESRestart(gmresrestart)
    ksp.setTolerances(rtol=ksprtol, atol=kspatol, divtol=kspdivtol, max_it=kspmaxit)

    # >>> PC
    #pc = PETSc.PC().create()
    pc = ksp.getPC()
    pc.setType(precond_type) # PETSc.PC.Type. LU(?) NONE works
    pc.setFactorSolverType('mumps')
    # FIELDSPLIT? need block matrix
    #pc.setFieldSplitType(pc.SchurFactType.UPPER)

    # >> KSP.PC
    #ksp.setPC(pc)
    #ksp set null space?
    if verbose:
        def kspmonitor(kspstate, it, rnorm):
            # eps = SLEPc.EPS
            if not it%100:
                logger.info('--- ksp monitor --- nit: %d +++ res %f', it, rnorm)
                #print('it: ', it)
                ##print('state: ', kspstate)
                #print('rnorm: ', rnorm)
        ksp.setMonitor(kspmonitor)

    # > ST.KSP
    #st.setKSP(ksp)

    # EPS.ST
    #eigensolver.setST(st)

    # Options?
    eigensolver.setFromOptions()

    if verbose:
        def epsmonitor(eps, it, nconv, eig, err):
            # eps = SLEPc.EPS
            logger.info('***** eps monitor ***** nit: %d +++ cvg %d / %d', it, nconv, n)
            #print('eps it: ', it)
            #print('converged: ', nconv)
            ##print('eig: ', eig)
            ##print('err: ', err)
        eigensolver.setMonitor(epsmonitor)

    # Solve
    eigensolver.solve()

    nconv =  eigensolver.getConverged()
    #niters = eigensolver.getIterationNumber()

    if verbose:
        logger.info('------ Computation terminated ------')

    sz = A.size[0]
    n = nconv
    valp = np.zeros(n, dtype=complex)
    vecp = np.zeros((sz, n), dtype=complex)
    vecp_re = np.zeros((sz, n), dtype=float)
    vecp_im = np.zeros((sz, n), dtype=float)

    vr = A.createVecRight()
    vi = A.createVecRight()

    for i in range(nconv):
        valp[i] = eigensolver.getEigenpair(i, Vr=vr, Vi=vi)

        # get ownership range (possibly different for real/imag)
        istart_r, iend_r = vr.getOwnershipRange()
        istart_i, iend_i = vi.getOwnershipRange()
        # fill in reserved part in vector
        vecp_re[istart_r:iend_r, i] = vr.array
        vecp_im[istart_r:iend_i, i] = vi.array

        if verbose:
            logger.info('Eigenvalue %2d is: %f+i %f' % (i+1, np.real(valp[i]), np.imag(valp[i])))

    vecp = vecp_re + 1j*vecp_im

    eigz = (valp, vecp)
    if return_eigensolver or DEBUG:
        eigz = (eigz, eigensolver)

    return eigz


def make_mat_to_test_slepc(view=False, singular=False, neigpairs=3, density_B=1.0, rand=False):
    sz = 2*neigpairs
    if neigpairs==3:
        # known problem
        re = [0.2, -0.5, -0.7] # real part of eigenval
        im = [0.25, 0.8,  0.0] # imag part of eigenval
    else:
        if rand:
            re = np.random.randn(neigpairs) # real part of eigenval
            im = np.random.randn(neigpairs) # imag part of eigenval
        else:
            eigstep = 0.1
            re=np.arange(start=-neigpairs*eigstep, stop=0, step=eigstep)
            im=np.arange(start=-neigpairs*eigstep, stop=0, step=eigstep)
            # eigenvalues from -10 to -1 (both re and im)
            #re = np.linspace(-10, -1, neigpairs)
            #im = np.linspace(-10, -1, neigpairs)

    def makemat_conjeig(a,b): # make 2x2 matrix with given conj eigenval
        return [[a, -b], [b, a]]

    # slow because of reallocation
    #a = makemat_conjeig(re[0], im[0])
    #for i in range(1, neigpairs): # make block matrix with a+1jb as eigenval
    #    a = la.block_diag(a, makemat_conjeig(re[i], im[i]))
    # hopefully faster
    A = np.zeros((sz, sz))
    for i in range(neigpairs):
        j = 2*i
        A[j:j+2, j:j+2] = makemat_conjeig(re[i], im[i]) # i+2 excluded

    # make b in numpy to set values easily, then convert
    B = np.eye(sz)
    if singular: # makes b non invertible
        if neigpairs==3:
            B[1, 1] = 0
            B[2, 2] = 0
        else:
            # number of zeros in B
            nrzeros = int(sz * (1-density_B))
            if rand:
                # where zeros are located
                wherezeros = np.random.permutation(sz)[:nrzeros]
            else:
                wherezeros = np.arange(nrzeros)
            B[wherezeros, wherezeros] = 0

    if view:
        return A, B
    else:
        return array_to_petscmat(A), array_to_petscmat(B)


def load_mat_from_file(mat_type='fem32'):
    prefix = '/stck/wjussiau/fenics-python/ns/data/matrices/'
    suffix = '.npz'
    def mkpath(name):
        return prefix + name + suffix
    if mat_type=='fem32':
        Afile = mkpath('A_sparse_nx32')
        Qfile = mkpath('Q_sparse_nx32')
    else:
        if mat_type=='fem64':
            Afile = mkpath('A_sparse_nx64')
            Qfile = mkpath('Q_sparse_nx64')
        else:
            Afile = mkpath('A_rand')
            Qfile = mkpath('B_rand')
    return spr.load_npz(Afile), spr.load_npz(Qfile)


def geig_singular(A, B, n=2, DEBUG=False, target=None, solve_dense=False):
    '''Get n eigenvales of generalized problem (A, B)
    Where B is a singular matrix (hence no inv(B))
    Direct SLEPc backend.'''

    # Setup problem
    # works with classic matrices for now, not PETSc
    s = 1.2
    nn = A.shape[0]
    sznp = nn-np.linalg.matrix_rank(B)
    C = np.ones((nn-sznp, nn-sznp))

    At, Bt = augment_matrices(A, B, s, C)
    #print('matrices A, B are: \n', At, '\n', Bt)

    if solve_dense:
        Dt, Vt = la.eig(At, Bt)
    else:
        At = dense_to_sparse(At)
        Bt = dense_to_sparse(Bt)
        if target is None:
            target = 0.0
        Asb = At - target*Bt
        Asb = Asb.tocsc()
        LU = spr_la.splu(Asb)
        OPinv = spr_la.LinearOperator(matvec=lambda x: LU.solve(x), shape=Asb.shape)
        OPinv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Asb, x, tol=1e-5)[0], shape=Asb.shape)
        Dt, Vt = spr_la.eigs(A=At, k=n, M=Bt, tol=0, sigma=target, OPinv=OPinv)
        logger.info('Embedded sparse eig: %f', Dt)

    Vt_zero = Vt[-sznp:, :]
    EPS = np.finfo(float).eps
    idx_true_eig = np.all(np.abs(Vt_zero) < EPS*1e3, axis=0)

    true_eigval = Dt[idx_true_eig]
    true_eigvec = Vt[:-sznp, idx_true_eig]

    return true_eigval, true_eigvec


def augment_matrices(A, B, s, C):
    '''Augment matrices to make eigenproblem nonsingular
    Inputs are dense matrices [A, B, C] and scalar [s]'''
    N = la.null_space(B)
    M = la.null_space(B.T).T
    #sznp = N.shape[1]

    s = 1.2
    MA = M @ A
    AN = A @ N
    C = np.ones((MA.shape[0], AN.shape[1]))

    Aa = np.block([[A, s*AN], [MA, s*C]])
    Ba = np.block([[B, AN], [MA, C]])

    return Aa, Ba


def get_all_enum_petsc_slepc(enum):
    '''Get every possible value of enum in PETSc/SLEPc subpackage
    e.g. PETSc.KSP.Type, PETSc.PC.Type... '''
    allenum = enum.__dict__.copy()
    keys_to_pop = []
    for key, value in allenum.items():
        if key[0]=='_':
            keys_to_pop.append(key)
    for key in keys_to_pop:
        allenum.pop(key)
    return allenum


def get_mat_vp(A, B=None, n=3, DEBUG=False):
    '''Get n eigenvales of matrix A
    Possibly with generalized problem (A, B).
    dolfin backend (with slepc back-backend).
    A::dolfin.cpp.la.dolfin.PETScMatrix'''
    eigensolver = dolfin.SLEPcEigenSolver(A, B)
    eigensolver.solve(n)
    nconv = eigensolver.get_number_converged()
    logger.info('Tried: %d, converged: %d' % (n, nconv))

    sz = A.size(0)
    valp = np.zeros(nconv, dtype=complex)
    c = np.zeros_like(valp)
    vecp = np.zeros((sz, nconv), dtype=complex)
    cx = np.zeros_like(vecp)

    for i in range(nconv):
        valp[i], c[i], vecp[:, i], cx[:, i] = eigensolver.get_eigenpair(i)
        logger.info('Eigenvalue %d is: %f+i %f' % (i+1, np.real(valp[i]), np.imag(valp[i])))

    if DEBUG:
        return (valp, vecp), eigensolver
    else:
        return valp, vecp


# Export utility # GOTO #######################################################
def export_field(cfields, W, V, P, save_dir=None, time_steps=None):
    '''Export complex field to files, for visualizing in function space
    May be used to export any matrix defined on the function spaces W=(V,P),
    such as the actuation and sensing matrices B, C
    cfields: array with cfields as columns
    Usage: export_field(cfields, fs.W, fs.V, fs.P, ...)'''
    if save_dir is None:
        save_dir = '/stck/wjussiau/fenics-python/ns/data/export/vec_'
    vec_v_file = save_dir + '_v'
    vec_p_file = save_dir + '_p'
    xdmf = '.xdmf'
    def mkfilename(filename, part):
        return filename + '_' + part + xdmf

    ww = dolfin.Function(W)
    vv = dolfin.Function(V)
    pp = dolfin.Function(P)
    fa = dolfin.FunctionAssigner([V, P], W)

    if time_steps is None:
        time_steps = list(range(cfields.shape[1]))

    is_append = False
    for i in range(cfields.shape[1]): # nr of vecp
        cfield = cfields[:,i]

        # split re, im
        cfield_re = np.real(cfield)
        cfield_im = np.imag(cfield)
        cfield_abs = np.abs(cfield)
        cfield_arg = np.angle(cfield)

        # for re, im: export
        for cfield_part, cfield_part_name in zip([cfield_re, cfield_im, cfield_abs, cfield_arg],
            ['re', 'im', 'abs', 'arg']):
            # copy eigen vector to W-function
            ww.vector().set_local(cfield_part)
            # split W to V, P
            fa.assign([vv, pp], ww)
            # write v, p separately (no choice with FEniCS)
            write_xdmf(mkfilename(vec_v_file, cfield_part_name),
                       vv,
                       'v_eig_' + cfield_part_name,
                       time_step=time_steps[i], append=is_append)
            write_xdmf(mkfilename(vec_p_file, cfield_part_name),
                       pp,
                       'p_eig_' + cfield_part_name,
                       time_step=time_steps[i], append=is_append)

        # next eigenvec: append to file
        is_append = True
        logger.info('Writing eigenvector: %d' %(i+1))


def export_sparse_matrix(A, figname=None):
    '''Export sparse matrix to spy plot
    A==dolfin.cpp.la.dolfin.PETScMatrix or scipy.sparse.csr_matrix'''
    if spr.issparse(A):
        Acsr = A
    else:
        # probably: A is dolfin.PETScMatrix
        Acsr = dense_to_sparse(A)

    # Make plot
    fig, ax = plt.subplots()
    ax.spy(Acsr, markersize=1)
    ax.set_title('Sparse matrix plot')
    # Export plot
    if figname is None:
        figname = 'spy.png'
    fig.savefig(figname)


def export_to_mat(infile, outfile, matname, option='sparse'):
    '''Load sparse matrix from infile.npz and export it to outfile.mat'''
    if option=='sparse':
        Msp = spr.load_npz(infile)
        sio.savemat(outfile, mdict={matname: Msp.tocsc()})
    else: # as dict
        sio.savemat(outfile, mdict=np.load(infile))


def export_flowsolver_matrices(fs, path, suffix=''):
    '''Export A, B, C, Q matrices of high dim state-space representation
    of flow, in sparse and mat formats'''
    # Gather matrices
    A = fs.get_A()
    B = fs.get_B()
    C = fs.get_C()
    Q = fs.get_mass_matrix()

    for mat, matname in zip([A, B, C, Q], ['A', 'B', 'C', 'Q']):
        # Convert to sparse
        mat_spr = dense_to_sparse(mat,
            eliminate_zeros=matname=='B', eliminate_under=1e-14)

        # Mat sparse
        mat_file = path + matname + suffix + '.mat'
        sio.savemat(mat_file, mdict={matname: mat_spr.tocsc()})

        ## Mat struct
        #mat_file_struct = path + matname + suffix + '_struct.mat'
        #sio.savemat(mat_file_struct, mdict=np.load(sparse_file))

        ## Python NPZ
        #sparse_file = path + matname + suffix + '.npz'
        #spr.save_npz(sparse_file, mat_spr)

        ## Mat COO
        ##mat_file_coo = path + matname + '_coo.mat'
        ##sio.savemat(mat_file_coo, mdict={matname: mat_spr.tocoo()})


def export_dof_map(W, plotsz=None):
    '''Create an image of size W.dim*W.dim with each column
    being the colour of the underlying function space
    of the corresponding dof
    E.G. (column i==red) if (dof i==u)
    By default, print all dofs, but one can print less'''
    dofmap = get_subspace_dofs(W)
    sz = W.dim()

    if plotsz is None:
        plotsz = sz

    dofim = np.zeros((plotsz, plotsz), dtype=int)

    for i, subs in enumerate(['u','v','p']):
        dofmap_idx = dofmap[subs]
        subs_low = dofmap_idx[dofmap_idx < plotsz]
        dofim[:, subs_low] = i

    fig, ax = plt.subplots()
    im = ax.imshow(dofim, cmap=cm.get_cmap('binary'))
    ax.set_title('Distribution of DOFs by index: (u,v,p)')
    fig.colorbar(im)
    fig.savefig('dofmap.png')


def export_subdomains(mesh, subdomains_list, filename='subdomains.xdmf'):
    """Export subdomins of FlowSolver object to be displayed
    Usage: export_subdomains(fs.mesh, fs.boundaries.subdomain, ...)"""
    subd = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)
    for i, subdomain in enumerate(subdomains_list):
        subdnr = 10*(i+1)
        subdomain.mark(subd, subdnr)
        logger.info('Marking subdomain nr: {0} ({1})'.format(i+1, subdnr))
    logger.info('Writing subdomains file: %s', filename)
    with dolfin.XDMFFile(filename) as fsubd:
        fsubd.write(subd)


def export_facetnormals(mesh, ds, n=None, filename='facet_normals.xdmf',
                        name='facet_normals'):
    '''Write input:{mesh} facet normals {n} to file
    Can be used to export forces (n << dot(sigma, n))'''
    # Define facet normals of mesh
    if n is None:
        n = dolfin.FacetNormal(mesh)
    # Define CG1 fspace
    V = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
    # Define variational projection problem
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    a_lhs = inner(u, v)*ds
    l_rhs = inner(n, v)*ds
    A = dolfin.assemble(a_lhs, keep_diagonal=True)
    L = dolfin.assemble(l_rhs)
    A.ident_zeros()
    # Project
    nh = dolfin.Function(V, name=name)
    dolfin.solve(A, nh.vector(), L)
    # Write
    write_xdmf(filename, nh, name)


def export_stress_tensor(sigma, mesh, filename='stress_tensor.xdmf',
                         export_forces=False,
                         ds=None,
                         name='stress_tensor'):
    '''Write input:{stress tensor} to file'''
    # Make tensor function space
    TT = dolfin.TensorFunctionSpace(mesh, 'DG', degree=0)
    # Project
    sigma_ = dolfin.Function(TT, name=name)
    sigma_.assign(projectm(sigma, TT))
    # Write
    write_xdmf(filename, sigma_, name)

    if export_forces:
        n = dolfin.FacetNormal(mesh)
        export_facetnormals(mesh=mesh, n=-dot(sigma, n), ds=ds, filename='forces.xdmf', name='forces')
###############################################################################


# Signal processing and array utility # GOTO ##################################
def compute_signal_frequency(sig, Tf, nzp=10):
    '''Compute frequency of periodic signal
    with FFT+zero-padding (nzp*len(sig)).
    Can be used to compute Strouhal number
    by setting sig=Cl...'''
    fftstart = int((Tf/2)/dt)
    sig_cp = sig
    sig_cp = sig_cp[fftstart:]
    sig_cp = sig_cp - np.mean(sig_cp)
    Fs = 1/dt
    nn = len(sig_cp)*nzp
    frq = np.arange(nn)*Fs/nn
    frq = frq[:len(frq)//2]
    Y = np.fft.fft(sig_cp, nn)/nn
    Y = Y[:len(Y)//2]
    return frq[np.argmax(np.abs(Y))]


def sample_lco(Tlco, Tstartlco, nsim): # dt, save_every_old
    '''Define times for sampling a LCO in simulation
    Tlco: period of LCO
    Tstartlco: beginning of simulations - has to match a saved step
    nsim: number of simulations to be run'''
    tcl = Tstartlco + Tlco/nsim * np.arange(nsim)
    return tcl
    ##tcl = [Tstartlco, Tstartlco+Tlco/2]
    #tsave = 2000*0.005
    #stl = [tsave*(i//tsave) for i in tcl] # closest lesser multiple of tsave

    #stl.reverse()
    #for i in range(len(stl)):
    #    if i+1<len(stl) and stl[i]==stl[i+1]:
    #        stl[i+1:] = [el-tsave for el in stl[i+1:]]
    #stl.reverse()
    #return stl, tcl


def pad_upto(L, N, v=0):
    '''Pad list/np.ndarray L with v, so that it has N elements
    It is assumed that len(L)<N'''
    if type(L) is list:
       return L + (N-len(L))*[v]
    if type(L) is np.ndarray:
        return np.pad(L, pad_width=(0,N-L.shape[0]), mode='constant', constant_values=(v))
    else:
        logger.info('Type for padding not supported')
        return L


def saturate(x, xmin , xmax):
    '''Saturate a signal x between [xmin, xmax]
    This implementation (native Py) is faster for signals of small dimension'''
    # max(xmin, min(xmax, x))
    # sorted((x, xmin, xmax))[1]
    # np.clip
    return xmin if x < xmin else xmax if x > xmax else x
###############################################################################


# Controller utility # GOTO ###################################################
def step_controller(K, x, e, dt):
    '''Wrapper for stepping controller on one time step, from state (x),
    with input(e), up to time (dt) >> u=K*e
    Return controller output u and controller new state x'''
    #import pdb
    #pdb.set_trace()
    e_rep = np.repeat(np.atleast_2d(e), repeats=2, axis=0).T

    #import pdb
    #pdb.set_trace()

    Tsim = [0, dt]
    _, yout, xout = control.forced_response(
        K, U=e_rep, T=Tsim, X0=x,
        interpolate=False, return_x=True) # y=y(t)
    u = np.ravel(yout)[0]
    x = xout[:, 1] # this is x(t+dt)
    return u, x


def read_matfile(path):
    '''Read mat file without Duplicate variable warning'''
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Duplicate variable name*')
        return sio.loadmat(path)


def read_regulator(path):
    '''Read matrices of scipy.signal.StateSpace from provided .mat file path'''
    rd = read_matfile(path)
    return ss.StateSpace(rd['A'], rd['B'], rd['C'], rd['D'])


def read_ss(path):
    '''Read matrices of control.StateSpace from provided .mat file path
    (same as read_regulator but with control toolbox)'''
    rd = read_matfile(path)
    return control.StateSpace(rd['A'], rd['B'], rd['C'], rd['D'])


def write_ss(sys, path):
    '''Write control.StateSpace to file'''
    ssdict = {'A': sys.A, 'B': sys.B, 'C': sys.C, 'D': sys.D}
    sio.savemat(path, ssdict)
    return 0
###############################################################################


# FlowSolver direct utility # GOTO ############################################
def get_subspace_dofs(W):
    '''Return a map of which function space contain which dof'''
    def get_dofs(V): return np.array(V.dofmap().dofs(), dtype=int)
    subspace_dof = {'u': get_dofs(W.sub(0).sub(0)),
                    'v': get_dofs(W.sub(0).sub(1)),
                    'p': get_dofs(W.sub(1))}
    return subspace_dof


def end_simulation(fs, t0=None):
    '''Display information about simulation done by FlowSolver fs'''
    if fs.num_steps > 3:
            if t0 is not None:
                logger.info('Total time is: %f', time.time() - t0)
            logger.info('Iteration 1 time     --- %f', fs.timeseries.loc[1, 'runtime'])
            logger.info('Iteration 2 time     --- %f', fs.timeseries.loc[2, 'runtime'])
            logger.info('Mean iteration time  --- %f', np.mean(fs.timeseries.loc[3:, 'runtime']))
            logger.info('Time/iter/dof        --- %f', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])


def make_mesh_param(meshname):
    '''Make mesh parameters from given name
    Essentially a shortcut to defining mesh characteristics'''
    meshname = meshname.lower()
    # nx32
    if meshname=='nx32':
        params_mesh = {'genmesh': True,
                       'remesh': False,
                       'nx': 32,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/',
                       'meshname': 'S53.xdmf',
                       'xinf': 20, #50, # 20
                       'xinfa': -5, #-30, # -5
                       'yinf': 8, #30, # 8
                       'segments': 360}
    # m1
    if meshname=='m1':
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 32,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/',
                       'meshname': 'M1.xdmf',
                       'xinf': 40,
                       'xinfa': -25,
                       'yinf': 25,
                       'segments': 360}
    # n1
    if meshname=='n1':
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 1,
                       'meshpath': '/stck/wjussiau/fenics-python/mesh/',
                       'meshname': 'N1.xdmf',
                       'xinf': 20,
                       'xinfa': -10,
                       'yinf': 10,
                       'segments': 360}
    # o1
    if meshname=='o1':
        params_mesh = {'genmesh': False,
                       'remesh': False,
                       'nx': 1,
                       #'meshpath': '/stck/wjussiau/fenics-python/mesh/',
                       'meshpath': '/Volumes/Samsung_T5/Travail/ONERA/Travail/Productions/Avancement/ALL_FILES/stck/fenics-python/mesh/',
                       'meshname': 'O1.xdmf',
                       'xinf': 20,
                       'xinfa': -10,
                       'yinf': 10,
                       'segments': 360}
    return params_mesh
###############################################################################


# Cross optimization-Flowsolver utility # GOTO ################################
def compute_cost(fs, criterion, u_penalty, fullstate=True, scaling=None,
    verbose=True, diverged=False, diverged_penalty=50):
    '''Compute cost associated to FlowSolver object
    criterion = integral or terminal (str)
    u_penalty = penalty to control energy
    fullstate = True:xQx, False:yQy
    diverged & diverged_penalty = if simulation diverged, use arbitrary cost'''
    # Divergence -> arbitrary cost
    if diverged:
        return diverged_penalty

    # Default scaling = Id
    if scaling is None:
        def scaling(x):
            return x
    # examples are: scaling = lambda x: np.log10(x)

    # Normalization = time average
    #Tnorm = fs.dt / (fs.timeseries.loc[:, 'time'].iloc[-1] - fs.Tc)
    Tnorm = fs.dt / (fs.t - fs.Tc) # Tc most likely false most of the time

    # State-related cost
    if criterion == 'integral': # integral of energy
        if fullstate:
            xQx = np.sum(scaling(fs.timeseries.loc[:, 'dE']))
        else: # only y
            y_meas_str = fs.make_y_dataframe_column_name()
            y2_arr = (fs.timeseries.loc[:, y_meas_str]**2).to_numpy()
            xQx = np.sum(y2_arr.ravel())
        xQx *= Tnorm

    else: # terminal energy
        if fullstate:
            xQx = scaling(fs.timeseries.loc[:, 'dE'].iloc[-1])
        else: # only y
            y_meas_str = fs.make_y_dataframe_column_name()
            y2_end = (fs.timeseries.loc[:, y_meas_str].iloc[-1]**2).to_numpy()
            xQx = np.sum(y2_end)

    # Control input regularization
    uRu = np.sum(fs.timeseries.u_ctrl**2) * Tnorm

    # Sum
    J = xQx + u_penalty*uRu

    #import pdb
    #pdb.set_trace()

    # Show
    if verbose:
        logger.info('grep [energy, regularization]: %f', [xQx, u_penalty*uRu])
        logger.info('grep full cost: %f', J)

    return J


def write_optim_csv(fs, x, J, diverged, write=True):
    '''Write csv file associated to 1 controller of parameter x,
    with associated costfunction J'''
    if write:# and not diverged:
        # Ensure x is 1D: x.shape=(nx,)
        x = x.reshape(-1,)
        # Export CSV
        # theta as string list
        sl = ['{:.3f}'.format(xi).replace('.',',') for xi in x]
        # J as string
        jstr = '{:.5f}'.format(J).replace('.',',')
        # turn string list to file extension
        file_extension = '_'.join(sl)
        # make timeseries file path
        timeseries_path = fs.savedir0 + 'timeseries/timeseries_' + \
                    'J='+ jstr + '_' + 'x=' + file_extension + '.csv'
        # write timeseries
        if MpiUtils.get_rank()==0:
            fs.timeseries.to_csv(timeseries_path, sep=',', index=False)
    return 1
###############################################################################


# FlowSolver frequency response utility # GOTO ################################
def get_Hw(fs, A=None, B=None, C=None, D=None, Q=None, logwmin=-2, logwmax=2, nw=10,
           save_dir='/scratchm/wjussiau/fenics-python/cylinder/data/temp/', save_suffix='', verbose=True):
    '''Get frequency response of infinite-dimensional system
    One can pass A, B, C, D read from file or small dimension'''

    MpiUtils.check_process_rank()

    # solve: given w, (jwQ-A)x=B, then y=Cx
    ww = np.logspace(logwmin, logwmax, num=nw)
    ns = fs.sensor_nr
    # 1 line per sensor
    # 1 column per freq
    Hw = np.zeros((ns, len(ww)), dtype=complex)

    # If input is none, get A B C from FlowSolver object
    if A is None:
        A = fs.get_A()
    if B is None:
        B = fs.get_B()
    if C is None:
        C = fs.get_C()
    if D is None:
        D = 0
    if Q is None:
        Q = fs.get_mass_matrix()
        # should we do that???
        #[bc.apply(E) for bc in fs.bc['bcu']]
    Q = dense_to_sparse(Q)

    # Sub blocks (before loop)
    if verbose:
        logger.info('Defining sub-blocks (A, E)')
    Acsr = dense_to_sparse(A)
    sz = Acsr.shape[0]  # fs.W.dim()
    #print('size of acsr is:', sz)

    petsc4py.init()
    Rb = PETSc.Mat().create()
    Rb.setSizes((2*sz, 2*sz))
    Rb.setUp()
    Rb.assemble()

    rstart, rend = Rb.getOwnershipRange()
    #print('my ownership range is: ', rstart, rend)

    # B, C
    #C0 = np.zeros_like(C)
    B0 = np.zeros_like(B)
    #Czero = np.hstack([C, C0])
    #zeroC = np.hstack([C0, C])
    Bzero = np.vstack([B, B0])
    CjC = np.hstack([C, 1j*C])

    # solver
    solvR = dolfin.LUSolver('mumps')
    #solvR = PETSc.KSP().create()
    # setup ksp here

    hw_timings = {'stack': 0, 'createAIJ': 0, 'createmat': 0, 'solve': 0, 'decompose': 0}
    # 0: make block matrix (scipy)
    # 1: create petsc matrix
    # 2: create petsc vectors + transform to dolfin
    # 3: solve system
    # 4: multiply x
    tb = time.time()
    for ii, w in enumerate(ww):
        if verbose:
            logger.info('Computing %d/%d with puls: %5.3f...' %(ii+1, len(ww), w))
        # Block matrix
        t00 = time.time()

        # from here to......
        Ablk = spr.bmat([[-Acsr, -w*Q], [w*Q, -Acsr]]) # TODO convert to PETSc
        Ablk = Ablk.tocsr()
        #Ablk_csr = (Ablk.indptr, Ablk.indices, Ablk.data)

        Ablk_csr = (  Ablk.indptr[rstart:rend+1] - Ablk.indptr[rstart],
                      Ablk.indices[Ablk.indptr[rstart]:Ablk.indptr[rend]],
                      Ablk.data[Ablk.indptr[rstart]:Ablk.indptr[rend]]     )

        hw_timings['stack'] += - t00 + time.time()

        t01 = time.time()
        Rb.createAIJ(size=(2*sz, 2*sz), csr=Ablk_csr, comm=PETSc.COMM_WORLD)
        #Rb.createAIJWithArrays(size=[2*sz, 2*sz], csr=Ablk_csr, comm=PETSc.COMM_WORLD)

        Rb.assemblyBegin()
        Rb.assemblyEnd()
        hw_timings['createAIJ'] += - t01 + time.time()

        # Create solution (in LHS) & RHS
        t02 = time.time()
        vecx, vecb = Rb.createVecs()
        #vecx = Rb.createVecRight()
        #print('size of vx after createvec: ', vecx.size)
        #print('size of vb after createvec: ', vecb.size)
        #print('size of bzero: ', Bzero.shape)
        #vecb = PETSc.Vec().createWithArray(Bzero, comm=PETSc.COMM_WORLD)
        rbstart, rbend = vecb.getOwnershipRange()
        #print('ownership range is: ', rbstart, rbend)
        #vecb.createWithArray(Bzero, comm=PETSc.COMM_WORLD)
        localidx = list(range(rbstart, rbend))
        vecb.setValues(localidx, Bzero[localidx])
        #print('size of vb after createarray: ', vecb.size)
        vecx, vecb = dolfin.PETScVector(vecx), dolfin.PETScVector(vecb)
        #print('size of vx after convert: ', vecx.size())
        #print('size of vb after convert: ', vecb.size())

        # Casting matrix
        ############################## warning: this is back to dolfin
        #Rmat = Rb
        #solvR.setOperators(Rmat)
        #solvR.setFromOptions()
        Rmat = dolfin.PETScMatrix(Rb)
        hw_timings['createmat'] += - t02 + time.time()

        # Solving system
        t03 = time.time()
        #exitFlag = solvR.solve(Rmat, vecx, vecb)
        solvR.solve(Rmat, vecx, vecb)
        #solvR.solve(vecb, vecx)
        hw_timings['solve'] += - t03 + time.time()
        # .............. here
        # the function determines the field response x = inv(jwQ-A)*B for given w

        ########################################################
        # this part has to take into account the nr of sensors #
        ########################################################
        #print('max of x is:', max(vecx.get_local()))
        t04 = time.time()

        rxstart, rxend = vecx.vec().getOwnershipRange()
        localidx = list(range(rxstart, rxend))
        #print('size of vecx local: ', vecx.get_local().shape)
        #print('size of vecx local slice: ', vecx.get_local()[:sz].shape)
        #print('size of C: ', C.shape)

        vecy = spr.csr_matrix(CjC[:,localidx]).dot(vecx.get_local())
        #vecy = spr.csr_matrix(C).dot(vecx.get_local()[:sz]) + D
        #ivecy = spr.csr_matrix(C).dot(vecx.get_local()[sz:])
        hw_timings['decompose'] += - t04 + time.time()

        #print('  --- y is:', vecy)
        #print('  --- iy is:', ivecy)

        #Hw[:, ii] = vecy + 1j * ivecy
        #Hw[:, ii] = vecy # mpi.allgather ?
        Hw[:, ii] = dolfin.MPI.comm_world.reduce(vecy, root=0) # mpi.allgather ?

        if verbose:
            for sn in range(ns):
                logger.info('\t magnitude is: %5.4f' %(np.abs(Hw[sn, ii])))

    if verbose:
        logger.info('Elapsed computing {0} pulsations: {1}'.format(len(ww), time.time()-tb))


    if save_dir and MpiUtils.get_rank()==0: # if dir not empty string & rank 0
        # save whole frequency response with sensor positions
        suffix = '_' + str(ns) + 'sensors'
        savepath = save_dir + 'Hw_nw' + str(nw) + save_suffix + suffix + '.mat'
        sio.savemat(savepath, {'H': Hw, 'w': ww, 'xs': fs.sensor_location,
                               'comment': '1 line = 1 sensor'})

        # save every sensor response as separate file
        for sn in range(ns):
            xs_i = fs.sensor_location[sn]
            Hw_i = Hw[sn, :]

            suffix = '_dx=' + str(xs_i[0]) + '_dy=' + str(xs_i[1])

            savepath = save_dir + 'Hw_nw' + str(nw) + save_suffix + suffix + '.mat'
            sio.savemat(savepath, {'H': Hw_i, 'w': ww, 'xs': xs_i})
            logger.info('Saving frequency response to: %s', savepath)

            fig, axs = plt.subplots(2, 1)

            axs[0].grid(which='both')
            axs[0].scatter(ww, 20*np.log10(np.abs(Hw_i)), marker='.')
            axs[0].set_title('Amplitude response')
            axs[0].set_xlabel('Frequency')
            axs[0].set_xscale('log')

            axs[1].grid(which='both')
            axs[1].scatter(ww, (180/np.pi)*np.unwrap(np.angle(Hw_i)), marker='.')
            axs[1].set_title('Phase response')
            axs[1].set_xlabel('Frequency')
            axs[1].set_xscale('log')

            fig.tight_layout()
            fig.savefig(save_dir + 'bodeplot_' + suffix + save_suffix + '.png')
            plt.close(fig)

    return Hw, ww, hw_timings


def get_field_response(fs, w, A=None, B=None, Q=None, verbose=True):
    '''Get field response at frequency w
    The field is defined by: x=inv(sQ-A)*B
    This function should be used in get_Hw()
    WARNING: FUNCTION UNDER WORK'''
    if A is None:
        A = fs.get_A()
    if B is None:
        B = fs.get_B()
    if Q is None:
        Q = fs.get_mass_matrix()
    Acsr = dense_to_sparse(A)
    sz = Acsr.shape[0]#fs.W.dim()
    Q = dense_to_sparse(Q)

    # B, C
    Bzero = np.vstack([B, np.zeros_like(B)])

    # solver
    solvR = dolfin.LUSolver('mumps')
    Rb = PETSc.Mat()

    hw_timings = {'stack': 0, 'createAIJ': 0, 'createmat': 0, 'solve': 0, 'decompose': 0}
    # 0: make block matrix (scipy)
    # 1: create petsc matrix
    # 2: create petsc vectors + transform to dolfin
    # 3: solve system
    # 4: multiply x
    #tb = time.time()
    #for ii, w in enumerate(ww):
    # Block matrix
    t00 = time.time()
    Ablk = spr.bmat([[-Acsr, -w*Q], [w*Q, -Acsr]])
    Ablk = Ablk.tocsr()
    Ablk_csr = (Ablk.indptr, Ablk.indices, Ablk.data)
    hw_timings['stack'] += - t00 + time.time()

    t01 = time.time()
    Rb.createAIJWithArrays(size=[2*sz, 2*sz], csr=Ablk_csr, comm=PETSc.COMM_WORLD)
    Rb.assemblyBegin()
    Rb.assemblyEnd()
    hw_timings['createAIJ'] += - t01 + time.time()

    # Create solution (in LHS) & RHS
    t02 = time.time()
    vecx, vecb = Rb.createVecs()
    vecb.createWithArray(Bzero)
    vecx, vecb = dolfin.PETScVector(vecx), dolfin.PETScVector(vecb)

    # Casting matrix
    Rmat = dolfin.PETScMatrix(Rb)
    hw_timings['createmat'] += - t02 + time.time()

    # Solving system
    t03 = time.time()
    #exitFlag = solvR.solve(Rmat, vecx, vecb)
    solvR.solve(Rmat, vecx, vecb)
    hw_timings['solve'] += - t03 + time.time()

    # Assign to function (2 func: Re & Im)
    return vecx.get_local()


def get_Hw_lifting(fs, A=None, C=None, Q=None, logwmin=-2, logwmax=2, nw=10,
           save_dir='/scratchm/wjussiau/fenics-python/cylinder/data/temp/',
           save_suffix='', verbose=True):
    '''Get frequency response of infinite-dimensional system WITH LIFTING TRANSFORM
    See Barbagallo & Sipp pp.7,34 (2009) for derivation
    Lifting formulation:
    Xfull(t) = rho(t)*S1 + X(t) (for time simulation only)
    dX/dt = A*x + Q*S1*c
    y = C*x
    where S1 = steady state with rho = 1
    Hence for resolvent response:
        A = A
        B = Q*S1
        C = C
    Or with double-lifting? Do we have to do that?
    Maybe yes, because Xfull(t) has rho(t) in it, so we need the dynamics
     of the full system (including rho)
    d/dt(X, rho) = diag(A, 0)(X, rho) + (B, -1)c
    y = (C, 0)(X, rho)
    Qhat = diag(Q, 1)
    So for lifting matrices:
    Al = diag(A, 0)
    Bl = (B; -1)
    Cl = (C, 0)
    Ql = diag(Q, 1)
    And we compute H(s)=Cl*inv(s*Ql - Al)*Bl
    Then resolvent response gives (s-omega): m/c = m/(-drho/dt) = -1/jw m/rho
    So we must find: H(m/rho) = -jw * H(m/c)
    WARNING: ONLY RUNS FOR A SINGLE SENSOR (but might not even give satisfactory ans)
    '''
    # Get matrices
    if A is None:
        A = fs.get_A()
    if C is None:
        C = fs.get_C()
    if Q is None:
        Q = fs.get_mass_matrix()
        # should we do that???
        #[bc.apply(E) for bc in fs.bc['bcu']]
    Q = dense_to_sparse(Q)

    # Determine matrices in lifting formulation
    Al, Bl, Cl, Ql = fs.get_matrices_lifting(A=A, C=C, Q=Q)

    # Get frequency response with prescribed matrices A, B, C, D, E
    Hw_l, ww, hw_timings = get_Hw(fs, A=Al, B=Bl, C=Cl, D=0, Q=Ql,
        logwmin=logwmin, logwmax=logwmax, nw=nw, save_dir=save_dir, save_suffix=save_suffix,
        verbose=verbose)

    # warning; what is exported is Hw_l and not Hw
    Hw = Hw_l * -1j*ww

    return Hw, Hw_l, ww, hw_timings





#################################################################################
#################################################################################
#################################################################################
# Run
if __name__=='__main__':
    ##Av, Bv = make_mat_to_test_slepc(view=True, singular=False)
    ##A = dense_to_sparse(Av)
    ##B = dense_to_sparse(Bv)

    ##k = 4
    ##D = spr_la.eigs(A=A, k=k, M=B, sigma=None, Minv=None, OPinv=None, return_eigenvectors=False)

    ##print('Eigenvalues are:')
    ##print(D)
    ###print('Eigenvectors are:')
    ###print(V)

    ##k=6
    ##Ad, Bd = make_mat_to_test_slepc(view=True, singular=True)

    ##print('----------------------------------- begin embedded fn')
    ##solve_dense = True
    ##sigma = 0.0
    ##eva, eve = geig_singular(Ad, Bd, n=k, solve_dense=solve_dense, target=sigma)
    ##print('solving with dense version: ', solve_dense)
    ##print('eigenvalues: \n', eva)
    ###print(eve)
    ##
    ##Z = Ad@eve - Bd@eve@np.diag(eva)
    ###print('residual: \n', Z)
    ##print('----------------------------------- end')



    #print('----- converting to sparse ----- begin')

    ##n = 4 # size of problem
    ##re = [0.2, -0.5] # real part of eigenval
    ##im = [0.25, 0.8] # imag part of eigenval
    ##makemat_conjeig = lambda a, b: [[a, -b], [b, a]] # make 2x2 matrix with given conj eigenval
    ##a = makemat_conjeig(re[0], im[0])
    ##for i in range(1, n//2): # make block matrix with a+1jb as eigenval
    ##    a = la.block_diag(a, makemat_conjeig(re[i], im[i]))
    ### make b in numpy to set values easily, then convert
    ##b = np.eye(n)
    ##singular = True
    ##if singular: # makes b non invertible
    ##    b[1, 1] = 0
    ##    b[2, 2] = 0
    ##Ad = a
    ##Bd = b

    #As = dense_to_sparse(Ad)
    #Bs = dense_to_sparse(Bd)
    #
    #sigma = 0.
    #II = spr.eye(As.shape[0])

    #rhs = Bs # II

    #Asb = As - sigma*rhs
    #Asb = Asb.tocsc()
    #LU = spr_la.splu(Asb)

    ## should not be specified if sigma is not None
    ## Minv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Bs, x, tol=1e-5)[0], shape=Bs.shape)
    #
    #OPinv = spr_la.LinearOperator(matvec=lambda x: LU.solve(x), shape=Asb.shape)
    ###OPinv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Asb, x, tol=1e-5)[0], shape=Asb.shape)

    ##############Ds, Vs = spr_la.eigs(A=spr_la.aslinearoperator(As), k=k, M=Bs, sigma=sigma, which='LM', OPinv=OPinv)

    ### try: M = None, OPinv = A_as_LU.solve(B @ x)
    ##ALU = spr_la.splu(As.tocsc())
    ### this solves: Ax=Bb with virtually: inv(B)*Ax=b
    ##OPinv_AB = spr_la.LinearOperator(matvec=lambda x: ALU.solve(rhs@x), shape=Asb.shape)
    ##Ds, Vs = spr_la.eigs(A=As, k=k, M=None, sigma=sigma, OPinv=OPinv_AB, which='LM')
    #
    ##Ds_shifted = Ds - sigma

    ##print('eigenvalues: \n', Ds)
    ##print('eigenvalues (shifted back): \n', Ds_shifted)
    ##print('eigenvectors: \n', Vs)

    #print('----- scipy.sparse ----- end')


    logger.info('----- using slepc ----- begin')

    logger.info('...............................')
    logger.info('Small matrix')
    nit = 1
    dt = 0
    for i in range(nit):
        t0 = time.time()
        Ad, Bd = make_mat_to_test_slepc(view=True, singular=True)
        A = array_to_petscmat(Ad)
        B = array_to_petscmat(Bd)
        eigz, eigensolver = get_mat_vp_slepc(A, B, target=0.1, n=4, DEBUG=False, verbose=False,
            return_eigensolver=True, gmresrestart=1000, tol=1e-5)
        dt += time.time() - t0
    logger.info('Elapsed (avg on %d iter): %f' %(nit, dt/nit))
    logger.info(eigz[0][:,None])
    logger.info('...............................')


    # Test routines with random sparse matrices
    if 1:
        logger.info('')
        tc = time.time()
        AA, BB = make_mat_to_test_slepc(view=True, singular=True,
            neigpairs=2000, density_B=7/10, rand=False)
        #AA, BB = load_mat_from_file('rand')
        logger.info('...............................')
        logger.info('Random matrices of size: %i' %(BB.shape[0]))
        logger.info('Create - elapsed: %f' %(time.time() - tc))


        # BENCHMARK
        if 1:
            logger.info('...............................')
            logger.info('Benchmark')
            # Get all preconditioners available
            all_pc = get_all_enum_petsc_slepc(PETSc.PC.Type)
            pc_except = ['MAT','ICC', 'NONE', 'DEFLATION']
            for pc in pc_except:
                all_pc.pop(pc)
            # We should do: for all pair (solver, pc), try eig solve, log time elapsed
            pc_time = all_pc.copy()
            for pc_key, pc_name in all_pc.items():
                try:
                    t0 = time.time()
                    eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                        target=-0.1, n=4, return_eigensolver=True, precond_type=pc_name, verbose=True)
                    tpc = time.time() - t0
                    logger.info('PC success: %s --- %f' %(pc_name, tpc))
                except Exception as e:
                    logger.info('PC failed:  %s ' %(pc_name))
                    tpc = np.inf
                    raise(e)
                pc_time[pc_key] = tpc


            # Get all solvers available
            all_solvers = get_all_enum_petsc_slepc(SLEPc.EPS.Type)
            solvers_except = ['POWER', 'SUBSPACE', 'ARNOLDI']
            for solver in solvers_except:
                all_solvers.pop(solver)
            solver_time = all_solvers.copy()
            for solver_key, solver_name in all_solvers.items():
                try:
                    t0 = time.time()
                    eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                        target=-0.1, n=4, return_eigensolver=True,
                        precond_type=PETSc.PC.Type.NONE, verbose=False,
                        eps_type=solver_name)
                    tsl = time.time() - t0
                    logger.info('Solver success: %s --- %f' %(solver_name, tsl))
                except Exception as e:
                    logger.info('Solver failed: %s' %(solver_name))
                    tsl = np.inf
                    raise(e)
                solver_time[solver_key] = tsl
            logger.info('...............................')

        # BEST
        # Try best + best
        logger.info('...............................')
        logger.info('Best + best')
        t0b = time.time()
        eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                target=-2.103, n=4, return_eigensolver=True, verbose=False,
                precond_type=PETSc.PC.Type.HYPRE,
                eps_type=SLEPc.EPS.Type.KRYLOVSCHUR)
        logger.info('Best + best: %f ' %(time.time() - t0b) )
        Z = AA@eigz[1] - BB@eigz[1]@np.diag(eigz[0])
        logger.info('Sanity check (norm): ', np.linalg.norm(Z))
        logger.info('...............................')

    # True FEM matrices
    if 0:
        logger.info('...............................')
        logger.info('Start with FEM matrices')
        t0 = time.time()
        logger.info('--- Loading matrices')
        #save_npz_path = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/matrices/'
        save_npz_path = '/scratchm/wjussiau/fenics-python/cylinder/data/o1/matrices/'
        #save_npz_path = '/scratchm/wjussiau/fenics-python/cavity/data/matrices/'
        logger.info('--- from sparse...')
        AA = spr.load_npz(save_npz_path + 'A.npz')
        BB = spr.load_npz(save_npz_path + 'Q.npz')
        logger.info('--- ... to petscmat')
        AA = array_to_petscmat(AA, eliminate_zeros=False)
        BB = array_to_petscmat(BB, eliminate_zeros=False)
        # make empty entries -> 0
        #AA.setdiag(AA.diagonal())
        #BB.setdiag(BB.diagonal())
        # solve
        logger.info('--- Starting solve')
        eigz, eigensolver = get_mat_vp_slepc(
            A=AA,
            B=BB,
            target=0.1,
            n=10,
            return_eigensolver=True,
            verbose=True,
            precond_type=PETSc.PC.Type.LU,
            eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
            ksp_type=PETSc.KSP.Type.PREONLY,
            gmresrestart=500,
            tol=1e-8,
            kspmaxit=5000)
        logger.info('Big matrix - elapsed: %f' %(time.time() - t0))
        logger.info('...............................')

    logger.info('...............................')
    #print('Summary .......................')
    #st = eigensolver.getST()
    #ksp = st.getKSP()
    #pc = ksp.getPC()

    #Print = PETSc.Sys.Print
    #its = eigensolver.getIterationNumber()
    #Print( "Number of iterations of the method: %d" % its )

    #eps_type = eigensolver.getType()
    #Print( "Solution method: %s" % eps_type )

    #nev, ncv, mpd = eigensolver.getDimensions()
    #Print( "Number of requested eigenvalues: %d" % nev )

    #tol, maxit = eigensolver.getTolerances()
    #Print( "Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit) )

    #nconv = eigensolver.getConverged()
    #Print( "Number of converged eigenpairs %d" % nconv )
    #
    #Print("Overview of eigenvalues: \n ", eigz[0][:,None])
