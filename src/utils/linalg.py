"""Linear algebra utilities: matrix conversions, eigenvalue solvers, frequency response."""

import logging
import time
from typing import Optional

import dolfin
import numpy as np
import petsc4py
import scipy.io as sio
import scipy.linalg as la
import scipy.sparse as spr
import scipy.sparse.linalg as spr_la
from joblib import Parallel, delayed
from mpi4py import MPI as mpi
from petsc4py import PETSc
from slepc4py import SLEPc

import utils.utils_flowsolver as flu
from utils.mpi import MpiUtils

logger = logging.getLogger(__name__)


# --- PETSc / scipy.sparse conversions ---


def dense_to_sparse(A, eliminate_under: Optional[float] = None):
    """Cast PETSc or dolfin Matrix to scipy.sparse CSR.
    (Misleadingly named: A need not be dense.)"""

    def eliminate_zeros(A):
        if eliminate_under is None:
            A.eliminate_zeros()
        else:
            Adense = A.toarray()
            Adense[Adense <= eliminate_under] = 0
            A = spr.csr_matrix(Adense)
        return A

    if spr.issparse(A):
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
    """Cast a scipy sparse matrix to PETSc.Mat (square CSR)."""
    A = A.tocsr()
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


def array_to_petscmat(A, eliminate_zeros=True):
    """Cast a numpy array to PETSc.Mat."""
    return sparse_to_petscmat(dense_to_sparse(A, eliminate_zeros))


def numpy_to_petsc(M: np.ndarray):
    """Cast a numpy array to a PETSc matrix (dense → AIJ)."""
    mat = PETSc.Mat().createAIJ(size=M.shape, comm=PETSc.COMM_WORLD)
    mat.setValues(range(M.shape[0]), range(M.shape[1]), M.flatten())
    mat.assemble()
    return mat


def numpy_to_scipy_csr(M: np.ndarray):
    """Cast a numpy array to scipy.sparse.csr_matrix."""
    return spr.csr_matrix(M)


def dolfin_petsc_to_petsc(M: dolfin.cpp.la.PETScMatrix):
    """Cast a dolfin PETScMatrix to a raw PETSc.Mat."""
    return dolfin.as_backend_type(M).mat()


def petsc_to_scipy(mat: PETSc.Mat) -> spr.csr_matrix:
    """Convert a PETSc.Mat to a scipy CSR matrix."""
    indptr, indices, data = mat.getValuesCSR()
    return spr.csr_matrix((data, indices, indptr), shape=mat.getSize())


# --- Eigenvalue solvers ---


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
    """Compute n eigenvalues of A (or generalized (A, B)) via SLEPc shift-and-invert."""
    eigensolver = SLEPc.EPS().create()
    eigensolver.setType(eps_type)

    pbtype = (
        eigensolver.ProblemType.NHEP if B is None else eigensolver.ProblemType.GNHEP
    )
    eigensolver.setProblemType(pbtype)
    eigensolver.setTarget(target)
    eigensolver.setWhichEigenpairs(eigensolver.Which.TARGET_REAL)
    eigensolver.setTolerances(tol, niter)
    eigensolver.setOperators(A, B)
    eigensolver.setDimensions(nev=n, mpd=mpd, ncv=ncv)

    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)

    ksp = st.getKSP()
    ksp.setType(ksp_type)
    ksp.setGMRESRestart(gmresrestart)
    ksp.setTolerances(rtol=ksprtol, atol=kspatol, divtol=kspdivtol, max_it=kspmaxit)

    pc = ksp.getPC()
    pc.setType(precond_type)
    pc.setFactorSolverType("mumps")

    if verbose:

        def kspmonitor(kspstate, it, rnorm):
            if not it % 100:
                logger.info("--- ksp monitor --- nit: %d +++ res %f", it, rnorm)

        ksp.setMonitor(kspmonitor)

        def epsmonitor(eps, it, nconv, eig, err):
            logger.info("***** eps monitor ***** nit: %d +++ cvg %d / %d", it, nconv, n)

        eigensolver.setMonitor(epsmonitor)

    eigensolver.setFromOptions()
    eigensolver.solve()

    nconv = eigensolver.getConverged()
    if verbose:
        logger.info("------ Computation terminated ------")

    sz = A.size[0]
    n = nconv
    valp = np.zeros(n, dtype=complex)
    vecp_re = np.zeros((sz, n), dtype=float)
    vecp_im = np.zeros((sz, n), dtype=float)

    vr = A.createVecRight()
    vi = A.createVecRight()

    for i in range(nconv):
        valp[i] = eigensolver.getEigenpair(i, Vr=vr, Vi=vi)
        istart_r, iend_r = vr.getOwnershipRange()
        istart_i, iend_i = vi.getOwnershipRange()
        vecp_re[istart_r:iend_r, i] = vr.array
        vecp_im[istart_r:iend_i, i] = vi.array
        if verbose:
            logger.info(
                "Eigenvalue %2d is: %f+i %f"
                % (i + 1, np.real(valp[i]), np.imag(valp[i]))
            )

    vecp = vecp_re + 1j * vecp_im
    eigz = (valp, vecp)
    if return_eigensolver or DEBUG:
        eigz = (eigz, eigensolver)
    return eigz


def make_mat_to_test_slepc(
    view=False, singular=False, neigpairs=3, density_B=1.0, rand=False
):
    """Build a small test matrix pair (A, B) with known eigenvalues for SLEPc testing."""
    sz = 2 * neigpairs
    if neigpairs == 3:
        re = [0.2, -0.5, -0.7]
        im = [0.25, 0.8, 0.0]
    else:
        if rand:
            re = np.random.randn(neigpairs)
            im = np.random.randn(neigpairs)
        else:
            eigstep = 0.1
            re = np.arange(start=-neigpairs * eigstep, stop=0, step=eigstep)
            im = np.arange(start=-neigpairs * eigstep, stop=0, step=eigstep)

    def makemat_conjeig(a, b):
        return [[a, -b], [b, a]]

    A = np.zeros((sz, sz))
    for i in range(neigpairs):
        j = 2 * i
        A[j : j + 2, j : j + 2] = makemat_conjeig(re[i], im[i])

    B = np.eye(sz)
    if singular:
        if neigpairs == 3:
            B[1, 1] = 0
            B[2, 2] = 0
        else:
            nrzeros = int(sz * (1 - density_B))
            wherezeros = (
                np.random.permutation(sz)[:nrzeros] if rand else np.arange(nrzeros)
            )
            B[wherezeros, wherezeros] = 0

    if view:
        return A, B
    return array_to_petscmat(A), array_to_petscmat(B)


def load_mat_from_file(mat_type="fem32"):
    """Load A and Q sparse matrices from .npz files (test/benchmark use)."""
    prefix = "/stck/wjussiau/fenics-python/ns/data/matrices/"

    def mkpath(name):
        return prefix + name + ".npz"

    if mat_type == "fem32":
        return spr.load_npz(mkpath("A_sparse_nx32")), spr.load_npz(
            mkpath("Q_sparse_nx32")
        )
    if mat_type == "fem64":
        return spr.load_npz(mkpath("A_sparse_nx64")), spr.load_npz(
            mkpath("Q_sparse_nx64")
        )
    return spr.load_npz(mkpath("A_rand")), spr.load_npz(mkpath("B_rand"))


def geig_singular(A, B, n=2, DEBUG=False, target=None, solve_dense=False):
    """Compute n eigenvalues of generalized problem (A, B) where B is singular."""
    s = 1.2
    nn = A.shape[0]
    sznp = nn - np.linalg.matrix_rank(B)
    C = np.ones((nn - sznp, nn - sznp))

    At, Bt = augment_matrices(A, B, s, C)

    if solve_dense:
        Dt, Vt = la.eig(At, Bt)
    else:
        At = dense_to_sparse(At)
        Bt = dense_to_sparse(Bt)
        if target is None:
            target = 0.0
        Asb = (At - target * Bt).tocsc()
        LU = spr_la.splu(Asb)
        OPinv = spr_la.LinearOperator(
            matvec=lambda x: spr_la.minres(Asb, x, tol=1e-5)[0], shape=Asb.shape
        )
        Dt, Vt = spr_la.eigs(A=At, k=n, M=Bt, tol=0, sigma=target, OPinv=OPinv)
        logger.info("Embedded sparse eig: %f", Dt)

    Vt_zero = Vt[-sznp:, :]
    EPS = np.finfo(float).eps
    idx_true_eig = np.all(np.abs(Vt_zero) < EPS * 1e3, axis=0)
    return Dt[idx_true_eig], Vt[:-sznp, idx_true_eig]


def augment_matrices(A, B, s, C):
    """Augment (A, B) to make the generalized eigenproblem non-singular."""
    N = la.null_space(B)
    M = la.null_space(B.T).T
    s = 1.2
    MA = M @ A
    AN = A @ N
    C = np.ones((MA.shape[0], AN.shape[1]))
    Aa = np.block([[A, s * AN], [MA, s * C]])
    Ba = np.block([[B, AN], [MA, C]])
    return Aa, Ba


def get_all_enum_petsc_slepc(enum):
    """Return all public values in a PETSc/SLEPc enum subpackage."""
    return {k: v for k, v in enum.__dict__.items() if not k.startswith("_")}


def get_mat_vp(A, B=None, n=3, DEBUG=False):
    """Compute n eigenvalues of A (or (A, B)) via the dolfin SLEPc wrapper."""
    eigensolver = dolfin.SLEPcEigenSolver(A, B)
    eigensolver.solve(n)
    nconv = eigensolver.get_number_converged()
    logger.info("Tried: %d, converged: %d" % (n, nconv))

    sz = A.size(0)
    valp = np.zeros(nconv, dtype=complex)
    c = np.zeros_like(valp)
    vecp = np.zeros((sz, nconv), dtype=complex)
    cx = np.zeros_like(vecp)

    for i in range(nconv):
        valp[i], c[i], vecp[:, i], cx[:, i] = eigensolver.get_eigenpair(i)
        logger.info(
            "Eigenvalue %d is: %f+i %f" % (i + 1, np.real(valp[i]), np.imag(valp[i]))
        )

    if DEBUG:
        return (valp, vecp), eigensolver
    return valp, vecp


# --- Frequency response (scipy-based) ---


def _get_matrix_size(
    M: PETSc.Mat | spr.spmatrix | np.ndarray, axis: int = 0
) -> tuple[int, int]:
    """Return global (nrows, ncols) from a PETSc, scipy sparse, or numpy matrix."""
    if isinstance(M, PETSc.Mat):
        return M.getSize()
    elif isinstance(M, spr.spmatrix):
        return M.shape
    elif isinstance(M, np.ndarray):
        if M.ndim != 2:
            raise ValueError(f"numpy matrix must be 2D, got shape {M.shape}.")
        return mpi.COMM_WORLD.reduce(M.shape[axis], op=mpi.SUM, root=0)
    else:
        raise TypeError(
            f"Unsupported matrix type: {type(M)}. "
            f"Expected PETSc.Mat, scipy sparse, or np.ndarray."
        )


def _get_freqresp_sizes(
    A: PETSc.Mat | spr.spmatrix,
    B: np.ndarray,
    C: np.ndarray,
    ww: np.ndarray,
) -> tuple[int, int, int, int]:
    """Extract and validate (n, nu, ny, nw) from system matrices."""
    B = np.asarray(B)
    C = np.asarray(C)
    n_A, m_A = _get_matrix_size(A)
    n = n_A
    nu = _get_matrix_size(B, axis=1) if B.ndim == 2 else 1
    ny = _get_matrix_size(C, axis=0) if C.ndim == 2 else 1
    nw = len(ww)
    if n_A != m_A:
        raise ValueError(f"A must be square, got shape ({n_A}, {m_A}).")
    if nw < 1:
        raise ValueError(f"ww must be non-empty, got length {nw}.")
    return n, nu, ny, nw


def _format_freqresp_matrices(
    A: PETSc.Mat | spr.spmatrix,
    B: np.ndarray,
    Q: PETSc.Mat | spr.spmatrix,
) -> tuple[spr.csc_matrix, spr.csc_matrix, np.ndarray]:
    """Convert A, Q to scipy CSC and normalise B to 2D for scipy-based solvers."""

    def _to_csc(M: PETSc.Mat | spr.spmatrix, name: str) -> spr.csc_matrix:
        if isinstance(M, PETSc.Mat):
            indptr, indices, data = M.getValuesCSR()
            return spr.csr_matrix((data, indices, indptr), shape=M.getSize()).tocsc()
        elif isinstance(M, spr.spmatrix):
            return M.tocsc()
        raise TypeError(f"{name} must be PETSc.Mat or scipy sparse, got {type(M)}.")

    Acsc = _to_csc(A, "A")
    Qcsc = _to_csc(Q, "Q")
    B = np.asarray(B)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    elif B.ndim != 2:
        raise ValueError(f"B must be 1D or 2D, got shape {B.shape}.")
    return Acsc, Qcsc, B


def _show_freqresp_info(n, nu, ny, nw, ww):
    logger.info(
        "System dimensions: n=%d, nu=%d, ny=%d | Frequency points: nw=%d, "
        "w in [1e%g, 1e%g]",
        n,
        nu,
        ny,
        nw,
        np.log10(ww[0]),
        np.log10(ww[-1]),
    )


def get_frequency_response_sequential(
    A: PETSc.Mat,
    B: np.ndarray,
    C: np.ndarray,
    Q: PETSc.Mat,
    ww: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H(w) = C * inv(jwQ - A) * B sequentially via scipy splu.

    Returns H of shape (ny, nu, nw) complex, and ww.
    """
    n, nu, ny, nw = _get_freqresp_sizes(A=A, B=B, C=C, ww=ww)
    Acsc, Qcsc, B = _format_freqresp_matrices(A=A, B=B, Q=Q)
    _show_freqresp_info(n, nu, ny, nw, ww)

    H = np.zeros((ny, nu, nw), dtype=complex)
    rhs = np.vstack([B, np.zeros_like(B)])

    t_start = time.time()
    for ii, w in enumerate(ww):
        t_step = time.time()
        Ablk = spr.bmat([[-Acsc, -w * Qcsc], [w * Qcsc, -Acsc]], format="csc")

        lu = spr_la.splu(Ablk)
        x = lu.solve(rhs)

        H[:, :, ii] = C @ x[:n, :] + 1j * C @ x[n:, :]

        if verbose:
            logger.info(
                "  [%d/%d] w=%.4e | max|H|=%.4e | elapsed: %.3fs",
                ii + 1,
                nw,
                w,
                np.max(np.abs(H[:, :, ii])),
                time.time() - t_step,
            )

    logger.info("Frequency response computed in %.3fs total.", time.time() - t_start)
    return H, ww


def get_frequency_response_parallel(
    A: PETSc.Mat,
    B: np.ndarray,
    C: np.ndarray,
    Q: PETSc.Mat,
    ww: np.ndarray,
    verbose: bool = True,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H(w) = C * inv(jwQ - A) * B with joblib parallelism over frequencies."""
    n, nu, ny, nw = _get_freqresp_sizes(A=A, B=B, C=C, ww=ww)
    Acsc, Qcsc, B = _format_freqresp_matrices(A=A, B=B, Q=Q)
    _show_freqresp_info(n, nu, ny, nw, ww)

    rhs = np.vstack([B, np.zeros_like(B)])
    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_solve_at_frequency)(w, Acsc, Qcsc, rhs, C, n) for w in ww
    )

    H = np.zeros((ny, nu, nw), dtype=complex)
    t_start = time.time()
    for ii, (w, H_w) in enumerate(zip(ww, results)):
        H[:, :, ii] = H_w
        if verbose:
            logger.info(
                "  [%d/%d] w=%.4e | max|H|=%.4e | elapsed: %.3fs",
                ii + 1,
                nw,
                w,
                np.max(np.abs(H_w)),
                time.time() - t_start,
            )

    logger.info("Frequency response computed in %.3fs total.", time.time() - t_start)
    return H, ww


def get_frequency_response_mpi(
    A: PETSc.Mat,
    B: np.ndarray,
    C: np.ndarray,
    Q: PETSc.Mat,
    ww: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H(w) = C * inv(jwQ - A) * B using PETSc/MUMPS for parallel solves.

    Must be launched with mpirun. H is meaningful on all ranks.
    """
    comm = A.getComm()
    rank = comm.getRank()

    n = A.getSize()[0]
    nu = B.shape[1] if B.ndim > 1 else 1
    ny = C.shape[0]
    B = B.reshape(n, 1) if B.ndim == 1 else B
    nw = ww.shape[0]

    if rank == 0:
        _show_freqresp_info(n, nu, ny, nw, ww)
        logger.info("Building RHS vectors...")

    rhs_vecs = _build_rhs(B, n, comm)
    H = np.zeros((ny, nu, nw), dtype=complex)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setFromOptions()

    t_start = time.time()
    for ii, w in enumerate(ww):
        t_step = time.time()
        Ablk = _build_block_matrix(A, Q, w)
        ksp.setOperators(Ablk)
        ksp.setUp()
        sol_blk = Ablk.createVecRight()

        for iu in range(nu):
            ksp.solve(rhs_vecs[iu], sol_blk)
            local_vals = sol_blk.getArray()
            rstart, rend = sol_blk.getOwnershipRange()
            xr_local = np.zeros(n)
            xi_local = np.zeros(n)
            for local_i, global_i in enumerate(range(rstart, rend)):
                if global_i < n:
                    xr_local[global_i] = local_vals[local_i]
                else:
                    xi_local[global_i - n] = local_vals[local_i]
            xr = np.zeros(n)
            xi = np.zeros(n)
            comm.tompi4py().Allreduce(xr_local, xr)
            comm.tompi4py().Allreduce(xi_local, xi)
            H[:, iu, ii] = C @ xr + 1j * C @ xi

        if verbose and rank == 0:
            logger.info(
                "  [%d/%d] w=%.4e | max|H|=%.4e | step: %.3fs | total: %.3fs",
                ii + 1,
                nw,
                w,
                np.max(np.abs(H[:, :, ii])),
                time.time() - t_step,
                time.time() - t_start,
            )
        Ablk.destroy()

    ksp.destroy()
    if rank == 0:
        logger.info(
            "Frequency response computed in %.3fs total.", time.time() - t_start
        )
    return H, ww


def _solve_at_frequency(w, Acsr, Qcsr, rhs, C, n):
    """Solve the block system at a single frequency (used by parallel variant)."""
    Ablk = spr.bmat([[-Acsr, -w * Qcsr], [w * Qcsr, -Acsr]], format="csc")
    lu = spr_la.splu(Ablk)
    x = lu.solve(rhs)
    return C @ x[:n, :] + 1j * C @ x[n:, :]


def _build_block_matrix(A: PETSc.Mat, Q: PETSc.Mat, w: float) -> PETSc.Mat:
    """Build the 2n×2n PETSc block matrix [[-A, -wQ], [wQ, -A]] for frequency w."""
    wQ = Q.copy()
    wQ.scale(w)
    mA = A.copy()
    mA.scale(-1.0)
    mwQ = wQ.copy()
    mwQ.scale(-1.0)
    block = PETSc.Mat().createNest([[mA, mwQ], [wQ, mA]], comm=A.getComm())
    block.setUp()
    block.assemble()
    return block


def _build_rhs(B: np.ndarray, n: int, comm: PETSc.Comm) -> list[PETSc.Vec]:
    """Build distributed PETSc Vecs for the block-system RHS [B[:, iu]; 0]."""
    nu = B.shape[1]
    vecs = []
    for iu in range(nu):
        rhs = PETSc.Vec().create(comm=comm)
        rhs.setSizes(2 * n)
        rhs.setUp()
        rstart, rend = rhs.getOwnershipRange()
        for i in range(rstart, rend):
            if i < n:
                rhs.setValue(i, B[i, iu])
        rhs.assemblyBegin()
        rhs.assemblyEnd()
        vecs.append(rhs)
    return vecs


def _build_rhs_mat(B: np.ndarray, n: int, comm: PETSc.Comm) -> PETSc.Mat:
    """Build distributed PETSc Mat from B for the block system RHS, shape (2n, nu)."""
    nu = B.shape[1]
    rhs_mat = PETSc.Mat().create(comm=comm)
    rhs_mat.setSizes([2 * n, nu])
    rhs_mat.setType(PETSc.Mat.Type.DENSE)
    rhs_mat.setUp()
    rstart, rend = rhs_mat.getOwnershipRange()
    for i in range(rstart, rend):
        if i < n:
            for iu in range(nu):
                rhs_mat.setValue(i, iu, B[i, iu])
    rhs_mat.assemblyBegin()
    rhs_mat.assemblyEnd()
    return rhs_mat


# --- Legacy FlowSolver frequency response (old dolfin/MUMPS path) ---


def get_Hw(
    fs,
    A=None,
    B=None,
    C=None,
    D=None,
    Q=None,
    logwmin=-2,
    logwmax=2,
    nw=10,
    save_dir="/scratchm/wjussiau/fenics-python/cylinder/data/temp/",
    save_suffix="",
    verbose=True,
):
    """Compute frequency response via dolfin LUSolver (legacy MPI path).

    Prefer get_frequency_response_sequential / get_frequency_response_mpi for new code.
    """
    MpiUtils.check_process_rank()
    ww = np.logspace(logwmin, logwmax, num=nw)
    ns = fs.params_control.sensor_number
    Hw = np.zeros((ns, len(ww)), dtype=complex)

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

    if verbose:
        logger.info("Defining sub-blocks (A, E)")

    Acsr = dense_to_sparse(A)
    Q = dense_to_sparse(Q)
    sz = Acsr.shape[0]

    petsc4py.init()
    Rb = PETSc.Mat().create()
    Rb.setSizes((2 * sz, 2 * sz))
    Rb.setUp()
    Rb.assemble()

    rstart, rend = Rb.getOwnershipRange()
    B0 = np.zeros_like(B)
    Bzero = np.vstack([B, B0])
    CjC = np.hstack([C, 1j * C])
    solvR = dolfin.LUSolver("mumps")

    hw_timings = {
        "stack": 0,
        "createAIJ": 0,
        "createmat": 0,
        "solve": 0,
        "decompose": 0,
    }
    tb = time.time()

    for ii, w in enumerate(ww):
        if verbose:
            logger.info("Computing %d/%d with puls: %5.3f..." % (ii + 1, len(ww), w))
        t00 = time.time()
        Ablk = spr.bmat([[-Acsr, -w * Q], [w * Q, -Acsr]]).tocsr()
        Ablk_csr = (
            Ablk.indptr[rstart : rend + 1] - Ablk.indptr[rstart],
            Ablk.indices[Ablk.indptr[rstart] : Ablk.indptr[rend]],
            Ablk.data[Ablk.indptr[rstart] : Ablk.indptr[rend]],
        )
        hw_timings["stack"] += -t00 + time.time()

        t01 = time.time()
        Rb.createAIJ(size=(2 * sz, 2 * sz), csr=Ablk_csr, comm=PETSc.COMM_WORLD)
        Rb.assemblyBegin()
        Rb.assemblyEnd()
        hw_timings["createAIJ"] += -t01 + time.time()

        t02 = time.time()
        vecx, vecb = Rb.createVecs()
        rbstart, rbend = vecb.getOwnershipRange()
        localidx = list(range(rbstart, rbend))
        vecb.setValues(localidx, Bzero[localidx])
        vecx, vecb = dolfin.PETScVector(vecx), dolfin.PETScVector(vecb)
        Rmat = dolfin.PETScMatrix(Rb)
        hw_timings["createmat"] += -t02 + time.time()

        t03 = time.time()
        solvR.solve(Rmat, vecx, vecb)
        hw_timings["solve"] += -t03 + time.time()

        t04 = time.time()
        rxstart, rxend = vecx.vec().getOwnershipRange()
        localidx = list(range(rxstart, rxend))
        vecy = spr.csr_matrix(CjC[:, localidx]).dot(vecx.get_local())
        hw_timings["decompose"] += -t04 + time.time()

        Hw[:, ii] = dolfin.MPI.comm_world.reduce(vecy, root=0)

        if verbose:
            for sn in range(ns):
                logger.info("\t magnitude is: %5.4f" % (np.abs(Hw[sn, ii])))

    if verbose:
        logger.info(
            "Elapsed computing {0} pulsations: {1}".format(len(ww), time.time() - tb)
        )

    if save_dir and MpiUtils.get_rank() == 0:
        suffix = "_" + str(ns) + "sensors"
        savepath = save_dir + "Hw_nw" + str(nw) + save_suffix + suffix + ".mat"
        sio.savemat(
            savepath,
            {
                "H": Hw,
                "w": ww,
                "xs": fs.sensor_location,
                "comment": "1 line = 1 sensor",
            },
        )
        for sn in range(ns):
            xs_i = fs.sensor_location[sn]
            suffix = "_dx=" + str(xs_i[0]) + "_dy=" + str(xs_i[1])
            savepath = save_dir + "Hw_nw" + str(nw) + save_suffix + suffix + ".mat"
            sio.savemat(savepath, {"H": Hw[sn, :], "w": ww, "xs": xs_i})
            logger.info("Saving frequency response to: %s", savepath)

    return Hw, ww, hw_timings


def get_field_response(fs, w, A=None, B=None, Q=None, verbose=True):
    """Compute the field response x = inv(jwQ - A) * B at a single frequency w."""
    if A is None:
        A = fs.get_A()
    if B is None:
        B = fs.get_B()
    if Q is None:
        Q = fs.get_mass_matrix()
    Acsr = dense_to_sparse(A)
    sz = Acsr.shape[0]
    Q = dense_to_sparse(Q)
    Bzero = np.vstack([B, np.zeros_like(B)])
    solvR = dolfin.LUSolver("mumps")
    Rb = PETSc.Mat()
    Ablk = spr.bmat([[-Acsr, -w * Q], [w * Q, -Acsr]]).tocsr()
    Ablk_csr = (Ablk.indptr, Ablk.indices, Ablk.data)
    Rb.createAIJWithArrays(size=[2 * sz, 2 * sz], csr=Ablk_csr, comm=PETSc.COMM_WORLD)
    Rb.assemblyBegin()
    Rb.assemblyEnd()
    vecx, vecb = Rb.createVecs()
    vecb.createWithArray(Bzero)
    vecx, vecb = dolfin.PETScVector(vecx), dolfin.PETScVector(vecb)
    Rmat = dolfin.PETScMatrix(Rb)
    solvR.solve(Rmat, vecx, vecb)
    return vecx.get_local()


def get_Hw_lifting(
    fs,
    A=None,
    C=None,
    Q=None,
    logwmin=-2,
    logwmax=2,
    nw=10,
    save_dir="/scratchm/wjussiau/fenics-python/cylinder/data/temp/",
    save_suffix="",
    verbose=True,
):
    """Compute frequency response with lifting transform (Barbagallo & Sipp 2009)."""
    if A is None:
        A = fs.get_A()
    if C is None:
        C = fs.get_C()
    if Q is None:
        Q = fs.get_mass_matrix()
    Q = dense_to_sparse(Q)
    Al, Bl, Cl, Ql = fs.get_matrices_lifting(A=A, C=C, Q=Q)
    Hw_l, ww, hw_timings = get_Hw(
        fs,
        A=Al,
        B=Bl,
        C=Cl,
        D=0,
        Q=Ql,
        logwmin=logwmin,
        logwmax=logwmax,
        nw=nw,
        save_dir=save_dir,
        save_suffix=save_suffix,
        verbose=verbose,
    )
    Hw = Hw_l * -1j * ww
    return Hw, Hw_l, ww, hw_timings


def get_matrices_lifting(self, A, C, Q):
    """Return matrices A, B, C, Q resulting form lifting transform (Barbagallo et al. 2009)
    See get_Hw_lifting for details"""
    # Steady field with rho=1: S1
    logger.info("Computing steady actuated field...")
    self.actuator_expression.ampl = 1.0
    S1 = self.compute_steady_state_newton()
    S1v = S1.vector()

    # Q*S1 (as vector)
    sz = self.W.dim()
    QS1v = dolfin.Vector(S1v.copy())
    QS1v.set_local(
        np.zeros(
            sz,
        )
    )
    QS1v.apply("insert")
    Q.mult(S1v, QS1v)  # QS1v = Q * S1v

    # Bl = [Q*S1; -1]
    Bl = np.hstack((QS1v.get_local(), -1))  # stack -1
    Bl = np.atleast_2d(Bl).T  # as column

    # Cl = [C, 0]
    Cl = np.hstack((C, np.atleast_2d(0)))

    # Ql = diag(Q, 1)
    Qsp = flu.dense_to_sparse(Q)
    Qlsp = flu.spr.block_diag((Qsp, 1))

    # Al = diag(A, 0)
    Asp = flu.dense_to_sparse(A)
    Alsp = flu.spr.block_diag((Asp, 0))

    return Alsp, Bl, Cl, Qlsp
