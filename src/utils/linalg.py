"""Linear algebra utilities: matrix conversions, eigenvalue solvers, freq response."""

import logging
import time

import dolfin
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as spr_la
from joblib import Parallel, delayed
from petsc4py import PETSc
from slepc4py import SLEPc

logger = logging.getLogger(__name__)


# --- PETSc / scipy.sparse conversions ---


def petsc_to_scipy(A: PETSc.Mat) -> spr.csr_matrix:
    """Convert a PETSc.Mat to a scipy CSR matrix."""
    indptr, indices, data = A.getValuesCSR()
    Acsr = spr.csr_matrix((data, indices, indptr), shape=A.getSize())
    Acsr.eliminate_zeros()
    return Acsr


def dolfin_to_scipy(A: dolfin.cpp.la.PETScMatrix) -> spr.csr_matrix:
    """Convert a dolfin PETScMatrix to a scipy CSR matrix."""
    return petsc_to_scipy(dolfin.as_backend_type(A).mat())


def scipy_to_petsc(A: spr.spmatrix) -> PETSc.Mat:
    """Convert a scipy sparse matrix to a PETSc.Mat."""
    A = A.tocsr()
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


def numpy_to_petsc(A: np.ndarray) -> PETSc.Mat:
    """Convert a numpy array to a PETSc.Mat (via sparse CSR)."""
    return scipy_to_petsc(spr.csr_matrix(A))


def dolfin_to_petsc(A: dolfin.cpp.la.PETScMatrix) -> PETSc.Mat:
    """Convert a dolfin PETScMatrix to a raw PETSc.Mat."""
    return dolfin.as_backend_type(A).mat()


# --- Eigenvalue solvers ---


def get_mat_vp_slepc(
    A: PETSc.Mat,
    B: PETSc.Mat | None = None,
    n: int = 10,
    target: float = 0.0,
    return_eigensolver: bool = False,
    verbose: bool = False,
    eps_type: str = SLEPc.EPS.Type.KRYLOVSCHUR,
    precond_type: str = PETSc.PC.Type.LU,
    ksp_type: str = PETSc.KSP.Type.PREONLY,
    tol: float = 1e-5,
    niter: int = 1000,
    ncv: int = PETSc.DECIDE,
    mpd: int = PETSc.DECIDE,
) -> tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], SLEPc.EPS]:
    """Compute n eigenvalues of A (or generalized (A, B)) nearest to target via SLEPc.

    Uses shift-and-invert with a MUMPS direct solve by default, which handles
    singular B correctly as long as (A - target*B) is non-singular.
    KSP/PC options can be overridden via ksp_type/precond_type, or through
    PETSc command-line options picked up by setFromOptions().
    """
    eigensolver = SLEPc.EPS().create()
    eigensolver.setType(eps_type)

    pbtype = eigensolver.ProblemType.NHEP if B is None else eigensolver.ProblemType.GNHEP
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

    pc = ksp.getPC()
    pc.setType(precond_type)
    pc.setFactorSolverType("mumps")

    if verbose:

        def epsmonitor(eps, it, nconv, eig, err):
            logger.info("EPS it=%d  converged=%d/%d", it, nconv, n)

        eigensolver.setMonitor(epsmonitor)

    eigensolver.setFromOptions()
    eigensolver.solve()

    nconv = eigensolver.getConverged()
    if verbose:
        logger.info("Converged: %d / %d requested", nconv, n)

    sz = A.size[0]
    valp = np.zeros(nconv, dtype=complex)
    vecp_re = np.zeros((sz, nconv), dtype=float)
    vecp_im = np.zeros((sz, nconv), dtype=float)

    vr = A.createVecRight()
    vi = A.createVecRight()

    for i in range(nconv):
        valp[i] = eigensolver.getEigenpair(i, Vr=vr, Vi=vi)
        istart_r, iend_r = vr.getOwnershipRange()
        istart_i, iend_i = vi.getOwnershipRange()
        vecp_re[istart_r:iend_r, i] = vr.array
        vecp_im[istart_i:iend_i, i] = vi.array
        if verbose:
            logger.info("Eigenvalue %2d: %+.6f %+.6fj", i + 1, valp[i].real, valp[i].imag)

    vecp = vecp_re + 1j * vecp_im
    if return_eigensolver:
        return (valp, vecp), eigensolver
    return valp, vecp


# --- Frequency response (scipy-based) ---


def _get_freqresp_sizes(
    A: PETSc.Mat | spr.spmatrix,
    B: np.ndarray,
    C: np.ndarray,
    ww: np.ndarray,
) -> tuple[int, int, int, int]:
    """Extract and validate (n, nu, ny, nw) from system matrices."""
    B = np.asarray(B)
    C = np.asarray(C)
    n, m = A.getSize() if isinstance(A, PETSc.Mat) else A.shape
    nu = B.shape[1] if B.ndim == 2 else 1
    ny = C.shape[0] if C.ndim == 2 else 1
    nw = len(ww)
    if n != m:
        raise ValueError(f"A must be square, got shape ({n}, {m}).")
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


def _show_freqresp_info(n: int, nu: int, ny: int, nw: int, ww: np.ndarray) -> None:
    logger.info(
        "System dimensions: n=%d, nu=%d, ny=%d | Frequency points: nw=%d, w in [1e%g, 1e%g]",
        n,
        nu,
        ny,
        nw,
        np.log10(ww[0]),
        np.log10(ww[-1]),
    )


def get_frequency_response_sequential(
    A: PETSc.Mat | spr.spmatrix,
    B: np.ndarray,
    C: np.ndarray,
    Q: PETSc.Mat | spr.spmatrix,
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
    A: PETSc.Mat | spr.spmatrix,
    B: np.ndarray,
    C: np.ndarray,
    Q: PETSc.Mat | spr.spmatrix,
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
    B = _ensure_global_B(B, n, comm)
    nu = B.shape[1]
    ny = C.shape[0]
    nw = len(ww)

    if rank == 0:
        _show_freqresp_info(n, nu, ny, nw, ww)

    rhs_vecs = _build_rhs(B, n, comm)
    H = np.zeros((ny, nu, nw), dtype=complex)
    ksp = _create_mumps_ksp(comm)

    t_start = time.time()
    for ii, w in enumerate(ww):
        t_step = time.time()
        Ablk = _build_block_matrix(A, Q, w)
        ksp.setOperators(Ablk)
        ksp.setUp()
        sol_blk = Ablk.createVecRight()

        for iu in range(nu):
            ksp.solve(rhs_vecs[iu], sol_blk)
            xr, xi = _gather_solution(sol_blk, n, comm)
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
        logger.info("Frequency response computed in %.3fs total.", time.time() - t_start)
    return H, ww


def get_field_response(
    A: PETSc.Mat,
    B: np.ndarray,
    Q: PETSc.Mat,
    ww: float | np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute field response X(w) = inv(jwQ - A) * B for each frequency in ww.

    Returns X of shape (n, nu, nw) complex, where n is the state dimension,
    nu the number of inputs, and nw the number of frequencies.

    To export to XDMF use io.export_complex_field.
    """
    ww = np.atleast_1d(np.asarray(ww, dtype=float))
    nw = len(ww)
    comm = A.getComm()
    rank = comm.getRank()

    n = A.getSize()[0]
    B = _ensure_global_B(B, n, comm)
    nu = B.shape[1]

    if rank == 0:
        _show_freqresp_info(n, nu, ny=n, nw=nw, ww=ww)

    rhs_vecs = _build_rhs(B, n, comm)
    X = np.zeros((n, nu, nw), dtype=complex)
    ksp = _create_mumps_ksp(comm)

    t_start = time.time()
    for ii, w in enumerate(ww):
        t_step = time.time()
        Ablk = _build_block_matrix(A, Q, w)
        ksp.setOperators(Ablk)
        ksp.setUp()
        sol_blk = Ablk.createVecRight()

        for iu in range(nu):
            ksp.solve(rhs_vecs[iu], sol_blk)
            xr, xi = _gather_solution(sol_blk, n, comm)
            X[:, iu, ii] = xr + 1j * xi

        if verbose and rank == 0:
            logger.info(
                "  [%d/%d] w=%.4e | step: %.3fs | total: %.3fs",
                ii + 1,
                nw,
                w,
                time.time() - t_step,
                time.time() - t_start,
            )
        Ablk.destroy()

    ksp.destroy()
    if rank == 0:
        logger.info("Field response computed in %.3fs total.", time.time() - t_start)
    return X


def _solve_at_frequency(
    w: float,
    Acsr: spr.csc_matrix,
    Qcsr: spr.csc_matrix,
    rhs: np.ndarray,
    C: np.ndarray,
    n: int,
) -> np.ndarray:
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
    """Build distributed PETSc Vecs for the block-system RHS [B[:, iu]; 0].

    B must be globally replicated on all ranks (shape (n, nu)).
    """
    nu = B.shape[1]
    vecs = []
    for iu in range(nu):
        rhs = PETSc.Vec().create(comm=comm)
        rhs.setSizes(2 * n)
        rhs.setUp()
        rstart, rend = rhs.getOwnershipRange()
        indices = np.arange(rstart, rend)
        top = indices[indices < n]
        if len(top) > 0:
            rhs.setValues(top, B[top, iu])
        rhs.assemblyBegin()
        rhs.assemblyEnd()
        vecs.append(rhs)
    return vecs


def _ensure_global_B(B: np.ndarray, n: int, comm: PETSc.Comm) -> np.ndarray:
    """Return B as a globally replicated (n, nu) array on all ranks.

    If B is already shape (n, nu) it is returned as-is. If each rank holds
    only its local DOF slice (shape (local_n, nu)), the chunks are allgathered
    into the full array. B is reshaped to 2D (n, 1) if 1D.
    """
    B = B.reshape(-1, 1) if B.ndim == 1 else B
    if B.shape[0] != n:
        B = np.vstack(comm.tompi4py().allgather(B))
    return B


def _create_mumps_ksp(comm: PETSc.Comm) -> PETSc.KSP:
    """Create a PREONLY / LU / MUMPS KSP solver."""
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp


def _gather_solution(sol_blk: PETSc.Vec, n: int, comm: PETSc.Comm) -> tuple[np.ndarray, np.ndarray]:
    """Gather distributed block solution [xr; xi] into full arrays on all ranks."""
    local_vals = sol_blk.getArray()
    rstart, rend = sol_blk.getOwnershipRange()
    global_idx = np.arange(rstart, rend)

    xr_local = np.zeros(n)
    xi_local = np.zeros(n)
    top = global_idx < n
    xr_local[global_idx[top]] = local_vals[top]
    xi_local[global_idx[~top] - n] = local_vals[~top]

    xr, xi = np.zeros(n), np.zeros(n)
    comm.tompi4py().Allreduce(xr_local, xr)
    comm.tompi4py().Allreduce(xi_local, xi)
    return xr, xi
