from pathlib import Path

import scipy.sparse as spr
from petsc4py import PETSc
from slepc4py import SLEPc

from utils.eig.eig_utils import get_mat_vp_slepc, sparse_to_petscmat


def main():
    opts = PETSc.Options()
    n = opts.getInt("n", 30)

    ## Make matrix
    A, B = make_operators(n=n)
    # operators_path = Path("src", "examples", "operators", "cylinder", "data_output")
    # operators_path = Path("src", "examples", "operators", "cavity", "data_output")
    # operators_path = Path("src", "examples", "operators", "lidcavity", "data_output")
    # A, B = load_operators(operators_path=operators_path)

    ## Make EPS
    E = SLEPc.EPS()
    E.create()

    E.setOperators(A, B)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    E.setFromOptions()

    sigma = 4
    # sigma = 0.13 + 0.7j
    # sigma = 1 + 13j
    # sigma = 0 + 2.85j
    E.setTarget(sigma)

    # # Spectral transform: shift-and-invert
    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(sigma)  # target value
    st.setTransform(True)  # activates spectral transform

    # # Access the linear solver (KSP) used in the ST
    # ksp = st.getKSP()
    # ksp.setType("gmres")  # or 'bcgs', 'fgmres', etc.

    # # Preconditioner
    # pc = ksp.getPC()
    # pc.setType("ilu")  # or 'hypre', 'gamg', 'lu', 'none'

    # # Optional: tweak tolerances
    # ksp.setTolerances(rtol=1e-8)

    ## Solve EPS
    E.solve()

    ## Show solution
    Print = PETSc.Sys.Print

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    if nconv > 0:
        # Create the results vectors
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()
        #
        Print()
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)
            if k.imag != 0.0:
                Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f      %12g" % (k.real, error))
        Print()


def make_operators(n):
    """Make dummy operators A (sym), B (singular)"""
    # A
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    rstart, rend = A.getOwnershipRange()

    # first row
    if rstart == 0:
        A[0, :2] = [2, -1]
        rstart += 1
    # last row
    if rend == n:
        A[n - 1, -2:] = [-1, 2]
        rend -= 1
    # other rows
    for i in range(rstart, rend):
        A[i, i - 1 : i + 2] = [-1, 2, -1]

    A.assemble()

    # B
    B = PETSc.Mat().create()
    B.setSizes([n, n])
    B.setFromOptions()
    B.setUp()
    rstart, rend = B.getOwnershipRange()
    for i in range(rstart, rend):
        B[i, i] = 1
    B[1, 1] = 0
    B.assemble()

    return A, B


def load_operators(operators_path):
    """Load operators A, B from path"""
    # Read
    AAspr = spr.load_npz(operators_path / "A.npz")
    BBspr = spr.load_npz(operators_path / "E.npz")
    # To PETSc
    seq = True
    A = sparse_to_petscmat(AAspr, sequential=seq)
    B = sparse_to_petscmat(BBspr, sequential=seq)

    return A, B


if __name__ == "__main__":
    main()
