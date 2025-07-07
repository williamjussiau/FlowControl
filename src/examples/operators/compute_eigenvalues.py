"""
----------------------------------------------------------------------
Compute eigenvalues and eigenvectors for example problems
Run compute_operators.py first to save operators as .npz
Run compute_eigenvalues.py after, pointing to the right path
----------------------------------------------------------------------
Warning: these functions should be run in a dedicated slepc4py environment,
including: slepc4py=*=*complex*, scipy, matplotlib, conda-build
and conda-develop src (to include src/utils/eig)
----------------------------------------------------------------------
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as sprl
from petsc4py import PETSc
from slepc4py import SLEPc

from utils.eig.eig_utils import get_mat_vp_slepc, sparse_to_petscmat

#################################################################################
#################################################################################
#################################################################################


def main(operators_path):
    # intro
    print("----- using slepc ----- begin")
    print("...............................")

    # process matrices
    print("--- Loading matrices")
    print("--- \t from sparse...")
    t0 = time.time()
    AAspr = spr.load_npz(operators_path / "A.npz")
    BBspr = spr.load_npz(operators_path / "E.npz")
    print("--- \t ... to petscmat")
    seq = True
    AA = sparse_to_petscmat(AAspr, sequential=seq)
    BB = sparse_to_petscmat(BBspr, sequential=seq)
    sz = AA.size[0]

    ########################################
    # Expected unstable eigenvalues
    # (choose targets appropriately)
    ########################################

    # Cylinder @ Re=100 -> 1 eigenpair
    # 0.132643 +  0.770015j

    # Open cavity @ Re=7500 -> 4 eigenpairs
    # 0.889 + 10.899j
    # 0.727 + 13.800j
    # 0.461 + 7.881j
    # 0.0318 + 16.726j

    # Lid-driven cavity @ Re=?

    ########################################

    # Specify targets for shift-invert
    targets = np.array(
        [
            -0.0 + 0j,
            -0.0 + 1j,
            -0.0 + 2j,
            -0.0 + 3j,
        ]
    )
    # targets = np.array([0.1 + 0.8j, 0.0])
    # targets = np.array([-0.0 + 0j, 1 + 13j, 0.5 + 8j, 0 + 16j])

    # Mesh targets (for fine mapping)
    # targets_re = np.arange(start=0, step=-0.5, stop=-2)
    # targets_im = np.arange(start=0, step=0.5, stop=2)
    # txx, tyy = np.meshgrid(targets_re, targets_im)
    # targets = txx.flatten() + 1j * tyy.flatten()

    # Number of eigenvalues to compute at each target
    neig_at_target = 1
    neig_list = neig_at_target * np.ones(shape=targets.shape, dtype=int)

    # Solve A@x = L@Q@x
    LAMBDA = np.zeros((0,), dtype=complex)
    V = np.zeros((sz, 0), dtype=complex)
    print("--- Starting solve A x = Q L x ")
    for target, neig in zip(targets, neig_list):
        print(f"---> Current target: {target} <---")
        # L, v, eigensolver = get_mat_vp_slepc(
        #     A=AA,
        #     B=BB,
        #     target=target,
        #     n=neig,
        #     return_eigensolver=True,
        #     verbose=True,
        #     precond_type=PETSc.PC.Type.LU,
        #     eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
        #     ksp_type=PETSc.KSP.Type.PREONLY,
        #     tol=1e-12,
        # )

        L, v = sprl.eigs(A=AAspr, k=neig, M=BBspr, sigma=target)

        if np.any(np.real(L) > 0):
            print(
                f"!!! Warning !!! eigenvalue with positive real part: {L[np.real(L) > 0]}"
            )

        LAMBDA = np.hstack((LAMBDA, L))
        V = np.hstack((V, v))

    # Export
    # as npz
    np.savez_compressed(operators_path / "eigenValues", LAMBDA)  # npz
    np.savez_compressed(operators_path / "eigenVectors", V)  # npz
    # as txt
    np.savetxt(operators_path / "eigenValues.txt", LAMBDA, delimiter=",")  # txt
    # np.savetxt(operators_path / "eigenVectors.txt", V, delimiter=",")  # txt, not recommended

    # exit
    print("Elapsed: %f" % (time.time() - t0))
    print("...............................")
    return LAMBDA, V


def plot_eig(LAMBDA, operators_path):
    """Plot eigenvalues in the complex plane
    Conjugate eigenvalues are added by hand, even if not computed"""
    fig, ax = plt.subplots()
    ax.plot(
        np.real(LAMBDA),
        np.imag(LAMBDA),
        color="green",
        marker=".",
        linestyle="None",
        markersize=6,
    )
    ax.plot(
        np.real(LAMBDA),
        -np.imag(LAMBDA),
        color="red",
        marker=".",
        linestyle="None",
        markersize=6,
    )
    plt.axhline(y=0, color="k", linestyle="--")
    plt.axvline(x=0, color="k", linestyle="--")
    ax.grid(visible=True, which="major")
    ax.set_title("Eigenvalues")
    # export plot
    fig.savefig(operators_path / "eigenValues.png")


if __name__ == "__main__":
    # operators_path = Path("src", "examples", "operators", "cylinder", "data_output")
    # operators_path = Path("src", "examples", "operators", "cavity", "data_output")
    operators_path = Path("src", "examples", "operators", "lidcavity", "data_output")
    LAMBDA, V = main(operators_path=operators_path)

    PLOT_EIG = True
    if PLOT_EIG:
        plot_eig(LAMBDA=LAMBDA, operators_path=operators_path)
