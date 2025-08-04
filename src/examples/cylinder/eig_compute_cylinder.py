"""
----------------------------------------------------------------------
Eigenvalues utilities
Warning: does not run with MPI
----------------------------------------------------------------------
Warning: these functions should be run in a dedicated slepc4py environment,
including: slepc4py=*=*complex*, scipy, matplotlib, conda-build
with conda-develop src
----------------------------------------------------------------------
"""

# from dolfin import *
# import functools

# import matplotlib.pyplot as plt
# from matplotlib import cm
# import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import scipy as scp
import scipy.io as sio
import scipy.sparse as spr
from petsc4py import PETSc
from slepc4py import SLEPc

from utils.eig.eig_utils import get_mat_vp_slepc, sparse_to_petscmat

# import warnings


#################################################################################
#################################################################################
#################################################################################


def main():
    # intro
    print("----- using slepc ----- begin")
    print("...............................")

    # process matrices
    print("--- Loading matrices")
    save_npz_path = Path("src", "examples", "cylinder", "data_output", "operators")
    print("--- \t from sparse...")
    t0 = time.time()
    AA = spr.load_npz(save_npz_path / "A.npz")
    BB = spr.load_npz(save_npz_path / "E.npz")
    print("--- \t ... to petscmat")
    seq = True
    AA = sparse_to_petscmat(AA, sequential=seq)
    BB = sparse_to_petscmat(BB, sequential=seq)
    sz = AA.size[0]

    # targets
    # targets = np.array([0 + 0j, 1j, 2j, 3j])
    # neiglist = np.array([3, 3, 3, 3])

    targets = np.array(
        [
            -0.0 + 0j,
            -0.0 + 1j,
            -0.0 + 2j,
            -0.0 + 3j,
        ]
    )

    # targets_re = np.arange(start=0, step=-0.5, stop=-2)
    # targets_im = np.arange(start=0, step=0.5, stop=2)
    # txx, tyy = np.meshgrid(targets_re, targets_im)
    # txx = txx.flatten()
    # tyy = tyy.flatten()
    # targets = txx + 1j * tyy

    neig1 = 3
    neiglist = neig1 * np.ones(shape=targets.shape, dtype=int)

    # solve A@x = Q*L@x
    LAMBDA = np.zeros((0,), dtype=complex)
    V = np.zeros((sz, 0), dtype=complex)
    print("--- Starting solve A x = Q L x ")
    for target, neig in zip(targets, neiglist):
        print(f"---> Current target: {target} <---")
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
            tol=1e-12,
        )

        if np.any(np.real(L) > 0):
            print(
                f"!!! Warning !!! eigenvalue with positive real part: {np.real(L[np.real(L) > 0])}"
            )

        LAMBDA = np.hstack((LAMBDA, L))
        V = np.hstack((V, v))

    # export
    # as npz
    np.savez_compressed(save_npz_path / "eigenValues", LAMBDA)  # npz
    np.savez_compressed(save_npz_path / "eigenVectors", V)  # npz
    # as txt
    np.savetxt(save_npz_path / "eigenValues.txt", LAMBDA, delimiter=",")  # txt
    # np.savetxt(save_npz_path / "eigenVectors.txt", V, delimiter=",")  # txt, not recommended

    # exit
    print("Elapsed: %f" % (time.time() - t0))
    print("...............................")
    return LAMBDA, V


def plot_eig(LAMBDA):
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
    fig.savefig(
        Path(
            "src",
            "examples",
            "cylinder",
            "data_output",
            "operators",
            "eigenValues.png",
        )
    )


if __name__ == "__main__":
    LAMBDA, V = main()

    PLOT_EIG = True
    if PLOT_EIG:
        plot_eig(LAMBDA=LAMBDA)
