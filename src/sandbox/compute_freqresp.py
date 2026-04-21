import logging
from pathlib import Path

import numpy as np
import scipy.sparse as spr

import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.operatorgetter import OperatorGetter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent


def export_square_operators(path, operators, operators_names):
    """Export given square operators as png and sparse npz"""
    path.mkdir(parents=True, exist_ok=True)
    for Mat, Matname in zip(operators, operators_names):
        flu.export_sparse_matrix(Mat, path / f"{Matname}.png")

        Matc, Mats, Matr = Mat.mat().getValuesCSR()
        Acsr = spr.csr_matrix((Matr, Mats, Matc))
        Acoo = Acsr.tocoo()
        spr.save_npz(path / f"{Matname}.npz", Acsr)
        spr.save_npz(path / f"{Matname}_coo.npz", Acoo)


def compute_operators(operator_getter: OperatorGetter):
    flowsolver = operator_getter.flowsolver
    opget = OperatorGetter(flowsolver)
    A0 = opget.get_A(UP0=flowsolver.fields.UP0, autodiff=True)
    E = opget.get_mass_matrix()
    B = opget.get_B()
    C = opget.get_C()
    return A0, B, C, E


if __name__ == "__main__":
    fs = CylinderFlowSolver.make_default(Re=100)
    fs.load_steady_state()

    operator_getter = OperatorGetter(fs)
    A, B, C, E = compute_operators(operator_getter)
    A = flu.dolfin_petsc_to_petsc(A)
    Q = flu.dolfin_petsc_to_petsc(E)

    logwmin = -2
    logwmax = 2
    nw = 50
    ww = np.logspace(logwmin, logwmax, nw)
    H, ww = flu.get_frequency_response_parallel(A, B, C, Q, ww, verbose=True, n_jobs=2)
    # H, ww = flu.get_frequency_response_sequential(A, B, C, Q, ww, verbose=True)

    if flu.MpiUtils.get_rank() == 0:
        flu.save_Hw(
            H,
            ww,
            save_dir=cwd / "frequency_response/",
            input_labels=["up", "lo"],
            output_labels=["fb", "perf1", "perf2"],
        )
        flu.plot_Hw(
            H,
            ww,
            save_dir=cwd / "frequency_response/",
            input_labels=["up", "lo"],
            output_labels=["fb", "perf1", "perf2"],
        )
