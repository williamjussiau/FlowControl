"""Compute and save the frequency response H(jw) for the cylinder flow.

Linearizes the flow around the steady state, assembles (A, E, B, C),
then evaluates H(jw) = C (jwE - A)^{-1} B over a log-spaced frequency grid.
Results are saved as .mat files and Bode plots.
"""

import logging
from pathlib import Path

import numpy as np

import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.operatorgetter import OperatorGetter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent


if __name__ == "__main__":
    fs = CylinderFlowSolver.make_default(Re=100, path_out=cwd / "cylinder" / "data_output")
    fs.load_steady_state()

    opget = OperatorGetter(fs)
    A, E, B, C = opget.get_all()
    A = flu.dolfin_to_petsc(A)
    Q = flu.dolfin_to_petsc(E)

    ww = np.logspace(-2, 2, 50)
    H, ww = flu.get_frequency_response_parallel(A, B, C, Q, ww, verbose=True, n_jobs=2)

    if flu.MpiUtils.get_rank() == 0:
        save_dir = cwd / "cylinder" / "frequency_response"
        flu.save_Hw(
            H,
            ww,
            save_dir=save_dir,
            input_labels=["up", "lo"],
            output_labels=["fb", "perf1", "perf2"],
        )
        flu.plot_Hw(
            H,
            ww,
            save_dir=save_dir,
            input_labels=["up", "lo"],
            output_labels=["fb", "perf1", "perf2"],
        )
