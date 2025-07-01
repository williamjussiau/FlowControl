"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=7700
----------------------------------------------------------------------
This file exports operators A and E from a LidCavityFlowSolver
It should be run in a conda environment for FEniCS
----------------------------------------------------------------------
"""

import logging
from pathlib import Path

import dolfin
import scipy.sparse as spr

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCUniformU
from flowcontrol.operatorgetter import OperatorGetter


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)

    # Instantiate LidCavityFlowSolver
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=8000, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=100, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        # meshpath=cwd / "data_input" / "mesh128.xdmf"
        meshpath=cwd / "data_input" / "lidcavity_5.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    actuator_bc_up = ActuatorBCUniformU()
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[],
        actuator_list=[actuator_bc_up],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=0.1, yloc=0.1, radius=0.1, amplitude=0.1
    )

    fs = LidCavityFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=10,
    )

    logger.info("__init__(): successful!")

    logger.info("Compute steady state...")
    U00 = dolfin.Function(fs.V)
    P00 = dolfin.Function(fs.P)
    steady_state_filename_U0 = params_save.path_out / "steady" / f"U0_Re=8000.xdmf"
    steady_state_filename_P0 = params_save.path_out / "steady" / f"P0_Re=8000.xdmf"
    flu.read_xdmf(steady_state_filename_U0, U00, "U0")
    flu.read_xdmf(steady_state_filename_P0, P00, "P0")
    initial_guess = fs.merge(U00, P00)
    uctrl0 = [0.0]
    fs.compute_steady_state(method="picard", max_iter=20, tol=1e-8, u_ctrl=uctrl0, initial_guess=initial_guess)
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )
    # or load

    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(fs)
    A0 = opget.get_A(UP0=fs.fields.UP0, autodiff=True)

    # Compute operator: E
    E = opget.get_mass_matrix()

    export = True
    if export:
        path_out = cwd / "data_output" / "operators"
        path_out.mkdir(parents=True, exist_ok=True)
        for Mat, Matname in zip([A0, E], ["A", "E"]):
            # Export as png only
            flu.export_sparse_matrix(Mat, path_out / f"{Matname}.png")

            # Export as npz
            Matc, Mats, Matr = Mat.mat().getValuesCSR()
            Acsr = spr.csr_matrix((Matr, Mats, Matc))
            spr.save_npz(path_out / f"{Matname}.npz", Acsr)

    logger.info("Lidcavity -- Finished properly.")
    logger.info("*" * 50)


if __name__ == "__main__":
    main()
