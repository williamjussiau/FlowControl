"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=7700
----------------------------------------------------------------------
This file demonstrates the following possibilites:
    - Use initial guess for steady-state computation
    - Compute steady-states at increasing Re
----------------------------------------------------------------------
"""

import logging
from pathlib import Path

import dolfin

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV

Re_final = 8000


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=1000, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=10, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "lidcavity_5.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    actuator_bc_up = ActuatorBCParabolicV(width=0)
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[],
        actuator_list=[actuator_bc_up],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=0.1, yloc=0.1, radius=0.1, amplitude=0.1
    )

    Re_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500, Re_final]
    first_loop = True
    steady_state_filename_U0 = ""
    steady_state_filename_P0 = ""
    for Re in Re_list:
        logger.info("*" * 100)
        logger.info(f"--- Computing steady state for Reynolds number Re={Re}")
        logger.info("*" * 100)
        params_flow.Re = Re

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
        uctrl0 = [0.0]

        U00 = dolfin.Function(fs.V)
        P00 = dolfin.Function(fs.P)
        if first_loop:
            # use default
            initial_guess = None
        else:
            # use previous Reynolds
            flu.read_xdmf(steady_state_filename_U0, U00, "U0")
            flu.read_xdmf(steady_state_filename_P0, P00, "P0")
            initial_guess = fs.merge(U00, P00)

        fs.compute_steady_state(
            method="picard",
            max_iter=10,
            tol=1e-7,
            u_ctrl=uctrl0,
            initial_guess=initial_guess,
        )

        fs.compute_steady_state(
            method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
        )

        steady_state_filename_U0 = params_save.path_out / "steady" / f"U0_Re={Re}.xdmf"
        steady_state_filename_P0 = params_save.path_out / "steady" / f"P0_Re={Re}.xdmf"
        flu.write_xdmf(steady_state_filename_U0, fs.fields.U0, "U0")
        flu.write_xdmf(steady_state_filename_P0, fs.fields.P0, "P0")

        first_loop = False


if __name__ == "__main__":
    main()
