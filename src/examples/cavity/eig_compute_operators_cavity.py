"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=8000
----------------------------------------------------------------------
This file exports operators A and E from a LidCavityFlowSolver
It should be run in a conda environment for FEniCS
----------------------------------------------------------------------
"""

import logging
from pathlib import Path

import dolfin
import numpy as np
import scipy.sparse as spr
import scipy.io

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cavity.cavityflowsolver import CavityFlowSolver
from flowcontrol.actuator import ActuatorForceGaussianV
from flowcontrol.sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint
# from flowcontrol.actuator import ActuatorBCUniformU
from flowcontrol.operatorgetter import OperatorGetter
from examples.cavity.compute_steady_state import Re


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)

    # Instantiate LidCavityFlowSolver
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["L"] = 1.0
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=1000, dt=0.0004, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=100, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "cavity_coarse.xdmf"
    )
    params_mesh.user_data["xinf"] = 2.5
    params_mesh.user_data["xinfa"] = -1.2
    params_mesh.user_data["yinf"] = 0.5
    params_mesh.user_data["x0ns_left"] = -0.4  # sf left from x0ns, ns right from x0ns
    params_mesh.user_data["x0ns_right"] = 1.75  # ns left from x0ns, sf right from x0ns

    params_restart = flowsolverparameters.ParamRestart()

    actuator_force = ActuatorForceGaussianV(
        sigma=0.0849, position=np.array([-0.1, 0.02])
    )

    sensor_feedback = SensorHorizontalWallShear(
        sensor_index=100,
        x_sensor_left=1.0,
        x_sensor_right=1.1,
        y_sensor=0.0,
        sensor_type=SENSOR_TYPE.OTHER,
    )
    sensor_perf_1 = SensorPoint(
        sensor_type=SENSOR_TYPE.U, position=np.array([0.1, 0.1])
    )
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1],
        actuator_list=[actuator_force],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CavityFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=5,
    )

    logger.info("__init__(): successful!")

    # logger.info("Compute steady state...")
    # # U00 = dolfin.Function(fs.V)
    # # P00 = dolfin.Function(fs.P)
    # # steady_state_filename_U0 = params_save.path_out / "steady" / f"U0_Re={Re}.xdmf"
    # # steady_state_filename_P0 = params_save.path_out / "steady" / f"P0_Re={Re}.xdmf"
    # # flu.read_xdmf(steady_state_filename_U0, U00, "U0")
    # # flu.read_xdmf(steady_state_filename_P0, P00, "P0")
    # # initial_guess = fs.merge(U00, P00)
    # uctrl0 = [0.0]
    # fs.compute_steady_state(method="picard", max_iter=20, tol=3e-7, u_ctrl=uctrl0, initial_guess=None)
    # fs.compute_steady_state(
    #     method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    # )
    # or load
    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0.xdmf",
            cwd / "data_output" / "steady" / f"P0.xdmf",
        ]
    )

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

            # Export as mat
            scipy.io.savemat(
                path_out / f"{Matname}_sparse.mat",
                {
                    f"{Matname}_data": Acsr.data,
                    f"{Matname}_indices": Acsr.indices,
                    f"{Matname}_indptr": Acsr.indptr,
                    f"{Matname}_shape": Acsr.shape,
                },
            )

    logger.info("Lidcavity -- Finished properly.")
    logger.info("*" * 50)


if __name__ == "__main__":
    main()
