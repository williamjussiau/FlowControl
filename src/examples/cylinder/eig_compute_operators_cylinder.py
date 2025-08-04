import logging
import time
from pathlib import Path

import dolfin
import numpy as np
import scipy.sparse as spr
import scipy.io

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint
from examples.cylinder.compute_steady_state import Re
from flowcontrol.operatorgetter import OperatorGetter
from examples.cylinder.compute_steady_state import Re


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "O1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart()

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    angular_size_deg = 10
    actuator_bc_1 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    actuator_bc_2 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CylinderFlowSolver(
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
