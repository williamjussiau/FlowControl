import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spr

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cavity.cavityflowsolver import CavityFlowSolver
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV, ActuatorForceGaussianV
from flowcontrol.operatorgetter import OperatorGetter
from flowcontrol.sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint

OPERATORS_CYLINDER = True
OPERATORS_CAVITY = False
OPERATORS_LIDCAVITY = False  # not implemented yet
OPERATORS_PINBALL = False  # not implemented yet

# Set logger
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

cwd = Path(__file__).parent

# Get Cylinder operators
if OPERATORS_CYLINDER:
    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "cylinder" / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd.parent / "cylinder" / "data_input" / "O1.xdmf"
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

    fscyl = CylinderFlowSolver(
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
    logger.info("Compute steady state...")
    uctrl0 = [0.0, 0.0]
    fscyl.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)
    fscyl.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fscyl.fields.UP0
    )
    # fscyl.load_steady_state()

    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(fscyl)
    A0 = opget.get_A(UP0=fscyl.fields.UP0, autodiff=True)

    # Compute operator: B
    B = opget.get_B(as_function_list=False, interpolate=True)

    # Compute operator: C
    # import time
    # t0 = time.time()
    C = opget.get_C(check=False, fast=True)
    # dt1 = time.time() - t0
    # logger.info(dt1)
    # C = opget.get_C(check=False, fast=False)
    # dt2 = time.time() - (t0 + dt1)
    # logger.info(dt2)
    # logger.info(f"Error in optimization: {np.linalg.norm(C - C_opt)}")

    # Compute operator: E
    E = opget.get_mass_matrix()

    export = True
    # if export:
    #     path_out = cwd / "cylinder" / "data_output"
    #     flu.export_sparse_matrix(A0, path_out / "A.png")
    #     flu.export_field(B, fscyl.W, fscyl.V, fscyl.P, save_dir=path_out / "B")
    #     flu.export_field(C.T, fscyl.W, fscyl.V, fscyl.P, save_dir=path_out / "C")
    #     flu.export_sparse_matrix(E, path_out / "E.png")

    # Exporting like in eig_compute_operators_lidcavity.py
    if export:
        path_out = cwd / "cylinder" / "data_output"
        path_out.mkdir(parents=True, exist_ok=True)
        for Mat, Matname in zip([A0, E], ["A", "E"]):
            # Export as png only
            flu.export_sparse_matrix(Mat, path_out / f"{Matname}.png")

            # Export as npz
            Matc, Mats, Matr = Mat.mat().getValuesCSR()
            Acsr = spr.csr_matrix((Matr, Mats, Matc))
            spr.save_npz(path_out / f"{Matname}.npz", Acsr)

    logger.info("Cylinder -- Finished properly.")
    logger.info("*" * 50)


# Get Cavity operators
if OPERATORS_CAVITY:
    params_flow = flowsolverparameters.ParamFlow(Re=7500, uinf=1.0)
    params_flow.user_data["L"] = 1.0
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.0004, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "cavity" / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd.parent / "cavity" / "data_input" / "cavity_coarse.xdmf"
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

    fscav = CavityFlowSolver(
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
    logger.info("Compute steady state...")
    uctrl0 = [0.0]
    fscav.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=uctrl0)
    fscav.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fscav.fields.UP0
    )
    # fscav.load_steady_state()

    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(fscav)

    t0 = time.time()
    A0 = opget.get_A(UP0=fscav.fields.UP0, autodiff=True)
    dt1 = time.time() - t0
    logger.info(dt1)

    # Compute operator: B
    t0 = time.time()
    B = opget.get_B(as_function_list=False, interpolate=True)
    dt1 = time.time() - t0
    logger.info(dt1)

    # Compute operator: C
    t0 = time.time()
    C = opget.get_C(check=True, fast=True)
    dt1 = time.time() - t0
    for jj in range(C.shape[0]):
        print(f"max of Cj is: {max(C[jj, :])}")
    # max of Cj is: 3.038870378001052
    # max of Cj is: 0.9999999999994703
    # result is somewhat different in parallel, nice!
    # ---> check matrices in Paraview
    # dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
    logger.info(dt1)

    # Compute operator: E
    t0 = time.time()
    E = opget.get_mass_matrix()
    dt1 = time.time() - t0
    logger.info(dt1)

    export = True
    if export:
        path_out = cwd / "cavity" / "data_output"
        flu.export_sparse_matrix(A0, path_out / "A.png")
        flu.export_field(B, fscav.W, fscav.V, fscav.P, save_dir=path_out / "B")
        flu.export_field(C.T, fscav.W, fscav.V, fscav.P, save_dir=path_out / "C")
        flu.export_sparse_matrix(E, path_out / "E.png")

    logger.info("Cavity -- Finished properly.")
