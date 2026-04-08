import logging
from pathlib import Path

import numpy as np
import scipy.sparse as spr

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.operatorgetter import OperatorGetter
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

# Set logger
logger = logging.getLogger(__name__)

cwd = Path(__file__).parent


def export_square_operators(path, operators, operators_names):
    """Export given square operators as png and sparse npz"""
    path.mkdir(parents=True, exist_ok=True)
    for Mat, Matname in zip(operators, operators_names):
        # Export as png only
        flu.export_sparse_matrix(Mat, path / f"{Matname}.png")

        # Export as npz
        Matc, Mats, Matr = Mat.mat().getValuesCSR()
        Acsr = spr.csr_matrix((Matr, Mats, Matc))
        Acoo = Acsr.tocoo()
        spr.save_npz(path / f"{Matname}.npz", Acsr)
        spr.save_npz(path / f"{Matname}_coo.npz", Acoo)


def export_column_operators(path, operators, operators_names):
    """Export given column operators as xdmf"""
    return 1


####################################################################################
# Get Cylinder operators
####################################################################################
def make_cylinder(Re=100):
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(save_every=5, path_out=cwd)

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd.parent / "examples" / "cylinder" / "data_input" / "O1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart()

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

    params_ic = flowsolverparameters.ParamIC()

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
    # fscyl.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)
    # fscyl.compute_steady_state(
    #     method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fscyl.fields.UP0
    # )
    fscyl.load_steady_state()

    return fscyl


####################################################################################
# Compute operators
####################################################################################
def compute_operators(operator_getter: OperatorGetter):
    flowsolver = operator_getter.flowsolver

    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(flowsolver)
    A0 = opget.get_A(UP0=flowsolver.fields.UP0, autodiff=True)

    # Compute operator: E
    E = opget.get_mass_matrix()

    # Could also compute B, C
    B = opget.get_B(interpolate=False)
    C = opget.get_C(fast=True)

    return A0, B, C, E


####################################################################################
# Main
####################################################################################
if __name__ == "__main__":
    cylinder_flow = make_cylinder(Re=100)
    operator_getter = OperatorGetter(cylinder_flow)
    A, B, C, E = compute_operators(operator_getter)
    A = flu.dolfin_petsc_to_petsc(A)
    Q = flu.dolfin_petsc_to_petsc(E)

    logwmin = -2
    logwmax = 2
    nw = 100
    ww = np.logspace(logwmin, logwmax, nw)
    # H, ww = flu.get_frequency_response(A, B, C, Q, ww)
    H, ww = flu.get_frequency_response_parallel(
        A, B, C, Q, ww, n_jobs=12
    )  # 14 logical procs

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
