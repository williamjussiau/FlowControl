import logging
from pathlib import Path

import numpy as np
import scipy.sparse as spr

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cavity.cavityflowsolver import CavityFlowSolver
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import (
    ActuatorBCParabolicV,
    ActuatorBCUniformU,
    ActuatorForceGaussianV,
)
from flowcontrol.operatorgetter import OperatorGetter
from flowcontrol.sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint

# Set logger
logger = logging.getLogger(__name__)

cwd = Path(__file__).parent


def export_operators(path, operators, operators_names):
    """Export given operators as png and sparse npz"""
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


####################################################################################
# Get Cylinder operators
####################################################################################
def make_cylinder(Re=100):
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
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

    return fscyl


####################################################################################
# Get Cavity operators
####################################################################################
def make_cavity(Re=7500):
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
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

    return fscav


####################################################################################
# Get Lid-driven Cavity operators
####################################################################################
def make_lidcavity(Re=4000):
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=100, path_out=cwd / "lidcavity" / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd.parent / "lidcavity" / "data_input" / "mesh64.xdmf"
    )
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

    fslid = LidCavityFlowSolver(
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
    # fslid.compute_steady_state(
    #     method="picard", max_iter=16, tol=1e-8, u_ctrl=uctrl0, initial_guess=None
    # )
    # fslid.compute_steady_state(
    #     method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fslid.fields.UP0
    # )
    root_path_load = cwd.parent
    fslid.load_steady_state(
        path_u_p=[
            root_path_load
            / Path(
                "lidcavity",
                "data_output",
                "steady",
                "U0_Re=8000.xdmf",
            ),
            root_path_load
            / Path(
                "lidcavity",
                "data_output",
                "steady",
                "P0_Re=8000.xdmf",
            ),
        ]
    )

    return fslid


####################################################################################
# Get Pinball operators
####################################################################################
def make_pinball():
    raise NotImplementedError()
    return None


####################################################################################
# Compute operators
####################################################################################
def compute_operators_flowsolver(flowsolver, export):
    # Compute operator: A
    logger.info("Now computing operators...")
    opget = OperatorGetter(flowsolver)
    A0 = opget.get_A(UP0=flowsolver.fields.UP0, autodiff=True)

    # Compute operator: E
    E = opget.get_mass_matrix()

    # Could also compute B, C

    # Export
    if export:
        export_operators(
            path=cwd / "lidcavity" / "data_output",
            operators=[A0, E],
            operators_names=["A", "E"],
        )


####################################################################################
# Main
####################################################################################
if __name__ == "__main__":
    COMPUTE_OPERATORS_CYLINDER = False
    COMPUTE_OPERATORS_CAVITY = False
    COMPUTE_OPERATORS_LIDCAVITY = True
    COMPUTE_OPERATORS_PINBALL = False  # not implemented yet

    if COMPUTE_OPERATORS_CYLINDER:
        compute_operators_flowsolver(make_cylinder(Re=100), export=True)

    if COMPUTE_OPERATORS_CAVITY:
        compute_operators_flowsolver(make_cavity(Re=7500), export=True)

    if COMPUTE_OPERATORS_LIDCAVITY:
        compute_operators_flowsolver(make_lidcavity(Re=8000), export=True)

    if COMPUTE_OPERATORS_PINBALL:
        pass
