"""
Fluidic pinball — rotation actuation
3 cylinders, nondimensional incompressible Navier-Stokes.
Suggested Re=100.

Demonstrates:
- ActuatorBCRotation: tangential velocity on the full cylinder surface
- compute_steady_state with force-coefficient reporting
- open-loop Gaussian-bump rotation actuation
"""

import logging
from pathlib import Path

import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.pinball.pinballflowsolver import (
    PinballCustomInitialGuess,
    PinballFlowSolver,
)
from flowcontrol.actuator import CYLINDER_ACTUATION_MODE, ActuatorBCRotation
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent


def main():
    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=20, dt=0.005, Tstart=0.0)
    params_save = flowsolverparameters.ParamSave(
        save_every=10, path_out=cwd / "data_output"
    )
    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )
    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "mesh_middle_gmsh.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -6
    params_mesh.user_data["yinf"] = 6

    cylinder_diameter = params_flow.user_data["D"]
    position_mid = [-1.5 * np.cos(np.pi / 6), 0.0]
    position_top = [0.0, +0.75]

    actuator_mid = ActuatorBCRotation(
        position_x=position_mid[0],
        position_y=position_mid[1],
        diameter=cylinder_diameter,
        boundary_name="actuator_mid",
    )
    actuator_top = ActuatorBCRotation(
        position_x=position_top[0],
        position_y=+position_top[1],
        diameter=cylinder_diameter,
        boundary_name="actuator_top",
    )
    actuator_bot = ActuatorBCRotation(
        position_x=position_top[0],
        position_y=-position_top[1],
        diameter=cylinder_diameter,
        boundary_name="actuator_bot",
    )

    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([8.0, 0.0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([10.0, 0.0]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([12.0, 0.0]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_mid, actuator_top, actuator_bot],
        user_data={"mode_actuation": CYLINDER_ACTUATION_MODE.ROTATION},
    )
    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = PinballFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_control=params_control,
        params_ic=params_ic,
        verbose=10,
    )

    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )
    initial_guess = PinballCustomInitialGuess(mode="antisymmetric_bot")
    fs.compute_steady_state(
        method="picard",
        max_iter=15,
        tol=1e-7,
        u_ctrl=[0.0, 0.0, 0.0],
        initial_guess=initial_guess.as_dolfin_function(fs.W),
    )
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=[0.0, 0.0, 0.0], initial_guess=fs.fields.UP0
    )
    fs.initialize_time_stepping(ic=None)

    # Open-loop: Gaussian-bump rotation on each cylinder
    tlen = 0.10
    tpeak = [0.25, 0.5, 0.75]
    u0peak = [+2.0, -1.5, -2.0]

    def gaussian_bump(t, tp):
        return np.exp(-0.5 * (t - tp) ** 2 / tlen**2)

    for _ in range(fs.params_time.num_steps):
        u_ctrlm = u0peak[0] * gaussian_bump(fs.t, tpeak[0])
        u_ctrlt = u0peak[1] * gaussian_bump(fs.t, tpeak[1])
        u_ctrlb = u0peak[2] * gaussian_bump(fs.t, tpeak[2])
        fs.step(u_ctrl=[u_ctrlm, u_ctrlt, u_ctrlb])

    fs.write_timeseries()

    cl_cd_dict = fs.compute_force_coefficients(fs.fields.u_, fs.fields.p_)
    for surface, (cl, cd) in cl_cd_dict.items():
        print(f"{surface}: Cl={cl:.4f}, Cd={cd:.4f}")


if __name__ == "__main__":
    main()
