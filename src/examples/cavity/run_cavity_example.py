"""
Flow over an open cavity
Nondimensional incompressible Navier-Stokes equations
Suggested Re=7500.

Demonstrates:
- compute_steady_state (Picard then Newton)
- unactuated time simulation with a volume-force actuator
"""

import logging
from pathlib import Path

import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cavity.cavityflowsolver import CavityFlowSolver
from flowcontrol.actuator import ActuatorForceGaussianV
from flowcontrol.sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint

logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent


def main():
    params_flow = flowsolverparameters.ParamFlow(Re=7500, uinf=1.0)
    params_flow.user_data["L"] = 1.0
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=100, dt=0.0004, Tstart=0.0)
    params_save = flowsolverparameters.ParamSave(
        save_every=20, path_out=cwd / "data_output"
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
    params_mesh.user_data["x0ns_left"] = -0.4
    params_mesh.user_data["x0ns_right"] = 1.75

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
        params_control=params_control,
        params_ic=params_ic,
        verbose=10,
    )

    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )
    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=[0.0])
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=[0.0], initial_guess=fs.fields.UP0
    )
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    fs.write_timeseries()


if __name__ == "__main__":
    main()
