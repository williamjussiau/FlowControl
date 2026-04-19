"""
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c≈7700. Suggested Re=8000.

Demonstrates:
- compute_steady_state with Re close to bifurcation (Picard only)
- unactuated time simulation
"""

import logging
from pathlib import Path

import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCUniformU
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent
Re = 8000


def main():
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=100, dt=0.005, Tstart=0.0)
    params_save = flowsolverparameters.ParamSave(
        save_every=20, path_out=cwd / "data_output"
    )
    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )
    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "mesh64.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    actuator_bc_up = ActuatorBCUniformU(boundary_name="lid")
    sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5]))
    sensor_2 = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.95]))
    sensor_3 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.5, 0.95]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_1, sensor_2, sensor_3],
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
        params_control=params_control,
        params_ic=params_ic,
        verbose=10,
    )

    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )
    fs.compute_steady_state(method="picard", max_iter=40, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        fs.step(u_ctrl=[0.0 * y_meas[0]])

    fs.write_timeseries()


if __name__ == "__main__":
    main()
