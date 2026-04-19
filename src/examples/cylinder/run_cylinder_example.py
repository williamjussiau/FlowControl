"""
Flow past a cylinder
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation at Re_c≈46. Suggested Re=100.

Demonstrates:
- compute_steady_state (Picard then Newton)
- closed-loop time-stepping with a state-space controller
- restart from a JSON-sidecar checkpoint (no ParamRestart needed)
"""

import logging
from pathlib import Path

import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

logging.basicConfig(level=logging.INFO)

cwd = Path(__file__).parent


def make_params(path_out, num_steps, save_every, Tstart=0.0):
    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(
        num_steps=num_steps, dt=0.005, Tstart=Tstart
    )
    params_save = flowsolverparameters.ParamSave(
        save_every=save_every, path_out=path_out
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

    angular_size_deg = 10
    radius = params_flow.user_data["D"] / 2
    width = ActuatorBCParabolicV.angular_size_deg_to_width(angular_size_deg, radius)
    actuator_bc_1 = ActuatorBCParabolicV(
        width=width, position_x=0.0, boundary_name="actuator_up"
    )
    actuator_bc_2 = ActuatorBCParabolicV(
        width=width, position_x=0.0, boundary_name="actuator_lo"
    )

    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0.0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1.0]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1.0]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )
    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    return dict(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_control=params_control,
        params_ic=params_ic,
    )


def main():
    path_out = cwd / "data_output"

    # ── First run ─────────────────────────────────────────────────────────────
    kw = make_params(path_out=path_out, num_steps=10, save_every=5)
    fs = CylinderFlowSolver(**kw, verbose=5)
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, path_out / "subdomains.xdmf"
    )

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=[0.0, 0.0], initial_guess=fs.fields.UP0
    )
    fs.initialize_time_stepping(ic=None)

    Kss = Controller.from_file(file=cwd / "data_input" / "Kopt_reduced13.mat", x0=None)

    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)
        fs.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    fs.write_timeseries()

    # ── Restart from Tstart=0.05 (JSON sidecar, no ParamRestart needed) ──────
    kw_restart = make_params(path_out=path_out, num_steps=10, save_every=5, Tstart=0.05)
    fs_restart = CylinderFlowSolver(**kw_restart, verbose=5)

    fs_restart.load_steady_state()
    fs_restart.initialize_time_stepping(Tstart=fs_restart.params_time.Tstart)

    for _ in range(fs_restart.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs_restart.params_time.dt)
        fs_restart.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    fs_restart.write_timeseries()


if __name__ == "__main__":
    main()
