"""Integration tests for the cylinder flow example using flowsolver2.

Mirrors test_cylinder.py exactly, but exercises flowsolver2.FlowSolver
(with NSForms / SteadyStateSolver / FlowExporter) instead of flowsolver.FlowSolver.

The solver mechanics are identical so all reference values are the same.
The main differences exercised here:
- params_restart is optional (omitted in the smoke test, JSON-based in the regression)
- Restart discovery uses the JSON sidecar written by flowsolver2 automatically
- timeseries is a property returning a DataFrame on demand
"""

from pathlib import Path

import numpy as np
import pytest

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver2 import CylinderFlowSolver2
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "src" / "examples" / "cylinder"
MESH_PATH = EXAMPLE_DIR / "data_input" / "O1.xdmf"
CONTROLLER_PATH = EXAMPLE_DIR / "data_input" / "Kopt_reduced13.mat"


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _make_base_params(path_out, num_steps=3, save_every=0):
    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(
        num_steps=num_steps, dt=0.005, Tstart=0.0
    )
    params_save = flowsolverparameters.ParamSave(
        save_every=save_every, path_out=path_out
    )
    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )
    params_mesh = flowsolverparameters.ParamMesh(meshpath=MESH_PATH)
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

    sensor_feedback = SensorPoint(
        sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0.0])
    )
    sensor_perf_1 = SensorPoint(
        sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1.0])
    )
    sensor_perf_2 = SensorPoint(
        sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1.0])
    )
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


# ── Smoke test ────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_cylinder2_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("cylinder2_smoke")

    kw = _make_base_params(path_out=path_out, num_steps=3, save_every=0)
    fs = CylinderFlowSolver2(**kw, verbose=0)

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_cylinder2_regression(tmp_path_factory):
    """10-step closed-loop run + JSON-based restart must reproduce reference values."""
    u_max_ref = 2.2855984664058986
    u_mean_ref = 0.3377669778983669

    path_out = tmp_path_factory.mktemp("cylinder2_regression")

    # ── First run (10 steps, saves at step 5 and 10) ─────────────────────────
    kw = _make_base_params(path_out=path_out, num_steps=10, save_every=5)
    fs = CylinderFlowSolver2(**kw, verbose=0)

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=[0.0, 0.0], initial_guess=fs.fields.UP0
    )
    fs.initialize_time_stepping(ic=None)

    Kss = Controller.from_file(file=CONTROLLER_PATH, x0=None)

    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)
        fs.step(u_ctrl=[u_ctrl[0], u_ctrl[0]])

    fs.write_timeseries()

    # ── Restart from Tstart=0.05 using JSON sidecar (no ParamRestart needed) ─
    params_time_restart = flowsolverparameters.ParamTime(
        num_steps=10, dt=0.005, Tstart=0.05
    )

    kw_restart = _make_base_params(path_out=path_out, num_steps=10, save_every=5)
    kw_restart["params_time"] = params_time_restart
    # params_restart omitted — flowsolver2 discovers the checkpoint via JSON sidecar

    fs_restart = CylinderFlowSolver2(**kw_restart, verbose=0)
    fs_restart.load_steady_state()
    fs_restart.initialize_time_stepping(Tstart=fs_restart.params_time.Tstart)

    for _ in range(fs_restart.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs_restart.params_time.dt)
        fs_restart.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    fs_restart.write_timeseries()

    u_max = flu.apply_fun(fs_restart.fields.Usave, np.max)
    u_mean = flu.apply_fun(fs_restart.fields.Usave, np.mean)

    assert np.isclose(u_max, u_max_ref, rtol=1e-6), (
        f"u_max mismatch: {u_max} != {u_max_ref}"
    )
    assert np.isclose(u_mean, u_mean_ref, rtol=1e-6), (
        f"u_mean mismatch: {u_mean} != {u_mean_ref}"
    )

    last = fs_restart.timeseries.iloc[-1]
    assert np.isclose(last["time"], 0.100, rtol=1e-6), f"time: {last['time']}"
    assert np.isclose(last["y_meas_1"], 0.131695, rtol=1e-4), (
        f"y_meas_1: {last['y_meas_1']}"
    )
    assert np.isclose(last["y_meas_2"], 0.009738, rtol=1e-4), (
        f"y_meas_2: {last['y_meas_2']}"
    )
    assert np.isclose(last["y_meas_3"], 0.009810, rtol=1e-4), (
        f"y_meas_3: {last['y_meas_3']}"
    )
    assert np.isclose(last["dE"], 0.122620, rtol=1e-4), f"dE: {last['dE']}"
