"""Integration tests for the cylinder flow example using flowsolver.

Mirrors test_cylinder.py exactly, but exercises flowsolver.FlowSolver
(with NSForms / SteadyStateSolver / FlowExporter) instead of flowsolver.FlowSolver.

The solver mechanics are identical so all reference values are the same.
The main differences exercised here:
- params_restart is optional (omitted in the smoke test, JSON-based in the regression)
- Restart discovery uses the JSON sidecar written by flowsolver automatically
- timeseries is a property returning a DataFrame on demand
"""

from pathlib import Path

import numpy as np
import pytest

import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.controller import Controller

CONTROLLER_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "examples"
    / "cylinder"
    / "data_input"
    / "Kopt_reduced13.mat"
)


# ── Fast CI test with coarse generated mesh ───────────────────────────────────


def test_cylinder_fast(coarse_cylinder_mesh, tmp_path_factory):
    """Fast smoke test with coarse generated mesh - runs in CI on every push."""
    path_out = tmp_path_factory.mktemp("cylinder_fast")

    # Override mesh path in make_default
    fs = CylinderFlowSolver.make_default(Re=100, path_out=path_out, num_steps=3)
    fs.params_mesh.meshpath = coarse_cylinder_mesh

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Smoke test ────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_cylinder_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("cylinder_smoke")

    fs = CylinderFlowSolver.make_default(Re=100, path_out=path_out, num_steps=3)
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────

_U_MAX_REF = np.float64(2.2855984664058986)
_U_MEAN_REF = np.float64(0.3377669778983669)
_LAST_TIME_REF = np.float64(0.100)
_LAST_Y_MEAS_1_REF = np.float64(0.131695)
_LAST_Y_MEAS_2_REF = np.float64(0.009738)
_LAST_Y_MEAS_3_REF = np.float64(0.009810)
_LAST_DE_REF = np.float64(0.122620)


@pytest.mark.slow
def test_cylinder_regression(tmp_path_factory):
    """10-step closed-loop run + JSON-based restart must reproduce reference values."""
    path_out = tmp_path_factory.mktemp("cylinder_regression")

    # ── First run (10 steps, saves at step 5 and 10) ─────────────────────────
    fs = CylinderFlowSolver.make_default(
        Re=100, path_out=path_out, num_steps=10, save_every=5
    )
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
    fs_restart = CylinderFlowSolver.make_default(
        Re=100, path_out=path_out, num_steps=10, save_every=5, Tstart=0.05
    )
    fs_restart.load_steady_state()
    fs_restart.initialize_time_stepping(Tstart=fs_restart.params_time.Tstart)

    for _ in range(fs_restart.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs_restart.params_time.dt)
        fs_restart.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    fs_restart.write_timeseries()

    u_max = flu.apply_fun(fs_restart.fields.Usave, np.max)
    u_mean = flu.apply_fun(fs_restart.fields.Usave, np.mean)
    last = fs_restart.timeseries.iloc[-1]

    assert np.isclose(u_max, _U_MAX_REF, rtol=1e-6), f"u_max: {u_max} != {_U_MAX_REF}"
    assert np.isclose(u_mean, _U_MEAN_REF, rtol=1e-6), (
        f"u_mean: {u_mean} != {_U_MEAN_REF}"
    )
    assert np.isclose(last["time"], _LAST_TIME_REF, rtol=1e-6), f"time: {last['time']}"
    assert np.isclose(last["y_meas_1"], _LAST_Y_MEAS_1_REF, rtol=1e-4), (
        f"y_meas_1: {last['y_meas_1']}"
    )
    assert np.isclose(last["y_meas_2"], _LAST_Y_MEAS_2_REF, rtol=1e-4), (
        f"y_meas_2: {last['y_meas_2']}"
    )
    assert np.isclose(last["y_meas_3"], _LAST_Y_MEAS_3_REF, rtol=1e-4), (
        f"y_meas_3: {last['y_meas_3']}"
    )
    assert np.isclose(last["dE"], _LAST_DE_REF, rtol=1e-4), f"dE: {last['dE']}"
