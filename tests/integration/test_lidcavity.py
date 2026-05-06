"""Integration smoke test for the lid-driven cavity using flowsolver."""

import numpy as np
import pytest

from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver


# ── Fast CI test with coarse generated mesh ───────────────────────────────────


def test_lidcavity_fast(coarse_lidcavity_mesh, tmp_path_factory):
    """Fast smoke test with coarse generated mesh - runs in CI on every push."""
    path_out = tmp_path_factory.mktemp("lidcavity_fast")

    fs = LidCavityFlowSolver.make_default(
        Re=1000, path_out=path_out, num_steps=3, meshpath=coarse_lidcavity_mesh
    )

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Slow tests with pre-generated meshes ─────────────────────────────────────


@pytest.mark.slow
def test_lidcavity_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("lidcavity_smoke")

    fs = LidCavityFlowSolver.make_default(Re=1000, path_out=path_out, num_steps=3)
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────

_U_MAX_REF = np.float64(1.000000000000008)
_U_MEAN_REF = np.float64(0.0020232695187553038)
_LAST_TIME_REF = np.float64(0.05)
_LAST_Y_MEAS_1_REF = np.float64(0.010125139096606742)
_LAST_Y_MEAS_2_REF = np.float64(0.0014633392005555207)
_LAST_DE_REF = np.float64(0.0004309249670384312)


@pytest.mark.slow
def test_lidcavity_regression(tmp_path_factory):
    """10-step unactuated run must reproduce reference values."""
    import utils.utils_flowsolver as flu

    path_out = tmp_path_factory.mktemp("lidcavity_regression")

    fs = LidCavityFlowSolver.make_default(
        Re=1000, path_out=path_out, num_steps=10, save_every=5
    )
    fs.compute_steady_state(method="picard", max_iter=40, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    fs.write_timeseries()

    u_max = flu.apply_fun(fs.fields.Usave, np.max)
    u_mean = flu.apply_fun(fs.fields.Usave, np.mean)
    last = fs.timeseries.iloc[-1]

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
    assert np.isclose(last["dE"], _LAST_DE_REF, rtol=1e-4), f"dE: {last['dE']}"
