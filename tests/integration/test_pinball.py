"""Integration smoke test for the fluidic pinball using flowsolver."""

import numpy as np
import pytest

from examples.pinball.pinballflowsolver import PinballFlowSolver
from flowcontrol.actuator import CYLINDER_ACTUATION_MODE

# ── Fast CI test with coarse generated mesh ───────────────────────────────────


def test_pinball_fast(coarse_pinball_mesh, tmp_path_factory):
    """Fast smoke test with coarse generated mesh - runs in CI on every push."""
    path_out = tmp_path_factory.mktemp("pinball_fast")

    fs = PinballFlowSolver.make_default(
        Re=30,
        mode_actuation=CYLINDER_ACTUATION_MODE.SUCTION,
        path_out=path_out,
        num_steps=3,
        meshpath=coarse_pinball_mesh,
    )

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0, 0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Slow tests with pre-generated meshes ─────────────────────────────────────


@pytest.mark.slow
def test_pinball_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("pinball_smoke")

    fs = PinballFlowSolver.make_default(
        Re=30,
        mode_actuation=CYLINDER_ACTUATION_MODE.SUCTION,
        path_out=path_out,
        num_steps=3,
    )
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0, 0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────
_U_MAX_REF = np.float64(1.5168848768060617)
_U_MEAN_REF = np.float64(0.14938830691642546)
_LAST_TIME_REF = np.float64(0.05)
_LAST_Y_MEAS_1_REF = np.float64(-0.0007170095882221647)
_LAST_DE_REF = np.float64(0.05725656971818885)


@pytest.mark.slow
def test_pinball_regression(tmp_path_factory):
    """10-step unactuated run must reproduce reference values."""
    import utils.utils_flowsolver as flu

    path_out = tmp_path_factory.mktemp("pinball_regression")

    fs = PinballFlowSolver.make_default(
        Re=30,
        mode_actuation=CYLINDER_ACTUATION_MODE.SUCTION,
        path_out=path_out,
        num_steps=10,
        save_every=5,
    )
    fs.compute_steady_state(method="picard", max_iter=15, tol=1e-7, u_ctrl=[0.0, 0.0, 0.0])
    fs.compute_steady_state(
        method="newton",
        max_iter=10,
        u_ctrl=[0.0, 0.0, 0.0],
        initial_guess=fs.fields.UP0,
    )
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0, 0.0])

    fs.write_timeseries()

    u_max = flu.apply_fun(fs.fields.Usave, np.max)
    u_mean = flu.apply_fun(fs.fields.Usave, np.mean)
    last = fs.timeseries.iloc[-1]

    assert np.isclose(u_max, _U_MAX_REF, rtol=1e-4), f"u_max: {u_max} != {_U_MAX_REF}"
    assert np.isclose(u_mean, _U_MEAN_REF, rtol=1e-6), f"u_mean: {u_mean} != {_U_MEAN_REF}"
    assert np.isclose(last["time"], _LAST_TIME_REF, rtol=1e-6), f"time: {last['time']}"
    assert np.isclose(last["y_meas_1"], _LAST_Y_MEAS_1_REF, rtol=1e-4), f"y_meas_1: {last['y_meas_1']}"
    assert np.isclose(last["dE"], _LAST_DE_REF, rtol=1e-4), f"dE: {last['dE']}"
