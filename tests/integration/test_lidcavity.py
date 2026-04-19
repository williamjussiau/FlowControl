"""Integration smoke test for the lid-driven cavity using flowsolver."""

from pathlib import Path

import numpy as np
import pytest

import flowcontrol.flowsolverparameters as flowsolverparameters
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCUniformU
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "src" / "examples" / "lidcavity"
MESH_PATH = EXAMPLE_DIR / "data_input" / "mesh64.xdmf"


def _make_base_params(path_out, num_steps=3, save_every=0):
    params_flow = flowsolverparameters.ParamFlow(Re=1000, uinf=1.0)
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
    params_mesh.user_data["yup"] = 1.0
    params_mesh.user_data["ylo"] = 0.0
    params_mesh.user_data["xri"] = 1.0
    params_mesh.user_data["xle"] = 0.0

    # boundary_name="lid" so ActuatorBC.load_expression auto-wires the boundary
    actuator = ActuatorBCUniformU(boundary_name="lid")
    sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5]))
    sensor_2 = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.95]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_1, sensor_2],
        actuator_list=[actuator],
    )
    params_ic = flowsolverparameters.ParamIC(
        xloc=0.1, yloc=0.1, radius=0.1, amplitude=0.1
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


@pytest.mark.slow
def test_lidcavity_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("lidcavity_smoke")

    kw = _make_base_params(path_out=path_out, num_steps=3, save_every=0)
    fs = LidCavityFlowSolver(**kw, verbose=0)

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────

# TODO: run once and fill these reference values, then remove the pytest.skip call
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

    kw = _make_base_params(path_out=path_out, num_steps=10, save_every=5)
    fs = LidCavityFlowSolver(**kw, verbose=0)

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
