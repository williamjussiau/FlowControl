"""Integration smoke test for the fluidic pinball using flowsolver."""

from pathlib import Path

import numpy as np
import pytest

import flowcontrol.flowsolverparameters as flowsolverparameters
from examples.pinball.pinballflowsolver import PinballFlowSolver
from flowcontrol.actuator import CYLINDER_ACTUATION_MODE, ActuatorBCParabolicV
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "src" / "examples" / "pinball"
MESH_PATH = EXAMPLE_DIR / "data_input" / "mesh_middle_gmsh.xdmf"


def _make_base_params(path_out, num_steps=3, save_every=0):
    params_flow = flowsolverparameters.ParamFlow(Re=30, uinf=1.0)
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
    params_mesh.user_data["xinfa"] = -6
    params_mesh.user_data["yinf"] = 6

    mode_actuation = CYLINDER_ACTUATION_MODE.SUCTION
    cylinder_diameter = params_flow.user_data["D"]
    angular_size_deg = 10
    actuator_width = ActuatorBCParabolicV.angular_size_deg_to_width(
        angular_size_deg=angular_size_deg,
        cylinder_radius=cylinder_diameter / 2,
    )
    position_mid = [-1.5 * np.cos(np.pi / 6), 0.0]
    position_top = [0.0, +0.75]

    # boundary_name matches the keys returned by _make_boundaries (SUCTION mode)
    actuator_mid = ActuatorBCParabolicV(
        width=actuator_width, position_x=position_mid[0], boundary_name="actuator_mid"
    )
    actuator_top = ActuatorBCParabolicV(
        width=actuator_width, position_x=position_top[0], boundary_name="actuator_top"
    )
    actuator_bot = ActuatorBCParabolicV(
        width=actuator_width, position_x=position_top[0], boundary_name="actuator_bot"
    )

    sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([8.0, 0.0]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_1],
        actuator_list=[actuator_mid, actuator_top, actuator_bot],
        user_data={"mode_actuation": mode_actuation},
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


@pytest.mark.slow
def test_pinball_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("pinball_smoke")

    kw = _make_base_params(path_out=path_out, num_steps=3, save_every=0)
    fs = PinballFlowSolver(**kw, verbose=0)

    fs.compute_steady_state(
        method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0, 0.0]
    )
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0, 0.0, 0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"


# ── Regression test ───────────────────────────────────────────────────────────

# TODO: run once and fill these reference values, then remove the pytest.skip call
_U_MAX_REF = np.float64(1.4635364453393656)
_U_MEAN_REF = np.float64(0.14009906606265646)
_LAST_TIME_REF = np.float64(0.05)
_LAST_Y_MEAS_1_REF = np.float64(2.0334968617544303e-06)
_LAST_DE_REF = np.float64(0.0954510563847507)


@pytest.mark.slow
def test_pinball_regression(tmp_path_factory):
    """10-step unactuated run must reproduce reference values."""
    import utils.utils_flowsolver as flu

    path_out = tmp_path_factory.mktemp("pinball_regression")

    kw = _make_base_params(path_out=path_out, num_steps=10, save_every=5)
    fs = PinballFlowSolver(**kw, verbose=0)

    fs.compute_steady_state(
        method="picard", max_iter=15, tol=1e-7, u_ctrl=[0.0, 0.0, 0.0]
    )
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

    assert np.isclose(u_max, _U_MAX_REF, rtol=1e-6), f"u_max: {u_max} != {_U_MAX_REF}"
    assert np.isclose(u_mean, _U_MEAN_REF, rtol=1e-6), (
        f"u_mean: {u_mean} != {_U_MEAN_REF}"
    )
    assert np.isclose(last["time"], _LAST_TIME_REF, rtol=1e-6), f"time: {last['time']}"
    assert np.isclose(last["y_meas_1"], _LAST_Y_MEAS_1_REF, rtol=1e-4), (
        f"y_meas_1: {last['y_meas_1']}"
    )
    assert np.isclose(last["dE"], _LAST_DE_REF, rtol=1e-4), f"dE: {last['dE']}"
