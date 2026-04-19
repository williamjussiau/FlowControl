"""Integration smoke test for the lid-driven cavity using flowsolver2."""

from pathlib import Path

import numpy as np
import pytest

import flowcontrol.flowsolverparameters as flowsolverparameters
from examples.lidcavity.lidcavityflowsolver2 import LidCavityFlowSolver2
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
def test_lidcavity2_smoke(tmp_path_factory):
    """Pipeline runs without crashing; velocity values are finite after 3 steps."""
    path_out = tmp_path_factory.mktemp("lidcavity2_smoke")

    kw = _make_base_params(path_out=path_out, num_steps=3, save_every=0)
    fs = LidCavityFlowSolver2(**kw, verbose=0)

    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0])
    fs.initialize_time_stepping(ic=None)

    for _ in range(fs.params_time.num_steps):
        fs.step(u_ctrl=[0.0])

    u_vals = fs.fields.u_.vector().get_local()
    assert np.all(np.isfinite(u_vals)), "velocity field contains non-finite values"
