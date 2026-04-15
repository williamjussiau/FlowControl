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
import flowcontrol.flowsolver2 as flowsolver2
import utils.utils_extract as flu2
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.flowfield import BoundaryConditions
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

import dolfin
import pandas

EXAMPLE_DIR = Path(__file__).parent.parent.parent / "src" / "examples" / "cylinder"
MESH_PATH = EXAMPLE_DIR / "data_input" / "O1.xdmf"
CONTROLLER_PATH = EXAMPLE_DIR / "data_input" / "Kopt_reduced13.mat"


# ── Cylinder solver using flowsolver2 ─────────────────────────────────────────


class CylinderFlowSolver2(flowsolver2.FlowSolver):
    """Flow past a cylinder, backed by flowsolver2.FlowSolver."""

    def _make_boundaries(self):
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        or_cpp = flu.or_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS

        inlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinfa", "MESH_TOL"),
            xinfa=self.params_mesh.user_data["xinfa"],
            MESH_TOL=MESH_TOL,
        )
        outlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinf", "MESH_TOL"),
            xinf=self.params_mesh.user_data["xinf"],
            MESH_TOL=MESH_TOL,
        )
        walls = dolfin.CompiledSubDomain(
            on_boundary_cpp
            + and_cpp
            + "("
            + near_cpp("x[1]", "-yinf", "MESH_TOL")
            + or_cpp
            + near_cpp("x[1]", "yinf", "MESH_TOL")
            + ")",
            yinf=self.params_mesh.user_data["yinf"],
            MESH_TOL=MESH_TOL,
        )

        radius = self.params_flow.user_data["D"] / 2
        ldelta = self.params_control.actuator_list[0].width

        close_to_cylinder_cpp = (
            between_cpp("x[0]", "-radius", "radius")
            + and_cpp
            + between_cpp("x[1]", "-radius", "radius")
        )
        cylinder_boundary_cpp = on_boundary_cpp + and_cpp + close_to_cylinder_cpp
        cone_up_cpp = (
            between_cpp("x[0]", "-ldelta", "ldelta", tol="0.01")
            + and_cpp
            + between_cpp("x[1]", "0", "radius")
        )
        cone_lo_cpp = (
            between_cpp("x[0]", "-ldelta", "ldelta", tol="0.01")
            + and_cpp
            + between_cpp("x[1]", "-radius", "0")
        )
        cone_le_cpp = between_cpp("x[0]", "-radius", "-ldelta")
        cone_ri_cpp = between_cpp("x[0]", "ldelta", "radius")

        cylinder = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + "(" + cone_le_cpp + or_cpp + cone_ri_cpp + ")",
            radius=radius,
            ldelta=ldelta,
        )
        actuator_up = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_up_cpp, radius=radius, ldelta=ldelta
        )
        actuator_lo = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_lo_cpp, radius=radius, ldelta=ldelta
        )

        return pandas.DataFrame(
            index=["inlet", "outlet", "walls", "cylinder", "actuator_up", "actuator_lo"],
            data={"subdomain": [inlet, outlet, walls, cylinder, actuator_up, actuator_lo]},
        )

    def _make_bcs(self):
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("inlet")
        )
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0).sub(1), dolfin.Constant(0), self.get_subdomain("walls")
        )
        bcu_cylinder = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("cylinder")
        )
        bcu_actuation_up = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[0].expression,
            self.get_subdomain("actuator_up"),
        )
        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[1].expression,
            self.get_subdomain("actuator_lo"),
        )

        return BoundaryConditions(
            bcu=[bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo],
            bcp=[],
        )

    def compute_steady_state(self, u_ctrl, method="newton", **kwargs):
        super().compute_steady_state(method=method, u_ctrl=u_ctrl, **kwargs)
        cl, cd = self.compute_force_coefficients(self.fields.U0, self.fields.P0)
        self.cl0 = cl
        self.cd0 = cd

    def compute_force_coefficients(self, u, p):
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re
        sigma = flu2.stress_tensor(nu, u, p)
        facet_normals = dolfin.FacetNormal(self.mesh)
        Fo = -dolfin.dot(sigma, facet_normals)
        surfaces_idx = [
            self.boundaries.loc[nm].idx
            for nm in ["cylinder", "actuator_up", "actuator_lo"]
        ]
        drag = dolfin.assemble(sum(Fo[0] * self.ds(int(i)) for i in surfaces_idx))
        lift = dolfin.assemble(sum(Fo[1] * self.ds(int(i)) for i in surfaces_idx))
        cd = drag / (0.5 * self.params_flow.uinf**2 * D)
        cl = lift / (0.5 * self.params_flow.uinf**2 * D)
        return cl, cd


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
    actuator_bc_1 = ActuatorBCParabolicV(width=width, position_x=0.0, boundary_name="actuator_up")
    actuator_bc_2 = ActuatorBCParabolicV(width=width, position_x=0.0, boundary_name="actuator_lo")

    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0.0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1.0]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1.0]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )
    params_ic = flowsolverparameters.ParamIC(xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0)

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

    assert np.isclose(u_max, u_max_ref, rtol=1e-6), f"u_max mismatch: {u_max} != {u_max_ref}"
    assert np.isclose(u_mean, u_mean_ref, rtol=1e-6), f"u_mean mismatch: {u_mean} != {u_mean_ref}"

    last = fs_restart.timeseries.iloc[-1]
    assert np.isclose(last["time"], 0.100, rtol=1e-6), f"time: {last['time']}"
    assert np.isclose(last["y_meas_1"], 0.131695, rtol=1e-4), f"y_meas_1: {last['y_meas_1']}"
    assert np.isclose(last["y_meas_2"], 0.009738, rtol=1e-4), f"y_meas_2: {last['y_meas_2']}"
    assert np.isclose(last["y_meas_3"], 0.009810, rtol=1e-4), f"y_meas_3: {last['y_meas_3']}"
    assert np.isclose(last["dE"], 0.122620, rtol=1e-4), f"dE: {last['dE']}"