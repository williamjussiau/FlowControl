"""Flow past a cylinder, using flowsolver.FlowSolver."""

import logging

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions
from utils.physics import stress_tensor

logger = logging.getLogger(__name__)


class CylinderFlowSolver(flowsolver.FlowSolver):
    """Flow past a cylinder. Proposed Re=100."""

    def _make_boundaries(self):
        """Return subdomains for the cylinder geometry.

        Boundaries: inlet, outlet, far-field walls (top and bottom), cylinder
        body (no-slip, excluding slots), and two actuator slots (actuator_up,
        actuator_lo) at the top and bottom poles of the cylinder.
        """
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
        """Return perturbation-field BCs: zero on inlet/walls/cylinder; actuator expressions on the two slots."""
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
        """Compute steady state then cache lift/drag coefficients as self.cl0, self.cd0."""
        super().compute_steady_state(method=method, u_ctrl=u_ctrl, **kwargs)
        self.cl0, self.cd0 = self.compute_force_coefficients(self.fields.U0, self.fields.P0)

    def compute_force_coefficients(self, u, p):
        """Compute lift and drag coefficients on the cylinder."""
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re
        sigma = stress_tensor(nu, u, p)
        Fo = -dolfin.dot(sigma, dolfin.FacetNormal(self.mesh))
        surfaces_idx = [
            self.boundaries.loc[nm].idx
            for nm in ["cylinder", "actuator_up", "actuator_lo"]
        ]
        drag = dolfin.assemble(sum(Fo[0] * self.ds(int(i)) for i in surfaces_idx))
        lift = dolfin.assemble(sum(Fo[1] * self.ds(int(i)) for i in surfaces_idx))
        cd = drag / (0.5 * self.params_flow.uinf**2 * D)
        cl = lift / (0.5 * self.params_flow.uinf**2 * D)
        return cl, cd

    @classmethod
    def make_default(
        cls,
        Re: float = 100,
        path_out=None,
        num_steps: int = 10,
        save_every: int = 0,
        Tstart: float = 0.0,
        verbose: int = 0,
    ) -> "CylinderFlowSolver":
        """Return a CylinderFlowSolver with standard parameters (Re=100, 2 BC actuators, 3 sensors)."""
        from pathlib import Path

        import numpy as np

        import flowcontrol.flowsolverparameters as fsp
        from flowcontrol.actuator import ActuatorBCParabolicV
        from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

        if path_out is None:
            path_out = Path(__file__).parent / "data_output"

        params_flow = fsp.ParamFlow(Re=Re, uinf=1.0)
        params_flow.user_data["D"] = 1.0

        params_time = fsp.ParamTime(num_steps=num_steps, dt=0.005, Tstart=Tstart)
        params_save = fsp.ParamSave(save_every=save_every, path_out=path_out)
        params_solver = fsp.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)
        params_mesh = fsp.ParamMesh(
            meshpath=Path(__file__).parent / "data_input" / "O1.xdmf"
        )
        params_mesh.user_data.update({"xinf": 20, "xinfa": -10, "yinf": 10})

        radius = params_flow.user_data["D"] / 2
        width = ActuatorBCParabolicV.angular_size_deg_to_width(10, radius)
        params_control = fsp.ParamControl(
            sensor_list=[
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.0, 0.0])),
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1.0])),
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1.0])),
            ],
            actuator_list=[
                ActuatorBCParabolicV(width=width, position_x=0.0, boundary_name="actuator_up"),
                ActuatorBCParabolicV(width=width, position_x=0.0, boundary_name="actuator_lo"),
            ],
        )
        params_ic = fsp.ParamIC()

        return cls(
            params_flow=params_flow,
            params_time=params_time,
            params_save=params_save,
            params_solver=params_solver,
            params_mesh=params_mesh,
            params_control=params_control,
            params_ic=params_ic,
            verbose=verbose,
        )
