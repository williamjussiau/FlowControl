"""Flow past a cylinder, using flowsolver2.FlowSolver."""

import logging

import dolfin
import pandas

import flowcontrol.flowsolver2 as flowsolver2
import utils.utils_extract as flu2
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

logger = logging.getLogger(__name__)


class CylinderFlowSolver2(flowsolver2.FlowSolver):
    """Flow past a cylinder. Proposed Re=100."""

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
        self.cl0, self.cd0 = self.compute_force_coefficients(self.fields.U0, self.fields.P0)

    def compute_force_coefficients(self, u, p):
        """Compute lift and drag coefficients on the cylinder."""
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re
        sigma = flu2.stress_tensor(nu, u, p)
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
