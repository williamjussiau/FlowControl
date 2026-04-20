"""Lid-driven cavity flow, using flowsolver.FlowSolver.

Supercritical Hopf bifurcation near Re_c=7700. Proposed Re=8000.
"""

import logging

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

logger = logging.getLogger(__name__)


class LidCavityFlowSolver(flowsolver.FlowSolver):
    """Lid-driven cavity flow. Proposed Re=8000."""

    def _make_boundaries(self):
        """Return subdomains for the unit-square lid-driven cavity.

        Boundaries: lid (top wall, actuated), leftwall, rightwall, bottomwall.
        Domain is the upper-right quadrant [xle,xri] × [ylo,yup] by default.
        """
        near_cpp = flu.near_cpp
        and_cpp = flu.and_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS

        lid = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "yup", "MESH_TOL"),
            yup=self.params_mesh.user_data["yup"],
            MESH_TOL=MESH_TOL,
        )
        leftwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xle", "MESH_TOL"),
            xle=self.params_mesh.user_data["xle"],
            MESH_TOL=MESH_TOL,
        )
        rightwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xri", "MESH_TOL"),
            xri=self.params_mesh.user_data["xri"],
            MESH_TOL=MESH_TOL,
        )
        bottomwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "ylo", "MESH_TOL"),
            ylo=self.params_mesh.user_data["ylo"],
            MESH_TOL=MESH_TOL,
        )

        return pandas.DataFrame(
            index=["lid", "leftwall", "rightwall", "bottomwall"],
            data={"subdomain": [lid, leftwall, rightwall, bottomwall]},
        )

    def _make_bcs(self):
        """Return perturbation-field BCs: actuator expression on lid; no-slip on the three other walls."""
        # Actuated lid (perturbation BC: zero at rest)
        bcu_lid = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[0].expression,
            self.get_subdomain("lid"),
        )
        bcu_leftwall = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("leftwall")
        )
        bcu_rightwall = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("rightwall")
        )
        bcu_bottomwall = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("bottomwall")
        )
        return BoundaryConditions(
            bcu=[bcu_lid, bcu_leftwall, bcu_rightwall, bcu_bottomwall], bcp=[]
        )

    def _make_BCs(self) -> BoundaryConditions:
        """Steady-state BCs: lid moves at uinf; walls no-slip."""
        bcu_lid_ss = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.get_subdomain("lid"),
        )
        bcs = self._make_bcs()
        return BoundaryConditions(bcu=[bcu_lid_ss] + bcs.bcu[1:], bcp=[])

    def _default_steady_state_initial_guess(self) -> dolfin.UserExpression:
        """Zero everywhere — cavity starts from rest."""
        class _ZeroFlow(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 0.0
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        return _ZeroFlow()
