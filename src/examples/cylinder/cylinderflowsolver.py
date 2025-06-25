"""
----------------------------------------------------------------------
Flow past a cylinder
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation at Re_c=46
Suggested Re=100
----------------------------------------------------------------------
"""

import logging

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import utils.utils_extract as flu2
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class CylinderFlowSolver(flowsolver.FlowSolver):
    """Flow past a cylinder. Proposed Re=100."""

    def _make_boundaries(self):
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        or_cpp = flu.or_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS

        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinfa", "MESH_TOL"),
            xinfa=self.params_mesh.user_data["xinfa"],
            MESH_TOL=MESH_TOL,
        )
        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinf", "MESH_TOL"),
            xinf=self.params_mesh.user_data["xinf"],
            MESH_TOL=MESH_TOL,
        )
        ## Walls
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

        ## Cylinder
        radius = self.params_flow.user_data["D"] / 2
        ldelta = self.params_control.actuator_list[0].width

        # close_to_cylinder_cpp = between_cpp("x[0]*x[0] + x[1]*x[1]", "0", "2*radius*radius")
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
            cylinder_boundary_cpp
            + and_cpp
            + "("
            + cone_le_cpp
            + or_cpp
            + cone_ri_cpp
            + ")",
            radius=radius,
            ldelta=ldelta,
        )
        actuator_up = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_up_cpp, radius=radius, ldelta=ldelta
        )
        actuator_lo = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_lo_cpp, radius=radius, ldelta=ldelta
        )

        # assign boundaries as pd.DataFrame
        boundaries_names = [
            "inlet",
            "outlet",
            "walls",
            "cylinder",
            "actuator_up",
            "actuator_lo",
        ]
        subdomains_list = [inlet, outlet, walls, cylinder, actuator_up, actuator_lo]

        boundaries_df = pandas.DataFrame(
            index=boundaries_names, data={"subdomain": subdomains_list}
        )

        return boundaries_df

    def _make_bcs(self):
        # Free boundaries
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("inlet"),
        )
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("walls"),
        )
        bcu_cylinder = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cylinder"),
        )

        # Actuated boundaries
        bcu_actuation_up = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[0].expression,
            self.get_subdomain("actuator_up"),
        )
        self.params_control.actuator_list[0].boundary = self.get_subdomain(
            "actuator_up"
        )

        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[1].expression,
            self.get_subdomain("actuator_lo"),
        )
        self.params_control.actuator_list[1].boundary = self.get_subdomain(
            "actuator_lo"
        )

        bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]

        return BoundaryConditions(bcu=bcu, bcp=[])

    # Steady state
    def compute_steady_state(self, u_ctrl, method="newton", **kwargs):
        super().compute_steady_state(method=method, u_ctrl=u_ctrl, **kwargs)
        # assign steady cl, cd
        cl, cd = self.compute_force_coefficients(self.fields.U0, self.fields.P0)

        self.cl0 = cl
        self.cd0 = cd
        if self.verbose:
            logger.info(f"Lift coefficient is: cl = {cl}")
            logger.info(f"Drag coefficient is: cd = {cd}")

    # Additional, case-specific func
    def compute_force_coefficients(
        self, u: dolfin.Function, p: dolfin.Function
    ) -> tuple[float, float]:  # keep this one in here
        """Compute lift & drag coefficients acting on the cylinder."""
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re

        sigma = flu2.stress_tensor(nu, u, p)
        facet_normals = dolfin.FacetNormal(self.mesh)
        Fo = -dolfin.dot(sigma, facet_normals)

        # integration surfaces names
        surfaces_names = ["cylinder", "actuator_up", "actuator_lo"]
        # integration surfaces indices
        surfaces_idx = [self.boundaries.loc[nm].idx for nm in surfaces_names]

        # define drag & lift expressions
        # sum symbolic forces
        drag_sym = sum(
            [Fo[0] * self.ds(int(sfi)) for sfi in surfaces_idx]
        )  # (forced int)
        lift_sym = sum(
            [Fo[1] * self.ds(int(sfi)) for sfi in surfaces_idx]
        )  # (forced int)
        # integrate sum of symbolic forces
        lift = dolfin.assemble(lift_sym)
        drag = dolfin.assemble(drag_sym)

        # define force coefficients by normalizing
        cd = drag / (1 / 2 * self.params_flow.uinf**2 * D)
        cl = lift / (1 / 2 * self.params_flow.uinf**2 * D)
        return cl, cd


###############################################################################
###############################################################################
############################ END CLASS DEFINITION #############################
###############################################################################
###############################################################################


###############################################################################
###############################################################################
############################     RUN EXAMPLE      #############################
###############################################################################
###############################################################################
if __name__ == "__main__":
    from examples.cylinder import run_cylinder_example

    run_cylinder_example.main()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
