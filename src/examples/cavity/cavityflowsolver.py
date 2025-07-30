"""
----------------------------------------------------------------------
Flow over an open cavity
Nondimensional incompressible Navier-Stokes equations
Suggested Re=7500
----------------------------------------------------------------------
"""

import logging

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CavityFlowSolver(flowsolver.FlowSolver):
    """Flow over an open cavity. Proposed Re=7500."""

    # @abstractmethod
    # def _make_boundaries(self) -> pd.DataFrame:
    def _make_boundaries(self):
        #                    sf
        #   ------------------------------------------
        #   |                                        |
        # in|                                        |out
        #   |                                        |
        #   -----x0nsl---      -----x0nsr-------------
        #      sf     ns|      | ns           sf
        #               |      |
        #               |      |
        #               --------
        #                  ns
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS
        L = self.params_flow.user_data["L"]
        D = self.params_flow.user_data["D"]
        xinfa = self.params_mesh.user_data["xinfa"]
        xinf = self.params_mesh.user_data["xinf"]
        yinf = self.params_mesh.user_data["yinf"]
        x0ns_left = self.params_mesh.user_data["x0ns_left"]
        x0ns_right = self.params_mesh.user_data["x0ns_right"]

        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinfa, MESH_TOL)",
            xinfa=xinfa,
            MESH_TOL=MESH_TOL,
        )

        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinf, MESH_TOL)",
            xinf=xinf,
            MESH_TOL=MESH_TOL,
        )

        ## Upper wall
        upper_wall = dolfin.CompiledSubDomain(
            "on_boundary && \
                     near(x[1], yinf, MESH_TOL)",
            yinf=yinf,
            MESH_TOL=MESH_TOL,
        )

        # Open cavity
        if 0:  # not-compiled syntax (kept for information)
            # cavity left
            class bnd_cavity_left(dolfin.SubDomain):
                """Left wall of cavity"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[1], (-D, 0))
                        and dolfin.near(x[0], 0)
                    )

            cavity_left = bnd_cavity_left()

            # cavity bottom
            class bnd_cavity_botm(dolfin.SubDomain):
                """Bottom wall of cavity"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[0], (0, L))
                        and dolfin.near(x[1], -D)
                    )

            cavity_botm = bnd_cavity_botm()

            # cavity right
            class bnd_cavity_right(dolfin.SubDomain):
                """Right wall of cavity"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[1], (-D, 0))
                        and dolfin.near(x[0], L)
                    )

            cavity_right = bnd_cavity_right()

            # Lower wall
            # left
            # stress free
            class bnd_lower_wall_left_sf(dolfin.SubDomain):
                """Lower wall left, stress free"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and x[0] >= xinfa
                        and x[0] <= x0ns_left + 10 * MESH_TOL
                        and dolfin.near(x[1], 0)
                    )
                    # add MESH_TOL to force all cells to belong to a dolfin.SubDomain

            lower_wall_left_sf = bnd_lower_wall_left_sf()

            # no slip
            class bnd_lower_wall_left_ns(dolfin.SubDomain):
                """Lower wall left, no stress"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and x[0] >= x0ns_left - 10 * MESH_TOL
                        and x[0] <= 0
                        and dolfin.near(x[1], 0)
                    )
                    # add MESH_TOL to force all cells to belong to a dolfin.SubDomain

            lower_wall_left_ns = bnd_lower_wall_left_ns()

            # right
            # no slip
            class bnd_lower_wall_right_ns(dolfin.SubDomain):
                """Lower wall right, no slip"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[0], (L, x0ns_right))
                        and dolfin.near(x[1], 0)
                    )

            lower_wall_right_ns = bnd_lower_wall_right_ns()

            # # stress free
            class bnd_lower_wall_right_sf(dolfin.SubDomain):
                """Lower wall right, stress free"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[0], (x0ns_right, xinf))
                        and dolfin.near(x[1], 0)
                    )

            lower_wall_right_sf = bnd_lower_wall_right_sf()

        else:  # compiled
            cavity_left = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[0]", 0, "MESH_TOL")
                + and_cpp
                + between_cpp("x[1]", "-D", "0"),
                MESH_TOL=MESH_TOL,
                D=D,
            )

            cavity_botm = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", "-D", "MESH_TOL")
                + and_cpp
                + between_cpp("x[0]", "0", "L"),
                L=L,
                D=D,
                MESH_TOL=MESH_TOL,
            )

            cavity_right = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "near(x[0], L, MESH_TOL)"
                + and_cpp
                + between_cpp("x[1]", -D, "0"),
                L=L,
                D=D,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_left_sf = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "x[0] >= xinfa"
                + and_cpp
                + "x[0] <= x0ns_left + 10*MESH_TOL"
                + and_cpp
                + near_cpp("x[1]", 0),
                xinfa=xinfa,
                x0ns_left=x0ns_left,
                MESH_TOL=MESH_TOL,
            )

            actuator_start_x = -7.0 / 20

            lower_wall_left_ns = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "x[0] >= x0ns_left - 10*MESH_TOL"
                + and_cpp
                + "x[0] <= actuator_start_x"
                + and_cpp
                + near_cpp("x[1]", 0),
                x0ns_left=x0ns_left,
                actuator_start_x=actuator_start_x,
                MESH_TOL=MESH_TOL,
            )

            actuated_bc = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "x[0] >= actuator_start_x"
                + and_cpp
                + "x[0] <= 0"
                + and_cpp
                + near_cpp("x[1]", 0),
                actuator_start_x=actuator_start_x,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_right_ns = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", 0)
                + and_cpp
                + between_cpp("x[0]", "L", "x0ns_right"),
                x0ns_right=x0ns_right,
                L=L,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_right_sf = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", 0)
                + and_cpp
                + between_cpp("x[0]", "x0ns_right", "xinf"),
                x0ns_right=x0ns_right,
                xinf=xinf,
                MESH_TOL=MESH_TOL,
            )

        # Concatenate all boundaries
        subdomains_list = [
            inlet,
            outlet,
            upper_wall,
            cavity_left,
            cavity_botm,
            cavity_right,
            lower_wall_left_sf,
            lower_wall_left_ns,
            actuated_bc,
            lower_wall_right_ns,
            lower_wall_right_sf,
        ]

        boundaries_names = [
            "inlet",
            "outlet",
            "upper_wall",
            "cavity_left",
            "cavity_botm",
            "cavity_right",
            "lower_wall_left_sf",
            "lower_wall_left_ns",
            "actuated_bc",
            "lower_wall_right_ns",
            "lower_wall_right_sf",
        ]

        boundaries_df = pandas.DataFrame(
            index=boundaries_names, data={"subdomain": subdomains_list}
        )

        return boundaries_df

    # @abstractmethod
    def _make_bcs(self):
        # inlet : u=uinf, v=0
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("inlet"),
        )
        # upper wall : dy(u)=0
        bcu_upper_wall = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("upper_wall"),
        )
        # lower wall left sf : v=0 + dy(u)=0
        bcu_lower_wall_left_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_left_sf"),
        )
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_left_ns"),
        )
        # actuated bc
        bcu_actuated = dolfin.DirichletBC(
        self.W.sub(0),
        self.params_control.actuator_list[0].expression,
        self.get_subdomain("actuated_bc"),
        )
        # additional line required for actuated boundary
        self.params_control.actuator_list[0].boundary = self.get_subdomain("actuated_bc")

        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_right_ns"),
        )
        # lower wall right sf : v=0 + dy(u)=0
        bcu_lower_wall_right_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_right_sf"),
        )
        # cavity : no slip, u=0; v=0
        bcu_cavity_left = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_left"),
        )
        bcu_cavity_botm = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_botm"),
        )
        bcu_cavity_right = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_right"),
        )

        bcu = [
            bcu_inlet,
            bcu_upper_wall,
            bcu_lower_wall_left_sf,
            bcu_lower_wall_left_ns,
            bcu_actuated,
            bcu_lower_wall_right_ns,
            bcu_lower_wall_right_sf,
            bcu_cavity_left,
            bcu_cavity_botm,
            bcu_cavity_right,
        ]

        # pressure on outlet -> p=0 or p free (standard outflow)
        # bcp_outlet = dolfin.DirichletBC(
        #     self.W.sub(1),
        #     dolfin.Constant(0),
        #     self.get_subdomain("outlet"),
        # )
        # bcp = [bcp_outlet]
        bcp = []

        return BoundaryConditions(bcu=bcu, bcp=bcp)

    def _steady_state_default_initial_guess(self) -> dolfin.UserExpression:
        class default_initial_guess(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0
                if x[1] <= 0:  # inside cavity
                    value[0] = 0.0

            def value_shape(self):
                return (3,)

        return default_initial_guess()


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
    from examples.cavity import run_cavity_example

    run_cavity_example.main()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
