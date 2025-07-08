"""
----------------------------------------------------------------------
Fluidic pinball (3 cylinders)
Nondimensional incompressible Navier-Stokes equations
Several supercritical Hopf bifurcations
Recommended Re<100
----------------------------------------------------------------------
"""

import logging

import dolfin
import pandas as pd

import flowcontrol.flowsolver as flowsolver
import utils.utils_extract as flu2
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class PinballFlowSolver(flowsolver.FlowSolver):
    """Flow past 3 cylinders"""

    def _make_boundaries(self):
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        or_cpp = flu.or_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS
        mode_actuation = self.params_flow.user_data["mode_actuation"]

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

        ## Pinball
        
        radius = self.params_flow.user_data["D"] / 2
        close_to_cylinder_top_cpp = (
               between_cpp("x[0]", "-radius", "radius")
               + and_cpp
               + between_cpp("x[1]", "radius/2", "5*radius/2")
               )
        close_to_cylinder_bot_cpp = (
               between_cpp("x[0]", "-radius", "radius")
               + and_cpp
               + between_cpp("x[1]", "-5*radius/2", "-radius/2")
               )
        close_to_cylinder_charm_cpp = (
               between_cpp("x[0]", "-radius-1.5*cos(pi/6)", "+radius-1.5*cos(pi/6)")
               + and_cpp
               + between_cpp("x[1]", "-radius", "radius")
               )
        
        cylinder_boundary_top_cpp = (
               on_boundary_cpp + and_cpp + close_to_cylinder_top_cpp
        )
        cylinder_boundary_bot_cpp = (
               on_boundary_cpp + and_cpp + close_to_cylinder_bot_cpp
        )
        cylinder_boundary_charm_cpp = (
               on_boundary_cpp + and_cpp + close_to_cylinder_charm_cpp
        )
        
        boundaries_names = ["inlet", "outlet", "walls"]
        subdomains_list = [inlet, outlet, walls]
        
        if mode_actuation == 'suc':
           
           ldelta = self.params_control.actuator_list[0].width
              
           cone_charm_act_cpp = between_cpp(
               "x[0]", "-ldelta-1.5*cos(pi/6)", "-1.5*cos(pi/6)+ldelta"
           )
           cone_top_act_cpp = between_cpp("x[0]", "-ldelta", "+ldelta")
           cone_bot_act_cpp = between_cpp("x[0]", "-ldelta", "+ldelta")

           cylinder_top = dolfin.CompiledSubDomain(
               cylinder_boundary_top_cpp, radius=radius, ldelta=ldelta
           )
           cylinder_bot = dolfin.CompiledSubDomain(
               cylinder_boundary_bot_cpp, radius=radius, ldelta=ldelta
           )
           cylinder_charm = dolfin.CompiledSubDomain(
               cylinder_boundary_charm_cpp, radius=radius, ldelta=ldelta
           )

           actuator_top = dolfin.CompiledSubDomain(
               cylinder_boundary_top_cpp + and_cpp + cone_top_act_cpp,
               radius=radius,
               ldelta=ldelta,
           )
           actuator_bot = dolfin.CompiledSubDomain(
               cylinder_boundary_bot_cpp + and_cpp + cone_bot_act_cpp,
               radius=radius,
               ldelta=ldelta,
           )
           actuator_charm = dolfin.CompiledSubDomain(
               cylinder_boundary_charm_cpp + and_cpp + cone_charm_act_cpp,
               radius=radius,
               ldelta=ldelta,
           )

        # assign boundaries as pd.DataFrame
           boundaries_names += [
            "cylinder_top", "cylinder_bot", "cylinder_charm",
            "actuator_charm", "actuator_top", "actuator_bot"
           ]
           subdomains_list += [
            cylinder_top, cylinder_bot, cylinder_charm,
            actuator_charm, actuator_top, actuator_bot
           ]
           
           
        else:
            
            
            #actuator_top = dolfin.CompiledSubDomain(
            #   cylinder_boundary_top_cpp + and_cpp + '(x[0]*x[0] + (x[1] - 3*radius/2)*(x[1] - 3*radius/2) <= radius*radius + 1e-9)',
            #   radius=radius,
            #)
            actuator_top = dolfin.CompiledSubDomain(
               cylinder_boundary_top_cpp,
               radius=radius,
            )
            #actuator_bot = dolfin.CompiledSubDomain(
            #   cylinder_boundary_bot_cpp + and_cpp + '(x[0]*x[0] + (x[1] + 3*radius/2)*(x[1] + 3*radius/2) <= radius*radius + 1e-9)',
            #   radius=radius,
            #)
            actuator_bot = dolfin.CompiledSubDomain(
               cylinder_boundary_bot_cpp,
               radius=radius,
            )
            #actuator_charm = dolfin.CompiledSubDomain(
            #    cylinder_boundary_charm_cpp + and_cpp + '((x[0]+1.5*cos(pi/6))*(x[0]+1.5*cos(pi/6)) + x[1]*x[1] <= radius*radius + 1e-9)',
            #    radius=radius,
            #)
            actuator_charm = dolfin.CompiledSubDomain(
                cylinder_boundary_charm_cpp,
                radius=radius,
            )
             # assign boundaries as pd.DataFrame
            boundaries_names += [
            "actuator_charm", "actuator_top", "actuator_bot"
            ]
            subdomains_list += [
            actuator_charm, actuator_top, actuator_bot
            ]
        
                 
        
        boundaries_df = pd.DataFrame(
            index=boundaries_names, data={"subdomain": subdomains_list}
        )




        return boundaries_df

    def _make_bcs(self):
        # Free boundaries
        mode_actuation = self.params_flow.user_data["mode_actuation"]
        
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
        bcu = [
            bcu_inlet,
            bcu_walls,]
        
        if mode_actuation == 'suc':
            
            bcu_cylinder_top = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.get_subdomain("cylinder_top"),
                )
            bcu_cylinder_bot = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.get_subdomain("cylinder_bot"),
            )
            bcu_cylinder_charm = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.get_subdomain("cylinder_charm"),
            )   
            bcu += [
                bcu_cylinder_top,
                bcu_cylinder_bot,
                bcu_cylinder_charm,
            ]

        # Actuated boundaries
        bcu_actuation_charm = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[0].expression,
            self.get_subdomain("actuator_charm"),
        )
        self.params_control.actuator_list[0].boundary = self.get_subdomain(
            "actuator_charm"
        )
        bcu_actuation_top = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[1].expression,
            self.get_subdomain("actuator_top"),
        )
        self.params_control.actuator_list[1].boundary = self.get_subdomain(
            "actuator_top"
        )
        bcu_actuation_bot = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[2].expression,
            self.get_subdomain("actuator_bot"),
        )
        self.params_control.actuator_list[2].boundary = self.get_subdomain(
            "actuator_bot"
        )

        bcu += [
            bcu_actuation_charm,
            bcu_actuation_top,
            bcu_actuation_bot,
        ]

        return BoundaryConditions(bcu=bcu, bcp=[])

    def _make_BCs(self):
        # inlet : u = uinf, v = 0
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        # walls :  u = uinf, v = 0
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["walls"].subdomain,
        )
        bcs = self._make_bcs()
        BC = BoundaryConditions(bcu=[bcu_inlet, bcu_walls] + bcs.bcu[2:], bcp=[])

        return BC

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
        mode_actuation = self.params_flow.user_data["mode_actuation"]

        sigma = flu2.stress_tensor(nu, u, p)
        facet_normals = dolfin.FacetNormal(self.mesh)
        Fo = -dolfin.dot(sigma, facet_normals)

        # integration surfaces names
        if mode_actuation == 'suc':
            surfaces_names = ["cylinder_charm", "actuator_charm", "cylinder_top", "actuator_top", "cylinder_bot", "actuator_bot"]
        else:
            surfaces_names = ["actuator_charm", "actuator_top", "actuator_bot"]
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
    from examples.pinball import run_pinball_example

    run_pinball_example.main()


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
