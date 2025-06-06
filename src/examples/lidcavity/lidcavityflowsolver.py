"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

import logging
import time
from pathlib import Path

import dolfin
import numpy as np
import pandas

import flowcontrol.flowsolver as flowsolver
import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_extract as flu2
import utils.utils_flowsolver as flu
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.flowfield import BoundaryConditions
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class LidCavityFlowSolver(flowsolver.FlowSolver):
    """Lid-driven cavity flow. Proposed Re=8000."""

    def _make_boundaries(self):
        near_cpp = flu.near_cpp
        and_cpp = flu.and_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS

        ## Lid (top), actuated
        lid = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "yup", "MESH_TOL"),
            yup=self.params_mesh.user_data["yup"],
            MESH_TOL=MESH_TOL,
        )

        ## Left wall
        leftwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xle", "MESH_TOL"),
            xle=self.params_mesh.user_data["xle"],
            MESH_TOL=MESH_TOL,
        )

        ## Right wall
        rightwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xri", "MESH_TOL"),
            xri=self.params_mesh.user_data["xri"],
            MESH_TOL=MESH_TOL,
        )

        ## Bottom wall
        bottomwall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "ylo", "MESH_TOL"),
            ylo=self.params_mesh.user_data["ylo"],
            MESH_TOL=MESH_TOL,
        )

        # assign boundaries as pd.DataFrame
        boundaries_names = [
            "lid",
            "leftwall",
            "rightwall",
            "bottomwall",
        ]
        subdomains_list = [lid, leftwall, rightwall, bottomwall]

        boundaries_df = pandas.DataFrame(
            index=boundaries_names, data={"subdomain": subdomains_list}
        )

        return boundaries_df

    def _make_bcs(self):
        # Actuated lid, also inlet-like (see make_BCs)
        bcu_lid = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[0].expression,
            self.get_subdomain("lid"),
        )
        # additional line required for actuated boundary
        self.params_control.actuator_list[0].boundary = self.get_subdomain("lid")

        # No-slip on walls
        bcu_leftwall = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("leftwall"),
        )
        bcu_rightwall = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("rightwall"),
        )
        bcu_bottomwall = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("bottomwall"),
        )

        bcu = [bcu_lid, bcu_leftwall, bcu_rightwall, bcu_bottomwall]

        return BoundaryConditions(bcu=bcu, bcp=[])

    # override
    def _make_BCs(self):
        bcu_lid = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["lid"].subdomain,
        )
        bcs = self._make_bcs()
        BC = BoundaryConditions(bcu=[bcu_lid] + bcs.bcu[1:], bcp=[])

        return BC


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
    from examples.lidcavity import run_lidcavity_example

    run_lidcavity_example.main()

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
