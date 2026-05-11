"""Flow over an open cavity, using flowsolver.FlowSolver.

Nondimensional incompressible Navier-Stokes. Suggested Re=7500.
"""

import logging
from pathlib import Path

import dolfin
import pandas

import flowcontrol.flowsolver as flowsolver
import utils.utils_flowsolver as flu
from flowcontrol.flowfield import BoundaryConditions

logger = logging.getLogger(__name__)


class CavityFlowSolver(flowsolver.FlowSolver):
    """Flow over an open cavity. Proposed Re=7500."""

    def _make_boundaries(self):
        """Return subdomains for the open-cavity geometry.

        Channel has inlet/outlet, an upper slip wall, a cavity cut into the
        lower wall (left/bottom/right no-slip walls), and lower walls split
        into slip (sf) and no-slip (ns) segments upstream/downstream of the cavity.
        """
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

        inlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinfa", "MESH_TOL"),
            xinfa=xinfa,
            MESH_TOL=MESH_TOL,
        )
        outlet = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", "xinf", "MESH_TOL"),
            xinf=xinf,
            MESH_TOL=MESH_TOL,
        )
        upper_wall = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "yinf", "MESH_TOL"),
            yinf=yinf,
            MESH_TOL=MESH_TOL,
        )
        cavity_left = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[0]", 0, "MESH_TOL") + and_cpp + between_cpp("x[1]", "-D", "0"),
            MESH_TOL=MESH_TOL,
            D=D,
        )
        cavity_botm = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", "-D", "MESH_TOL") + and_cpp + between_cpp("x[0]", "0", "L"),
            L=L,
            D=D,
            MESH_TOL=MESH_TOL,
        )
        cavity_right = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + "near(x[0], L, MESH_TOL)" + and_cpp + between_cpp("x[1]", -D, "0"),
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
        lower_wall_left_ns = dolfin.CompiledSubDomain(
            on_boundary_cpp
            + and_cpp
            + "x[0] >= x0ns_left - 10*MESH_TOL"
            + and_cpp
            + "x[0] <= 0"
            + and_cpp
            + near_cpp("x[1]", 0),
            x0ns_left=x0ns_left,
            MESH_TOL=MESH_TOL,
        )
        lower_wall_right_ns = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", 0) + and_cpp + between_cpp("x[0]", "L", "x0ns_right"),
            x0ns_right=x0ns_right,
            L=L,
            MESH_TOL=MESH_TOL,
        )
        lower_wall_right_sf = dolfin.CompiledSubDomain(
            on_boundary_cpp + and_cpp + near_cpp("x[1]", 0) + and_cpp + between_cpp("x[0]", "x0ns_right", "xinf"),
            x0ns_right=x0ns_right,
            xinf=xinf,
            MESH_TOL=MESH_TOL,
        )

        return pandas.DataFrame(
            index=[
                "inlet",
                "outlet",
                "upper_wall",
                "cavity_left",
                "cavity_botm",
                "cavity_right",
                "lower_wall_left_sf",
                "lower_wall_left_ns",
                "lower_wall_right_ns",
                "lower_wall_right_sf",
            ],
            data={
                "subdomain": [
                    inlet,
                    outlet,
                    upper_wall,
                    cavity_left,
                    cavity_botm,
                    cavity_right,
                    lower_wall_left_sf,
                    lower_wall_left_ns,
                    lower_wall_right_ns,
                    lower_wall_right_sf,
                ]
            },
        )

    def _make_bcs(self):
        """Return perturbation-field BCs: zero on inlet; slip (v=0) on sf walls; no-slip on ns walls and cavity faces."""
        bcu_inlet = dolfin.DirichletBC(self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("inlet"))
        bcu_upper_wall = dolfin.DirichletBC(self.W.sub(0).sub(1), dolfin.Constant(0), self.get_subdomain("upper_wall"))
        bcu_lower_wall_left_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_left_sf"),
        )
        bcu_lower_wall_left_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_left_ns"),
        )
        bcu_lower_wall_right_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_right_ns"),
        )
        bcu_lower_wall_right_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_right_sf"),
        )
        bcu_cavity_left = dolfin.DirichletBC(self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("cavity_left"))
        bcu_cavity_botm = dolfin.DirichletBC(self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("cavity_botm"))
        bcu_cavity_right = dolfin.DirichletBC(
            self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("cavity_right")
        )
        return BoundaryConditions(
            bcu=[
                bcu_inlet,
                bcu_upper_wall,
                bcu_lower_wall_left_sf,
                bcu_lower_wall_left_ns,
                bcu_lower_wall_right_ns,
                bcu_lower_wall_right_sf,
                bcu_cavity_left,
                bcu_cavity_botm,
                bcu_cavity_right,
            ],
            bcp=[],
        )

    def _default_steady_state_initial_guess(self) -> dolfin.UserExpression:
        """u=1 in the channel, u=0 inside the cavity (x[1] < 0)."""

        class _CavityFlow(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0 if x[1] >= 0 else 0.0
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        return _CavityFlow()

    @classmethod
    def make_default(
        cls,
        Re: float = 7500,
        path_out=None,
        num_steps: int = 10,
        save_every: int = 0,
        Tstart: float = 0.0,
        verbose: int = 0,
        meshpath: str | Path | None = None,
    ) -> "CavityFlowSolver":
        """Return a CavityFlowSolver with standard parameters (Re=7500, 1 FORCE actuator, 2 sensors)."""
        from pathlib import Path

        import numpy as np

        import flowcontrol.flowsolverparameters as fsp
        from flowcontrol.actuator import ActuatorForceGaussianV
        from flowcontrol.sensor import (
            SENSOR_TYPE,
            SensorHorizontalWallShear,
            SensorPoint,
        )

        if path_out is None:
            path_out = Path(__file__).parent / "data_output"

        params_flow = fsp.ParamFlow(Re=Re, uinf=1.0)
        params_flow.user_data.update({"L": 1.0, "D": 1.0})

        params_time = fsp.ParamTime(num_steps=num_steps, dt=0.0004, Tstart=Tstart)
        params_save = fsp.ParamSave(save_every=save_every, path_out=path_out)
        params_solver = fsp.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)

        default_mesh = Path(__file__).parent / "data_input" / "cavity_coarse.xdmf"
        params_mesh = fsp.ParamMesh(meshpath=meshpath or default_mesh)
        params_mesh.user_data.update(
            {
                "xinf": 2.5,
                "xinfa": -1.2,
                "yinf": 0.5,
                "x0ns_left": -0.4,
                "x0ns_right": 1.75,
            }
        )
        params_control = fsp.ParamControl(
            sensor_list=[
                SensorHorizontalWallShear(
                    sensor_index=100,
                    x_sensor_left=1.0,
                    x_sensor_right=1.1,
                    y_sensor=0.0,
                    sensor_type=SENSOR_TYPE.OTHER,
                ),
                SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.1, 0.1])),
            ],
            actuator_list=[
                ActuatorForceGaussianV(sigma=0.0849, position=np.array([-0.1, 0.02])),
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
