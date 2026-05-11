"""Fluidic pinball (3 cylinders), using flowsolver.FlowSolver.

Several supercritical Hopf bifurcations. Recommended Re < 100.
"""

import logging
from pathlib import Path

import dolfin
import numpy as np
import pandas as pd

import flowcontrol.flowsolver as flowsolver
import utils.utils_flowsolver as flu
from flowcontrol.actuator import CYLINDER_ACTUATION_MODE
from flowcontrol.flowfield import BoundaryConditions
from utils.physics import stress_tensor

logger = logging.getLogger(__name__)


class PinballFlowSolver(flowsolver.FlowSolver):
    """Flow past 3 cylinders (fluidic pinball). Proposed Re=100."""

    def _make_boundaries(self):
        """Return subdomains for the fluidic-pinball geometry.

        Always defines inlet, outlet, far-field walls, and three cylinder
        surfaces.  In SUCTION mode each cylinder is split into a body (no-slip)
        and an actuator slot; in ROTATION mode the full cylinder surface is the
        actuator.  Cylinder positions follow the equilateral triangle layout.
        """
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        or_cpp = flu.or_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS
        mode_actuation = self.params_control.user_data["mode_actuation"]

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
        close_to_cylinder_top_cpp = (
            between_cpp("x[0]", "-radius", "radius") + and_cpp + between_cpp("x[1]", "radius/2", "5*radius/2")
        )
        close_to_cylinder_bot_cpp = (
            between_cpp("x[0]", "-radius", "radius") + and_cpp + between_cpp("x[1]", "-5*radius/2", "-radius/2")
        )
        close_to_cylinder_mid_cpp = (
            between_cpp("x[0]", "-radius-1.5*cos(pi/6)", "+radius-1.5*cos(pi/6)")
            + and_cpp
            + between_cpp("x[1]", "-radius", "radius")
        )

        cylinder_boundary_top_cpp = on_boundary_cpp + and_cpp + close_to_cylinder_top_cpp
        cylinder_boundary_bot_cpp = on_boundary_cpp + and_cpp + close_to_cylinder_bot_cpp
        cylinder_boundary_mid_cpp = on_boundary_cpp + and_cpp + close_to_cylinder_mid_cpp

        boundaries_names = ["inlet", "outlet", "walls"]
        subdomains_list = [inlet, outlet, walls]

        if mode_actuation == CYLINDER_ACTUATION_MODE.SUCTION:
            ldelta = self.params_control.actuator_list[0].width

            cone_charm_act_cpp = between_cpp("x[0]", "-ldelta-1.5*cos(pi/6)", "-1.5*cos(pi/6)+ldelta")
            cone_top_act_cpp = between_cpp("x[0]", "-ldelta", "+ldelta")
            cone_bot_act_cpp = between_cpp("x[0]", "-ldelta", "+ldelta")

            cylinder_top = dolfin.CompiledSubDomain(cylinder_boundary_top_cpp, radius=radius, ldelta=ldelta)
            cylinder_bot = dolfin.CompiledSubDomain(cylinder_boundary_bot_cpp, radius=radius, ldelta=ldelta)
            cylinder_mid = dolfin.CompiledSubDomain(cylinder_boundary_mid_cpp, radius=radius, ldelta=ldelta)
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
            actuator_mid = dolfin.CompiledSubDomain(
                cylinder_boundary_mid_cpp + and_cpp + cone_charm_act_cpp,
                radius=radius,
                ldelta=ldelta,
            )
            boundaries_names += [
                "cylinder_top",
                "cylinder_bot",
                "cylinder_mid",
                "actuator_mid",
                "actuator_top",
                "actuator_bot",
            ]
            subdomains_list += [
                cylinder_top,
                cylinder_bot,
                cylinder_mid,
                actuator_mid,
                actuator_top,
                actuator_bot,
            ]
        else:
            actuator_top = dolfin.CompiledSubDomain(cylinder_boundary_top_cpp, radius=radius)
            actuator_bot = dolfin.CompiledSubDomain(cylinder_boundary_bot_cpp, radius=radius)
            actuator_mid = dolfin.CompiledSubDomain(cylinder_boundary_mid_cpp, radius=radius)
            boundaries_names += ["actuator_mid", "actuator_top", "actuator_bot"]
            subdomains_list += [actuator_mid, actuator_top, actuator_bot]

        return pd.DataFrame(index=boundaries_names, data={"subdomain": subdomains_list})

    def _make_bcs(self):
        """Return perturbation-field BCs for the pinball.

        In SUCTION mode: no-slip on cylinder bodies and actuator expressions on
        the three slots.  In ROTATION mode: actuator expressions cover the full
        cylinder surfaces (no separate body BC needed).
        """
        mode_actuation = self.params_control.user_data["mode_actuation"]

        bcu_inlet = dolfin.DirichletBC(self.W.sub(0), dolfin.Constant((0, 0)), self.get_subdomain("inlet"))
        bcu_walls = dolfin.DirichletBC(self.W.sub(0).sub(1), dolfin.Constant(0), self.get_subdomain("walls"))
        bcu = [bcu_inlet, bcu_walls]

        if mode_actuation == CYLINDER_ACTUATION_MODE.SUCTION:
            bcu += [
                dolfin.DirichletBC(
                    self.W.sub(0),
                    dolfin.Constant((0, 0)),
                    self.get_subdomain("cylinder_top"),
                ),
                dolfin.DirichletBC(
                    self.W.sub(0),
                    dolfin.Constant((0, 0)),
                    self.get_subdomain("cylinder_bot"),
                ),
                dolfin.DirichletBC(
                    self.W.sub(0),
                    dolfin.Constant((0, 0)),
                    self.get_subdomain("cylinder_mid"),
                ),
            ]

        bcu += [
            dolfin.DirichletBC(
                self.W.sub(0),
                self.params_control.actuator_list[0].expression,
                self.get_subdomain("actuator_mid"),
            ),
            dolfin.DirichletBC(
                self.W.sub(0),
                self.params_control.actuator_list[1].expression,
                self.get_subdomain("actuator_top"),
            ),
            dolfin.DirichletBC(
                self.W.sub(0),
                self.params_control.actuator_list[2].expression,
                self.get_subdomain("actuator_bot"),
            ),
        ]

        return BoundaryConditions(bcu=bcu, bcp=[])

    def _make_BCs(self) -> BoundaryConditions:
        """Steady-state BCs: uniform flow at inlet and walls."""
        uniform = dolfin.Constant((self.params_flow.uinf, 0))
        bcu_inlet = dolfin.DirichletBC(self.W.sub(0), uniform, self.get_subdomain("inlet"))
        bcu_walls = dolfin.DirichletBC(self.W.sub(0), uniform, self.get_subdomain("walls"))
        bcs = self._make_bcs()
        return BoundaryConditions(bcu=[bcu_inlet, bcu_walls] + bcs.bcu[2:], bcp=[])

    def compute_steady_state(self, u_ctrl, method="newton", **kwargs):
        """Compute steady state then log force coefficients for each cylinder surface."""
        super().compute_steady_state(method=method, u_ctrl=u_ctrl, **kwargs)
        force_coeffs = self.compute_force_coefficients(self.fields.U0, self.fields.P0)
        if self.verbose:
            for name, (cl, cd) in force_coeffs.items():
                logger.info(f"{name}: Cl={cl:.4f}, Cd={cd:.4f}")

    def compute_force_coefficients(self, u, p) -> dict:
        """Return {surface_name: (cl, cd)} for each cylinder surface."""
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re
        mode_actuation = self.params_control.user_data["mode_actuation"]

        sigma = stress_tensor(nu, u, p)
        Fo = -dolfin.dot(sigma, dolfin.FacetNormal(self.mesh))

        if mode_actuation == CYLINDER_ACTUATION_MODE.SUCTION:
            surfaces = [
                "cylinder_mid",
                "actuator_mid",
                "cylinder_top",
                "actuator_top",
                "cylinder_bot",
                "actuator_bot",
            ]
        else:
            surfaces = ["actuator_mid", "actuator_top", "actuator_bot"]

        result = {}
        for name in surfaces:
            idx = int(self.boundaries.loc[name].idx)
            drag = dolfin.assemble(Fo[0] * self.ds(idx))
            lift = dolfin.assemble(Fo[1] * self.ds(idx))
            result[name] = (
                lift / (0.5 * self.params_flow.uinf**2 * D),
                drag / (0.5 * self.params_flow.uinf**2 * D),
            )
        return result

    @classmethod
    def make_default(
        cls,
        Re: float = 50,
        mode_actuation=None,
        path_out=None,
        num_steps: int = 10,
        save_every: int = 0,
        Tstart: float = 0.0,
        verbose: int = 0,
        meshpath: str | Path | None = None,
    ) -> "PinballFlowSolver":
        """Return a PinballFlowSolver with standard parameters (Re=50, rotation actuation, 3 sensors)."""
        from pathlib import Path

        import numpy as np

        import flowcontrol.flowsolverparameters as fsp
        from flowcontrol.actuator import (
            CYLINDER_ACTUATION_MODE,
            ActuatorBCParabolicV,
            ActuatorBCRotation,
        )
        from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

        if path_out is None:
            path_out = Path(__file__).parent / "data_output"
        if mode_actuation is None:
            mode_actuation = CYLINDER_ACTUATION_MODE.ROTATION

        params_flow = fsp.ParamFlow(Re=Re, uinf=1.0)
        params_flow.user_data["D"] = 1.0

        params_time = fsp.ParamTime(num_steps=num_steps, dt=0.005, Tstart=Tstart)
        params_save = fsp.ParamSave(save_every=save_every, path_out=path_out)
        params_solver = fsp.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)

        default_mesh = Path(__file__).parent / "data_input" / "mesh_middle_gmsh.xdmf"
        params_mesh = fsp.ParamMesh(meshpath=meshpath or default_mesh)
        params_mesh.user_data.update({"xinf": 20, "xinfa": -6, "yinf": 6})

        cylinder_diameter = params_flow.user_data["D"]
        position_mid = [-1.5 * np.cos(np.pi / 6), 0.0]
        position_top = [0.0, +0.75]

        if mode_actuation == CYLINDER_ACTUATION_MODE.SUCTION:
            width = ActuatorBCParabolicV.angular_size_deg_to_width(10, cylinder_diameter / 2)
            actuator_list = [
                ActuatorBCParabolicV(width=width, position_x=position_mid[0]),
                ActuatorBCParabolicV(width=width, position_x=position_top[0]),
                ActuatorBCParabolicV(width=width, position_x=position_top[0]),
            ]
        else:
            actuator_list = [
                ActuatorBCRotation(
                    position_x=position_mid[0],
                    position_y=position_mid[1],
                    diameter=cylinder_diameter,
                ),
                ActuatorBCRotation(
                    position_x=position_top[0],
                    position_y=+position_top[1],
                    diameter=cylinder_diameter,
                ),
                ActuatorBCRotation(
                    position_x=position_top[0],
                    position_y=-position_top[1],
                    diameter=cylinder_diameter,
                ),
            ]

        params_control = fsp.ParamControl(
            sensor_list=[
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([8.0, 0.0])),
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([10.0, 0.0])),
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([12.0, 0.0])),
            ],
            actuator_list=actuator_list,
            user_data={"mode_actuation": mode_actuation},
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


class PinballCustomInitialGuess(dolfin.UserExpression):
    """Custom initial guess for Picard iteration on the pinball."""

    def __init__(self, mode="symmetric", **kwargs):
        self.mode = mode
        super().__init__(**kwargs)

    def eval(self, value, x):
        """Set (u, v, p) by mode: symmetric=(1,0,0), antisymmetric_top=(1/√2,+1/√2,0), antisymmetric_bot=(1/√2,−1/√2,0)."""
        if self.mode == "symmetric":
            value[0] = 1.0
            value[1] = 0.0
            value[2] = 0.0
        elif self.mode == "antisymmetric_top":
            value[0] = 1.0 / np.sqrt(2)
            value[1] = +1.0 / np.sqrt(2)
            value[2] = 0.0
        elif self.mode == "antisymmetric_bot":
            value[0] = 1.0 / np.sqrt(2)
            value[1] = -1.0 / np.sqrt(2)
            value[2] = 0.0
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def value_shape(self):
        """Return (3,) — mixed (u, v, p) field shape."""
        return (3,)

    def as_dolfin_function(self, function_space, interp=True):
        """Project or interpolate this expression onto function_space and return as dolfin.Function."""
        return flu.expression_to_dolfin_function(self, function_space=function_space, interp=interp)
