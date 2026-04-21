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

    @classmethod
    def make_default(
        cls,
        Re: float = 8000,
        path_out=None,
        num_steps: int = 10,
        save_every: int = 0,
        Tstart: float = 0.0,
        verbose: int = 0,
    ) -> "LidCavityFlowSolver":
        """Return a LidCavityFlowSolver with standard parameters (Re=8000, 1 BC actuator, 2 sensors)."""
        from pathlib import Path

        import numpy as np

        import flowcontrol.flowsolverparameters as fsp
        from flowcontrol.actuator import ActuatorBCUniformU
        from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

        if path_out is None:
            path_out = Path(__file__).parent / "data_output"

        params_flow = fsp.ParamFlow(Re=Re, uinf=1.0)
        params_flow.user_data["D"] = 1.0

        params_time = fsp.ParamTime(num_steps=num_steps, dt=0.005, Tstart=Tstart)
        params_save = fsp.ParamSave(save_every=save_every, path_out=path_out)
        params_solver = fsp.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)
        params_mesh = fsp.ParamMesh(
            meshpath=Path(__file__).parent / "data_input" / "mesh64.xdmf"
        )
        params_mesh.user_data.update({"yup": 1, "ylo": 0, "xri": 1, "xle": 0})
        params_control = fsp.ParamControl(
            sensor_list=[
                SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5])),
                SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.95])),
            ],
            actuator_list=[ActuatorBCUniformU(boundary_name="lid")],
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
