"""State-space operator computation for FlowSolver instances.

Provides get_A, get_mass_matrix, get_B, and get_C for assembling the
linearized operators (A, E, B, C) needed for stability analysis and control design.
"""

import logging
from typing import Optional

import dolfin
import numpy as np
from dolfin import div, dot, inner, nabla_grad
from numpy.typing import NDArray

from flowcontrol import flowsolver
from flowcontrol.actuator import ACTUATOR_TYPE
from flowcontrol.sensor import SENSOR_TYPE, SensorIntegral, SensorPoint

logger = logging.getLogger(__name__)


class OperatorGetter:
    def __init__(self, flowsolver: flowsolver.FlowSolver):
        self.flowsolver = flowsolver

    def get_A(
        self,
        UP0: Optional[dolfin.Function] = None,
        autodiff: bool = True,
        u_ctrl: Optional[NDArray[np.float64]] = None,
    ) -> dolfin.PETScMatrix:
        """Get state-space dynamic matrix A = -dF/dUP0 linearized around UP0."""
        logger.info("Computing jacobian A...")

        if UP0 is None:
            UP0 = self.flowsolver.fields.UP0

        if u_ctrl is None:
            self.flowsolver.flush_actuators_u_ctrl()
        else:
            self.flowsolver.set_actuators_u_ctrl(u_ctrl)

        if autodiff:
            f = self.flowsolver._gather_actuators_expressions()
            F0 = self.flowsolver.forms.steady(UP0, f)
            du = dolfin.TrialFunction(self.flowsolver.W)
            dF0 = dolfin.derivative(-F0, u=UP0, du=du)
        else:
            dx = self.flowsolver.dx
            U0, _ = dolfin.split(UP0)
            u, p = dolfin.TrialFunctions(self.flowsolver.W)
            v, q = dolfin.TestFunctions(self.flowsolver.W)
            iRe = dolfin.Constant(1 / self.flowsolver.params_flow.Re)
            dF0 = (
                -dot(dot(U0, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(U0)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
            )

        Jac = dolfin.PETScMatrix()
        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in self.flowsolver.bc.bcu]

        return Jac

    def get_mass_matrix(self) -> dolfin.PETScMatrix:
        """Get mass matrix associated to spatial discretization"""
        logger.info("Computing mass matrix E...")
        up = dolfin.TrialFunction(self.flowsolver.W)
        vq = dolfin.TestFunction(self.flowsolver.W)

        E = dolfin.PETScMatrix()

        mf = (up[0] * vq[0] + up[1] * vq[1]) * self.flowsolver.dx  # sum u, v but not p
        dolfin.assemble(mf, tensor=E)

        return E

    def get_B(
        self,
        UP0: Optional[dolfin.Function] = None,
    ) -> NDArray[np.float64]:
        """Get actuation matrix B.

        For FORCE actuators: B column = load vector ∫ b(x)·v dx, i.e. B = -∂F/∂u_ctrl.
        For BC actuators: B column = A_raw · w_lift, where A_raw is the raw Jacobian
            (no BC rows zeroed) and w_lift is the unit lifting function for the actuator BC.

        Sign convention consistent with get_A: A = -dF/dq, B = -dF/du_ctrl.
        """
        logger.info("Computing actuation matrix B...")

        if UP0 is None:
            UP0 = self.flowsolver.fields.UP0

        W = self.flowsolver.W
        mpi_local_size = len(W.dofmap().dofs())
        actuator_list = self.flowsolver.params_control.actuator_list
        actuator_number = self.flowsolver.params_control.actuator_number

        B = np.zeros((mpi_local_size, actuator_number))
        v = dolfin.TestFunction(W)

        has_bc_actuators = any(
            a.actuator_type is ACTUATOR_TYPE.BC for a in actuator_list
        )

        u_ctrl_old = self.flowsolver.get_actuators_u_ctrl()

        try:
            # Assemble raw Jacobian (no BCs applied) — needed for BC actuator lifting
            if has_bc_actuators:
                self.flowsolver.flush_actuators_u_ctrl()
                f = self.flowsolver._gather_actuators_expressions()
                F0 = self.flowsolver.forms.steady(UP0, f)
                du = dolfin.TrialFunction(W)
                dF0 = dolfin.derivative(-1 * F0, u=UP0, du=du)
                A_raw = dolfin.PETScMatrix()
                dolfin.assemble(dF0, tensor=A_raw)
                # intentionally no bc.apply(A_raw)

            # Set u_ctrl = 1 to get unit shape functions
            self.flowsolver.set_actuators_u_ctrl(actuator_number * [1.0])

            for ii, actuator in enumerate(actuator_list):
                if actuator.actuator_type is ACTUATOR_TYPE.FORCE:
                    # B = -∂F/∂u_ctrl = ∫ b(x)·v dx (velocity components only)
                    b_form = (
                        actuator.expression[0] * v[0] + actuator.expression[1] * v[1]
                    ) * self.flowsolver.dx
                    B[:, ii] = dolfin.assemble(b_form).get_local()

                elif actuator.actuator_type is ACTUATOR_TYPE.BC:
                    # Lifting: zero function with unit actuator profile on boundary DOFs
                    w = dolfin.Function(W)
                    dolfin.DirichletBC(
                        W.sub(0), actuator.expression, actuator.boundary
                    ).apply(w.vector())

                    # B = A_raw · w
                    b_col = dolfin.PETScVector()
                    A_raw.init_vector(b_col, 0)
                    A_raw.mult(w.vector(), b_col)
                    B[:, ii] = b_col.get_local()

                else:
                    raise NotImplementedError(
                        f"Actuator type {actuator.actuator_type} not supported in get_B"
                    )

        finally:
            self.flowsolver.set_actuators_u_ctrl(u_ctrl_old)

        logger.info(f"Finished computing B of size {B.shape}")
        return B

    def get_C(self) -> NDArray[np.float64]:
        """Get measurement matrix C.

        For SensorPoint: uses dolfin.PointSource to assemble the basis function
            weights at the sensor location — correct even when the point falls
            between DOFs.
        For SensorIntegral: assembles sensor.linear_form(v) with a TestFunction,
            giving the C row as a single dolfin.assemble call.

        Both approaches are MPI-compatible and scale as O(sensor_number).
        """
        logger.info("Computing measurement matrix C...")

        W = self.flowsolver.W
        sensor_list = self.flowsolver.params_control.sensor_list
        sensor_number = self.flowsolver.params_control.sensor_number
        mpi_local_size = len(W.dofmap().dofs())

        C = np.zeros((sensor_number, mpi_local_size))
        b = dolfin.Function(W).vector()
        v = dolfin.TestFunction(W)

        for ii, sensor in enumerate(sensor_list):
            if isinstance(sensor, SensorPoint):
                b.zero()
                dolfin.PointSource(
                    _point_sensor_subspace(W, sensor.sensor_type),
                    dolfin.Point(sensor.position),
                    1.0,
                ).apply(b)
                C[ii, :] = b.get_local()
            elif isinstance(sensor, SensorIntegral):
                C[ii, :] = dolfin.assemble(sensor.linear_form(v)).get_local()
            else:
                raise TypeError(
                    f"Sensor type {type(sensor).__name__} not supported in get_C"
                )

        logger.info(f"Finished computing C of size {C.shape}")
        return C

    def get_all(
        self,
        autodiff: bool = True,
        u_ctrl: Optional[NDArray[np.float64]] = None,
    ) -> tuple:
        """Compute all four state-space operators (A, E, B, C) in one call.

        Returns
        -------
        A : dolfin.PETScMatrix  — Jacobian (dynamic matrix)
        E : dolfin.PETScMatrix  — Mass matrix
        B : NDArray             — Actuation matrix
        C : NDArray             — Measurement matrix
        """
        A = self.get_A(autodiff=autodiff, u_ctrl=u_ctrl)
        E = self.get_mass_matrix()
        B = self.get_B()
        C = self.get_C()
        return A, E, B, C


def _point_sensor_subspace(
    W: dolfin.FunctionSpace, sensor_type: SENSOR_TYPE
) -> dolfin.FunctionSpace:
    """Map a SensorPoint sensor_type to the corresponding subspace of W = V × Q."""
    mapping = {
        SENSOR_TYPE.U: W.sub(0).sub(0),
        SENSOR_TYPE.V: W.sub(0).sub(1),
        SENSOR_TYPE.P: W.sub(1),
    }
    if sensor_type not in mapping:
        raise ValueError(
            f"sensor_type {sensor_type} cannot be mapped to a subspace for PointSource"
        )
    return mapping[sensor_type]
