"""Tests for flowcontrol.actuator.

No-mesh section: abstract enforcement, instantiation, defaults, actuator_type,
    angular_size_deg_to_width.
With-mesh section: load_expression sets self.expression, u_ctrl initial value,
    load_expression return value.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dolfin

from flowcontrol.actuator import (
    ACTUATOR_TYPE,
    Actuator,
    ActuatorBC,
    ActuatorBCParabolicV,
    ActuatorBCRotation,
    ActuatorBCUniformU,
    ActuatorForceGaussianV,
)


# ── Shared mock ───────────────────────────────────────────────────────────────


class MockFlowSolver:
    """Minimal stand-in for FlowSolver — only the attributes load_expression needs."""

    def __init__(self):
        mesh = dolfin.UnitSquareMesh(4, 4)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W = dolfin.FunctionSpace(mesh, P2 * P1)
        self.mesh = mesh
        self.W = W
        self.V = W.sub(0).collapse()


@pytest.fixture(scope="module")
def mock_fs():
    return MockFlowSolver()


# ── No-mesh: abstract enforcement ─────────────────────────────────────────────


class TestAbstract:
    def test_actuator_is_abstract(self):
        with pytest.raises(TypeError):
            Actuator(actuator_type=ACTUATOR_TYPE.BC)  # type: ignore

    def test_actuator_bc_is_abstract(self):
        with pytest.raises(TypeError):
            ActuatorBC(actuator_type=ACTUATOR_TYPE.BC)  # type: ignore


# ── No-mesh: instantiation and defaults ───────────────────────────────────────


class TestInstantiation:
    def test_parabolic_v_defaults(self):
        a = ActuatorBCParabolicV(width=0.5, position_x=1.0)
        assert a.width == 0.5
        assert a.position_x == 1.0
        assert a.expression is None
        assert a.actuator_type is ACTUATOR_TYPE.BC

    def test_rotation_defaults(self):
        a = ActuatorBCRotation(position_x=0.0, position_y=0.0, diameter=1.0)
        assert a.diameter == 1.0
        assert a.expression is None
        assert a.actuator_type is ACTUATOR_TYPE.BC

    def test_uniform_u_defaults(self):
        a = ActuatorBCUniformU()
        assert a.expression is None
        assert a.actuator_type is ACTUATOR_TYPE.BC

    def test_force_gaussian_v(self):
        a = ActuatorForceGaussianV(sigma=0.01, position=np.array([0.1, 0.2]))
        assert a.sigma == 0.01
        assert a.expression is None
        assert a.actuator_type is ACTUATOR_TYPE.FORCE

    def test_expression_none_before_load(self):
        a = ActuatorBCParabolicV(width=0.1, position_x=0.0)
        assert a.expression is None


# ── No-mesh: angular_size_deg_to_width ────────────────────────────────────────


class TestAngularWidth:
    def test_90_degrees_unit_radius(self):
        # sin(45°) = sqrt(2)/2
        result = ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg=90, cylinder_radius=1.0
        )
        assert_allclose(result, np.sin(np.pi / 4), rtol=1e-10)

    def test_zero_degrees(self):
        result = ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg=0, cylinder_radius=1.0
        )
        assert_allclose(result, 0.0, atol=1e-15)

    def test_scales_with_radius(self):
        w1 = ActuatorBCParabolicV.angular_size_deg_to_width(10, 1.0)
        w2 = ActuatorBCParabolicV.angular_size_deg_to_width(10, 2.0)
        assert_allclose(w2, 2.0 * w1, rtol=1e-10)


# ── With mesh: load_expression ────────────────────────────────────────────────


class TestLoadExpression:
    def test_parabolic_v_sets_expression(self, mock_fs):
        a = ActuatorBCParabolicV(width=0.1, position_x=0.0)
        a.load_expression(mock_fs)
        assert a.expression is not None

    def test_parabolic_v_returns_expression(self, mock_fs):
        a = ActuatorBCParabolicV(width=0.1, position_x=0.0)
        returned = a.load_expression(mock_fs)
        assert returned is a.expression

    def test_parabolic_v_u_ctrl_is_zero(self, mock_fs):
        """After loading, u_ctrl should be 0 (actuator is off by default)."""
        a = ActuatorBCParabolicV(width=0.1, position_x=0.0)
        a.load_expression(mock_fs)
        assert a.expression.u_ctrl == 0.0

    def test_rotation_sets_expression(self, mock_fs):
        a = ActuatorBCRotation(position_x=0.5, position_y=0.5, diameter=1.0)
        a.load_expression(mock_fs)
        assert a.expression is not None

    def test_uniform_u_sets_expression(self, mock_fs):
        a = ActuatorBCUniformU()
        a.load_expression(mock_fs)
        assert a.expression is not None

    def test_force_gaussian_v_sets_expression(self, mock_fs):
        a = ActuatorForceGaussianV(sigma=0.1, position=np.array([0.5, 0.5]))
        a.load_expression(mock_fs)
        assert a.expression is not None

    def test_force_gaussian_v_u_ctrl_is_zero(self, mock_fs):
        """After normalization, u_ctrl should be 0 (ready for simulation)."""
        a = ActuatorForceGaussianV(sigma=0.1, position=np.array([0.5, 0.5]))
        a.load_expression(mock_fs)
        assert a.expression.u_ctrl == 0.0

    def test_force_gaussian_v_eta_normalizes(self, mock_fs):
        """eta should be set so that the norm of the expression is 1 at u_ctrl=1."""
        a = ActuatorForceGaussianV(sigma=0.1, position=np.array([0.5, 0.5]))
        a.load_expression(mock_fs)
        a.expression.u_ctrl = 1.0
        norm = dolfin.norm(a.expression, mesh=mock_fs.mesh)
        assert_allclose(norm, 1.0, rtol=1e-6)