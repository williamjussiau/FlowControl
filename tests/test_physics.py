"""Tests for utils.physics — requires FEniCS/dolfin."""

import dolfin
import numpy as np
import pytest
from numpy.testing import assert_allclose

from utils.physics import get_div0_u, stress_tensor


@pytest.fixture(scope="module")
def mesh_spaces():
    mesh = dolfin.UnitSquareMesh(8, 8)
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, P2 * P1)
    V = W.sub(0).collapse()
    P = W.sub(1).collapse()
    return mesh, V, P


# ── get_div0_u ────────────────────────────────────────────────────────────────


class TestGetDiv0U:
    def test_returns_function(self, mesh_spaces):
        mesh, V, P = mesh_spaces
        u = get_div0_u(V=V, P=P, xloc=0.5, yloc=0.5, size=0.1)
        assert isinstance(u, dolfin.Function)

    def test_divergence_is_numerically_zero(self):
        """The stream-function construction should produce a near-divergence-free field.

        Uses a fine mesh (32x32) so the Gaussian (size=0.1) is well-resolved by P2
        elements; the L2-projection error is then small enough to check.
        """
        mesh = dolfin.UnitSquareMesh(32, 32)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W = dolfin.FunctionSpace(mesh, P2 * P1)
        V = W.sub(0).collapse()
        P = W.sub(1).collapse()
        u = get_div0_u(V=V, P=P, xloc=0.5, yloc=0.5, size=0.1)
        div_u = dolfin.project(dolfin.div(u), P)
        div_norm = dolfin.norm(div_u, norm_type="L2")
        assert div_norm < 1e-2

    def test_field_is_not_identically_zero(self, mesh_spaces):
        mesh, V, P = mesh_spaces
        u = get_div0_u(V=V, P=P, xloc=0.5, yloc=0.5, size=0.1)
        assert u.vector().norm("l2") > 1e-12


# ── stress_tensor ─────────────────────────────────────────────────────────────


class TestStressTensor:
    def test_returns_ufl_expression(self, mesh_spaces):
        """stress_tensor returns a UFL form, not a dolfin.Function."""
        import ufl

        mesh, V, P = mesh_spaces
        W = V  # use V as a proxy — we just need any Function
        u = dolfin.Function(V)
        p = dolfin.Function(P)
        sigma = stress_tensor(nu=1.0, u=u, p=p)
        assert isinstance(sigma, ufl.core.expr.Expr)

    def test_assembles_without_error(self, mesh_spaces):
        """The stress tensor should be integrable over the domain."""
        mesh, V, P = mesh_spaces
        u = dolfin.Function(V)
        p = dolfin.Function(P)
        sigma = stress_tensor(nu=1.0, u=u, p=p)
        # Integrate the Frobenius norm of sigma over the domain
        val = dolfin.assemble(dolfin.inner(sigma, sigma) * dolfin.dx)
        assert np.isfinite(val)

    def test_linear_shear_stress_value(self, mesh_spaces):
        """For u=(y, 0) and p=0, sigma_12 = sigma_21 = nu (linear shear flow)."""
        mesh, V, P = mesh_spaces
        u_expr = dolfin.Expression(("x[1]", "0.0"), degree=1)
        p_expr = dolfin.Expression("0.0", degree=0)
        u = dolfin.interpolate(u_expr, V)
        p = dolfin.interpolate(p_expr, P)
        nu = 1.0
        sigma = stress_tensor(nu=nu, u=u, p=p)
        # sigma_12 = nu * (du_1/dx_2 + du_2/dx_1) = nu * 1 = nu
        W_ten = dolfin.TensorFunctionSpace(mesh, "DG", 1)
        sigma_h = dolfin.project(sigma, W_ten)
        # Evaluate at centre of domain
        val = sigma_h([0.5, 0.5])
        # sigma is 2x2; in row-major order: [s00, s01, s10, s11]
        # s01 = s10 = nu = 1.0
        assert abs(val[1] - nu) < 1e-10
