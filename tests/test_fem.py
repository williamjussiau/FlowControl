"""Tests for utils.fem — requires FEniCS/dolfin for function tests;
near_cpp / between_cpp are pure string helpers (no FEniCS needed).
"""

import dolfin
import numpy as np
import pytest
from numpy.testing import assert_allclose

from utils.fem import between_cpp, expression_to_dolfin_function, near_cpp


# ── near_cpp / between_cpp — pure string, no FEniCS ──────────────────────────


class TestNearCpp:
    def test_basic_output(self):
        s = near_cpp("x[0]", 1.0)
        assert "near" in s
        assert "x[0]" in s
        assert "1.0" in s

    def test_custom_tolerance(self):
        s = near_cpp("x[1]", 0.5, tol="1e-10")
        assert "1e-10" in s

    def test_returns_string(self):
        assert isinstance(near_cpp("x[0]", 0.0), str)


class TestBetweenCpp:
    def test_basic_output(self):
        s = between_cpp("x[0]", "0.1", "0.9")
        assert "x[0]" in s
        assert "0.1" in s
        assert "0.9" in s

    def test_returns_string(self):
        assert isinstance(between_cpp("x[0]", "0.0", "1.0"), str)

    def test_contains_comparison_operators(self):
        s = between_cpp("x[0]", "0.0", "1.0")
        assert ">=" in s or "<=" in s


# ── expression_to_dolfin_function — requires FEniCS ──────────────────────────


@pytest.fixture(scope="module")
def scalar_space(unit_square_mesh):
    """CG1 scalar space on 4×4 unit square."""
    return dolfin.FunctionSpace(unit_square_mesh, "CG", 1)


class TestExpressionToDolfinFunction:
    def test_interpolate_returns_function(self, scalar_space):
        expr = dolfin.Expression("x[0]*x[0] + x[1]", degree=2)
        f = expression_to_dolfin_function(expr, scalar_space, interp=True)
        assert isinstance(f, dolfin.Function)

    def test_project_returns_function(self, scalar_space):
        expr = dolfin.Expression("sin(x[0])", degree=2)
        f = expression_to_dolfin_function(expr, scalar_space, interp=False)
        assert isinstance(f, dolfin.Function)

    def test_interpolate_values_correct(self, scalar_space):
        """Interpolating the constant expression 1.0 should give a uniform function."""
        expr = dolfin.Expression("1.0", degree=0)
        f = expression_to_dolfin_function(expr, scalar_space, interp=True)
        vals = f.vector().get_local()
        assert_allclose(vals, 1.0, atol=1e-12)

    def test_interpolate_vs_project_close(self, scalar_space):
        """For a polynomial expressible in the space, interpolation ≈ projection."""
        expr = dolfin.Expression("x[0]", degree=1)
        f_interp = expression_to_dolfin_function(expr, scalar_space, interp=True)
        f_proj = expression_to_dolfin_function(expr, scalar_space, interp=False)
        diff_norm = dolfin.norm(f_interp.vector() - f_proj.vector())
        assert diff_norm < 1e-10
