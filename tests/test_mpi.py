"""Tests for utils.mpi — serial-only (not marked mpi; MPI tests live in integration/)."""

import dolfin
import numpy as np
import pytest
from numpy.testing import assert_allclose

from utils.mpi import get_rank, peval


@pytest.fixture(scope="module")
def scalar_function():
    mesh = dolfin.UnitSquareMesh(4, 4)
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    # f(x, y) = x + 2*y
    f = dolfin.interpolate(dolfin.Expression("x[0] + 2*x[1]", degree=1), V)
    return f


# ── get_rank ──────────────────────────────────────────────────────────────────


class TestGetRank:
    def test_returns_zero_in_serial(self):
        assert get_rank() == 0

    def test_returns_int(self):
        assert isinstance(get_rank(), int)


# ── peval ─────────────────────────────────────────────────────────────────────


class TestPeval:
    def test_correct_value_at_corner(self, scalar_function):
        """f(0, 0) = 0 + 2*0 = 0."""
        val = peval(scalar_function, [0.0, 0.0])
        assert_allclose(val, 0.0, atol=1e-12)

    def test_correct_value_at_interior_point(self, scalar_function):
        """f(0.5, 0.5) = 0.5 + 2*0.5 = 1.5."""
        val = peval(scalar_function, [0.5, 0.5])
        assert_allclose(val, 1.5, atol=1e-10)

    def test_correct_value_at_top_right(self, scalar_function):
        """f(1, 1) = 1 + 2*1 = 3."""
        val = peval(scalar_function, [1.0, 1.0])
        assert_allclose(val, 3.0, atol=1e-10)

    def test_returns_scalar_for_scalar_function(self, scalar_function):
        val = peval(scalar_function, [0.5, 0.5])
        assert np.isscalar(val) or (hasattr(val, "shape") and val.ndim == 0)
