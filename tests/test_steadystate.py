"""Tests for flowcontrol.steadystate.SteadyStateSolver — requires FEniCS/dolfin."""

import dolfin
import numpy as np
import pytest
from numpy.testing import assert_allclose

from flowcontrol.nsforms import NSForms
from flowcontrol.steadystate import SteadyStateSolver


def _lid_cavity_setup(Re=1.0):
    """Build a lid-cavity on a 6×6 mesh at given Re.

    Boundary conditions:
    - Lid (top, y=1): u = (1, 0)
    - All other walls: u = (0, 0)

    Returns (W, bcu, forms).
    """
    mesh = dolfin.UnitSquareMesh(6, 6)
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, P2 * P1)
    V = W.sub(0)

    noslip = dolfin.Constant((0.0, 0.0))
    lid = dolfin.Constant((1.0, 0.0))

    bc_lid = dolfin.DirichletBC(V, lid, "near(x[1], 1.0)")
    bc_walls = dolfin.DirichletBC(V, noslip, "on_boundary && !near(x[1], 1.0)")
    bcu = [bc_lid, bc_walls]

    forms = NSForms(W, Re=Re, dt=0.01, is_nonlinear=True, shift=0.0)
    return W, bcu, forms


# ── Newton ────────────────────────────────────────────────────────────────────


class TestNewton:
    def test_converges_on_stokes_problem(self):
        """At very low Re the nonlinear terms are tiny; Newton should converge in ~2 steps."""
        W, bcu, forms = _lid_cavity_setup(Re=1.0)
        solver = SteadyStateSolver(W=W, bcu=bcu, forms=forms, verbose=False)
        UP0 = dolfin.Function(W)
        result = solver.newton(UP0, f=dolfin.Constant((0.0, 0.0)), max_iter=25)
        # The solve mutates and returns UP0. Velocity DOFs should be non-trivial.
        u, p = result.split(deepcopy=True)
        vel_max = u.vector().norm("linf")
        assert vel_max > 1e-6, "Newton returned a zero solution"

    def test_returns_same_object(self):
        W, bcu, forms = _lid_cavity_setup(Re=1.0)
        solver = SteadyStateSolver(W=W, bcu=bcu, forms=forms, verbose=False)
        UP0 = dolfin.Function(W)
        result = solver.newton(UP0, f=dolfin.Constant((0.0, 0.0)))
        assert result is UP0


# ── Picard ────────────────────────────────────────────────────────────────────


class TestPicard:
    def test_converges_on_stokes_problem(self):
        W, bcu, forms = _lid_cavity_setup(Re=1.0)
        solver = SteadyStateSolver(W=W, bcu=bcu, forms=forms, verbose=False)
        UP0 = dolfin.Function(W)
        result = solver.picard(
            UP0, f=dolfin.Constant((0.0, 0.0)), max_iter=20, tol=1e-8
        )
        u, p = result.split(deepcopy=True)
        vel_max = u.vector().norm("linf")
        assert vel_max > 1e-6


# ── Newton and Picard agree on Stokes ─────────────────────────────────────────


class TestNewtonPicardAgreement:
    def test_same_solution_at_low_re(self):
        """Newton and Picard must converge to the same solution at Re=1."""
        W, bcu, forms = _lid_cavity_setup(Re=1.0)

        UP_newton = dolfin.Function(W)
        SteadyStateSolver(W=W, bcu=bcu, forms=forms, verbose=False).newton(
            UP_newton, f=dolfin.Constant((0.0, 0.0))
        )

        UP_picard = dolfin.Function(W)
        SteadyStateSolver(W=W, bcu=bcu, forms=forms, verbose=False).picard(
            UP_picard, f=dolfin.Constant((0.0, 0.0)), max_iter=50, tol=1e-10
        )

        diff = (UP_newton.vector() - UP_picard.vector()).norm("l2")
        ref = UP_newton.vector().norm("l2")
        assert diff / ref < 1e-3, f"Newton/Picard mismatch: rel diff={diff / ref:.3e}"
