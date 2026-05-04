"""Tests for flowcontrol.nsforms.NSForms — requires FEniCS/dolfin."""

import dolfin
import pytest
import ufl

from flowcontrol.nsforms import NSForms


@pytest.fixture(scope="module")
def setup():
    """Return (W, forms, U0, u_n, u_nn, f) built on a minimal mesh."""
    mesh = dolfin.UnitSquareMesh(4, 4)
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, P2 * P1)
    forms = NSForms(W, Re=100.0, dt=0.005, is_nonlinear=True, shift=0.0)
    U0 = dolfin.Function(W.sub(0).collapse())
    u_n = dolfin.Function(W.sub(0).collapse())
    u_nn = dolfin.Function(W.sub(0).collapse())
    f = dolfin.Constant((0.0, 0.0))
    return W, forms, U0, u_n, u_nn, f


# ── Construction ──────────────────────────────────────────────────────────────


class TestNSFormsConstruct:
    def test_builds_without_error(self, setup):
        _, forms, *_ = setup
        assert forms is not None

    def test_re_stored_as_inverse(self, setup):
        _, forms, *_ = setup
        assert float(forms.invRe) == pytest.approx(1.0 / 100.0)

    def test_dt_stored_as_constant(self, setup):
        _, forms, *_ = setup
        assert float(forms.dt) == pytest.approx(0.005)


# ── transient (order 1) ───────────────────────────────────────────────────────


class TestTransient:
    def test_order1_returns_form(self, setup):
        _, forms, U0, u_n, u_nn, f = setup
        F = forms.transient(order=1, U0=U0, u_n=u_n, f=f)
        assert isinstance(F, ufl.Form)

    def test_order2_returns_form(self, setup):
        _, forms, U0, u_n, u_nn, f = setup
        F = forms.transient(order=2, U0=U0, u_n=u_n, f=f, u_nn=u_nn)
        assert isinstance(F, ufl.Form)

    def test_order2_without_unn_raises(self, setup):
        _, forms, U0, u_n, u_nn, f = setup
        with pytest.raises(ValueError):
            forms.transient(order=2, U0=U0, u_n=u_n, f=f, u_nn=None)

    def test_cn_returns_form(self, setup):
        _, forms, U0, u_n, u_nn, f = setup
        F = forms.transient(order="cn", U0=U0, u_n=u_n, f=f)
        assert isinstance(F, ufl.Form)

    def test_unknown_order_raises(self, setup):
        _, forms, U0, u_n, u_nn, f = setup
        with pytest.raises(ValueError):
            forms.transient(order=3, U0=U0, u_n=u_n, f=f)


# ── steady ────────────────────────────────────────────────────────────────────


class TestSteady:
    def test_steady_returns_form(self, setup):
        W, forms, *_ = setup
        f = dolfin.Constant((0.0, 0.0))
        UP0 = dolfin.Function(W)
        F = forms.steady(UP0, f=f)
        assert isinstance(F, ufl.Form)


# ── picard ────────────────────────────────────────────────────────────────────


class TestPicard:
    def test_picard_returns_two_forms(self, setup):
        W, forms, *_ = setup
        UP0 = dolfin.Function(W)
        U0 = dolfin.as_vector((UP0[0], UP0[1]))
        f = dolfin.Constant((0.0, 0.0))
        a, L = forms.picard(U0, f=f)
        assert isinstance(a, ufl.Form)
        assert isinstance(L, ufl.Form)
