"""Tests for flowcontrol.nsforms.NSForms — requires FEniCS/dolfin."""

import dolfin
import pytest
import ufl

from flowcontrol.nsforms import NSForms


@pytest.fixture(scope="module")
def nsforms_setup(mixed_space):
    """Return (W, forms, U0, u_n, u_nn, f) built on a minimal mesh."""
    forms = NSForms(mixed_space, Re=100.0, dt=0.005, is_nonlinear=True, shift=0.0)
    V = mixed_space.sub(0).collapse()
    U0 = dolfin.Function(V)
    u_n = dolfin.Function(V)
    u_nn = dolfin.Function(V)
    f = dolfin.Constant((0.0, 0.0))
    return mixed_space, forms, U0, u_n, u_nn, f


# ── Construction ──────────────────────────────────────────────────────────────


class TestNSFormsConstruct:
    def test_builds_without_error(self, nsforms_setup):
        _, forms, *_ = nsforms_setup
        assert forms is not None

    def test_re_stored_as_inverse(self, nsforms_setup):
        _, forms, *_ = nsforms_setup
        assert float(forms.invRe) == pytest.approx(1.0 / 100.0)

    def test_dt_stored_as_constant(self, nsforms_setup):
        _, forms, *_ = nsforms_setup
        assert float(forms.dt) == pytest.approx(0.005)


# ── transient (order 1) ───────────────────────────────────────────────────────


class TestTransient:
    def test_order1_returns_form(self, nsforms_setup):
        _, forms, U0, u_n, u_nn, f = nsforms_setup
        F = forms.transient(order=1, U0=U0, u_n=u_n, f=f)
        assert isinstance(F, ufl.Form)

    def test_order2_returns_form(self, nsforms_setup):
        _, forms, U0, u_n, u_nn, f = nsforms_setup
        F = forms.transient(order=2, U0=U0, u_n=u_n, f=f, u_nn=u_nn)
        assert isinstance(F, ufl.Form)

    def test_order2_without_unn_raises(self, nsforms_setup):
        _, forms, U0, u_n, u_nn, f = nsforms_setup
        with pytest.raises(ValueError):
            forms.transient(order=2, U0=U0, u_n=u_n, f=f, u_nn=None)

    def test_cn_returns_form(self, nsforms_setup):
        _, forms, U0, u_n, u_nn, f = nsforms_setup
        F = forms.transient(order="cn", U0=U0, u_n=u_n, f=f)
        assert isinstance(F, ufl.Form)

    def test_unknown_order_raises(self, nsforms_setup):
        _, forms, U0, u_n, u_nn, f = nsforms_setup
        with pytest.raises(ValueError):
            forms.transient(order=3, U0=U0, u_n=u_n, f=f)


# ── steady ────────────────────────────────────────────────────────────────────


class TestSteady:
    def test_steady_returns_form(self, nsforms_setup):
        W, forms, *_ = nsforms_setup
        f = dolfin.Constant((0.0, 0.0))
        UP0 = dolfin.Function(W)
        F = forms.steady(UP0, f=f)
        assert isinstance(F, ufl.Form)


# ── picard ────────────────────────────────────────────────────────────────────


class TestPicard:
    def test_picard_returns_two_forms(self, nsforms_setup):
        W, forms, *_ = nsforms_setup
        UP0 = dolfin.Function(W)
        U0 = dolfin.as_vector((UP0[0], UP0[1]))
        f = dolfin.Constant((0.0, 0.0))
        a, L = forms.picard(U0, f=f)
        assert isinstance(a, ufl.Form)
        assert isinstance(L, ufl.Form)
