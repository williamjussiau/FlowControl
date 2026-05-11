"""Variational forms for incompressible Navier-Stokes.

All methods return pure UFL forms — no assembly, no solvers, no side-effects.
The caller is responsible for lhs/rhs splitting, assembly, and applying BCs.

Typical usage::

    forms = NSForms(W, Re=100.0, dt=0.005, is_nonlinear=True, shift=0.0)
    F1 = forms.transient(order=1, U0=U0, u_n=u_n, u_nn=None, f=f)
    F2 = forms.transient(order=2, U0=U0, u_n=u_n, u_nn=u_nn, f=f)
    F0 = forms.steady(UP0, f=f)
    a_pic, L_pic = forms.picard(U0_velocity, f=f)
"""

import logging
from typing import Any, Optional

import dolfin
import ufl
from dolfin import div, dot, dx, inner, nabla_grad

# UFL_Argument is not part of dolfin's public API (it lives in ufl).
# Use this alias to annotate trial/test function arguments.
UFL_Argument = Any

logger = logging.getLogger(__name__)


class NSForms:
    """Pure UFL variational forms for incompressible linearized Navier-Stokes.

    Parameters
    ----------
    W:
        Mixed Taylor-Hood function space (velocity × pressure).
    Re:
        Reynolds number.
    dt:
        Time step (used as a dolfin.Constant so it can be updated in place).
    is_nonlinear:
        Include nonlinear advection terms in transient forms.
    shift:
        Spectral shift applied to the linear operator (for eigenvalue problems).
    """

    def __init__(
        self,
        W: dolfin.FunctionSpace,
        Re: float,
        dt: float,
        is_nonlinear: bool = True,
        shift: float = 0.0,
    ) -> None:
        self.W = W
        self.invRe = dolfin.Constant(1.0 / Re)
        self.dt = dolfin.Constant(dt)
        self.is_nonlinear = is_nonlinear
        self.shift = dolfin.Constant(shift)

    # ── Public API ────────────────────────────────────────────────────────────

    def transient(
        self,
        order: int | str,
        U0: dolfin.Function,
        u_n: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
        u_nn: Optional[dolfin.Function] = None,
        f_n: Optional[dolfin.Function] = None,
    ) -> ufl.Form:
        """Linearized transient NS variational form of given BDF order.

        The linearization is around the base flow U0. The nonlinear advection
        terms use u_n (and u_nn for order 2) evaluated at previous time steps.

        Parameters
        ----------
        order:
            Time-integration order: 1 (BDF1/Euler), 2 (BDF2), or ``"cn"``
            (Crank-Nicolson).
        U0:
            Base-flow velocity field (steady state).
        u_n:
            Velocity perturbation at previous time step.
        f:
            Body-force term at the current step (sum of FORCE-type actuator
            expressions, or zero).
        u_nn:
            Velocity perturbation two time steps back. Required when order=2.
        f_n:
            Body-force term at the previous step. Required when order is
            ``"cn"`` for second-order accurate body-force averaging.

        Returns
        -------
        ufl.Form
            Bilinear-minus-linear (a - L) variational form, suitable for
            ``dolfin.lhs`` / ``dolfin.rhs`` splitting.
        """
        u, p = dolfin.TrialFunctions(self.W)
        v, q = dolfin.TestFunctions(self.W)

        if order == 1:
            return self._order1(u, p, v, q, U0, u_n, f)
        elif order == 2:
            if u_nn is None:
                raise ValueError("u_nn is required for order-2 form")
            return self._order2(u, p, v, q, U0, u_n, u_nn, f)
        elif order == "cn":
            if f_n is None:
                raise ValueError("f_n is required for Crank-Nicolson form")
            return self._cn(u, p, v, q, U0, u_n, f, f_n)
        else:
            raise ValueError(f"order must be 1, 2, or 'cn', got {order}")

    def steady(
        self,
        UP0: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
    ) -> ufl.Form:
        """Nonlinear steady-state NS variational form for Newton iteration.

        The form is nonlinear in UP0: evaluating it at the steady state gives
        the zero residual. Pass it to ``dolfin.solve(F == 0, UP0, bcu, ...)``.

        Parameters
        ----------
        UP0:
            Current guess for the mixed (velocity, pressure) field.
            Modified in-place by ``dolfin.solve``.
        f:
            Body-force term.

        Returns
        -------
        ufl.Form
            Nonlinear residual form F such that F == 0 at steady state.
        """
        v, q = dolfin.TestFunctions(self.W)
        U0, P0 = dolfin.split(UP0)  # symbolic split — keeps link to UP0
        return (
            dot(dot(U0, nabla_grad(U0)), v) * dx
            + self.invRe * inner(nabla_grad(U0), nabla_grad(v)) * dx
            - P0 * div(v) * dx
            - q * div(U0) * dx
            - dot(f, v) * dx
        )

    def picard(
        self,
        U0: UFL_Argument,  # dolfin.as_vector(...) — a UFL expression, not a plain Function
        f: dolfin.Expression | dolfin.Constant,
    ) -> tuple[ufl.Form, ufl.Form]:
        """Linearized (Picard) steady-state form for fixed-point iteration.

        The nonlinear advection velocity is frozen at U0 (a UFL expression
        that typically wraps the current iterate). The form must be
        re-assembled at each Picard step so that the updated U0 is used.

        Parameters
        ----------
        U0:
            Advection velocity for the current iterate. Usually created as
            ``dolfin.as_vector((UP0[0], UP0[1]))`` so that it stays linked
            to UP0 symbolically and updates automatically when UP0 is assigned.
        f:
            Body-force term.

        Returns
        -------
        a:
            Bilinear form (linearized momentum + incompressibility).
        L:
            Linear form (body-force right-hand side).
        """
        u, p = dolfin.TrialFunctions(self.W)
        v, q = dolfin.TestFunctions(self.W)

        a = (
            dot(dot(U0, nabla_grad(u)), v) * dx
            + self.invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )
        L = dot(f, v) * dx

        return a, L

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cn(
        self,
        u: UFL_Argument,
        p: UFL_Argument,
        v: UFL_Argument,
        q: UFL_Argument,
        U0: dolfin.Function,
        u_n: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
        f_n: dolfin.Function,
    ) -> ufl.Form:
        """Crank-Nicolson (θ=½) linearized NS form.

        Linear stiff terms (diffusion + base-flow advection) are averaged
        between t^n and t^{n+1}.  The nonlinear perturbation advection is
        treated explicitly at t^n.  Pressure is fully implicit so that
        lhs/rhs splitting yields a well-posed saddle-point system.
        Self-starting: no BDF1 warm-up step needed.

        The body force is also averaged: ½(f^{n+1} + f^n), where f^n is the
        force applied at the previous step, cached as a dolfin.Function.
        """
        b0 = dolfin.Constant(1.0 if self.is_nonlinear else 0.0)
        half = dolfin.Constant(0.5)
        return (
            # Time derivative
            dot((u - u_n) / self.dt, v) * dx
            # Implicit half of linear terms (u unknown → LHS)
            + half * dot(dot(U0, nabla_grad(u)), v) * dx
            + half * dot(dot(u, nabla_grad(U0)), v) * dx
            + half * self.invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            # Explicit half of linear terms (u_n known → RHS)
            + half * dot(dot(U0, nabla_grad(u_n)), v) * dx
            + half * dot(dot(u_n, nabla_grad(U0)), v) * dx
            + half * self.invRe * inner(nabla_grad(u_n), nabla_grad(v)) * dx
            # Nonlinear perturbation advection (explicit)
            + b0 * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            # Pressure / incompressibility
            - p * div(v) * dx
            - div(u) * q * dx
            # Body force: CN-averaged ½(f^{n+1} + f^n) for second-order accuracy
            - half * dot(f, v) * dx
            - half * dot(f_n, v) * dx
            # Spectral shift
            - self.shift * dot(u, v) * dx
        )

    def _order1(
        self,
        u: UFL_Argument,
        p: UFL_Argument,
        v: UFL_Argument,
        q: UFL_Argument,
        U0: dolfin.Function,
        u_n: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
    ) -> ufl.Form:
        """BDF1 (backward-Euler) linearized NS form."""
        b0 = dolfin.Constant(1.0 if self.is_nonlinear else 0.0)
        return (
            # BDF1 time derivative: (u^{n+1} - u^n) / dt
            dot((u - u_n) / self.dt, v) * dx
            # Linear advection by base flow: (U0·∇)u
            + dot(dot(U0, nabla_grad(u)), v) * dx
            # Linearization term: (u·∇)U0
            + dot(dot(u, nabla_grad(U0)), v) * dx
            # Viscous diffusion: (1/Re) ∇u : ∇v
            + self.invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            # Nonlinear perturbation advection (explicit): (u_n·∇)u_n
            + b0 * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            # Pressure gradient
            - p * div(v) * dx
            # Incompressibility constraint
            - div(u) * q * dx
            # Body force
            - dot(f, v) * dx
            # Spectral shift σ·u
            - self.shift * dot(u, v) * dx
        )

    def _order2(
        self,
        u: UFL_Argument,
        p: UFL_Argument,
        v: UFL_Argument,
        q: UFL_Argument,
        U0: dolfin.Function,
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
    ) -> ufl.Form:
        """BDF2 linearized NS form."""
        b0 = dolfin.Constant(2.0 if self.is_nonlinear else 0.0)
        b1 = dolfin.Constant(-1.0 if self.is_nonlinear else 0.0)
        return (
            # BDF2 time derivative: (3u - 4u_n + u_{n-1}) / (2dt)
            dot((3 * u - 4 * u_n + u_nn) / (2 * self.dt), v) * dx
            # Linear advection by base flow: (U0·∇)u
            + dot(dot(U0, nabla_grad(u)), v) * dx
            # Linearization term: (u·∇)U0
            + dot(dot(u, nabla_grad(U0)), v) * dx
            # Viscous diffusion: (1/Re) ∇u : ∇v
            + self.invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            # Nonlinear advection (Adams-Bashforth): 2(u_n·∇)u_n - (u_{n-1}·∇)u_{n-1}
            + b0 * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            + b1 * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
            # Pressure gradient
            - p * div(v) * dx
            # Incompressibility constraint
            - div(u) * q * dx
            # Body force
            - dot(f, v) * dx
            # Spectral shift σ·u
            - self.shift * dot(u, v) * dx
        )
