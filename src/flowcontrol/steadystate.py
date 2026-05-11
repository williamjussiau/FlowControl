"""Steady-state solver for incompressible Navier-Stokes.

Provides Newton and Picard iteration as a self-contained class that requires
no back-reference to FlowSolver — only the function space, boundary conditions,
and an NSForms instance.

Typical usage::

    forms = NSForms(W, Re=100.0, dt=0.005, is_nonlinear=True)
    solver = SteadyStateSolver(W, bcu=BC.bcu, forms=forms, verbose=True)

    UP0 = dolfin.Function(W)
    UP0.interpolate(initial_guess)

    # Warm start with Picard, then refine with Newton
    UP0 = solver.picard(UP0, f=f, max_iter=5, tol=1e-6)
    UP0 = solver.newton(UP0, f=f, max_iter=25)
"""

import logging

import dolfin

from flowcontrol.nsforms import NSForms

logger = logging.getLogger(__name__)


class SteadyStateSolver:
    """Newton and Picard solvers for the steady incompressible NS equations.

    Parameters
    ----------
    W:
        Mixed Taylor-Hood function space (velocity × pressure).
    bcu:
        List of DirichletBC for the *full* velocity field (not perturbation).
        These are applied both when solving the linear system and when
        evaluating the residual.
    forms:
        NSForms instance providing ``steady()`` and ``picard()`` form builders.
    verbose:
        If True, Newton prints residual at each iteration.
    """

    def __init__(
        self,
        W: dolfin.FunctionSpace,
        bcu: list[dolfin.DirichletBC],
        forms: NSForms,
        verbose: bool = True,
    ) -> None:
        self.W = W
        self.bcu = bcu
        self.forms = forms
        self.verbose = verbose

    # ── Public API ────────────────────────────────────────────────────────────

    def newton(
        self,
        UP0: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
        max_iter: int = 25,
    ) -> dolfin.Function:
        """Solve the steady NS equations with dolfin's built-in Newton solver.

        The solve is done in-place: UP0 is updated with the solution and also
        returned for convenience.

        Parameters
        ----------
        UP0:
            Initial guess as a mixed (velocity, pressure) Function in W.
            Modified in-place.
        f:
            Body-force term (e.g. from force-type actuators).
        max_iter:
            Maximum number of Newton iterations.

        Returns
        -------
        dolfin.Function
            Solution UP0 (same object, updated in-place).
        """
        F0 = self.forms.steady(UP0, f)
        solver_params = {
            "newton_solver": {
                "linear_solver": "mumps",
                "preconditioner": "default",
                "maximum_iterations": max_iter,
                "report": self.verbose,
            }
        }
        dolfin.solve(F0 == 0, UP0, self.bcu, solver_parameters=solver_params)
        return UP0

    def picard(
        self,
        UP0: dolfin.Function,
        f: dolfin.Expression | dolfin.Constant,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> dolfin.Function:
        """Solve the steady NS equations with fixed-point (Picard) iteration.

        Each iteration freezes the advection velocity at the previous solution
        and solves the resulting linear system. Convergence is typically slower
        than Newton but the basin of attraction is larger, making it useful as
        a warm-start before handing off to Newton.

        Parameters
        ----------
        UP0:
            Initial guess as a mixed Function in W. Updated in-place at each
            iteration and also returned for convenience.
        f:
            Body-force term.
        max_iter:
            Maximum number of Picard iterations.
        tol:
            Convergence tolerance on the normalised residual norm.

        Returns
        -------
        dolfin.Function
            Solution estimate after convergence or max_iter steps.
        """
        UP1 = dolfin.Function(self.W)

        # Symbolic reference to the velocity part of UP0. Because this is a
        # UFL expression (not a deep copy), it automatically reflects updates
        # to UP0 when the form is re-assembled at each iteration.
        U0 = dolfin.as_vector((UP0[0], UP0[1]))

        a, L = self.forms.picard(U0, f)
        bp = dolfin.assemble(L)  # zero RHS — assembled once, constant
        solver = dolfin.LUSolver("mumps")

        for i in range(max_iter):
            # Re-assemble LHS with updated U0 (reflects current UP0)
            Ap = dolfin.assemble(a)
            [bc.apply(Ap, bp) for bc in self.bcu]
            solver.solve(Ap, UP1.vector(), bp)

            # Relative change between iterates — computed before the assign so
            # UP0 still holds the previous solution
            diff = dolfin.norm(UP1.vector() - UP0.vector())
            base = dolfin.norm(UP0.vector())
            rel_err = diff / (base + 1e-14)

            UP0.assign(UP1)

            logger.info(f"Picard {i + 1}/{max_iter}  rel_err = {rel_err:.3e}")
            if rel_err < tol:
                logger.info(f"Picard converged (rel_err {rel_err:.3e} < tol {tol:.3e})")
                break

        return UP1
