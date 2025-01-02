import dolfin
from dolfin import inner, div, curl, sym, grad
import utils_flowsolver as flu

import numpy as np

# import scipy.sparse as spr
import sympy as sp

import logging

logger = logging.getLogger(__name__)


def get_mass_matrix(self, sparse=False, volume=True, uvp=False):
    """Compute the mass matrix associated to
    spatial discretization"""
    logger.info("Computing mass matrix Q...")
    up = dolfin.TrialFunction(self.W)
    vq = dolfin.TestFunction(self.W)

    M = dolfin.PETScMatrix()
    # volume integral or surface integral (unused)
    dOmega = self.dx if volume else self.ds

    mf = sum([up[i] * vq[i] for i in range(2 + uvp)]) * dOmega  # sum u, v but not p
    dolfin.assemble(mf, tensor=M)

    if sparse:
        return flu.dense_to_sparse(M)
    return M


def get_matrices_lifting(self, A, C, Q):
    """Return matrices A, B, C, Q resulting form lifting transform (Barbagallo et al. 2009)
    See get_Hw_lifting for details"""
    # Steady field with rho=1: S1
    logger.info("Computing steady actuated field...")
    self.actuator_expression.ampl = 1.0
    S1 = self.compute_steady_state_newton()
    S1v = S1.vector()

    # Q*S1 (as vector)
    sz = self.W.dim()
    QS1v = dolfin.Vector(S1v.copy())
    QS1v.set_local(
        np.zeros(
            sz,
        )
    )
    QS1v.apply("insert")
    Q.mult(S1v, QS1v)  # QS1v = Q * S1v

    # Bl = [Q*S1; -1]
    Bl = np.hstack((QS1v.get_local(), -1))  # stack -1
    Bl = np.atleast_2d(Bl).T  # as column

    # Cl = [C, 0]
    Cl = np.hstack((C, np.atleast_2d(0)))

    # Ql = diag(Q, 1)
    Qsp = flu.dense_to_sparse(Q)
    Qlsp = flu.spr.block_diag((Qsp, 1))

    # Al = diag(A, 0)
    Asp = flu.dense_to_sparse(A)
    Alsp = flu.spr.block_diag((Asp, 0))

    return Alsp, Bl, Cl, Qlsp


def get_Dxy(self):
    """Get derivation matrices Dx, Dy in V space
    such that Dx*u = u.dx(0), Dy*u = u.dx(1)"""
    u = dolfin.TrialFunction(self.V)
    ut = dolfin.TestFunction(self.V)
    Dx = dolfin.assemble(inner(u.dx(0), ut) * self.dx)
    Dy = dolfin.assemble(inner(u.dx(1), ut) * self.dx)
    return Dx, Dy


def compute_vorticity(self, u=None):
    """Compute vorticity field of given velocity field u"""
    if u is None:
        u = self.u_
    # should probably project on space of order n-1 --> self.P
    vorticity = flu.projectm(curl(u), V=self.V.sub(0).collapse())
    return vorticity


def compute_divergence(self, u=None):
    """Compute divergence field of given velocity field u"""
    if u is None:
        u = self.u_
    divergence = flu.projectm(div(u), self.P)
    return divergence


def stress_tensor(nu, u, p):
    """Compute stress tensor (eg for lift & drag)"""
    return 2.0 * nu * (sym(grad(u))) - p * dolfin.Identity(p.geometric_dimension())


def get_div0_u(fs, xloc, yloc, size):
    """Create velocity field with zero divergence"""
    # V = self.V
    P = fs.P

    # Define courant function
    xm, ym = sp.symbols("x[0], x[1]")
    rr = (xm - xloc) ** 2 + (ym - yloc) ** 2
    fpsi = 0.25 * sp.exp(-1 / 2 * rr / size**2)
    # Piecewise does not work too well
    # fpsi = sp.Piecewise(   (sp.exp(-1/2 * rr / sigm**2),
    # rr <= nsig**2 * sigm**2), (0, True) )
    dfx = fpsi.diff(xm, 1)
    dfy = fpsi.diff(ym, 1)

    # Take derivatives
    # psi = dolfin.Expression(sp.ccode(fpsi), element=V.ufl_element())
    dfx_expr = dolfin.Expression(sp.ccode(dfx), element=P.ufl_element())
    dfy_expr = dolfin.Expression(sp.ccode(dfy), element=P.ufl_element())

    # Check
    # psiproj = flu.projectm(psi, P)
    # flu.write_xdmf('psi.xdmf', psiproj, 'psi')
    # flu.write_xdmf('psi_dx.xdmf', flu.projectm(dfx_expr, P), 'psidx')
    # flu.write_xdmf('psi_dy.xdmf', flu.projectm(dfy_expr, P), 'psidy')

    # Make velocity field
    upsi = flu.projectm(dolfin.as_vector([dfy_expr, -dfx_expr]), fs.V)
    return upsi


def get_div0_u_random(fs, sigma=0.1, seed=0):
    """Create random velocity field with zero divergence"""
    # CG2 scalar
    P2 = fs.V.sub(0).collapse()

    # Make scalar potential field in CG2 (scalar)
    a0 = dolfin.Function(P2)
    np.random.seed(seed)
    a0.vector()[:] += sigma * np.random.randn(a0.vector()[:].shape[0])

    # Take curl, then by definition div(u0)=div(curl(a0))=0
    Ve = dolfin.VectorElement("CG", fs.mesh.ufl_cell(), 1)
    V1 = dolfin.FunctionSpace(fs.mesh, Ve)

    u0 = flu.projectm(curl(a0), V1)

    ##divu0 = flu.projectm(div(u0), self.P)
    return u0


class localized_perturbation_u(dolfin.UserExpression):
    """Perturbation localized in disk
    Use: u = dolfin.interpolate(localized_perturbation_u(), self.V)
    or something like that"""

    def eval(self, value, x):
        if (x[0] - -2.5) ** 2 + (x[1] - 0.1) ** 2 <= 1:
            value[0] = 0.05
            value[1] = 0.05
        else:
            value[0] = 0
            value[1] = 0

    def value_shape(self):
        return (2,)


# see end_simulation in flu
def print_progress(fs, runtime):
    """Single line to print progress"""
    logger.info(
        "--- iter: %5d/%5d --- time: %3.3f/%3.2f --- elapsed %5.5f ---"
        % (
            fs.iter,
            fs.params_time.num_steps,
            fs.t,
            fs.params_time.Tf + fs.params_time.Tstart,
            runtime,
        )
    )
