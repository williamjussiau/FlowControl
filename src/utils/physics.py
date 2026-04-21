"""Physics-based flow field utilities: vorticity, divergence, stress tensor,
divergence-free initial conditions."""

import logging

import dolfin
import numpy as np
import sympy as sp
from dolfin import curl, div, grad, sym

from utils.fem import projectm

logger = logging.getLogger(__name__)


def stress_tensor(nu, u, p):
    """Compute viscous stress tensor sigma = 2*nu*sym(grad(u)) - p*I."""
    return 2.0 * nu * sym(grad(u)) - p * dolfin.Identity(p.geometric_dimension())


def compute_vorticity(fs, u=None):
    """Compute vorticity field of velocity field u (defaults to fs.u_)."""
    if u is None:
        u = fs.u_
    return projectm(curl(u), V=fs.V.sub(0).collapse())


def compute_divergence(fs, u=None):
    """Compute divergence field of velocity field u (defaults to fs.u_)."""
    if u is None:
        u = fs.u_
    return projectm(div(u), fs.P)


def get_div0_u(fs, xloc, yloc, size):
    """Create velocity field with zero divergence using a Gaussian stream function."""
    xm, ym = sp.symbols("x[0], x[1]")
    rr = (xm - xloc) ** 2 + (ym - yloc) ** 2
    fpsi = 0.25 * sp.exp(-1 / 2 * rr / size**2)
    dfx_expr = dolfin.Expression(sp.ccode(fpsi.diff(xm, 1)), element=fs.P.ufl_element())
    dfy_expr = dolfin.Expression(sp.ccode(fpsi.diff(ym, 1)), element=fs.P.ufl_element())
    return projectm(dolfin.as_vector([dfy_expr, -dfx_expr]), fs.V)


def get_div0_u_random(fs, sigma=0.1, seed=0):
    """Create random velocity field with zero divergence via curl of a scalar potential."""
    P2 = fs.V.sub(0).collapse()
    a0 = dolfin.Function(P2)
    np.random.seed(seed)
    a0.vector()[:] += sigma * np.random.randn(a0.vector()[:].shape[0])
    V1 = dolfin.FunctionSpace(
        fs.mesh, dolfin.VectorElement("CG", fs.mesh.ufl_cell(), 1)
    )
    return projectm(curl(a0), V1)
