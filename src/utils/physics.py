"""Physics-based flow field utilities: vorticity, divergence, stress tensor,
divergence-free initial conditions."""

import logging
from typing import Any

import dolfin
import numpy as np
import sympy as sp
from dolfin import Function, FunctionSpace, curl, div, grad, sym

from utils.fem import projectm

logger = logging.getLogger(__name__)


def stress_tensor(nu: float, u: Function, p: Function) -> Any:
    """Compute viscous stress tensor sigma = 2*nu*sym(grad(u)) - p*I."""
    return 2.0 * nu * sym(grad(u)) - p * dolfin.Identity(p.geometric_dimension())


def compute_vorticity(u: Function, V: FunctionSpace) -> Function:
    """Compute vorticity (curl of u) projected onto scalar FunctionSpace V."""
    return projectm(curl(u), V=V)


def compute_divergence(u: Function, P: FunctionSpace) -> Function:
    """Compute divergence of u projected onto FunctionSpace P."""
    return projectm(div(u), P)


def get_div0_u(
    V: FunctionSpace,
    P: FunctionSpace,
    xloc: float,
    yloc: float,
    size: float,
) -> Function:
    """Create velocity field with zero divergence using a Gaussian stream function."""
    xm, ym = sp.symbols("x[0], x[1]")
    rr = (xm - xloc) ** 2 + (ym - yloc) ** 2
    fpsi = 0.25 * sp.exp(-1 / 2 * rr / size**2)
    dfx_expr = dolfin.Expression(sp.ccode(fpsi.diff(xm, 1)), element=P.ufl_element())
    dfy_expr = dolfin.Expression(sp.ccode(fpsi.diff(ym, 1)), element=P.ufl_element())
    return projectm(dolfin.as_vector([dfy_expr, -dfx_expr]), V)


def get_div0_u_random(
    V: FunctionSpace,
    mesh: dolfin.Mesh,
    sigma: float = 0.1,
    seed: int = 0,
) -> Function:
    """Create random velocity field with zero divergence via curl of a scalar potential."""
    P2 = V.sub(0).collapse()
    a0 = Function(P2)
    np.random.seed(seed)
    a0.vector()[:] += sigma * np.random.randn(a0.vector()[:].shape[0])
    V1 = FunctionSpace(mesh, dolfin.VectorElement("CG", mesh.ufl_cell(), 1))
    return projectm(curl(a0), V1)
