from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import dolfin


@dataclass
class FlowField:
    """
    Flow field for incompressible 2D flow.

    Args:
        u (dolfin.Function): velocity field in dolfin.FunctionSpace V
        p (dolfin.Function): pressure field in dolfin.FunctionSpace P
        up (dolfin.Function): mixed field in mixed dolfin.FunctionSpace W
    """

    u: dolfin.Function
    p: dolfin.Function
    up: dolfin.Function

    def __init__(self, up: dolfin.Function):
        u, p = up.split(deepcopy=True)
        self.u = u
        self.p = p
        self.up = up


@dataclass
class FlowFieldCollection:
    """Collection of fields used for computations in FlowSolver.

    Args:
        STEADY (FlowField): steady (full) U, P, UP as FlowField
        U0 (dolfin.Function): steady (full) U
        P0 (dolfin.Function): steady (full) P
        UP0 (dolfin.Function): steady (full) UP
        ic (FlowField): initial condition (pert) u, p, up as FlowField
        u_ (dolfin.Function): current (pert) field u
        p_ (dolfin.Function): current (pert) field p
        up_ (dolfin.Function): current (pert) field up
        u_n (dolfin.Function): previous (pert) field u
        u_nn (dolfin.Function): previous^2 (pert) field u
        p_n (dolfin.Function): previous (pert) field p
        Usave (dolfin.Function): (full) field U for saving -- preallocation
        Psave (dolfin.Function): (full) field P for saving -- preallocation
        Usave_n (dolfin.Function): (full) field UP for saving -- preallocation
    """

    # Base flow (full field)
    STEADY: Optional[FlowField] = None
    U0: Optional[dolfin.Function] = None
    P0: Optional[dolfin.Function] = None
    UP0: Optional[dolfin.Function] = None
    # Initial condition (pert field)
    ic: Optional[FlowField] = None
    # Current and past solutions (pert field)
    u_: Optional[dolfin.Function] = None
    p_: Optional[dolfin.Function] = None
    up_: Optional[dolfin.Function] = None
    u_n: Optional[dolfin.Function] = None
    u_nn: Optional[dolfin.Function] = None
    p_n: Optional[dolfin.Function] = None
    # Saved fields (full field)
    Usave: Optional[dolfin.Function] = None
    Psave: Optional[dolfin.Function] = None
    Usave_n: Optional[dolfin.Function] = None


@dataclass
class BoundaryConditions:
    bcu: list[dolfin.DirichletBC]
    bcp: list[dolfin.DirichletBC]
