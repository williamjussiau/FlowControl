from __future__ import annotations

from dataclasses import dataclass

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
    STEADY: FlowField | None = None
    U0: dolfin.Function | None = None
    P0: dolfin.Function | None = None
    UP0: dolfin.Function | None = None
    # Initial condition (pert field)
    ic: FlowField | None = None
    # Current and past solutions (pert field)
    u_: dolfin.Function | None = None
    p_: dolfin.Function | None = None
    up_: dolfin.Function | None = None
    u_n: dolfin.Function | None = None
    u_nn: dolfin.Function | None = None
    p_n: dolfin.Function | None = None
    # Saved fields (full field)
    Usave: dolfin.Function | None = None
    Psave: dolfin.Function | None = None
    Usave_n: dolfin.Function | None = None


@dataclass
class BoundaryConditions:
    bcu: list[dolfin.DirichletBC]
    bcp: list[dolfin.DirichletBC]
    bcu: list[dolfin.DirichletBC]
    bcp: list[dolfin.DirichletBC]
