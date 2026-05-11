"""Data containers for flow fields and simulation file paths.

Classes
-------
SimPaths            : frozen dataclass holding all file paths used by FlowSolver
FlowField           : paired (u, p, up) dolfin functions for a single flow state
FlowFieldCollection : all fields needed during time-stepping — base flow (U0/P0/UP0),
                      current/previous perturbations (u_, u_n, u_nn, …), and
                      pre-allocated save buffers (Usave, Psave, Usave_n)
BoundaryConditions  : simple holder for velocity and pressure DirichletBC lists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dolfin


@dataclass(frozen=True)
class SimPaths:
    """Typed container for all file paths used by FlowSolver."""

    U0: Path  # steady-state velocity (write/read)
    P0: Path  # steady-state pressure (write/read)
    U: Path  # velocity snapshot to load for restart (Trestartfrom)
    P: Path  # pressure snapshot to load for restart (Trestartfrom)
    Uprev: Path  # previous velocity snapshot to load for restart
    U_restart: Path  # velocity snapshot to write during this run (Tstart)
    Uprev_restart: Path  # previous velocity snapshot to write during this run
    P_restart: Path  # pressure snapshot to write during this run
    timeseries: Path  # CSV timeseries output
    metadata: Path  # JSON sidecar written at end of run (for restart discovery)
    steady_meta: Path  # JSON sidecar written alongside U0/P0 (mesh compatibility check)
    mesh: Path  # mesh file (read-only)


@dataclass
class FlowField:
    """Flow field for incompressible 2D flow.

    Attributes
    ----------
    up :
        Mixed (u, p) field in the mixed function space W.  Required at construction.
    u :
        Velocity field in V, split from ``up`` automatically.
    p :
        Pressure field in P, split from ``up`` automatically.
    """

    u: dolfin.Function = field(init=False)
    p: dolfin.Function = field(init=False)
    up: dolfin.Function

    def __post_init__(self) -> None:
        self.u, self.p = self.up.split(deepcopy=True)


@dataclass
class FlowFieldCollection:
    """Collection of all dolfin fields used during a FlowSolver simulation.

    Attributes
    ----------
    U0, P0, UP0 :
        Base-flow (full) velocity, pressure, and mixed field — linearization point.
    ic :
        Initial condition as a :class:`FlowField` (perturbation).
    u_, p_ :
        Current perturbation velocity and pressure fields.
    u_n, u_nn :
        Previous and second-previous perturbation velocity fields (for BDF2).
    p_n :
        Previous perturbation pressure field.
    Usave, Psave, Usave_n :
        Pre-allocated full-field buffers used when writing XDMF snapshots.
    """

    # Base flow (full field) — linearization point
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
    """Simple holder for velocity and pressure DirichletBC lists."""

    bcu: list[dolfin.DirichletBC]
    bcp: list[dolfin.DirichletBC]
