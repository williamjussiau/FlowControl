"""FEniCS/dolfin utility functions: projections, field helpers, DOF maps, CPP string helpers."""

import functools
import logging
import time

import dolfin
import numpy as np

from utils.mpi import get_rank

logger = logging.getLogger(__name__)

# Shortcut to define projection with MUMPS
projectm = functools.partial(dolfin.project, solver_type="mumps")


def apply_fun(u, fun):
    """Apply a numpy function to a dolfin.Function vector."""
    return fun(u.vector().get_local())


def show_max(u, name=""):
    """Log the max value of a dolfin.Function."""
    logger.info('Max of vector "%s" is : %f' % (name, apply_fun(u, np.max)))


def print0(*args, **kwargs):
    """Log on MPI rank 0 only."""
    if get_rank() == 0:
        logger.info(*args, **kwargs)


def expression_to_dolfin_function(
    expression: dolfin.Expression, functionSpace: dolfin.FunctionSpace, interp=True
) -> dolfin.Function:
    """Convert a dolfin.Expression to a dolfin.Function by interpolation or projection."""
    if interp:
        f = dolfin.Function(functionSpace)
        f.interpolate(expression)
    else:
        f = projectm(expression, functionSpace)
    return f


# --- CPP string helpers for dolfin boundary definitions ---


def near_cpp(x: str, xnear: float | str, tol: str = "MESH_TOL"):
    return f"near({x}, {xnear}, {tol})"


def between_cpp(x: str, xmin: str, xmax: str, tol: str = "0.0"):
    return f"{x}>={xmin}-{tol} && {x}<={xmax}+{tol}"


def or_cpp():
    return " || "


def and_cpp():
    return " && "


def on_boundary_cpp():
    return "on_boundary"


# --- FlowSolver helpers ---


def get_subspace_dofs(W):
    """Return a dict mapping subspace name to DOF indices for W = (u, v, p)."""

    def get_dofs(V):
        return np.array(V.dofmap().dofs(), dtype=int)

    return {
        "u": get_dofs(W.sub(0).sub(0)),
        "v": get_dofs(W.sub(0).sub(1)),
        "p": get_dofs(W.sub(1)),
    }


def summarize_timings(fs, t0=None, dolfin_timings=False):
    """Log timing summary for a completed FlowSolver run."""
    if fs.iter > 3:
        if t0 is not None:
            logger.info("Total time is: %f", time.time() - t0)
        logger.info("Iteration 1 time     --- %E", fs.timeseries.loc[1, "runtime"])
        logger.info("Iteration 2 time     --- %E", fs.timeseries.loc[2, "runtime"])
        logger.info(
            "Mean iteration time  --- %E", np.mean(fs.timeseries.loc[3:, "runtime"])
        )
        logger.info(
            "Time/iter/dof        --- %E",
            np.mean(fs.timeseries.loc[3:, "runtime"]) / fs.W.dim(),
        )
    if dolfin_timings:
        dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
