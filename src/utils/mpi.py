"""MPI utilities for parallel FEniCS/dolfin simulations."""

import logging

import dolfin
import numpy as np
from mpi4py import MPI as mpi

logger = logging.getLogger(__name__)


def get_rank() -> int:
    """Return MPI rank in COMM_WORLD."""
    return mpi.COMM_WORLD.Get_rank()


def check_process_rank():
    """Log the MPI rank of this process."""
    logger.info("================= Hello I am process %d", mpi.COMM_WORLD.Get_rank())


def peval(f, x):
    """Evaluate dolfin function f at point x, synced across MPI ranks.

    All ranks attempt f(x); ranks that don't own the point get a RuntimeError
    and fall back to inf. An Allreduce(MIN) then selects the real value.
    Uses COMM_WORLD — correct for single-mesh simulations.
    Note: incurs one RuntimeError per non-owning rank per call; see peval2 for
    a bounding-box approach that avoids this.
    """
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf * np.ones(f.value_shape())
    yglob = np.zeros_like(yloc)
    dolfin.MPI.comm_world.Allreduce(yloc, yglob, op=mpi.MIN)
    return yglob


def peval1(f, x):
    """Evaluate dolfin function f at point x, synced via the mesh communicator.

    Same approach as peval (all ranks attempt, RuntimeError on non-owning ranks,
    Allreduce(MIN) selects the real value), but uses the mesh's own communicator
    rather than COMM_WORLD — more correct when multiple meshes live on different
    sub-communicators.
    """
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf * np.ones(f.value_shape())
    try:
        comm = f.function_space().mesh().mpi_comm().tompi4py()
    except AttributeError:
        comm = f.function_space().mesh().mpi_comm()
    yglob = np.zeros_like(yloc)
    comm.Allreduce(yloc, yglob, op=mpi.MIN)
    return yglob


def peval2(f, x):
    """Evaluate dolfin function f at point x using bounding-box tree, synced across ranks.

    Only the rank owning the point evaluates f; result is broadcast to all ranks.
    Avoids the RuntimeError overhead of peval/peval1.
    TODO: cache the owning rank per sensor point to avoid repeated bounding-box queries.
    """
    mesh = f.function_space().mesh()
    comm = mesh.mpi_comm()
    if comm.size == 1:
        return f(*x)
    _, distance = mesh.bounding_box_tree().compute_closest_entity(dolfin.Point(*x))
    f_eval = f(*x) if distance < dolfin.DOLFIN_EPS else None
    computed_f = comm.gather(f_eval, root=0)
    if comm.rank == 0:
        global_f_evals = np.array(
            [y for y in computed_f if y is not None], dtype=np.double
        )
        assert np.all(np.abs(global_f_evals[0] - global_f_evals) < 1e-9)
        computed_f = global_f_evals[0]
    else:
        computed_f = None
    return comm.bcast(computed_f, root=0)


def mpi_broadcast(x):
    """Broadcast x from rank 0 to all ranks."""
    return dolfin.MPI.comm_world.bcast(x, root=0)


# Backward-compat alias — external code using flu.MpiUtils.get_rank() etc. still works.
class MpiUtils:
    get_rank = staticmethod(get_rank)
    check_process_rank = staticmethod(check_process_rank)
    peval = staticmethod(peval)
    peval1 = staticmethod(peval1)
    peval2 = staticmethod(peval2)
    mpi_broadcast = staticmethod(mpi_broadcast)
