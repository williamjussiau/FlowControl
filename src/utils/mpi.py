"""MPI utilities for parallel FEniCS/dolfin simulations."""

import logging
import os

import dolfin
import numpy as np
from mpi4py import MPI as mpi

logger = logging.getLogger(__name__)


class MpiUtils:
    @staticmethod
    def get_rank():
        """Access MPI rank in COMM WORLD"""
        return mpi.COMM_WORLD.Get_rank()

    @staticmethod
    def check_process_rank():
        """Check process MPI rank"""
        comm = mpi.COMM_WORLD
        ip = comm.Get_rank()
        logger.info("================= Hello I am process %d", ip)

    @staticmethod
    def mpi4py_comm(comm):
        """Get mpi4py communicator"""
        try:
            return comm.tompi4py()
        except AttributeError:
            return comm

    @staticmethod
    def peval(f, x):
        """Parallel synced eval"""
        try:
            yloc = f(x)
        except RuntimeError:
            yloc = np.inf * np.ones(f.value_shape())
        comm = dolfin.MPI.comm_world
        yglob = np.zeros_like(yloc)
        comm.Allreduce(yloc, yglob, op=mpi.MIN)
        return yglob

    @staticmethod
    def peval1(f, x):
        """Parallel synced eval"""
        try:
            yloc = f(x)
        except RuntimeError:
            yloc = np.inf * np.ones(f.value_shape())
        comm = MpiUtils.mpi4py_comm(f.function_space().mesh().mpi_comm())
        yglob = np.zeros_like(yloc)
        comm.Allreduce(yloc, yglob, op=mpi.MIN)
        return yglob

    @staticmethod
    def peval2(f, x):
        """Parallel synced eval, v2"""
        mesh = f.function_space().mesh()
        comm = mesh.mpi_comm()
        if comm.size == 1:
            return f(*x)
        cell, distance = mesh.bounding_box_tree().compute_closest_entity(
            dolfin.Point(*x)
        )
        f_eval = f(*x) if distance < dolfin.DOLFIN_EPS else None
        comm = mesh.mpi_comm()
        computed_f = comm.gather(f_eval, root=0)
        if comm.rank == 0:
            global_f_evals = np.array(
                [y for y in computed_f if y is not None], dtype=np.double
            )
            assert np.all(np.abs(global_f_evals[0] - global_f_evals) < 1e-9)
            computed_f = global_f_evals[0]
        else:
            computed_f = None
        computed_f = comm.bcast(computed_f, root=0)
        return computed_f

    @staticmethod
    def set_omp_num_threads():
        """Memo for getting/setting OMP_NUM_THREADS, most likely does not work as is"""
        try:
            logger.info("nb threads was: %s", os.environ["OMP_NUM_THREADS"])
        except Exception as e:
            os.environ["OMP_NUM_THREADS"] = "1"
            raise (e)
        logger.info("nb threads is: %s", os.environ["OMP_NUM_THREADS"])

    @staticmethod
    def mpi_broadcast(x):
        """Broadcast y to MPI (shortcut but longer)"""
        y = dolfin.MPI.comm_world.bcast(x, root=0)
        return y
