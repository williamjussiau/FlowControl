"""Optimizer wrappers: Nelder-Mead, COBYLA, BFGS, SLSQP, DFO, Bayesian Optimization.

These are thin wrappers around scipy.optimize, blackbox_opt, and SMT-EGO with a
unified interface and sensible defaults for flow-control optimization campaigns.
They depend on optional packages (smt, blackbox_opt) that are not required by
the core library.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import scipy.optimize as so
import smt.surrogate_models as smod
from blackbox_opt.bb_optimize import bb_optimize
from mpi4py import MPI as mpi
from smt.applications.ego import EGO
from smt.sampling_methods import LHS

from utils.optim import fun_array, parallel_function_wrapper

logger = logging.getLogger(__name__)

comm = mpi.COMM_WORLD
rank = comm.Get_rank()

_SCIPY_METHODS = {
    "nm": "Nelder-Mead",
    "cobyla": "COBYLA",
    "bfgs": "BFGS",
    "slsqp": "SLSQP",
}


def construct_simplex(
    x0: np.ndarray, rectangular: bool = True, edgelen: float | list[float] = 1
) -> np.ndarray:
    """Construct an initial simplex around x0 for Nelder-Mead.

    Parameters
    ----------
    x0 :
        Centre point of the simplex.
    rectangular :
        If ``True``, build a rectangular simplex with ``x0`` as a vertex and
        edges ``edgelen[i] * e_i``.  If ``False``, build a regular simplex.
    edgelen :
        Edge length(s).  Scalar applies uniformly; a list sets per-dimension lengths.

    Returns
    -------
    np.ndarray
        Simplex array of shape ``(n+1, n)``.
    """
    x0 = x0.ravel()
    n = x0.shape[0]

    if np.isscalar(edgelen):
        edgelen = [edgelen] * n

    if rectangular:
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for ii in range(n):
            simplex[ii + 1] = x0 + np.eye(n)[ii] * edgelen[ii]
    else:
        simplex = np.vstack((np.zeros((1, n)), np.diag(edgelen)))
        a = 1 / (n + 1)
        simplex = simplex - a + x0

    return simplex


def nm_select_evaluated_points(
    x_best: np.ndarray,
    x_all: list[np.ndarray],
    y_all: list[float],
    verbose: bool = False,
) -> tuple[list, list]:
    """Retrieve cost values for the best-so-far Nelder-Mead simplex vertices.

    Parameters
    ----------
    x_best :
        Sequence of best simplex vertices per NM iteration (``res.allvecs``).
    x_all :
        All evaluated points.
    y_all :
        Cost values corresponding to ``x_all``.

    Returns
    -------
    x_good :
        Deduplicated best-so-far points in order of first appearance.
    y_good :
        Corresponding cost values.
    """
    uidx = np.unique(x_best, axis=0, return_index=True)[1]
    x_good = [x_best[index] for index in sorted(uidx)]

    y_good = [None] * len(x_good)
    for ii, el in enumerate(x_good):
        for jj in range(len(x_all)):
            if np.allclose(x_all[jj], el):
                if verbose:
                    logger.debug("Best-so-far: idx=%d - value=%s", jj, y_all[jj])
                y_good[ii] = y_all[jj]
                break
        if y_good[ii] is None:
            raise ValueError(
                f"Point x_best[{ii}] not found in x_all — history mismatch."
            )

    return x_good, y_good


_DEFAULT_MAXFEV = 100
_SCIPY_EPS = np.sqrt(np.finfo(float).eps)

_DEFAULT_OPTIONS: dict[str, dict] = {
    "nm": {
        "maxiter": None,
        "maxfev": _DEFAULT_MAXFEV,
        "disp": False,
        "return_all": True,
        "initial_simplex": None,
        "xatol": 1e-4,
        "fatol": 1e-4,
        "adaptive": True,
    },
    "cobyla": {
        "rhobeg": 1.0,
        "maxiter": _DEFAULT_MAXFEV,
        "disp": False,
        "catol": 0.0002,
    },
    "bfgs": {
        "gtol": 1e-5,
        "norm": np.inf,
        "eps": _SCIPY_EPS,
        "maxiter": _DEFAULT_MAXFEV,
        "disp": True,
        "return_all": True,
        "finite_diff_rel_step": None,
    },
    "slsqp": {
        "maxiter": _DEFAULT_MAXFEV,
        "ftol": 1e-6,
        "iprint": 1,
        "disp": True,
        "eps": _SCIPY_EPS,
        "finite_diff_rel_step": None,
    },
    "dfo": {
        "maxfev": _DEFAULT_MAXFEV,
        "init_delta": 0.5,
        "tol_delta": 1e-6,
        "tol_f": 1e-4,
        "tol_norm_g": 1e-6,
        "sample_gen": "auto",
        "verbosity": 1,
    },
    "bo": {
        "theta0": [0.01],
        "n_start": 20,
        "corr": "squar_exp",
        "theta_bounds": [0.01, 20],
        "poly": "constant",
        "n_iter": 10,
        "criterion": "EI",
        "xlimits": [],
        "verbose": False,
        "random_state": 1,
        "n_doe": 10,
    },
}


def optimizer_default_options(alg: str) -> dict:
    """Return a fresh copy of default options for the given algorithm."""
    try:
        return dict(_DEFAULT_OPTIONS[alg])
    except KeyError:
        raise ValueError(f"Unknown optimization algorithm: {alg!r}") from None


def optimizer_check_options(default_options: dict, options: dict) -> dict:
    """Merge user options into defaults, silently ignoring unknown keys.

    Parameters
    ----------
    default_options :
        Full set of valid options with their default values.
    options :
        User-supplied overrides.

    Returns
    -------
    dict
        Merged options dict containing only keys present in ``default_options``.
    """
    return {k: options.get(k, v) for k, v in default_options.items()}


def _minimize_bo(costfun: Callable[..., float], options: dict) -> object:
    """Run Bayesian Optimization via SMT EGO with MPI-collective cost evaluation.

    Rank 0 drives the EGO loop; all other ranks spin in a worker loop until
    rank 0 signals termination (see :func:`~utils.optim.parallel_function_wrapper`).

    Parameters
    ----------
    costfun :
        MPI-collective cost function; must be called on all ranks simultaneously.
    options :
        BO options dict (see ``_DEFAULT_OPTIONS['bo']`` for valid keys).

    Returns
    -------
    object
        Result object; ``res.x`` and ``res.fun`` are populated on rank 0 only.
    """
    sampling = LHS(xlimits=options["xlimits"], random_state=options["random_state"])
    xdoe = sampling(options["n_doe"])
    ydoe = fun_array(xdoe, costfun)

    surrogate = smod.KRG(
        print_global=False,
        theta0=options["theta0"],
        n_start=options["n_start"],
        corr=options["corr"],
        theta_bounds=options["theta_bounds"],
        poly=options["poly"],
    )
    ego = EGO(
        n_iter=options["n_iter"],
        criterion=options["criterion"],
        xdoe=xdoe,
        ydoe=ydoe,
        xlimits=options["xlimits"],
        verbose=options["verbose"],
        n_start=options["n_start"],
        surrogate=surrogate,
    )

    def costfun_npt(x):
        return fun_array(x, costfun)

    def costfun_parallel_smt(x):
        return parallel_function_wrapper(x, [0], costfun_npt)

    res = type("obj", (object,), {"nfev": options["n_doe"] + options["n_iter"]})()
    if rank == 0:
        stop = [0]
        x_opt, y_opt, *_ = ego.optimize(fun=costfun_parallel_smt)
        stop = [1]
        parallel_function_wrapper(np.zeros(1), stop, costfun_npt)
        res.x = x_opt
        res.fun = float(y_opt)
    else:
        stop = [0]
        while stop[0] == 0:
            parallel_function_wrapper(np.zeros(1), stop, costfun_npt)
    return res


def minimize(
    costfun: Callable[..., float],
    x0: np.ndarray,
    alg: str,
    options: dict,
    verbose: bool = True,
) -> object:
    """Run an optimization algorithm on a cost function.

    Parameters
    ----------
    costfun :
        Scalar cost function to minimize.
    x0 :
        Initial parameter vector.
    alg :
        Algorithm name: ``'nm'``, ``'cobyla'``, ``'bfgs'``, ``'slsqp'``,
        ``'dfo'``, or ``'bo'``.
    options :
        Algorithm-specific options dict (merged with defaults).
    verbose :
        If ``True``, enable solver progress output.

    Returns
    -------
    object
        Optimization result object (scipy ``OptimizeResult`` or equivalent).

    Raises
    ------
    ValueError
        If ``alg`` is not a supported algorithm name.
    """
    tstart = time.time()
    alg = alg.lower()
    options = dict(options)  # don't mutate caller's dict
    options["disp"] = verbose
    options = optimizer_check_options(optimizer_default_options(alg), options)

    if alg in _SCIPY_METHODS:
        res = so.minimize(
            fun=costfun, x0=x0, method=_SCIPY_METHODS[alg], options=options
        )
    elif alg == "dfo":
        res = bb_optimize(func=costfun, x_0=x0, alg="DFO", options=options)
        res.nfev = res.func_eval
    elif alg == "bo":
        res = _minimize_bo(costfun, options)
    else:
        raise ValueError(f"Unknown optimization algorithm: {alg!r}")

    logger.info("Total time: %.1f s with %s method.", time.time() - tstart, alg)
    return res
