"""Optimization utilities: algorithms, cost functions, FlowSolver cost evaluation.

Typical optimization loop
-------------------------
At each iteration, after running a FlowSolver simulation::

    Tnorm = fs.dt / (fs.t - fs.Tc)

    # Signal selection: full-state energy or output measurements
    signal = fs.timeseries["dE"]               # full-state (xQx)
    # signal = fs.timeseries["y_meas_0"]       # single output (yQy)

    # Cost assembly
    xQx = compute_signal_cost(signal, Tnorm, criterion="integral")
    uRu = compute_control_cost(
        fs.timeseries[["u_ctrl_1", "u_ctrl_2"]], Tnorm
    )
    J = xQx + u_penalty * uRu

    # Per-iteration save
    write_optim_csv(fs.timeseries, fs.savedir0, diverged=diverged, iteration=i)

At the end of the full optimization run::

    write_results(x_data, y_data, optim_path)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as so
import smt.surrogate_models as smod
from blackbox_opt.bb_optimize import bb_optimize
from mpi4py import MPI as mpi
from scipy.stats.qmc import Sobol
from smt.applications.ego import EGO
from smt.sampling_methods import LHS

from utils.mpi import get_rank

logger = logging.getLogger(__name__)

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
sz = comm.Get_size()


def fun_array(x: np.ndarray, fun: Callable[..., float], **kwargs) -> np.ndarray:
    """Evaluate a scalar cost function on a batch of points.

    Parameters
    ----------
    x :
        Input array of shape ``(n_points, dim)``.
    fun :
        Scalar-valued function ``R^n -> R``.

    Returns
    -------
    np.ndarray
        Output array of shape ``(n_points, 1)``.
    """
    npt = x.shape[0]
    out = np.zeros((npt, 1))
    for i in range(npt):
        J = fun(x[i, :], **kwargs)
        out[i, :] = J
    return out


def parallel_function_wrapper(
    x: np.ndarray, stop_all: list[int], fun: Callable
) -> float:
    """Worker-loop wrapper for MPI-collective cost function evaluation.

    Must be called simultaneously by all MPI ranks.  Rank 0 drives the
    optimizer and proposes points; all ranks collaborate on each FEniCS
    solve.  Rank 0 signals termination by broadcasting ``stop_all=[1]``.

    Parameters
    ----------
    x :
        Candidate point (only rank 0's value is used after broadcast).
    stop_all :
        Mutable single-element list used as a stop flag — ``[0]`` to evaluate,
        ``[1]`` to terminate all ranks.
    fun :
        MPI-collective cost function; must be called on all ranks simultaneously.

    Returns
    -------
    float
        Reduced cost value on rank 0; ``0`` on all other ranks.
    """
    stop_all[0] = comm.bcast(stop_all[0], root=0)
    f = 0
    x = comm.bcast(x, root=0)
    if stop_all[0] == 0:
        fe = fun(x)
        # All ranks compute the same fe (collective FEM solve); dividing by sz
        # before MPI.SUM recovers the single scalar value on rank 0.
        f = comm.reduce(fe / sz, op=mpi.SUM, root=0)
        if rank == 0:
            logger.debug("arg=%s >>> cost=%s", x, f)
    else:
        logger.debug("stopping function evaluation on rank %d", rank)
    return f


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


def cummin(
    y: np.ndarray, return_index: bool = True
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Return the cumulative minimum of a column vector.

    Parameters
    ----------
    y :
        Input array of shape ``(n, 1)``.
    return_index :
        If ``True``, also return the index of first occurrence of each minimum.

    Returns
    -------
    y_cummin :
        Cumulative minimum, shape ``(n, 1)``.
    idx :
        Indices of first occurrence, shape ``(n,)``.  Only returned when
        ``return_index=True``.
    """
    y_cummin = np.minimum.accumulate(y)
    if return_index:
        # Outer comparison (n,1) vs (1,n) → (n,n): row i marks which original
        # index achieved the cumulative minimum at step i.
        where_cummin = np.isclose(y_cummin, y.T).astype(int)
        idx = where_cummin.argmax(1)
        return y_cummin, idx
    return y_cummin


_SCIPY_METHODS = {
    "nm": "Nelder-Mead",
    "cobyla": "COBYLA",
    "bfgs": "BFGS",
    "slsqp": "SLSQP",
}


def _minimize_bo(costfun: Callable[..., float], options: dict) -> object:
    """Run Bayesian Optimization via SMT EGO with MPI-collective cost evaluation.

    Rank 0 drives the EGO loop; all other ranks spin in a worker loop until
    rank 0 signals termination (see :func:`parallel_function_wrapper`).

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


def write_results(
    x_data: np.ndarray, y_data: np.ndarray, optim_path: str | Path, verbose: bool = True
) -> None:
    """Write all optimization evaluations and their cumulative minimum to CSV.

    Writes two files into ``optim_path`` with columns ``["J", "x0", "x1", ...]``:
    ``J_costfun.csv`` (all evaluations) and ``J_costfun_cummin.csv``
    (best-so-far at each iteration).

    Parameters
    ----------
    x_data :
        Parameter vectors, shape ``(n_iter, dim)``.
    y_data :
        Cost values, length ``n_iter``.
    optim_path :
        Directory to write files into.
    verbose :
        If ``True``, log the output path.
    """
    optim_path = Path(optim_path)
    x_data_wr = np.array(x_data)
    y_data_wr = np.atleast_2d(np.array(y_data)).T
    dim = x_data_wr.shape[1]
    colnames = ["J"] + [f"x{i}" for i in range(dim)]

    df = pd.DataFrame(data=np.hstack((y_data_wr, x_data_wr)), columns=colnames)
    if verbose:
        logger.info("Logging results to: %s", optim_path)
    df.to_csv(optim_path / "J_costfun.csv", index=False)

    y_cummin, idx_cummin = cummin(y_data_wr, return_index=True)
    x_cummin = x_data_wr[idx_cummin, :]
    df_cummin = pd.DataFrame(data=np.hstack((y_cummin, x_cummin)), columns=colnames)
    df_cummin.to_csv(optim_path / "J_costfun_cummin.csv", index=False)


def sobol_sample(
    ndim: int,
    npt: int,
    xlimits: np.ndarray | None = None,
    skip: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate samples from a Sobol low-discrepancy sequence.

    Parameters
    ----------
    ndim :
        Dimension of each sample point.
    npt :
        Number of points to generate.
    xlimits :
        Optional bounds, shape ``(ndim, 2)`` or ``(2, ndim)``.  Samples are
        in ``[0, 1]^ndim`` when ``None``.
    skip :
        Number of initial Sobol points to skip (default 1000 avoids the
        low-uniformity warm-up region of the sequence).
    seed :
        If not ``None``, add a random offset to ``skip`` for a different
        sub-sequence each run.

    Returns
    -------
    np.ndarray
        Sample array of shape ``(npt, ndim)``.
    """
    engine = Sobol(d=ndim, scramble=False)
    skip = int(skip)
    if seed is not None:
        rng = np.random.default_rng(seed)
        skip += int(rng.integers(10000))
    if skip > 0:
        engine.fast_forward(skip)
    X = engine.random(npt)
    if xlimits is not None:
        xlimits = np.array(xlimits)
        if xlimits.shape == (2, ndim):
            xlimits = xlimits.T
        if xlimits.shape != (ndim, 2):
            raise ValueError(
                f"xlimits has wrong shape {xlimits.shape}, expected ({ndim}, 2)"
            )
        X *= xlimits[:, 1] - xlimits[:, 0]
        X += xlimits[:, 0]
    return X


# --- FlowSolver cost evaluation ---


def compute_signal_cost(
    signal: pd.Series, Tnorm: float, criterion: str, scaling: Callable | None = None
) -> float:
    """Compute integral or terminal cost of a 1D timeseries signal.

    Parameters
    ----------
    signal :
        Time series of signal values.
    Tnorm :
        Time normalisation factor (``fs.dt / (fs.t - fs.Tc)``).
    criterion :
        ``'integral'`` for time-averaged cost or ``'terminal'`` for final value.
    scaling :
        Optional function applied to the signal before aggregation.

    Returns
    -------
    float
        Scalar cost value.

    Raises
    ------
    ValueError
        If ``criterion`` is not ``'integral'`` or ``'terminal'``.
    """
    if criterion not in ("integral", "terminal"):
        raise ValueError(
            f"Unknown criterion {criterion!r}: expected 'integral' or 'terminal'."
        )
    if scaling is None:

        def scaling(x):
            return x

    if criterion == "integral":
        return float(np.sum(scaling(signal)) * Tnorm)
    else:
        return float(scaling(signal.iloc[-1]))


def compute_control_cost(u_ctrl: pd.Series | pd.DataFrame, Tnorm: float) -> float:
    """Compute time-normalised control effort ``∫‖u‖²dt``.

    Parameters
    ----------
    u_ctrl :
        Control input time series.  Pass all actuator columns; all channels
        are summed (e.g. ``timeseries[["u_ctrl_1", "u_ctrl_2"]]``).
    Tnorm :
        Time normalisation factor (``fs.dt / (fs.t - fs.Tc)``).

    Returns
    -------
    float
        Scalar control cost.
    """
    return float(np.sum(u_ctrl**2) * Tnorm)


def write_optim_csv(
    timeseries: pd.DataFrame, savedir: str | Path, diverged: bool, iteration: int
) -> None:
    """Write the timeseries CSV for one controller evaluation.

    Parameters
    ----------
    timeseries :
        ``fs.timeseries`` DataFrame.
    savedir :
        Root save directory.
    diverged :
        If ``True``, appends ``_DIVERGED`` to the filename.
    iteration :
        Zero-padded iteration index used in the filename.
    """
    suffix = "_DIVERGED" if diverged else ""
    filename = f"timeseries_iter_{iteration:04d}{suffix}.csv"
    timeseries_path = Path(savedir) / "timeseries" / filename
    if get_rank() == 0:
        timeseries.to_csv(timeseries_path, index=False)
