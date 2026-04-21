"""Optimization utilities: algorithms, cost functions, FlowSolver cost evaluation."""

import logging
import time

import numpy as np
import pandas as pd
import scipy.optimize as so
import smt.surrogate_models as smod
import sobol_seq as qmc
from blackbox_opt.bb_optimize import bb_optimize
from blackbox_opt.DFO_src.dfo_tr import params
from mpi4py import MPI as mpi
from smt.applications.ego import EGO
from smt.sampling_methods import LHS

from utils.mpi import get_rank

logger = logging.getLogger(__name__)

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
sz = comm.Get_size()


def fun_array(x, fun, **kwargs):
    """Evaluate function fun on n points at once.
    x must be a 2d array (rows = nr of points, cols = dimension)"""
    npt = x.shape[0]
    out = np.zeros((npt, 1))
    for i in range(npt):
        J = fun(x[i, :], **kwargs)
        out[i, :] = J
    return out


def costfun(x, verbose=True, allout=False):
    """Evaluate cost function on one point x"""
    xlim = 1
    if x <= xlim and x >= -xlim:
        f = x**2
    else:
        f = xlim
    if verbose:
        print("costfun: evaluation %2.10f +++ %2.10f" % (x, f))
    if allout:
        return f, 777
    return f


def parallel_function_wrapper(x, stop_all, fun):
    """Allows for the evaluation of fun in parallel in an outer process"""
    stop_all[0] = comm.bcast(stop_all[0], root=0)
    f = 0
    x = comm.bcast(x, root=0)
    if stop_all[0] == 0:
        fe = fun(x)
        f = comm.reduce(fe / sz, op=mpi.SUM, root=0)
        if rank == 0:
            print("from rank 0: arg=", x, " >>> cost=", f)
    else:
        print("##### Stopping function evaluation on process: ", rank)
    return f


def construct_simplex(x0, rectangular=True, edgelen=1):
    """Construct simplex around x0 to initialize Nelder-Mead algorithm.
    A rectangular simplex has x0 as a vertex and edgelen[i]*e_i as edges."""
    x0 = x0.ravel()
    n = x0.shape[0]

    if type(edgelen) is float:
        edgelen = [edgelen] * n

    if rectangular:
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for ii in range(1, n + 1):
            e_i = np.zeros((n,))
            e_i[ii - 1] = 1
            simplex[ii] = x0 + e_i * edgelen[ii - 1]
    else:
        simplex = np.vstack((np.zeros((1, n)), np.diag(edgelen)))
        a = 1 / (n + 1)
        simplex = simplex - a + x0

    return simplex


def nm_select_evaluated_points(x_best, x_all, y_all, verbose=False):
    """For NM algorithm: from best-so-far points x_best retrieve corresponding
    value of cost function."""
    uidx = np.unique(x_best, axis=0, return_index=True)[1]
    alv = [x_best[index] for index in sorted(uidx)]

    x_good = alv
    y_good = [0] * len(x_good)
    for ii, el in enumerate(x_good):
        for jj in range(len(x_all)):
            if np.all(x_all[jj] == el):
                if verbose:
                    print("Best-so-far: idx=", jj, " - value=", y_all[jj])
                y_good[ii] = y_all[jj]

    return x_good, y_good


def cummin(y, return_index=True):
    """Return cumulative minimum of 1D array y, along with indices of cummin"""
    y_cummin = np.minimum.accumulate(y)
    if return_index:
        where_cummin = np.isclose(y_cummin, y.T).astype(int)
        idx = where_cummin.argmax(1)
        return y_cummin, idx
    return y_cummin


def minimize(costfun, x0, alg, options, verbose=True):
    """Wrapper for launching optimization algorithm.
    Supported algorithms: Scipy (NM, COBYLA, BFGS, SLSQP), DFO, BO."""
    tstart = time.time()
    alg = alg.lower()
    default_options = optimizer_default_options(alg=alg)
    options["disp"] = verbose
    options = optimizer_check_options(default_options, options)
    if alg == "nm":
        res = so.minimize(fun=costfun, x0=x0, method="Nelder-Mead", options=options)
    if alg == "cobyla":
        res = so.minimize(fun=costfun, x0=x0, method="COBYLA", options=options)
    if alg == "bfgs":
        res = so.minimize(fun=costfun, x0=x0, method="BFGS", options=options)
    if alg == "slsqp":
        res = so.minimize(fun=costfun, x0=x0, method="SLSQP", options=options)
    if alg == "dfo":
        res = bb_optimize(func=costfun, x_0=x0, alg="DFO", options=options)
        res.nfev = res.func_eval
    if alg == "bo":
        sampling = LHS(xlimits=options["xlimits"], random_state=options["random_state"])
        xdoe = sampling(options["n_doe"])
        ydoe = fun_array(xdoe, costfun)

        criterion = options["criterion"]
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

        costfun_npt = lambda x: fun_array(x, costfun)
        costfun_parallel_smt = lambda x: parallel_function_wrapper(x, [0], costfun_npt)

        if rank == 0:
            stop = [0]
            x = np.zeros(1)
            x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(
                fun=costfun_parallel_smt
            )
            stop = [1]
            parallel_function_wrapper(x, stop, costfun_npt)
        else:
            stop = [0]
            x = np.zeros(1)
            while stop[0] == 0:
                parallel_function_wrapper(x, stop, costfun_npt)
        res = type("obj", (object,), {"nfev": options["n_doe"] + options["n_iter"]})
    tend = time.time()
    if verbose:
        print(
            "Total time is {} seconds with ".format(tend - tstart) + alg + (" method.")
        )
    return res


def optimizer_default_options(alg):
    """Define default algorithm parameters."""
    alg = alg.lower()
    maxfev = 100
    if alg == "nm":
        options = {
            "maxiter": None,
            "maxfev": maxfev,
            "disp": False,
            "return_all": True,
            "initial_simplex": None,
            "xatol": 1e-4,
            "fatol": 1e-4,
            "adaptive": True,
        }
    if alg == "cobyla":
        options = {"rhobeg": 1.0, "maxiter": maxfev, "disp": False, "catol": 0.0002}
    if alg == "bfgs":
        options = {
            "gtol": 1e-05,
            "norm": np.inf,
            "eps": 1.4901161193847656e-08,
            "maxiter": maxfev,
            "disp": True,
            "return_all": True,
            "finite_diff_rel_step": None,
        }
    if alg == "slsqp":
        options = {
            "maxiter": maxfev,
            "ftol": 1e-06,
            "iprint": 1,
            "disp": True,
            "eps": 1.4901161193847656e-08,
            "finite_diff_rel_step": None,
        }
    if alg == "dfo":
        tol = 1e-6
        options = dict(
            maxfev=maxfev,
            init_delta=0.5,
            tol_delta=tol,
            tol_f=1e-4,
            tol_norm_g=tol,
            sample_gen="auto",
            verbosity=1,
        )
    if alg == "bo":
        options = dict(
            theta0=[0.01],
            n_start=20,
            corr="squar_exp",
            theta_bounds=[0.01, 20],
            poly="constant",
            n_iter=10,
            criterion="EI",
            xlimits=[],
            verbose=False,
            random_state=1,
            n_doe=10,
        )
    return options


def optimizer_check_options(default_options, options):
    """Keep only entries in options that are present in default_options,
    to prevent scipy errors with incompatible options."""
    new_options = default_options
    for key in options.keys():
        if key in new_options.keys():
            new_options[key] = options[key]
    return new_options


def get_results(func, x_0, alg, options):
    """Call minimize and print the best point found."""
    res = minimize(func, x_0, alg, options)
    print("Printing result for function " + func.__name__ + ":")
    print(
        "best point: {}, with obj: {:.6f}".format(
            np.around(res.x.T, decimals=5), float(res.fun)
        )
    )
    if alg.lower() == "dfo":
        nf = res.func_eval
    else:
        nf = res.nfev
    print("nr f evaluations: ", nf)
    print("------------- " + alg + " Finished ----------------------\n")
    return res


def write_results(
    x_data,
    y_data,
    optim_path,
    colnames,
    x_cummin_all=None,
    y_cummin_all=None,
    idx_current_slice=0,
    nfev=0,
    verbose=True,
):
    """Write optimization results to csv file. Write all iterations and compute cummin."""
    x_data_wr = np.array(x_data)
    y_data_wr = np.atleast_2d(np.array(y_data)).T

    df = pd.DataFrame(data=np.hstack((y_data_wr, x_data_wr)), columns=colnames)
    if verbose:
        print("Logging J file to: ", optim_path)
    df.to_csv(optim_path + "J_costfun.csv", sep=",", index=False, mode="w", header=True)

    if x_cummin_all is not None:
        y_data_i = y_data_wr[idx_current_slice : idx_current_slice + nfev]
        y_cummin, idx_cummin = cummin(y_data_i, return_index=True)
        x_cummin = x_data_wr[idx_cummin + idx_current_slice, :]
        idx_current_slice = idx_current_slice + nfev
        x_cummin_all = np.vstack((x_cummin_all, x_cummin))
        y_cummin_all = np.vstack((y_cummin_all, y_cummin))
        df_cummin = pd.DataFrame(
            data=np.hstack((y_cummin_all, x_cummin_all)), columns=colnames
        )
        df_cummin.to_csv(
            optim_path + "J_costfun_cummin.csv",
            sep=",",
            index=False,
            mode="w",
            header=True,
        )
        return x_cummin_all, y_cummin_all, idx_current_slice


def run_parallel(rank, costfun_parallel):
    pass


def sobol_sample(ndim, npt, xlimits=None, skip=1e3, shuffle=False):
    """Generate samples from Sobol set.
    ndim: size of points, npt: number of points,
    xlimits: limits of generated points (initially in [0,1])"""
    if shuffle:
        np.random.seed(shuffle)
        skip += np.random.randint(1e4)
    X = qmc.i4_sobol_generate(dim_num=ndim, n=npt, skip=skip)
    if xlimits is not None:
        xlimits = np.array(xlimits)
        if xlimits.shape == (2, ndim):
            xlimits = xlimits.T
        try:
            X *= xlimits[:, 1] - xlimits[:, 0]
            X += xlimits[:, 0]
        except Exception:
            print("xlimits has wrong size: ", xlimits.shape, " - should be (ndim, 2).")
    return X


# --- FlowSolver cost evaluation ---

def compute_cost(
    fs,
    criterion,
    u_penalty,
    fullstate=True,
    scaling=None,
    verbose=True,
    diverged=False,
    diverged_penalty=50,
):
    """Compute cost associated to FlowSolver object.
    criterion: 'integral' or 'terminal'
    u_penalty: penalty on control energy
    fullstate: True -> xQx, False -> yQy"""
    if diverged:
        return diverged_penalty

    if scaling is None:

        def scaling(x):
            return x

    Tnorm = fs.dt / (fs.t - fs.Tc)

    if criterion == "integral":
        if fullstate:
            xQx = np.sum(scaling(fs.timeseries.loc[:, "dE"]))
        else:
            y_meas_str = fs.make_y_dataframe_column_name()
            y2_arr = (fs.timeseries.loc[:, y_meas_str] ** 2).to_numpy()
            xQx = np.sum(y2_arr.ravel())
        xQx *= Tnorm
    else:
        if fullstate:
            xQx = scaling(fs.timeseries.loc[:, "dE"].iloc[-1])
        else:
            y_meas_str = fs.make_y_dataframe_column_name()
            y2_end = (fs.timeseries.loc[:, y_meas_str].iloc[-1] ** 2).to_numpy()
            xQx = np.sum(y2_end)

    uRu = np.sum(fs.timeseries.u_ctrl**2) * Tnorm
    J = xQx + u_penalty * uRu

    if verbose:
        logger.info("grep [energy, regularization]: %f", [xQx, u_penalty * uRu])
        logger.info("grep full cost: %f", J)

    return J


def write_optim_csv(fs, x, J, diverged, write=True):
    """Write csv file for 1 controller of parameter x with cost J."""
    if write:
        x = x.reshape(-1)
        sl = ["{:.3f}".format(xi).replace(".", ",") for xi in x]
        jstr = "{:.5f}".format(J).replace(".", ",")
        file_extension = "_".join(sl)
        timeseries_path = (
            fs.savedir0
            + "timeseries/timeseries_"
            + "J="
            + jstr
            + "_"
            + "x="
            + file_extension
            + ".csv"
        )
        if get_rank() == 0:
            fs.timeseries.to_csv(timeseries_path, sep=",", index=False)
    return 1
