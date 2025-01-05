from __future__ import print_function
import FlowSolverParameters
import dolfin
from dolfin import dot, nabla_grad, dx, inner, div
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod

import logging

import utils_flowsolver as flu
import utils_extract as flu2


logger = logging.getLogger(__name__)
FORMAT = (
    "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
)
logging.basicConfig(format=FORMAT, level=logging.INFO)


class FlowSolver(ABC):
    def __init__(
        self,
        params_flow: FlowSolverParameters.ParamControl,
        params_time: FlowSolverParameters.ParamTime,
        params_save: FlowSolverParameters.ParamSave,
        params_solver: FlowSolverParameters.ParamSolver,
        params_mesh: FlowSolverParameters.ParamMesh,
        verbose: int = 1,
    ):
        self.params_flow = params_flow
        self.params_time = params_time
        self.params_save = params_save
        self.params_solver = params_solver
        self.params_mesh = params_mesh
        self.verbose = verbose

        self.params_time.Tf = self.params_time.num_steps * self.params_time.dt

        self.params_flow.sensor_nr = self.params_flow.sensor_location.shape[0]

        ##
        self.paths = self.define_paths()
        self.mesh = self.make_mesh()
        self.n = dolfin.FacetNormal(self.mesh)
        self.V, self.P, self.W = self.make_function_spaces()
        self.boundaries = self.make_boundaries()  # @abstract
        self.mark_boundaries()
        self.actuator_expression = self.make_actuator()  # @abstract
        self.bc_p = self.make_bcs()  # @abstract
        ##

        self.u_ = dolfin.Function(self.V)
        self.first_step = True
        self.IC = self.make_IC(up=dolfin.Function(self.W), isdefined=False)
        self.steady = {"u": 0, "p": 0, "up": 0, "y": 0}

    def make_IC(self, up: dolfin.Function, isdefined=True) -> None:
        """Define initial state
        Not intended to be used by user directly (see self.initialize_time_stepping())
        """
        IC = dict()
        IC["up"] = up
        IC["u"], IC["p"] = up.split()
        IC["perturbation"] = None
        IC["isdefined"] = isdefined
        return IC

    def define_paths(self):  # TODO move to superclass
        """Define attribute (dict) containing useful paths (save, etc.)"""
        logger.debug("Currently defining paths...")

        path_out = self.params_save.path_out
        # start simulation from time...
        Tstart = self.params_time.Tstart
        # use older files starting from time...
        Trestartfrom = self.params_time.Trestartfrom

        def make_file_extension(T):
            return "_restart" + str(np.round(T, decimals=3)).replace(".", ",")

        file_start = make_file_extension(Tstart)
        file_restart = make_file_extension(Trestartfrom)

        ext_xdmf = ".xdmf"
        ext_csv = ".csv"

        path_out = self.params_save.path_out

        filename_u0 = path_out / "steady" / ("u0" + ext_xdmf)
        filename_p0 = path_out / "steady" / ("p0" + ext_xdmf)

        filename_u = path_out / ("u" + file_restart + ext_xdmf)
        filename_uprev = path_out / ("uprev" + file_restart + ext_xdmf)
        filename_p = path_out / ("p" + file_restart + ext_xdmf)

        filename_u_restart = path_out / ("u" + file_start + ext_xdmf)
        filename_uprev_restart = path_out / ("uprev" + file_start + ext_xdmf)
        filename_p_restart = path_out / ("p" + file_start + ext_xdmf)

        filename_timeseries = path_out / ("timeseries1D" + file_start + ext_csv)

        return {
            "u0": filename_u0,
            "p0": filename_p0,
            "u": filename_u,
            "p": filename_p,
            "uprev": filename_uprev,
            "u_restart": filename_u_restart,
            "uprev_restart": filename_uprev_restart,
            "p_restart": filename_p_restart,
            "timeseries": filename_timeseries,
            "mesh": self.params_mesh.meshpath,
        }

    def make_mesh(self):  # TODO move to superclass
        """Define mesh
        params_mesh contains either name of existing mesh
        or geometry parameters: xinf, yinf, xinfa, nx..."""

        if self.verbose:
            logger.debug(f"Mesh exists @: {self.params_mesh.meshpath}")
            logger.debug("--- Reading mesh...")

        mesh = dolfin.Mesh(dolfin.MPI.comm_world)
        with dolfin.XDMFFile(
            dolfin.MPI.comm_world, str(self.params_mesh.meshpath)
        ) as fm:
            fm.read(mesh)

        if self.verbose:
            logger.debug(f"Mesh has: {mesh.num_cells()} cells")

        return mesh

    def make_function_spaces(self):  # TODO move to superclass
        """Define function spaces (u, p) = (CG2, CG1)"""
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)  # was 'P'
        Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)  # was 'P'
        We = dolfin.MixedElement([Ve, Pe])
        V = dolfin.FunctionSpace(self.mesh, Ve)
        P = dolfin.FunctionSpace(self.mesh, Pe)
        W = dolfin.FunctionSpace(self.mesh, We)
        if self.verbose:
            logger.debug(
                f"Function Space [V(CG2), P(CG1)] has: {P.dim()}+{V.dim()}={W.dim()} DOFs"
            )

        return V, P, W

    def mark_boundaries(self):
        """Mark boundaries for num integration with assemble(F*dx(idx))"""
        bnd_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        cell_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        boundaries_idx = range(len(self.boundaries))

        for i, boundary_index in enumerate(boundaries_idx):
            self.boundaries.iloc[i].subdomain.mark(bnd_markers, boundary_index)
            self.boundaries.iloc[i].subdomain.mark(cell_markers, boundary_index)

        self.dx = dolfin.Measure("dx", domain=self.mesh, subdomain_data=cell_markers)
        self.ds = dolfin.Measure("ds", domain=self.mesh, subdomain_data=bnd_markers)
        self.boundary_markers = bnd_markers
        self.cell_markers = cell_markers
        self.boundaries["idx"] = list(boundaries_idx)

    def initialize_time_stepping(self, Tstart=0.0, IC=None):  # TODO move to superclass
        """Create varitional functions/forms & flush files & define u(0), p(0)
        If Tstart is 0: IC is set in IC (or, if IC is None: = 0)
        If Tstart is not 0: IC is computed from files
        """
        self.u_full = dolfin.Function(self.V)
        self.u_n_full = dolfin.Function(self.V)
        self.p_full = dolfin.Function(self.P)

        if self.verbose:
            logger.info(
                f"Starting or restarting from time: {Tstart} "
                f"with temporal scheme order: {self.params_time.restart_order}"
            )

        if Tstart == 0.0:
            logger.debug("START FROM ZERO")
            u_, p_, u_n, u_nn, p_n = self.initialize_with_IC(IC)
        else:
            logger.debug("START FROM NON ZERO")
            u_, p_, u_n, u_nn, p_n = self.initialize_at_time(Tstart)

        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n

        self.timeseries = self.initialize_timeseries()

    def initialize_with_IC(self, IC):
        self.order = 1

        if IC is None:  # then zero
            logger.debug("IC is set internally to 0")
            self.IC = self.make_IC(dolfin.Function(self.W))
        else:
            logger.debug("IC is already set by user")
            self.IC = self.make_IC(IC)

        # Impulse or state perturbation @ div0
        # Impulse if self.ic_add_perturbation is inf
        self.IC["perturbation"] = self.params_solver.ic_add_perturbation
        if np.isinf(self.IC["perturbation"]):
            # work in parallel?
            self.IC["up"].vector()[:] = self.get_B().reshape((-1,))
        elif self.IC["perturbation"] != 0.0:
            logger.debug(f"Found IC perturbation: {self.IC["perturbation"]}")
            udiv0 = flu2.get_div0_u(self, xloc=2, yloc=0, size=0.5)
            pert0 = self.split_merge(u=udiv0, p=self.p0)
            self.IC["up"].vector()[:] += self.IC["perturbation"] * pert0.vector()[:]
        self.IC["up"].vector().apply("insert")
        self.IC = self.make_IC(self.IC["up"])

        u_n = flu.projectm(v=self.IC["u"], V=self.V, bcs=self.bc_p["bcu"])
        u_nn = u_n.copy(deepcopy=True)
        p_n = flu.projectm(self.IC["p"], self.P)
        u_ = u_n.copy(deepcopy=True)
        p_ = p_n.copy(deepcopy=True)

        # Flush files and save IC as time_step 0
        if self.params_save.save_every:
            self.export_field_xdmf(
                u_n, u_nn, p_n, time=0, append=False, write_mesh=True
            )

        return u_, p_, u_n, u_nn, p_n

    def initialize_at_time(self, Tstart):
        self.order = self.params_time.restart_order  # 2

        idxstart = (self.params_time.Tstart - self.params_time.Trestartfrom) / (
            self.params_time.dt_old * self.params_save.save_every_old
        )
        idxstart = int(np.floor(idxstart))

        u_ = dolfin.Function(self.V)
        p_ = dolfin.Function(self.P)
        u_n = dolfin.Function(self.V)
        u_nn = dolfin.Function(self.V)
        p_n = dolfin.Function(self.P)

        flu.read_xdmf(self.paths["u"], u_, "u", counter=idxstart)
        flu.read_xdmf(self.paths["p"], p_, "p", counter=idxstart)
        flu.read_xdmf(self.paths["u"], u_n, "u", counter=idxstart)
        flu.read_xdmf(self.paths["uprev"], u_nn, "u_n", counter=idxstart)
        flu.read_xdmf(self.paths["p"], p_n, "p", counter=idxstart)

        # write in new file as first time step
        if self.params_save.save_every:
            self.export_field_xdmf(
                u_n,
                u_nn,
                p_n,
                time=self.params_time.Tstart,
                append=False,
                write_mesh=True,
            )

        # if perturbations, remove base flow from loaded file
        # because one prefers to write complete flow (not just perturbations)
        for ufield in [u_n, u_nn, u_]:
            ufield.vector()[:] -= self.u0.vector()[:]
        for pfield in [p_n, p_]:
            pfield.vector()[:] -= self.p0.vector()[:]

        # used for measurement y on IC in initialize_timeseries
        self.IC = self.make_IC(up=self.split_merge(u=u_, p=p_))

        return u_, p_, u_n, u_nn, p_n

    def initialize_timeseries(self):
        self.t = self.params_time.Tstart
        self.iter = 0
        self.IC["y"] = self.make_measurement(self.IC["u"])
        self.y_meas = self.IC["y"]
        y_meas_str = ["y_meas_" + str(i + 1) for i in range(self.params_flow.sensor_nr)]
        colnames = ["time", "u_ctrl"] + y_meas_str + ["dE", "runtime"]
        empty_data = np.zeros((self.params_time.num_steps + 1, len(colnames)))
        timeseries = pd.DataFrame(columns=colnames, data=empty_data)
        timeseries.loc[0, "time"] = self.params_time.Tstart
        self.assign_y_to_df(df=timeseries, y_meas=self.IC["y"], index=0)

        dEb = self.compute_energy()
        timeseries.loc[0, "dE"] = dEb
        return timeseries

    def make_solver(self, **kwargs):  # TODO could be utils
        """Define solvers"""
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def mark_varf(self, order, **kwargs):
        """Define equations"""
        if order == 1:
            F = self.make_varf_order1(**kwargs)
        elif order == 2:
            F = self.make_varf_order2(**kwargs)
        else:
            raise ValueError("Equation order not recognized")
            # There will be more important problems than this exception
        return F

    def make_varf_order1(self, up, vq, u0, u_n, shift):
        (u, p) = up
        (v, q) = vq
        b0_1 = 1 if self.params_flow.is_eq_nonlinear else 0
        iRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)
        F1 = (
            dot((u - u_n) / dt, v) * dx
            + dot(dot(u0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(u0)), v) * dx
            + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_1) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - shift * dot(u, v) * dx
        )
        return F1

    def make_varf_order2(self, up, vq, u0, u_n, u_nn, shift):
        (u, p) = up
        (v, q) = vq
        if self.params_flow.is_eq_nonlinear:
            b0_2, b1_2 = 2, -1
        else:
            b0_2, b1_2 = 0, 0
        iRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)
        F2 = (
            dot((3 * u - 4 * u_n + u_nn) / (2 * dt), v) * dx
            + dot(dot(u0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(u0)), v) * dx
            + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_2) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            + dolfin.Constant(b1_2) * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - shift * dot(u, v) * dx
        )
        return F2

    def prepare_systems(self, up, vq, u_n, u_nn):
        shift = dolfin.Constant(self.params_flow.shift)
        # 1st order integration
        F1 = self.mark_varf(
            order=1,
            up=up,
            vq=vq,
            u0=self.u0,
            u_n=u_n,
            shift=shift,
        )
        # 2nd order integration
        F2 = self.mark_varf(
            order=2,
            up=up,
            vq=vq,
            u0=self.u0,
            u_n=u_n,
            u_nn=u_nn,
            shift=shift,
        )

        self.assemblers = dict()
        self.solvers = dict()
        self.rhs = dolfin.Vector()
        for index, varf in enumerate([F1, F2]):
            order = index + 1
            a = dolfin.lhs(varf)
            L = dolfin.rhs(varf)
            systemAssembler = dolfin.SystemAssembler(a, L, self.bc_p["bcu"])
            solver = self.make_solver(order=order)
            operatorA = dolfin.Matrix()
            systemAssembler.assemble(operatorA)
            solver.set_operator(operatorA)
            self.assemblers[order] = systemAssembler
            self.solvers[order] = solver

    def step(self, u_ctrl: float) -> None:
        """Simulate system with perturbation formulation,
        possibly an actuation value, and a shift"""
        v, q = dolfin.TestFunctions(self.W)
        up = dolfin.TrialFunction(self.W)
        u, p = dolfin.split(up)

        up_ = dolfin.Function(self.W)
        u_, p_ = dolfin.split(up_)

        u_nn = self.u_nn
        u_n = self.u_n
        p_n = self.p_n

        if self.first_step:
            if self.verbose:
                logger.debug("Perturbations forms DO NOT exist: create...")
            self.prepare_systems((u, p), (v, q), u_n, u_nn)
            self.first_step = False

        # time
        t0i = time.time()

        # control
        self.actuator_expression.ampl = u_ctrl

        # Assign system of eqs
        assembler = self.assemblers[self.order]
        solver = self.solvers[self.order]

        try:
            assembler.assemble(self.rhs)
            solver.solve(up_.vector(), self.rhs)
            u_, p_ = up_.split(deepcopy=True)
            if self.solver_diverged(u_):
                raise RuntimeError()
        except RuntimeError:
            logger.critical("*** Solver diverged, Inf found ***")
            if not self.params.throw_error:
                logger.critical("*** Exiting step() ***")
                return -1  # -1 is error code
            else:
                raise RuntimeError("Failed solving: Inf found in solution")

        # Update time
        self.iter += 1
        self.t = self.params_time.Tstart + self.iter * self.params_time.dt
        self.order = 2

        # Assign
        self.u_ = u_
        self.p_ = p_
        self.up_ = up_
        # Shift
        self.u_nn.assign(u_n)
        self.u_n.assign(u_)
        self.p_n.assign(p_)

        ## Output
        # Probe
        self.y_meas = self.make_measurement()
        # Runtime
        runtime = time.time() - t0i
        if self.niter_multiple_of(self.iter, self.verbose):
            flu2.print_progress(self, runtime=runtime)
        # Timeseries
        self.log_timeseries(
            u_ctrl=u_ctrl,
            y_meas=self.y_meas,
            dE=self.compute_energy(),
            t=self.t,
            runtime=runtime,
        )

        # Export xdmf & csv
        if self.niter_multiple_of(self.iter, self.params_save.save_every):
            self.export_field_xdmf(u_n, u_nn, p_n, self.t)
            # this calls process 0 -> do async or what?
            self.write_timeseries()

        return self.y_meas

    def solver_diverged(self, field):
        return not np.isfinite(field.vector().get_local()[0])

    def niter_multiple_of(self, iter, divider):
        return divider and not iter % divider

    def split_merge(self, up=None, u=None, p=None):
        """Split or merge field(s)
        if instruction is split: up -> (u,p)
        if instruction is merge: (u,p) -> up
        FunctionAssigner(receiver, sender)"""
        if u is None:  # split up
            # probably equivalent to up.split() from dolfin
            fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            u = dolfin.Function(self.V)
            p = dolfin.Function(self.P)
            fa.assign([u, p], up)
            return (u, p)
        else:  # merge u, p
            fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
            up = dolfin.Function(self.W)
            fa.assign(up, [u, p])
            return up

    def export_field_xdmf(self, u_n, u_nn, p_n, time, append=True, write_mesh=False):
        # Reconstruct full field
        self.u_full.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
        self.u_n_full.vector()[:] = u_nn.vector()[:] + self.u0.vector()[:]
        self.p_full.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]

        if self.verbose:
            logger.debug(f"saving to files {self.params_save.path_out}")

        flu.write_xdmf(
            filename=self.paths["u_restart"],
            func=self.u_full,
            name="u",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths["uprev_restart"],
            func=self.u_n_full,
            name="u_n",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths["p_restart"],
            func=self.p_full,
            name="p",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )

    # Steady state
    def load_steady_state(self):
        u0 = dolfin.Function(self.V)
        p0 = dolfin.Function(self.P)
        flu.read_xdmf(self.paths["u0"], u0, "u")
        flu.read_xdmf(self.paths["p0"], p0, "p")

        # Assign u0, p0 >>> up0
        up0 = self.split_merge(u=u0, p=p0)

        self.u0 = u0  # full field (u+upert)
        self.p0 = p0
        self.up0 = up0
        self.steady["y"] = self.make_measurement(mixed_field=up0)

        self.Eb = (
            1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=self.mesh) ** 2
        )  # same as <up, Q@up>
        return u0, p0, up0

    def compute_steady_state(self, method="newton", u_ctrl=0.0, **kwargs):
        """Compute flow steady state with given steady control"""
        # Save old control value, just in case
        actuation_ampl_old = self.actuator_expression.ampl
        self.actuator_expression.ampl = u_ctrl

        # If start is zero (i.e. not restart): compute
        # Note : could add a flag 'compute_steady_state' to compute or read...
        if self.params_time.Tstart == 0:  # and compute_steady_state
            # Solve
            if method == "newton":
                up0 = self.compute_steady_state_newton(**kwargs)
            else:
                up0 = self.compute_steady_state_picard(**kwargs)

            u0, p0 = up0.split()

            # Save steady state
            if self.params_save.save_every:
                flu.write_xdmf(
                    self.paths["u0"],
                    u0,
                    "u",
                    time_step=0.0,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["p0"],
                    p0,
                    "p",
                    time_step=0.0,
                    append=False,
                    write_mesh=True,
                )
            if self.verbose:
                logger.debug(f"Stored base flow in: {self.params_save.path_out}")

            self.steady["y"] = self.make_measurement(mixed_field=up0)

        # If IC is not zero: read steady state (should exist - should check though...)
        else:
            u0, p0, up0 = self.load_steady_state()

        # Set old actuator amplitude
        self.actuator_expression.ampl = actuation_ampl_old

        # assign steady state
        self.up0 = up0
        self.u0 = u0
        self.p0 = p0
        # assign steady energy
        self.Eb = (
            1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=self.mesh) ** 2
        )  # same as <up, Q@up>

    def compute_steady_state_newton(
        self, max_iter: int = 25, initial_guess: dolfin.Function = None
    ):
        """Compute steady state with built-in nonlinear solver (Newton method)
        initial_guess is a (u,p)_0"""
        self.make_form_mixed_steady(initial_guess=initial_guess)
        if initial_guess is None:
            logger.info("Newton solver --- without initial guess")
        up_ = self.up_

        nl_solver_param = {
            "newton_solver": {
                "linear_solver": "mumps",
                "preconditioner": "default",
                "maximum_iterations": max_iter,
                "report": bool(self.verbose),
            }
        }
        dolfin.solve(
            self.F0 == 0, up_, self.bc["bcu"], solver_parameters=nl_solver_param
        )
        # Return
        return up_

    def compute_steady_state_picard(self, max_iter: int = 10, tol: float = 1e-14):
        """Compute steady state with fixed-point iteration
        Should have a larger convergence radius than Newton method
        if initialization is bad in Newton method (and it is)
        TODO: residual not 0 if u_ctrl not 0 (see bc probably)"""
        self.make_form_mixed_steady()  # for BC only
        iRe = dolfin.Constant(1 / self.params_flow.Re)

        # for residual computation
        bcu_inlet0 = self.bc_p["bcu"][0]
        bcu0 = self.bc["bcu"] + [bcu_inlet0]

        up0 = dolfin.Function(self.W)
        up1 = dolfin.Function(self.W)

        u, p = dolfin.TrialFunctions(self.W)
        v, q = dolfin.TestFunctions(self.W)

        class initial_condition(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        up0.interpolate(initial_condition())
        u0 = dolfin.as_vector((up0[0], up0[1]))

        ap = (
            dot(dot(u0, nabla_grad(u)), v) * dx
            + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )  # steady dolfin.lhs
        Lp = (
            dolfin.Constant(0) * inner(u0, v) * dx + dolfin.Constant(0) * q * dx
        )  # zero dolfin.rhs
        bp = dolfin.assemble(Lp)

        solverp = dolfin.LUSolver("mumps")
        ndof = self.W.dim()

        for i in range(max_iter):
            Ap = dolfin.assemble(ap)
            [bc.apply(Ap, bp) for bc in self.bc["bcu"]]

            solverp.solve(Ap, up1.vector(), bp)

            up0.assign(up1)
            u, p = up1.split()

            res = dolfin.assemble(dolfin.action(ap, up1))
            [bc.apply(res) for bc in bcu0]
            res_norm = dolfin.norm(res) / dolfin.sqrt(ndof)
            if self.verbose:
                logger.info(
                    f"Picard iteration: {i + 1}/{max_iter}, residual: {res_norm}"
                )
            if res_norm < tol:
                if self.verbose:
                    logger.info(f"Residual norm lower than tolerance {tol}")
                break

        return up1

    # Dataframe utility
    def make_y_df_colname(self, sensor_nr: int):
        """Return column names of different measurements y_meas_i"""
        return ["y_meas_" + str(i + 1) for i in range(sensor_nr)]

    def assign_y_to_df(self, df: pd.DataFrame, y_meas: float, index: int):
        """Assign measurement (array y_meas) to DataFrame at index"""
        df.loc[index, self.make_y_df_colname(len(y_meas))] = y_meas

    def write_timeseries(self):  # TODO this is an export utility
        """Write pandas DataFrame to file"""
        if flu.MpiUtils.get_rank() == 0:
            # zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths["timeseries"], sep=",", index=False)

    def log_timeseries(
        self, u_ctrl: float, y_meas: float, dE: float, t: float, runtime: float
    ):
        """Fill timeseries table with data"""
        self.timeseries.loc[self.iter - 1, "u_ctrl"] = (
            u_ctrl  # careful here: log the control that was applied at time t (iter-1) to get time t+dt (iter)
        )
        self.assign_y_to_df(df=self.timeseries, y_meas=y_meas, index=self.iter)
        self.timeseries.loc[self.iter, "dE"] = dE
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    # General utility # could go outside class because
    # one might want to compute energy of an arbitrary velocity field
    def compute_energy(self):
        """Compute energy of perturbation flow
        OPTIONS REMOVED FROM PREVIOUS VERSION:
        on full/restricted domain      (default:full=True)
        minus base flow                (default:diff=False)
        normalized by base flow energy (default:normalize=False)"""
        dE = 1 / 2 * dolfin.norm(self.u_, norm_type="L2", mesh=self.mesh) ** 2
        return dE

    def compute_energy_field(self, export: bool = False, filename: str = None):
        """Compute field dot(u, u) to see spatial location of perturbation kinetic energy
        Perturbation formulation only"""
        Efield = dot(self.u_, self.u_)
        # Note: E = 1/2 * assemble(Efield * fs.dx)
        Efield = flu.projectm(Efield, self.P)  # project to deg 1
        if export:
            flu.write_xdmf(filename, Efield, "E")
        return Efield

    # Abstract methods (to be reimplemented for each case)
    @abstractmethod
    def make_boundaries(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def make_bcs(self) -> dict:
        pass

    @abstractmethod
    def make_actuator(self) -> dolfin.Expression:
        pass

    @abstractmethod
    def make_measurement(self) -> np.array:
        pass

    # NOTE: inlet BC should be 1st always, and have (u,v)=(1,0)
    # Otherwise, reimplement this
    def make_form_mixed_steady(self, initial_guess=None):
        """Make nonlinear forms for steady state computation, in mixed element space.
        Can be used to assign self.F0 and compute state spaces matrices."""
        v, q = dolfin.TestFunctions(self.W)
        if initial_guess is None:
            up_ = dolfin.Function(self.W)
        else:
            up_ = initial_guess
        u_, p_ = dolfin.split(up_)  # not deep copy, we need the link
        iRe = dolfin.Constant(1 / self.params_flow.Re)
        # f = self.actuator_expression # TODO activate if actuator_type is VOL
        # Problem
        F0 = (
            dot(dot(u_, nabla_grad(u_)), v) * dx
            + iRe * inner(nabla_grad(u_), nabla_grad(v)) * dx
            - p_ * div(v) * dx
            - q * div(u_) * dx
            #    - dot(f, v) * dx
        )
        self.F0 = F0
        self.up_ = up_
        self.u_ = u_
        self.p_ = p_

        # NOTE
        # Impossible to modify existing BC without causing problems in the time-stepping
        # Solution: duplicate BC
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        self.bc = {"bcu": [bcu_inlet] + self.bc_p["bcu"][1:], "bcp": []}
