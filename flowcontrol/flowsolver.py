from __future__ import print_function
from typing import Any
import flowsolverparameters
from flowfield import FlowField, FlowFieldCollection
import dolfin
from dolfin import dot, nabla_grad, dx, inner, div
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from pathlib import Path

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
        params_flow: flowsolverparameters.ParamControl,
        params_time: flowsolverparameters.ParamTime,
        params_save: flowsolverparameters.ParamSave,
        params_solver: flowsolverparameters.ParamSolver,
        params_mesh: flowsolverparameters.ParamMesh,
        params_restart: flowsolverparameters.ParamRestart,
        params_control: flowsolverparameters.ParamControl,
        verbose: int = 1,
    ) -> None:
        self.params_flow = params_flow
        self.params_time = params_time
        self.params_save = params_save
        self.params_solver = params_solver
        self.params_mesh = params_mesh
        self.params_restart = params_restart
        self.params_control = params_control
        self.verbose = verbose

        ##
        self.paths = self._define_paths()
        self.mesh = self._make_mesh()
        self.V, self.P, self.W = self._make_function_spaces()
        self.boundaries = self._make_boundaries()  # @abstract
        self._mark_boundaries()
        self.actuator_expression = self._make_actuator()  # @abstract
        self.bc = self._make_bcs()  # @abstract
        self.BC = self._make_BCs()
        ##

        self.first_step = True
        self.fields = FlowFieldCollection()

    def _define_paths(self) -> dict[str, Path]:
        """Define attribute (dict) containing useful paths (save, etc.)"""
        logger.debug("Currently defining paths...")

        def make_file_extension(T):
            return "_restart" + str(np.round(T, decimals=3)).replace(".", ",")

        # start simulation from time...
        Tstart = self.params_time.Tstart
        ext_Tstart = make_file_extension(Tstart)
        # use older files starting from time...
        Trestartfrom = self.params_restart.Trestartfrom
        ext_Trestart = make_file_extension(Trestartfrom)

        ext_xdmf = ".xdmf"
        ext_csv = ".csv"
        path_out = self.params_save.path_out

        filename_U0 = path_out / "steady" / ("U0" + ext_xdmf)
        filename_P0 = path_out / "steady" / ("P0" + ext_xdmf)

        filename_U = path_out / ("U" + ext_Trestart + ext_xdmf)
        filename_Uprev = path_out / ("Uprev" + ext_Trestart + ext_xdmf)
        filename_P = path_out / ("P" + ext_Trestart + ext_xdmf)

        filename_U_restart = path_out / ("U" + ext_Tstart + ext_xdmf)
        filename_Uprev_restart = path_out / ("Uprev" + ext_Tstart + ext_xdmf)
        filename_P_restart = path_out / ("P" + ext_Tstart + ext_xdmf)

        filename_timeseries = path_out / ("timeseries1D" + ext_Tstart + ext_csv)

        return {
            "U0": filename_U0,
            "P0": filename_P0,
            "U": filename_U,
            "P": filename_P,
            "Uprev": filename_Uprev,
            "U_restart": filename_U_restart,
            "Uprev_restart": filename_Uprev_restart,
            "P_restart": filename_P_restart,
            "timeseries": filename_timeseries,
            "mesh": self.params_mesh.meshpath,
        }

    def _make_mesh(self) -> dolfin.Mesh:
        """Define mesh
        params_mesh contains either name of existing mesh
        or geometry parameters: xinf, yinf, xinfa, nx..."""

        logger.info(f"Mesh exists @: {self.params_mesh.meshpath}")

        mesh = dolfin.Mesh(dolfin.MPI.comm_world)
        with dolfin.XDMFFile(
            dolfin.MPI.comm_world, str(self.params_mesh.meshpath)
        ) as fm:
            fm.read(mesh)

        logger.info(f"Mesh has: {mesh.num_cells()} cells")

        return mesh

    def _make_function_spaces(self) -> tuple[dolfin.FunctionSpace, ...]:
        """Define function spaces (u, p) = (CG2, CG1)"""
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)  # was 'P'
        Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)  # was 'P'
        We = dolfin.MixedElement([Ve, Pe])
        V = dolfin.FunctionSpace(self.mesh, Ve)
        P = dolfin.FunctionSpace(self.mesh, Pe)
        W = dolfin.FunctionSpace(self.mesh, We)

        logger.debug(
            f"Function Space [V(CG2), P(CG1)] has: {P.dim()}+{V.dim()}={W.dim()} DOFs"
        )

        return V, P, W

    def _mark_boundaries(self) -> None:
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

    def initialize_time_stepping(
        self, Tstart: float = 0.0, ic: dolfin.Function | None = None
    ) -> None:
        """Create varitional functions/forms & flush files & define u(0), p(0)
        If Tstart is 0: ic is set in ic (or, if ic is None: = 0)
        If Tstart is not 0: ic is computed from files
        """

        logger.info(
            f"Starting or restarting from time: {Tstart} "
            f"with temporal scheme order: {self.params_restart.restart_order}"
        )

        if Tstart == 0.0:
            logger.debug("Starting simulation from zero with IC")
            u_, p_, u_n, u_nn, p_n = self._initialize_with_ic(ic)
        else:
            logger.debug("Starting simulation from nonzero")
            u_, p_, u_n, u_nn, p_n = self._initialize_at_time(Tstart)

        self.fields.u_ = u_
        self.fields.p_ = p_
        self.fields.u_n = u_n
        self.fields.u_nn = u_nn
        self.fields.p_n = p_n

        self.timeseries = self._initialize_timeseries()

    def _initialize_with_ic(
        self, ic: dolfin.Function | None
    ) -> tuple[dolfin.Function, ...]:
        self.order = 1

        if ic is None:  # then zero
            logger.debug("ic is set internally to 0")
            self.fields.ic = FlowField.generate(up=dolfin.Function(self.W))
        else:
            logger.debug("ic is already set by user")
            self.fields.ic = FlowField.generate(up=ic)

        # Add perturbation to IC
        addpert = self.params_solver.ic_add_perturbation
        if addpert:
            logger.debug("Found ic perturbation: {0}".format(addpert))
            udiv0 = flu2.get_div0_u(self, xloc=2, yloc=0, size=0.5)
            pert0 = self.merge(u=udiv0, p=flu.projectm(self.fields.STEADY.p, self.P))
            self.fields.ic.up.vector()[:] += addpert * pert0.vector()[:]
        self.fields.ic.up.vector().apply("insert")
        self.fields.ic = FlowField.generate(self.fields.ic.up)

        u_n = flu.projectm(v=self.fields.ic.u, V=self.V, bcs=self.bc["bcu"])
        u_nn = u_n.copy(deepcopy=True)
        p_n = flu.projectm(self.fields.ic.p, self.P)
        u_ = u_n.copy(deepcopy=True)
        p_ = p_n.copy(deepcopy=True)

        # Flush files and save ic as time_step 0
        if self.params_save.save_every:
            self._export_fields_xdmf(
                u_n,
                u_nn,
                p_n,
                time=0,
                append=False,
                write_mesh=True,
                adjust_baseflow=+1,
            )

        return u_, p_, u_n, u_nn, p_n

    def _initialize_at_time(self, Tstart: float) -> tuple[dolfin.Function, ...]:
        self.order = self.params_restart.restart_order  # 2

        idxstart = (Tstart - self.params_restart.Trestartfrom) / (
            self.params_restart.dt_old * self.params_restart.save_every_old
        )
        idxstart = int(np.floor(idxstart))

        U_ = dolfin.Function(self.V)
        P_ = dolfin.Function(self.P)
        U_n = dolfin.Function(self.V)
        U_nn = dolfin.Function(self.V)
        P_n = dolfin.Function(self.P)

        flu.read_xdmf(self.paths["U"], U_, "U", counter=idxstart)
        flu.read_xdmf(self.paths["P"], P_, "P", counter=idxstart)
        flu.read_xdmf(self.paths["U"], U_n, "U", counter=idxstart)
        flu.read_xdmf(self.paths["Uprev"], U_nn, "U_n", counter=idxstart)
        flu.read_xdmf(self.paths["P"], P_n, "P", counter=idxstart)

        # write in new file as first time step
        if self.params_save.save_every:
            self._export_fields_xdmf(
                U_n,
                U_nn,
                P_n,
                time=Tstart,
                append=False,
                write_mesh=True,
                adjust_baseflow=0,
            )

        # remove base flow from loaded file
        u_ = dolfin.Function(self.V)
        p_ = dolfin.Function(self.P)
        u_n = dolfin.Function(self.V)
        u_nn = dolfin.Function(self.V)
        p_n = dolfin.Function(self.P)
        for u, U in zip([u_n, u_nn, u_], [U_n, U_nn, U_]):
            u.vector()[:] = U.vector()[:] - self.fields.STEADY.u.vector()[:]
            u.vector().apply("insert")
        for p, P in zip([p_n, p_], [P_n, P_]):
            p.vector()[:] = P.vector()[:] - self.fields.STEADY.p.vector()[:]
            p.vector().apply("insert")

        self.fields.ic = FlowField.generate(up=self.merge(u=u_, p=p_))

        return u_, p_, u_n, u_nn, p_n

    def _initialize_timeseries(self) -> pd.DataFrame:
        self.t = self.params_time.Tstart
        self.iter = 0
        self.y_meas = self.make_measurement(self.fields.ic.u)
        y_meas_str = [
            "y_meas_" + str(i + 1) for i in range(self.params_control.sensor_number)
        ]
        colnames = ["time", "u_ctrl"] + y_meas_str + ["dE", "runtime"]
        empty_data = np.zeros((self.params_time.num_steps + 1, len(colnames)))
        timeseries = pd.DataFrame(columns=colnames, data=empty_data)
        timeseries.loc[0, "time"] = self.params_time.Tstart
        self._assign_y_to_df(df=timeseries, y_meas=self.y_meas, index=0)

        dE0 = self.compute_energy()
        timeseries.loc[0, "dE"] = dE0
        return timeseries

    def _make_solver(self, **kwargs) -> Any:
        """Define solvers"""
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def _make_varf(self, order: int, **kwargs) -> dolfin.Form:
        """Define equations"""
        if order == 1:
            F = self._make_varf_order1(**kwargs)
        elif order == 2:
            F = self._make_varf_order2(**kwargs)
        else:
            raise ValueError("Equation order not recognized")
            # There will be more important problems than this exception
        return F

    def _make_varf_order1(
        self,
        up: tuple[dolfin.TrialFunction, dolfin.TrialFunction],
        vq: tuple[dolfin.TestFunction, dolfin.TestFunction],
        U0: dolfin.Function,
        u_n: dolfin.Function,
        shift: float,
    ) -> dolfin.Form:
        (u, p) = up
        (v, q) = vq
        b0_1 = 1 if self.params_solver.is_eq_nonlinear else 0
        invRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)
        F1 = (
            dot((u - u_n) / dt, v) * dx
            + dot(dot(U0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_1) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - shift * dot(u, v) * dx
        )
        return F1

    def _make_varf_order2(
        self,
        up: tuple[dolfin.TrialFunction, dolfin.TrialFunction],
        vq: tuple[dolfin.TestFunction, dolfin.TestFunction],
        U0: dolfin.Function,
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
        shift: float,
    ) -> dolfin.Form:
        (u, p) = up
        (v, q) = vq
        if self.params_solver.is_eq_nonlinear:
            b0_2, b1_2 = 2, -1
        else:
            b0_2, b1_2 = 0, 0
        invRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)
        F2 = (
            dot((3 * u - 4 * u_n + u_nn) / (2 * dt), v) * dx
            + dot(dot(U0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_2) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            + dolfin.Constant(b1_2) * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - shift * dot(u, v) * dx
        )
        return F2

    def _prepare_systems(
        self,
        up: tuple[dolfin.TrialFunction, dolfin.TrialFunction],
        vq: tuple[dolfin.TestFunction, dolfin.TestFunction],
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
    ) -> None:
        shift = dolfin.Constant(self.params_solver.shift)
        # 1st order integration
        F1 = self._make_varf(
            order=1,
            up=up,
            vq=vq,
            U0=self.fields.STEADY.u,
            u_n=u_n,
            shift=shift,
        )
        # 2nd order integration
        F2 = self._make_varf(
            order=2,
            up=up,
            vq=vq,
            U0=self.fields.STEADY.u,
            u_n=u_n,
            u_nn=u_nn,
            shift=shift,
        )

        self.forms = {1: F1, 2: F2}
        self.assemblers = dict()
        self.solvers = dict()
        self.rhs = dolfin.Vector()
        for index, varf in enumerate([F1, F2]):
            order = index + 1
            a = dolfin.lhs(varf)
            L = dolfin.rhs(varf)
            systemAssembler = dolfin.SystemAssembler(a, L, self.bc["bcu"])
            solver = self._make_solver(order=order)
            operatorA = dolfin.Matrix()
            systemAssembler.assemble(operatorA)
            solver.set_operator(operatorA)
            self.assemblers[order] = systemAssembler
            self.solvers[order] = solver

    def step(self, u_ctrl: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """Simulate system with perturbation formulation,
        possibly an actuation value, and a shift"""
        v, q = dolfin.TestFunctions(self.W)
        up = dolfin.TrialFunction(self.W)
        u, p = dolfin.split(up)

        up_ = dolfin.Function(self.W)
        u_, p_ = dolfin.split(up_)

        u_nn = self.fields.u_nn
        u_n = self.fields.u_n
        p_n = self.fields.p_n

        if self.first_step:
            logger.debug("Perturbations forms DO NOT exist: create...")
            self._prepare_systems((u, p), (v, q), u_n, u_nn)
            self.first_step = False

        # time
        t0i = time.time()

        # control
        self.actuator_expression.ampl = u_ctrl

        try:
            self.assemblers[self.order].assemble(self.rhs)
            self.solvers[self.order].solve(up_.vector(), self.rhs)
            u_, p_ = up_.split(deepcopy=True)
            if self._solver_diverged(u_):
                raise RuntimeError()
        except RuntimeError:
            logger.critical("*** Solver diverged, Inf found ***")
            if not self.params_solver.throw_error:
                logger.critical("*** Exiting step() ***")
                return -1  # -1 is error code
            else:
                raise RuntimeError("Failed solving: Inf found in solution")
        except Exception as e:
            raise e

        # Update time
        self.iter += 1
        self.t = self.params_time.Tstart + self.iter * self.params_time.dt
        self.order = 2

        # Assign
        self.fields.u_ = u_
        self.fields.p_ = p_
        self.fields.up_ = up_
        # Shift
        self.fields.u_nn.assign(u_n)
        self.fields.u_n.assign(u_)
        self.fields.p_n.assign(p_)

        ## Output
        # Probe
        self.y_meas = self.make_measurement(self.fields.up_)
        # Runtime
        runtime = time.time() - t0i
        if self._niter_multiple_of(self.iter, self.verbose):
            flu2.print_progress(self, runtime=runtime)
        # Timeseries
        self._log_timeseries(
            u_ctrl=u_ctrl,
            y_meas=self.y_meas,
            dE=self.compute_energy(),
            t=self.t,
            runtime=runtime,
        )

        # Export xdmf & csv
        if self._niter_multiple_of(self.iter, self.params_save.save_every):
            self._export_fields_xdmf(u_n, u_nn, p_n, self.t, adjust_baseflow=+1)
            self.write_timeseries()

        return self.y_meas

    def _solver_diverged(self, field: dolfin.Function) -> bool:
        return not np.isfinite(field.vector().get_local()[0])

    def _niter_multiple_of(self, iter: int, divider: int) -> bool:
        return divider and not iter % divider

    def merge(
        self, u: dolfin.Function | None = None, p: dolfin.Function | None = None
    ) -> dolfin.Function:
        """Split or merge field(s)
        if instruction is split: up -> (u,p)
        if instruction is merge: (u,p) -> up
        FunctionAssigner(receiver, sender)"""
        # if u is None:  # split up
        #     # probably equivalent to up.split(deepcopy=True) from dolfin
        #     fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
        #     u = dolfin.Function(self.V)
        #     p = dolfin.Function(self.P)
        #     fa.assign([u, p], up)
        #     return (u, p)
        # else:  # merge u, p
        fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
        up = dolfin.Function(self.W)
        fa.assign(up, [u, p])
        return up

    def _export_fields_xdmf(
        self,
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
        p_n: dolfin.Function,
        time: float,
        append: bool = True,
        write_mesh: bool = False,
        adjust_baseflow: float = 0,
    ) -> None:
        if not (hasattr(self.fields, "Psave_n")):
            self.fields.Usave = dolfin.Function(self.V)
            self.fields.Usave_n = dolfin.Function(self.V)
            self.fields.Psave = dolfin.Function(self.P)

        # Reconstruct full field
        pmbf = adjust_baseflow
        self.fields.Usave.vector()[:] = (
            u_n.vector()[:] + pmbf * self.fields.STEADY.u.vector()[:]
        )
        self.fields.Usave_n.vector()[:] = (
            u_nn.vector()[:] + pmbf * self.fields.STEADY.u.vector()[:]
        )
        self.fields.Psave.vector()[:] = (
            p_n.vector()[:] + pmbf * self.fields.STEADY.p.vector()[:]
        )
        for vec in [self.fields.Usave, self.fields.Usave_n, self.fields.Psave]:
            vec.vector().apply("insert")

        logger.debug(f"saving to files {self.params_save.path_out}")

        flu.write_xdmf(
            filename=self.paths["U_restart"],
            func=self.fields.Usave,
            name="U",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths["Uprev_restart"],
            func=self.fields.Usave_n,
            name="U_n",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths["P_restart"],
            func=self.fields.Psave,
            name="P",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )

    # Steady state
    def _assign_steady_state(self, U0: dolfin.Function, P0: dolfin.Function) -> None:
        UP0 = self.merge(u=U0, p=P0)
        self.fields.STEADY = FlowField.generate(UP0)
        self.fields.U0 = self.fields.STEADY.u
        self.fields.P0 = self.fields.STEADY.p
        self.fields.UP0 = self.fields.STEADY.up
        self.E0 = 1 / 2 * dolfin.norm(U0, norm_type="L2", mesh=self.mesh) ** 2

    def load_steady_state(self) -> None:
        U0 = dolfin.Function(self.V)
        P0 = dolfin.Function(self.P)
        flu.read_xdmf(self.paths["U0"], U0, "U0")
        flu.read_xdmf(self.paths["P0"], P0, "P0")
        self._assign_steady_state(U0=U0, P0=P0)

    def compute_steady_state(
        self, method: str = "newton", u_ctrl: float = 0.0, **kwargs
    ) -> None:
        """Compute flow steady state with given steady control"""
        self.actuator_expression.ampl = u_ctrl

        if method == "newton":
            UP0 = self._compute_steady_state_newton(**kwargs)
        else:
            UP0 = self._compute_steady_state_picard(**kwargs)

        U0, P0 = UP0.split(deepcopy=True)
        U0 = flu.projectm(U0, self.V)
        P0 = flu.projectm(P0, self.P)

        if self.params_save.save_every:
            flu.write_xdmf(
                self.paths["U0"],
                U0,
                "U0",
                time_step=0.0,
                append=False,
                write_mesh=True,
            )
            flu.write_xdmf(
                self.paths["P0"],
                P0,
                "P0",
                time_step=0.0,
                append=False,
                write_mesh=True,
            )

        logger.debug(f"Stored base flow in: {self.params_save.path_out}")

        self._assign_steady_state(U0=U0, P0=P0)

    def _compute_steady_state_newton(
        self, max_iter: int = 25, initial_guess: dolfin.Function | None = None
    ) -> dolfin.Function:
        """Compute steady state with built-in nonlinear solver (Newton method)
        initial_guess is a (u,p)_0"""
        F0, UP0 = self._make_form_mixed_steady(initial_guess=initial_guess)
        BC = self._make_BCs()

        if initial_guess is None:
            logger.info("Newton solver --- without initial guess")

        nl_solver_param = {
            "newton_solver": {
                "linear_solver": "mumps",
                "preconditioner": "default",
                "maximum_iterations": max_iter,
                "report": bool(self.verbose),
            }
        }
        dolfin.solve(F0 == 0, UP0, BC["bcu"], solver_parameters=nl_solver_param)
        # Return
        return UP0

    def _compute_steady_state_picard(
        self, max_iter: int = 10, tol: float = 1e-14
    ) -> dolfin.Function:
        """Compute steady state with fixed-point iteration
        Should have a larger convergence radius than Newton method
        if initialization is bad in Newton method (and it is)
        TODO: residual not 0 if u_ctrl not 0 (see bc probably)"""
        BC = self._make_BCs()
        invRe = dolfin.Constant(1 / self.params_flow.Re)

        UP0 = dolfin.Function(self.W)
        UP1 = dolfin.Function(self.W)

        u, p = dolfin.TrialFunctions(self.W)
        v, q = dolfin.TestFunctions(self.W)

        class initial_condition(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        UP0.interpolate(initial_condition())
        U0 = dolfin.as_vector((UP0[0], UP0[1]))

        ap = (
            dot(dot(U0, nabla_grad(u)), v) * dx
            + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )  # steady dolfin.lhs
        Lp = (
            dolfin.Constant(0) * inner(U0, v) * dx + dolfin.Constant(0) * q * dx
        )  # zero dolfin.rhs
        bp = dolfin.assemble(Lp)

        solverp = dolfin.LUSolver("mumps")

        for iter in range(max_iter):
            Ap = dolfin.assemble(ap)
            [bc.apply(Ap, bp) for bc in BC["bcu"]]
            solverp.solve(Ap, UP1.vector(), bp)

            UP0.assign(UP1)
            u, p = UP1.split()

            # Residual computation
            res = dolfin.assemble(dolfin.action(ap, UP1))
            [bc.apply(res) for bc in self.bc["bcu"]]
            res_norm = dolfin.norm(res) / dolfin.sqrt(self.W.dim())
            logger.info(
                f"Picard iteration: {iter + 1}/{max_iter}, residual: {res_norm}"
            )
            if res_norm < tol:
                logger.info(f"Residual norm lower than tolerance {tol}")
                break

        return UP1

    # Otherwise, reimplement this
    def _make_form_mixed_steady(
        self, initial_guess: dolfin.Function | None = None
    ) -> tuple[dolfin.Form, dolfin.Function]:
        """Make nonlinear forms for steady state computation, in mixed element space.
        Can be used to assign self.F0 and compute state spaces matrices."""
        v, q = dolfin.TestFunctions(self.W)
        if initial_guess is None:
            UP0 = dolfin.Function(self.W)
        else:
            UP0 = initial_guess
        U0, P0 = dolfin.split(UP0)  # not deep copy, we need the link
        invRe = dolfin.Constant(1 / self.params_flow.Re)

        # f = self.actuator_expression # TODO activate if actuator_type is VOL
        # Problem
        F0 = (
            dot(dot(U0, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(U0), nabla_grad(v)) * dx
            - P0 * div(v) * dx
            - q * div(U0) * dx
            #    - dot(f, v) * dx
        )
        return F0, UP0

    def _make_BCs(self) -> dict[str, Any]:
        # NOTE: inlet BC should be 1st always, and have (u,v)=(1,0)
        # NOTE
        # Impossible to modify existing BC without causing problems in the time-stepping
        # Solution: duplicate BC
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        BC = {"bcu": [bcu_inlet] + self.bc["bcu"][1:], "bcp": []}

        return BC

    # Dataframe utility
    def _make_y_df_colname(self, sensor_nr: int) -> list[str]:
        """Return column names of different measurements y_meas_i"""
        return ["y_meas_" + str(i + 1) for i in range(sensor_nr)]

    def _assign_y_to_df(self, df: pd.DataFrame, y_meas: float, index: int) -> None:
        """Assign measurement (array y_meas) to DataFrame at index"""
        df.loc[index, self._make_y_df_colname(len(y_meas))] = y_meas

    def write_timeseries(self) -> None:
        """Write pandas DataFrame to file"""
        if flu.MpiUtils.get_rank() == 0:  # TODO async?
            # zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths["timeseries"], sep=",", index=False)

    def _log_timeseries(
        self, u_ctrl: float, y_meas: float, dE: float, t: float, runtime: float
    ) -> None:
        """Fill timeseries table with data"""
        self.timeseries.loc[self.iter - 1, "u_ctrl"] = (
            u_ctrl  # careful here: log the control that was applied at time t (iter-1) to get time t+dt (iter)
        )
        self._assign_y_to_df(df=self.timeseries, y_meas=y_meas, index=self.iter)
        self.timeseries.loc[self.iter, "dE"] = dE
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    # General utility
    def compute_energy(self) -> float:
        """Compute energy of perturbation flow
        OPTIONS REMOVED FROM PREVIOUS VERSION:
        on full/restricted domain      (default:full=True)
        minus base flow                (default:diff=False)
        normalized by base flow energy (default:normalize=False)"""
        dE = 1 / 2 * dolfin.norm(self.fields.u_, norm_type="L2", mesh=self.mesh) ** 2
        return dE

    def compute_energy_field(
        self, export: bool = False, filename: str = None
    ) -> dolfin.Function:
        """Compute field dot(u, u) to see spatial location of perturbation kinetic energy
        Perturbation formulation only"""
        Efield = dot(self.fields.u_, self.fields.u_)
        # Note: E = 1/2 * assemble(Efield * fs.dx)
        Efield = flu.projectm(Efield, self.P)  # project to deg 1
        if export:
            flu.write_xdmf(filename, Efield, "E")
        return Efield

    # Abstract methods
    @abstractmethod
    def _make_boundaries(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _make_bcs(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _make_actuator(self) -> dolfin.Expression:
        pass

    @abstractmethod
    def make_measurement(self) -> np.ndarray:
        pass
