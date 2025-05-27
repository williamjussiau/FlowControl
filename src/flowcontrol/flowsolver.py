from __future__ import print_function

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import dolfin
import flowsolverparameters
import numpy as np
import pandas as pd
import utils_extract as flu2
import utils_flowsolver as flu
from actuator import ACTUATOR_TYPE
from dolfin import div, dot, dx, inner, nabla_grad
from flowfield import BoundaryConditions, FlowField, FlowFieldCollection

logger = logging.getLogger(__name__)
FORMAT = (
    "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
)
logging.basicConfig(format=FORMAT, level=logging.INFO)


class FlowSolver(ABC):
    """Abstract base class for defining flow simulations and control problems.
    This class implements utility functions but does not correspond to a flow problem.
    It implements: defining file paths, reading mesh, defining function spaces,
    trial and test functions, variational formulations, boundaries (geometry) and
    boundary conditions and time-stepping utility.

    Abstract methods:
        _make_boundaries(self) -> pd.DataFrame
        _make_bcs(self) -> dict[str, Any]
    """

    def __init__(
        self,
        params_flow: flowsolverparameters.ParamFlow,
        params_time: flowsolverparameters.ParamTime,
        params_save: flowsolverparameters.ParamSave,
        params_solver: flowsolverparameters.ParamSolver,
        params_mesh: flowsolverparameters.ParamMesh,
        params_restart: flowsolverparameters.ParamRestart,
        params_control: flowsolverparameters.ParamControl,
        params_ic: flowsolverparameters.ParamIC,
        verbose: int = 1,
    ) -> None:
        """Initialize FlowSolver object with Parameters objects and
        setup FlowSolver object.

        Args:
            params_flow (flowsolverparameters.ParamControl): see flowsolverparameters
            params_time (flowsolverparameters.ParamTime): see flowsolverparameters
            params_save (flowsolverparameters.ParamSave): see flowsolverparameters
            params_solver (flowsolverparameters.ParamSolver): see flowsolverparameters
            params_mesh (flowsolverparameters.ParamMesh): see flowsolverparameters
            params_restart (flowsolverparameters.ParamRestart): see flowsolverparameters
            params_control (flowsolverparameters.ParamControl): see flowsolverparameters
            params_ic (flowsolverparameters.ParamIC): see flowsolerparameters
            verbose (int, optional): print every _verbose_ iteration. Defaults to 1.
        """
        self.params_flow = params_flow
        self.params_time = params_time
        self.params_save = params_save
        self.params_solver = params_solver
        self.params_mesh = params_mesh
        self.params_restart = params_restart
        self.params_control = params_control
        self.params_ic = params_ic
        self.verbose = verbose

        for param in [
            params_flow,
            params_time,
            params_save,
            params_solver,
            params_mesh,
            params_restart,
            params_control,
            params_ic,
        ]:
            logger.debug(param)

        self._setup()

    def _setup(self):
        """Define class attributes common to all FlowSolver problems."""
        self.first_step = True
        self.fields = FlowFieldCollection()

        self.paths = self._define_paths()
        self.mesh = self._make_mesh()
        self.V, self.P, self.W = self._make_function_spaces()
        self.boundaries = self._make_boundaries()  # @abstract
        self._mark_boundaries()
        self._load_actuators()
        self._load_sensors()
        self.bc = self._make_bcs()  # @abstract
        self.BC = self._make_BCs()

    def _define_paths(self) -> dict[str, Path]:
        """Define dictionary of file names for import/export.

        Returns:
            dict[str, Path]: dictionary of paths for importing/exporting files
        """
        logger.debug("Currently defining paths...")

        NUMBER_OF_DECIMALS = 3

        def make_file_extension(T):
            return "_restart" + str(np.round(T, decimals=NUMBER_OF_DECIMALS)).replace(
                ".", ","
            )

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
        """Read xdmf mesh from mesh file given in ParamMesh object.

        Returns:
            dolfin.Mesh: mesh read from file in ParamMesh.meshpath
        """

        logger.info(f"Mesh exists @: {self.params_mesh.meshpath}")

        mesh = dolfin.Mesh(dolfin.MPI.comm_world)
        with dolfin.XDMFFile(
            dolfin.MPI.comm_world, str(self.params_mesh.meshpath)
        ) as fm:
            fm.read(mesh)

        logger.info(f"Mesh has: {mesh.num_cells()} cells")

        return mesh

    def _make_function_spaces(self) -> tuple[dolfin.FunctionSpace, ...]:
        """Define function spaces for FEM formulation.

        Default is Continuous-Galerkin (CG)
        for each velocity component (order 2) and pressure (order 1).

        Returns:
            tuple[dolfin.FunctionSpace, ...]: all FunctionSpaces (V, P, W)
        """
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
        """Mark boundaries automatically (used for numerical integration)."""
        bnd_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        cell_markers = dolfin.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
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
        """Initialize the time-stepping process by reading or generating
        initial conditions. Initialize the timeseries (pandas DataFrame)
        containing simulation information.

        Args:
            Tstart (float, optional): if Tstart is not 0, restart simulation from Tstart
                using files from a previous simulation provided in ParamRestart. Defaults to 0.0.
            ic (dolfin.Function | None, optional): if Tstart is 0, use ic as (pert) initial condition.
                Defaults to None.
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
        """Initialize time-stepping with given initial condition (ic).
        ic is give in perturbation form. ic can be set by user or defined
        as None, in which case it is 0. A perturbation can be added onto the
        ic given by the user, thanks to ParamSolver.ic_add_perturbation.

        Args:
            ic (dolfin.Function | None): perturbation initial condition.
                ic is adjusted with ParamSolver.ic_add_perturbation.

        Returns:
            tuple[dolfin.Function, ...]: initial perturbation fields
        """
        self.order = 1

        if ic is None:  # then zero
            logger.debug("ic is set internally to 0")
            self.fields.ic = FlowField(up=dolfin.Function(self.W))
        else:
            logger.debug("ic is already set by user")
            self.fields.ic = FlowField(up=ic)

        # Add perturbation to IC
        if self.params_ic.amplitude:
            logger.debug("Found ic perturbation: {0}".format(self.params_ic))
            ic_perturbation = self._default_initial_perturbation(
                xloc=self.params_ic.xloc,
                yloc=self.params_ic.yloc,
                radius=self.params_ic.radius,
            )
            self.fields.ic.up.vector()[:] += (
                self.params_ic.amplitude * ic_perturbation.vector()[:]
            )
        self.fields.ic.up.vector().apply("insert")
        self.fields.ic = FlowField(self.fields.ic.up)

        u_n = flu.projectm(v=self.fields.ic.u, V=self.V, bcs=self.bc.bcu)
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
        """Initialize time-stepping from given time, by reading fields from files.

        Args:
            Tstart (float): starting time. It must correspond to saved time steps from
                another simulation (to do so, set ParamRestart accordingly).

        Returns:
            tuple[dolfin.Function, ...]: initial perturbation fields
        """
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

        self.fields.ic = FlowField(up=self.merge(u=u_, p=p_))

        return u_, p_, u_n, u_nn, p_n

    def _initialize_timeseries(self) -> pd.DataFrame:
        """Instantiante and initialize timeseries containing
        flow information at each time step (e.g. time, measurements, energy...)

        Returns:
            pd.DataFrame: timeseries of flow information at each time step
        """
        self.t = self.params_time.Tstart
        self.iter = 0
        self.y_meas = self.make_measurement(up=self.fields.ic.up)
        y_meas_str = self._make_colname_df("y_meas", self.params_control.sensor_number)
        u_meas_str = self._make_colname_df(
            "u_ctrl", self.params_control.actuator_number
        )
        colnames = ["time"] + u_meas_str + y_meas_str + ["dE", "runtime"]
        empty_data = np.zeros((self.params_time.num_steps + 1, len(colnames)))
        timeseries = pd.DataFrame(columns=colnames, data=empty_data)
        timeseries.loc[0, "time"] = self.params_time.Tstart
        self._assign_to_df(df=timeseries, name="y_meas", value=self.y_meas, index=0)

        dE0 = self.compute_energy()
        timeseries.loc[0, "dE"] = dE0
        return timeseries

    def _make_solver(self, **kwargs) -> Any:
        """Define solvers to be used for type-stepping. This method may
        be overridden in order to use custom solvers.

        Returns:
            Any: dolfin.LUSolver or dolfin.KrylovSolver or anything that has a .solve() method
        """
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def _make_varf(self, order: int, **kwargs) -> dolfin.Form:
        """Metamethod for defining variational formulations (varf) of order 1 and 2

        Args:
            order (int): order of varf to create (1 or 2)

        Raises:
            ValueError: order not 1 nor 2

        Returns:
            dolfin.Form: varf to integrate NS equations in time
        """
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
        """Define variational formulation (varf) of order 1. Nonlinear term
        is approximated with velocity fields at previous time.

        Args:
            up (tuple[dolfin.TrialFunction, dolfin.TrialFunction]): trial functions
            vq (tuple[dolfin.TestFunction, dolfin.TestFunction]): test functions
            U0 (dolfin.Function): base flow
            u_n (dolfin.Function): previous velocity perturbation field
            shift (float): shift equations

        Returns:
            dolfin.Form: 1st order varf for integrating NS
        """

        (u, p) = up
        (v, q) = vq
        b0_1 = 1 if self.params_solver.is_eq_nonlinear else 0
        invRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)

        f = self._gather_actuators_expressions()

        F1 = (
            dot((u - u_n) / dt, v) * dx
            + dot(dot(U0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_1) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - dot(f, v) * dx
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
        """Define variational formulation (varf) of order 2. Nonlinear term
        is approximated with velocity fields at previous and previous^2 times.

        Args:
            up (tuple[dolfin.TrialFunction, dolfin.TrialFunction]): trial functions
            vq (tuple[dolfin.TestFunction, dolfin.TestFunction]): test functions
            U0 (dolfin.Function): base flow
            u_n (dolfin.Function): previous velocity perturbation field
            u_nn (dolfin.Function): previous^2 velocity perturbation field
            shift (float): shift equations

        Returns:
            dolfin.Form: 2nd order varf for integrating NS
        """

        (u, p) = up
        (v, q) = vq
        if self.params_solver.is_eq_nonlinear:
            b0_2, b1_2 = 2, -1
        else:
            b0_2, b1_2 = 0, 0
        invRe = dolfin.Constant(1 / self.params_flow.Re)
        dt = dolfin.Constant(self.params_time.dt)

        f = self._gather_actuators_expressions()

        F2 = (
            dot((3 * u - 4 * u_n + u_nn) / (2 * dt), v) * dx
            + dot(dot(U0, nabla_grad(u)), v) * dx
            + dot(dot(u, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx
            + dolfin.Constant(b0_2) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
            + dolfin.Constant(b1_2) * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            - dot(f, v) * dx
            - shift * dot(u, v) * dx
        )
        return F2

    def _gather_actuators_expressions(self) -> dolfin.Expression | dolfin.Constant:
        """Gathers actuators that have type ACTUATOR_TYPE.FORCE
        and sums their expressions, in order to integrate them in
        the momentum equation.

        Returns:
            dolfin.Expression | dolfin.Constant: sum of all force
                actuators expressions as dolfin.Expression, or (0,0) if none
        """
        f = sum(
            [
                actuator.expression
                for actuator in self.params_control.actuator_list
                if actuator.actuator_type is ACTUATOR_TYPE.FORCE
            ]
        )

        if f == 0:  # > sum of empty list = no force actuator
            f = dolfin.Constant((0, 0))

        return f

    def _prepare_systems(
        self,
        up: tuple[dolfin.TrialFunction, dolfin.TrialFunction],
        vq: tuple[dolfin.TestFunction, dolfin.TestFunction],
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
    ) -> int:
        """Define systems to be solved at each time step (assemble
        LHS operator, preallocate RHS...).

        Args:
            up (tuple[dolfin.TrialFunction, dolfin.TrialFunction]): trial functions
            vq (tuple[dolfin.TestFunction, dolfin.TestFunction]): test functions
            u_n (dolfin.Function): previous velocity perturbation field
            u_nn (dolfin.Function): previous^2 velocity perturbation field

        Returns:
            int: sanity check int (unused)
        """
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
            systemAssembler = dolfin.SystemAssembler(a, L, self.bc.bcu)
            solver = self._make_solver(order=order)
            operatorA = dolfin.Matrix()
            systemAssembler.assemble(operatorA)
            solver.set_operator(operatorA)
            self.assemblers[order] = systemAssembler
            self.solvers[order] = solver

        return 1

    def step(self, u_ctrl: np.ndarray[int, float]) -> np.ndarray[int, float]:
        """Simulate the system on one time-step: up(t)->up(t+dt).
        The first time this method is run, it calls _prepare_systems.

        Args:
            u_ctrl (np.ndarray[int, float]): control input list

        Raises:
            RuntimeError: solver failed (a coordinate is inf or nan)
            e: any other exception

        Returns:
            np.ndarray[int, float]: value of measurement y after step
        """
        v, q = dolfin.TestFunctions(self.W)
        up = dolfin.TrialFunction(self.W)
        u, p = dolfin.split(up)

        up_ = dolfin.Function(self.W)
        u_, p_ = dolfin.split(up_)

        u_nn = self.fields.u_nn
        u_n = self.fields.u_n
        p_n = self.fields.p_n

        if self.first_step:
            logger.debug("Perturbation varfs DO NOT exist: create...")
            self._prepare_systems((u, p), (v, q), u_n, u_nn)
            self.first_step = False
            logger.debug("Perturbation varfs created.")

        # time
        t0i = time.time()

        # control
        self.set_actuators_u_ctrl(u_ctrl)

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
        self.y_meas = self.make_measurement(up=self.fields.up_)
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
        """Check whether the solver has diverged

        Args:
            field (dolfin.Function): field to probe for solver sanity check

        Returns:
            bool: True if solver failed, False else
        """

        return not np.isfinite(field.vector().get_local()[0])

    def _niter_multiple_of(self, iter: int, divider: int) -> bool:
        """Check multiplicity for outputting verbose information

        Args:
            iter (int): iteration number
            divider (int): usually ParamSave.save_every

        Returns:
            bool: True if iteration is suitable for exporting/verbing, False else
        """
        return divider and not iter % divider

    def merge(self, u: dolfin.Function, p: dolfin.Function) -> dolfin.Function:
        """Merge two fields: (u, p) in (V, P) -> (up) in (W).

        For the inverse operation, use: u, p = up.dolfin.split(deepcopy: bool).

        Args:
            u (dolfin.Function): velocity field (pert or full) in self.V
            p (dolfin.Function): pressure field (pert or full) in self.P

        Returns:
            dolfin.Function: mixed field (pert or full) in self.W
        """
        fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
        up = dolfin.Function(self.W)
        fa.assign(up, [u, p])
        return up

    def set_actuators_u_ctrl(self, u_ctrl: Iterable) -> None:
        """Set control amplitudes for each actuator from iterable u_ctrl

        Args:
            u_ctrl (list): iterable of control values to assign to each actuator
        """
        for ii, actuator in enumerate(self.params_control.actuator_list):
            actuator.expression.u_ctrl = u_ctrl[ii]

    def flush_actuators_u_ctrl(self) -> None:
        """Set control amplitudes for each actuator to zero."""
        self.set_actuators_u_ctrl([0] * self.params_control.actuator_number)

    def get_actuators_u_ctrl(self) -> Iterable:
        """Get amplitude of each actuator"""
        u_ctrl = []
        for ii, actuator in enumerate(self.params_control.actuator_list):
            u_ctrl.append(actuator.expression.u_ctrl)
        return u_ctrl

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
        """Export perturbation fields to xdmf. The exported flow can be
        adjusted by the base flow times a given float adjust_base_flow.

        Example: to export perturbation fields, adjust_baseflow=0. To export
        full fields, adjust_baseflow=1.

        Args:
            u_n (dolfin.Function): (previous) velocity field to export
            u_nn (dolfin.Function): (previous^2) velocity field to export
            p_n (dolfin.Function): (previous) pressure field to export
            time (float): time index (important in xdmf file)
            append (bool, optional): append to xdmf or replace contents. Defaults to True.
            write_mesh (bool, optional): write mesh in xdmf or not. Not working in dolfin. Defaults to False.
            adjust_baseflow (float, optional): adjust fields with base flow (e.g. add or subtract). Defaults to 0.
        """

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
        """Assign steady state (U0, P0) to FlowSolver object for easy access.

        Args:
            U0 (dolfin.Function): full steady velocity field
            P0 (dolfin.Function): full steady pressure field
        """
        UP0 = self.merge(u=U0, p=P0)
        self.fields.STEADY = FlowField(UP0)
        self.fields.U0 = self.fields.STEADY.u
        self.fields.P0 = self.fields.STEADY.p
        self.fields.UP0 = self.fields.STEADY.up
        self.E0 = 1 / 2 * dolfin.norm(U0, norm_type="L2", mesh=self.mesh) ** 2

    def load_steady_state(self) -> None:
        """Load steady state from file (from ParamSave.path_out)"""
        U0 = dolfin.Function(self.V)
        P0 = dolfin.Function(self.P)
        flu.read_xdmf(self.paths["U0"], U0, "U0")
        flu.read_xdmf(self.paths["P0"], P0, "P0")
        self._assign_steady_state(U0=U0, P0=P0)

    def compute_steady_state(
        self, u_ctrl: list, method: str = "newton", **kwargs
    ) -> None:
        """Compute flow steady state with given method and constant input u_ctrl.
        Two methods are available: Picard method (see _compute_steady_state_picard)
        and Newton method (_compute_steady_state_newton). This method is intended
        to be used directly, contrary to _compute_steady_state_*() methods.

        Args:
            method (str, optional): method to be used (picard or newton). Defaults to "newton".
            u_ctrl (float, optional): constant input to take into account. Defaults to 0.0.
        """
        self.set_actuators_u_ctrl(u_ctrl)

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
        self, max_iter: int = 10, initial_guess: dolfin.Function | None = None
    ) -> dolfin.Function:
        """Compute steady state with built-in nonlinear solver (Newton method).
        initial_guess is a mixed field (up). This method should not be used directly
        (see compute_steady_state())

        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 10.
            initial_guess (dolfin.Function | None, optional): initial guess to use for mixed field UP. Defaults to None.

        Returns:
            dolfin.Function: estimation of steady state UP0
        """
        F0, UP0 = self._make_varf_steady(initial_guess=initial_guess)
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
        dolfin.solve(F0 == 0, UP0, BC.bcu, solver_parameters=nl_solver_param)
        # Return
        return UP0

    def _compute_steady_state_picard(
        self, max_iter: int = 10, tol: float = 1e-14
    ) -> dolfin.Function:
        """Compute steady state with fixed-point Picard iteration.
        This method should have a larger convergence radius than Newton method,
        but convergence is slower. The field computed by this method may be used as
        an initial guess for Newton method. This method should not be used directly
        (see compute_steady_state())

        Args:
            max_iter (int, optional): maximum number of iterations. Defaults to 10.
            tol (float, optional): precision tolerance. Defaults to 1e-14.

        Returns:
            dolfin.Function: estimation of steady state UP0
        """
        BC = self._make_BCs()
        invRe = dolfin.Constant(1 / self.params_flow.Re)

        UP0 = dolfin.Function(self.W)
        UP1 = dolfin.Function(self.W)

        u, p = dolfin.TrialFunctions(self.W)
        v, q = dolfin.TestFunctions(self.W)

        UP0.interpolate(self._default_steady_state_initial_guess())
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
            [bc.apply(Ap, bp) for bc in BC.bcu]
            solverp.solve(Ap, UP1.vector(), bp)

            UP0.assign(UP1)
            u, p = UP1.split()

            # Residual computation
            res = dolfin.assemble(dolfin.action(ap, UP1))
            [bc.apply(res) for bc in self.bc.bcu]
            res_norm = dolfin.norm(res) / dolfin.sqrt(self.W.dim())
            logger.info(
                f"Picard iteration: {iter + 1}/{max_iter}, residual: {res_norm}"
            )
            if res_norm < tol:
                logger.info(f"Residual norm lower than tolerance {tol}")
                break

        return UP1

    def _make_varf_steady(
        self, initial_guess: dolfin.Function | None = None
    ) -> tuple[dolfin.Form, dolfin.Function]:
        """Make nonlinear forms for steady state computation, in mixed element space W.

        Args:
            initial_guess (dolfin.Function | None, optional): field UP0 around which varf is computed.
                Defaults to None. If None, use zero dolfin.Function(self.W).

        Returns:
            tuple[dolfin.Form, dolfin.Function]: varf and field UP0
        """
        v, q = dolfin.TestFunctions(self.W)
        if initial_guess is None:
            UP0 = dolfin.Function(self.W)
        else:
            UP0 = initial_guess
        U0, P0 = dolfin.split(UP0)  # not deep copy, need the link only
        invRe = dolfin.Constant(1 / self.params_flow.Re)

        f = self._gather_actuators_expressions()

        # Problem
        F0 = (
            dot(dot(U0, nabla_grad(U0)), v) * dx
            + invRe * inner(nabla_grad(U0), nabla_grad(v)) * dx
            - P0 * div(v) * dx
            - q * div(U0) * dx
            - dot(f, v) * dx
        )
        return F0, UP0

    def _make_BCs(self) -> BoundaryConditions:
        """Define boundary conditions for the full field (i.e. not perturbation
        field). By default, the perturbation bcs are replicated and the inlet
        boundary condition is replaced with uniform profile with amplitude (u,v)=(uinf, 0).
        Note: the inlet boundary condition in _make_bcs() should always be first.
        For more complex inlet profiles, override this method.

        Returns:
            BoundaryConditions: boundary conditions for full field
        """
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        bcs = self._make_bcs()
        BC = BoundaryConditions(bcu=[bcu_inlet] + bcs.bcu[1:], bcp=[])

        return BC

    # Dataframe utility
    def _make_colname_df(self, name, column_nr: int) -> list[str]:
        """Return future column names for sensor measurements or control input in DataFrame.

        Args:
            name (str): usually y_meas or u_ctrl
            column_nr (int): number of columns to generate

        Returns:
            list[str]: [name_1, name_2, ...]
        """

        return [name + "_" + str(i + 1) for i in range(column_nr)]

    def _assign_to_df(self, df: pd.DataFrame, name, value: float, index: int) -> None:
        """Assign measurement array to timeseries at given index."""
        df.loc[index, self._make_colname_df(name, len(value))] = value

    def write_timeseries(self) -> None:
        """Write timeseries (pandas DataFrame) to file."""
        if flu.MpiUtils.get_rank() == 0:  # TODO async?
            # zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths["timeseries"], sep=",", index=False)

    def _log_timeseries(
        self, u_ctrl: float, y_meas: float, dE: float, t: float, runtime: float
    ) -> None:
        """Fill timeseries with simulation data at given index."""
        self._assign_to_df(
            df=self.timeseries, name="u_ctrl", value=u_ctrl, index=self.iter - 1
        )
        self._assign_to_df(
            df=self.timeseries, name="y_meas", value=y_meas, index=self.iter
        )
        self.timeseries.loc[self.iter, "dE"] = dE
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    # General utility
    def compute_energy(self) -> float:
        """Compute perturbation kinetic energy (PKE) of flow.

        Returns:
            float: PKE
        """
        dE = 1 / 2 * dolfin.norm(self.fields.u_, norm_type="L2", mesh=self.mesh) ** 2
        return dE

    def compute_energy_field(
        self, export: bool = False, filename: str = None
    ) -> dolfin.Function:
        """Compute perturbation field dot(u, u) of spatial location of PKE.

        Args:
            export (bool, optional): if export then write xdmf file. Defaults to False.
            filename (str, optional): if export then write xdmf file at filename. Defaults to None.

        Returns:
            dolfin.Function: spatialization of PKE
        """
        Efield = dot(self.fields.u_, self.fields.u_)
        # Note: E = 1/2 * assemble(Efield * fs.dx)
        Efield = flu.projectm(Efield, self.P)  # project to deg 1
        if export:
            flu.write_xdmf(filename, Efield, "E")
        return Efield

    def get_subdomain(self, name) -> dolfin.SubDomain | dolfin.CompiledSubDomain:
        return self.boundaries.loc[name].subdomain

    def _default_steady_state_initial_guess(self) -> dolfin.UserExpression:
        """Default initial guess for computing steady state. The method may
        be overriden to propose an initial guess deemed closer to the steady state."""

        class default_initial_guess(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        return default_initial_guess()

    def _default_initial_perturbation(
        self, xloc: float = 0.0, yloc: float = 0.0, radius: float = 1.0
    ) -> dolfin.Function:
        """Default perturbation added to the initial state, modulated by the amplitude
        self.params_solver.ic_add_perturbation (float)."""
        u_nodiv = flu2.get_div0_u(self, xloc=xloc, yloc=yloc, size=radius)
        p_default = flu.projectm(self.fields.STEADY.p, self.P)
        return self.merge(u=u_nodiv, p=p_default)

    def _load_actuators(self) -> None:
        """Load expressions from actuators in actuator_list"""
        for actuator in self.params_control.actuator_list:
            actuator.load_expression(self)

    def _load_sensors(self) -> None:
        """Load sensors, in particular SensorIntegral"""
        for sensor in self.params_control.sensor_list:
            if sensor.require_loading:
                sensor.load(self)

    def make_measurement(
        self,
        up: dolfin.Function,
    ) -> np.ndarray:
        """Define procedure for extracting a measurement from a given
        mixed field (u,v,p)."""
        y_meas = np.zeros((self.params_control.sensor_number,))

        for ii, sensor_i in enumerate(self.params_control.sensor_list):
            y_meas[ii] = sensor_i.eval(up=up)

        return y_meas

    # Abstract methods
    @abstractmethod
    def _make_boundaries(self) -> pd.DataFrame:
        """Define boundaries of the mesh (geometry only)
        as dolfin.Subdomain or dolfin.CompiledSubDomain.
        This method should return a pandas DataFrame
        containing each boundary and its associated name.

        Returns:
            pd.DataFrame: boundaries of mesh with column "subdomain" and boundaries names as index
        """
        pass

    @abstractmethod
    def _make_bcs(self) -> BoundaryConditions:
        """Define boundary conditions on previously defined boundaries.
        This method should return a dictionary containing two lists:
        boundary conditions for (u,v) and boundary conditions for (p).

        Returns:
            BoundaryConditions: boundary conditions for perturbation field as dataclass object
        """
        pass
