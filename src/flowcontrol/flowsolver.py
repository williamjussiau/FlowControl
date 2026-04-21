"""Refactored FlowSolver using NSForms, SteadyStateSolver, and FlowExporter.

This module is a drop-in replacement for flowsolver.py.  The public API
(constructor signature, method names, attributes) is unchanged so that
existing subclasses (CylinderFlowSolver, etc.) and user scripts continue to
work without modification.

Internal changes vs flowsolver.py:
- Variational forms delegated to NSForms (nsforms.py)
- Newton/Picard steady-state iteration delegated to SteadyStateSolver (steadystate.py)
- All I/O and timeseries logging delegated to FlowExporter (exporter.py)
- step() is shorter: control → solve → shift → log, each in one place
- _make_varf / _make_varf_order1/2 / _make_varf_steady removed (live in NSForms)
- _compute_steady_state_newton/picard removed (live in SteadyStateSolver)
- _export_fields_xdmf / write_timeseries / _log_timeseries / _initialize_timeseries
  removed (live in FlowExporter)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import dolfin
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import flowcontrol.flowsolverparameters as flowsolverparameters
from utils.fem import projectm
from utils.io import read_xdmf, write_xdmf
from utils.mpi import get_rank
from utils.physics import get_div0_u
from flowcontrol.actuator import ACTUATOR_TYPE
from flowcontrol.exporter import FlowExporter
from flowcontrol.flowfield import (
    BoundaryConditions,
    FlowField,
    FlowFieldCollection,
    SimPaths,
)
from flowcontrol.nsforms import NSForms
from flowcontrol.steadystate import SteadyStateSolver

logger = logging.getLogger(__name__)


class FlowSolver(ABC):
    """Abstract base class for flow simulation and control.

    Subclasses must implement:
        _make_boundaries() -> pd.DataFrame
        _make_bcs()        -> BoundaryConditions
    """

    def __init__(
        self,
        params_flow: flowsolverparameters.ParamFlow,
        params_time: flowsolverparameters.ParamTime,
        params_save: flowsolverparameters.ParamSave,
        params_solver: flowsolverparameters.ParamSolver,
        params_mesh: flowsolverparameters.ParamMesh,
        params_control: flowsolverparameters.ParamControl,
        params_ic: flowsolverparameters.ParamIC,
        params_restart: Optional[flowsolverparameters.ParamRestart] = None,
        verbose: int = 1,
    ) -> None:
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
            params_control,
            params_ic,
        ]:
            logger.debug(param)

        self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Run the full setup sequence: mesh, spaces, boundaries, actuators, forms, exporter."""
        self.fields = FlowFieldCollection()
        self.E0: float = 0.0

        self.paths = self._define_paths()
        self.mesh = self._make_mesh()
        self.V, self.P, self.W = self._make_function_spaces()
        self.boundaries = self._make_boundaries()  # abstract
        self._mark_boundaries()
        # Actuators must be loaded before _make_bcs — their expressions may be
        # referenced by BC factories in subclasses.
        self._load_actuators()
        self._load_sensors()
        self.bc = self._make_bcs()  # abstract
        self._function_assigner = dolfin.FunctionAssigner(self.W, [self.V, self.P])

        self.forms = NSForms(
            W=self.W,
            Re=self.params_flow.Re,
            dt=self.params_time.dt,
            is_nonlinear=self.params_solver.is_eq_nonlinear,
            shift=self.params_solver.shift,
        )
        self.exporter = FlowExporter(
            paths=self.paths,
            fields=self.fields,
            V=self.V,
            P=self.P,
            Tstart=self.params_time.Tstart,
            dt=self.params_time.dt,
            save_every=self.params_save.save_every,
        )

    # ── Path / mesh / function spaces ─────────────────────────────────────────

    def _define_paths(self) -> SimPaths:
        """Build SimPaths from param objects, deriving all output and restart file names."""
        def ext(T: float) -> str:
            return f"_restart{T:.3f}".replace(".", ",")

        Tstart = self.params_time.Tstart
        # Trestartfrom is only used to build the legacy load-paths (U, P, Uprev).
        # These are fallback paths consulted by _find_restart_from_params when no
        # JSON sidecar is available; they are not used in the common case.
        Trestartfrom = self.params_restart.Trestartfrom if self.params_restart else 0.0
        path_out = self.params_save.path_out

        return SimPaths(
            U0=path_out / "steady" / "U0.xdmf",
            P0=path_out / "steady" / "P0.xdmf",
            steady_meta=path_out / "steady" / "meta.json",
            U=path_out / ("U" + ext(Trestartfrom) + ".xdmf"),
            P=path_out / ("P" + ext(Trestartfrom) + ".xdmf"),
            Uprev=path_out / ("Uprev" + ext(Trestartfrom) + ".xdmf"),
            U_restart=path_out / ("U" + ext(Tstart) + ".xdmf"),
            Uprev_restart=path_out / ("Uprev" + ext(Tstart) + ".xdmf"),
            P_restart=path_out / ("P" + ext(Tstart) + ".xdmf"),
            timeseries=path_out / ("timeseries1D" + ext(Tstart) + ".csv"),
            metadata=path_out / ("meta" + ext(Tstart) + ".json"),
            mesh=self.params_mesh.meshpath,
        )

    def _make_mesh(self) -> dolfin.Mesh:
        """Read the XDMF mesh file and return a dolfin.Mesh."""
        logger.info(f"Mesh @ {self.params_mesh.meshpath}")
        mesh = dolfin.Mesh(dolfin.MPI.comm_world)
        with dolfin.XDMFFile(
            dolfin.MPI.comm_world, str(self.params_mesh.meshpath)
        ) as f:
            f.read(mesh)
        logger.info(f"Mesh has {mesh.num_entities_global(mesh.topology().dim())} cells (global)")
        return mesh

    def _make_function_spaces(self) -> tuple[dolfin.FunctionSpace, ...]:
        """Create Taylor-Hood P2/P1 velocity, pressure, and mixed function spaces."""
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        V = dolfin.FunctionSpace(self.mesh, Ve)
        P = dolfin.FunctionSpace(self.mesh, Pe)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Pe]))
        logger.debug(f"DOFs: {W.dim()} ({V.dim()} velocity + {P.dim()} pressure)")
        return V, P, W

    def _mark_boundaries(self) -> None:
        """Mark each boundary subdomain and build ds/dx integration measures."""
        self.bnd_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        cell_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim()
        )
        indices = []
        for i, row in enumerate(self.boundaries.itertuples()):
            row.subdomain.mark(self.bnd_markers, i)
            row.subdomain.mark(cell_markers, i)
            indices.append(i)
        self.boundaries["idx"] = indices
        self.ds = dolfin.Measure("ds", domain=self.mesh, subdomain_data=self.bnd_markers)
        self.dx = dolfin.Measure("dx", domain=self.mesh, subdomain_data=cell_markers)

    # ── Actuators / sensors ───────────────────────────────────────────────────

    def _load_actuators(self) -> None:
        """Call load_expression on every actuator against the current mesh and spaces."""
        for actuator in self.params_control.actuator_list:
            actuator.load_expression(self)

    def _load_sensors(self) -> None:
        """Call load() on sensors that require post-setup initialization (e.g. integral sensors)."""
        for sensor in self.params_control.sensor_list:
            if sensor.require_loading:
                sensor.load(self)

    def set_actuators_u_ctrl(self, u_ctrl: Iterable) -> None:
        u_ctrl = list(u_ctrl)
        if len(u_ctrl) != self.params_control.actuator_number:
            raise ValueError(
                f"Expected {self.params_control.actuator_number} control inputs, "
                f"got {len(u_ctrl)}"
            )
        for actuator, val in zip(self.params_control.actuator_list, u_ctrl):
            actuator.expression.u_ctrl = val

    def flush_actuators_u_ctrl(self) -> None:
        """Set all actuator control amplitudes to zero."""
        self.set_actuators_u_ctrl([0] * self.params_control.actuator_number)

    def get_actuators_u_ctrl(self) -> list:
        """Return the current u_ctrl amplitude of each actuator as a list."""
        return [a.expression.u_ctrl for a in self.params_control.actuator_list]

    def _gather_actuators_expressions(self) -> dolfin.Expression | dolfin.Constant:
        """Sum all FORCE-type actuator expressions; return a zero Constant if none are present."""
        forces = [
            a.expression
            for a in self.params_control.actuator_list
            if a.actuator_type is ACTUATOR_TYPE.FORCE
        ]
        return sum(forces, dolfin.Constant((0, 0)))

    def make_measurement(self, up: dolfin.Function) -> NDArray[np.float64]:
        """Evaluate all sensors on the given mixed field and return the measurement vector."""
        return np.array([sensor.eval(up=up) for sensor in self.params_control.sensor_list])

    # ── Boundary conditions ───────────────────────────────────────────────────

    def _make_BCs(self) -> BoundaryConditions:
        """Build full-field BCs: uniform inlet profile merged with perturbation-field side BCs."""
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.params_flow.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        bcs = self._make_bcs()
        return BoundaryConditions(bcu=[bcu_inlet] + bcs.bcu[1:], bcp=[])

    # ── Steady state ──────────────────────────────────────────────────────────

    def compute_steady_state(
        self,
        u_ctrl: list,
        method: str = "newton",
        initial_guess: Optional[dolfin.Function] = None,
        max_iter: int = 10,
        **kwargs,
    ) -> None:
        """Compute steady state and store it in self.fields.U0/P0/UP0."""
        self.set_actuators_u_ctrl(u_ctrl)
        f = self._gather_actuators_expressions()

        UP0 = self._define_initial_guess(initial_guess)
        ss = SteadyStateSolver(
            W=self.W,
            bcu=self._make_BCs().bcu,
            forms=self.forms,
            verbose=bool(self.verbose),
        )

        if method == "newton":
            UP0 = ss.newton(UP0, f=f, max_iter=max_iter, **kwargs)
        elif method == "picard":
            UP0 = ss.picard(UP0, f=f, max_iter=max_iter, **kwargs)
        else:
            raise ValueError(f"method must be 'newton' or 'picard', got {method!r}")

        U0, P0 = UP0.split(deepcopy=True)
        U0 = projectm(U0, self.V)
        P0 = projectm(P0, self.P)

        if self.params_save.save_every:
            write_xdmf(
                self.paths.U0, U0, "U0", time_step=0.0, append=False, write_mesh=True
            )
            write_xdmf(
                self.paths.P0, P0, "P0", time_step=0.0, append=False, write_mesh=True
            )
            if get_rank() == 0:
                self.paths.steady_meta.parent.mkdir(parents=True, exist_ok=True)
                self.paths.steady_meta.write_text(
                    json.dumps({"mesh_cells": self.mesh.num_entities_global(self.mesh.topology().dim())}, indent=2)
                )

        self._assign_steady_state(U0, P0)

    def load_steady_state(self, path_u_p: Optional[Sequence[Path]] = None) -> None:
        """Load U0/P0 from XDMF files, verify mesh compatibility, and store in fields."""
        paths = path_u_p or (self.paths.U0, self.paths.P0)
        self._check_steady_state_compatible(Path(paths[0]))
        U0 = dolfin.Function(self.V)
        P0 = dolfin.Function(self.P)
        read_xdmf(paths[0], U0, "U0")
        read_xdmf(paths[1], P0, "P0")
        self._assign_steady_state(U0, P0)

    def _check_steady_state_compatible(self, u0_path: Path) -> None:
        """Raise ValueError if the steady-state checkpoint was written on a different mesh."""
        if get_rank() != 0:
            return  # rank-0 only — sidecar is written and checked by a single process
        meta_path = u0_path.parent / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except FileNotFoundError:
            return  # no sidecar — can't verify, proceed
        stored = meta.get("mesh_cells")
        current = self.mesh.num_entities_global(self.mesh.topology().dim())
        if stored is not None and stored != current:
            raise ValueError(
                f"Steady-state checkpoint at {u0_path.parent} was written with "
                f"{stored} mesh cells, but the current mesh has {current}. "
                "Load a checkpoint from the same mesh, or recompute the steady state."
            )

    def _assign_steady_state(self, U0: dolfin.Function, P0: dolfin.Function) -> None:
        """Store U0/P0 in fields, build the mixed UP0, and cache the base-flow energy E0."""
        self.fields.U0 = U0
        self.fields.P0 = P0
        self.fields.UP0 = self.merge(U0, P0)
        self.E0 = 0.5 * dolfin.norm(U0, norm_type="L2", mesh=self.mesh) ** 2

    def _define_initial_guess(
        self, initial_guess: Optional[dolfin.Function] = None
    ) -> dolfin.Function:
        """Return a valid initial guess for the steady-state solver.

        Falls back to a uniform-flow field at uinf when none is provided.
        """
        if initial_guess is None:
            logger.info(
                "Steady-state solver — no initial guess provided, using default"
            )
            UP0 = dolfin.Function(self.W)
            UP0.interpolate(self._default_steady_state_initial_guess())
        else:
            logger.info("Steady-state solver — using provided initial guess")
            UP0 = initial_guess
        return UP0

    # ── Time stepping ─────────────────────────────────────────────────────────

    def initialize_time_stepping(
        self, Tstart: float = 0.0, ic: Optional[dolfin.Function] = None
    ) -> None:
        restart_order = self.params_restart.restart_order if self.params_restart else "n/a"
        logger.info(f"Initialising from t={Tstart}, restart_order={restart_order}")

        if Tstart == 0.0:
            u_, p_, u_n, u_nn, p_n = self._initialize_with_ic(ic)
        else:
            u_, p_, u_n, u_nn, p_n = self._initialize_at_time(Tstart)

        self.fields.u_ = u_
        self.fields.p_ = p_
        self.fields.u_n = u_n
        self.fields.u_nn = u_nn
        self.fields.p_n = p_n

        self.first_step = True
        self.exporter.reset()
        self.y_meas = self.make_measurement(up=self.fields.ic.up)
        self.exporter.log_ic(
            t=self.params_time.Tstart,
            y_meas=self.y_meas,
            dE=self.compute_perturbation_energy(),
        )

    def _initialize_with_ic(
        self, ic: Optional[dolfin.Function] = None
    ) -> tuple[dolfin.Function, ...]:
        """Initialise time-stepping fields from an initial condition at t=0.

        Starts from zero perturbation when ic is None. A non-zero ParamIC amplitude
        adds a divergence-free Gaussian perturbation on top.

        Returns
        -------
        u_, p_, u_n, u_nn, p_n
            Perturbation-field dolfin.Functions for the time-stepper.
        """
        self.order = "cn" if self.params_solver.time_scheme == "cn" else 1
        self.iter = 0
        self.t = self.params_time.Tstart

        if ic is None:
            self.fields.ic = FlowField(up=dolfin.Function(self.W))
        else:
            self.fields.ic = FlowField(up=ic)

        if self.params_ic.amplitude:
            ic_pert = self._default_initial_perturbation(
                xloc=self.params_ic.xloc,
                yloc=self.params_ic.yloc,
                radius=self.params_ic.radius,
            )
            self.fields.ic.up.vector()[:] += (
                self.params_ic.amplitude * ic_pert.vector()[:]
            )
            self.fields.ic.up.vector().apply("insert")
            # Reconstruct so that ic.u / ic.p reflect the mutated vector.
            self.fields.ic = FlowField(self.fields.ic.up)

        u_n = projectm(v=self.fields.ic.u, V=self.V, bcs=self.bc.bcu)
        u_nn = u_n.copy(deepcopy=True)
        p_n = projectm(self.fields.ic.p, self.P)
        u_ = u_n.copy(deepcopy=True)
        p_ = p_n.copy(deepcopy=True)

        if self.params_save.save_every:
            self.exporter.export_xdmf(
                u_n,
                u_nn,
                p_n,
                time=0.0,
                append=False,
                write_mesh=True,
                adjust_baseflow=1.0,
            )

        return u_, p_, u_n, u_nn, p_n

    def _find_restart_source(self, Tstart: float) -> tuple[dict, int, Path]:
        """Return (metadata dict, counter, base_dir) for restarting at Tstart.

        Tries JSON sidecars first; falls back to ParamRestart if none found.
        """
        result = self._find_restart_from_json(Tstart)
        if result is not None:
            return result
        return self._find_restart_from_params(Tstart)

    def _find_restart_from_json(
        self, Tstart: float
    ) -> Optional[tuple[dict, int, Path]]:
        """Scan path_out for JSON sidecars and return the one covering Tstart."""
        path_out = self.params_save.path_out
        for json_path in sorted(path_out.glob("meta_restart*.json")):
            meta = json.loads(json_path.read_text())
            T0 = meta["Tstart"]
            step = meta["dt"] * meta["save_every"]
            n = meta["checkpoints_written"]
            if n == 0:
                continue
            Tend = T0 + step * n
            if T0 - 1e-10 <= Tstart <= Tend + 1e-10:
                counter = round((Tstart - T0) / step)
                logger.info(
                    f"Restart: found JSON sidecar {json_path.name}, counter={counter}"
                )
                return meta, counter, path_out
        return None

    def _find_restart_from_params(self, Tstart: float) -> tuple[dict, int, Path]:
        """Legacy fallback: derive restart info from ParamRestart fields."""
        if self.params_restart is None:
            raise FileNotFoundError(
                f"No JSON metadata sidecar found covering Tstart={Tstart} in "
                f"{self.params_save.path_out}, and no ParamRestart was provided."
            )
        pr = self.params_restart
        step = pr.dt_old * pr.save_every_old
        counter = round((Tstart - pr.Trestartfrom) / step)
        meta = {
            "restart_order": pr.restart_order,
            "files": {
                "U": self.paths.U.name,
                "Uprev": self.paths.Uprev.name,
                "P": self.paths.P.name,
            },
        }
        logger.info(f"Restart: using legacy ParamRestart, counter={counter}")
        return meta, counter, self.params_save.path_out

    def _initialize_at_time(self, Tstart: float) -> tuple[dolfin.Function, ...]:
        """Restart time-stepping from a checkpoint at Tstart > 0.

        Reads full-field (U, P) snapshots from disk, subtracts the base flow to
        recover perturbation fields, and writes the first XDMF checkpoint frame.

        Returns
        -------
        u_, p_, u_n, u_nn, p_n
            Perturbation-field dolfin.Functions for the time-stepper.
        """
        meta, counter, base_dir = self._find_restart_source(Tstart)
        self.order = meta["restart_order"]
        self.iter = 0
        self.t = Tstart

        U_path = base_dir / meta["files"]["U"]
        Uprev_path = base_dir / meta["files"]["Uprev"]
        P_path = base_dir / meta["files"]["P"]

        U_ = dolfin.Function(self.V)
        P_ = dolfin.Function(self.P)
        U_n = dolfin.Function(self.V)
        U_nn = dolfin.Function(self.V)
        P_n = dolfin.Function(self.P)

        # U_, U_n, U_nn, P_, P_n are full fields (base flow + perturbation),
        # as written by export_xdmf with adjust_baseflow=1.0.
        read_xdmf(U_path, U_, "U", counter=counter)
        read_xdmf(P_path, P_, "P", counter=counter)
        read_xdmf(U_path, U_n, "U", counter=counter)   # same snapshot as U_
        read_xdmf(Uprev_path, U_nn, "U_n", counter=counter)
        read_xdmf(P_path, P_n, "P", counter=counter)

        if self.params_save.save_every:
            # Full field already loaded — no base-flow adjustment needed.
            self.exporter.export_xdmf(
                U_n,
                U_nn,
                P_n,
                time=Tstart,
                append=False,
                write_mesh=True,
                adjust_baseflow=0.0,
            )

        # subtract base flow to recover perturbation fields
        U0v = self.fields.U0.vector()[:]
        P0v = self.fields.P0.vector()[:]

        u_ = dolfin.Function(self.V)
        u_n = dolfin.Function(self.V)
        u_nn = dolfin.Function(self.V)
        p_ = dolfin.Function(self.P)
        p_n = dolfin.Function(self.P)

        for pert, full in [(u_, U_), (u_n, U_n), (u_nn, U_nn)]:
            pert.vector()[:] = full.vector()[:] - U0v
            pert.vector().apply("insert")
        for pert, full in [(p_, P_), (p_n, P_n)]:
            pert.vector()[:] = full.vector()[:] - P0v
            pert.vector().apply("insert")

        self.fields.ic = FlowField(up=self.merge(u_, p_))
        return u_, p_, u_n, u_nn, p_n

    def _prepare_systems(
        self,
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
    ) -> None:
        """Assemble LHS matrices and pre-allocate RHS for the chosen scheme."""
        U0 = self.fields.U0
        f = self._gather_actuators_expressions()

        self.assemblers: dict[int | str, dolfin.SystemAssembler] = {}
        self.solvers: dict[int | str, Any] = {}
        self.rhs = dolfin.Vector()

        scheme = self.params_solver.time_scheme
        orders = ("cn",) if scheme == "cn" else (1, 2)

        for order in orders:
            F = self.forms.transient(order=order, U0=U0, u_n=u_n, u_nn=u_nn, f=f)
            a = dolfin.lhs(F)
            L = dolfin.rhs(F)
            assembler = dolfin.SystemAssembler(a, L, self.bc.bcu)
            solver = self._make_solver(order=order)
            A = dolfin.Matrix()
            assembler.assemble(A)
            solver.set_operator(A)
            self.assemblers[order] = assembler
            self.solvers[order] = solver

        self._up_work = dolfin.Function(self.W)  # reused every step to avoid per-step allocation

    def step(self, u_ctrl: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """Advance the simulation by one time step.

        Parameters
        ----------
        u_ctrl:
            Control amplitudes for each actuator.

        Returns
        -------
        NDArray[np.float64]
            Measurement vector y after the step, or None if the solver
            diverged and params_solver.throw_error is False.
        """
        if self.first_step:
            self._prepare_systems(self.fields.u_n, self.fields.u_nn)
            self.first_step = False

        t0 = time.time()

        # Apply control
        self.set_actuators_u_ctrl(u_ctrl)

        # Solve
        try:
            self.assemblers[self.order].assemble(self.rhs)
            self.solvers[self.order].solve(self._up_work.vector(), self.rhs)
            u_, p_ = self._up_work.split(deepcopy=True)
            if self._solver_diverged(u_):
                raise RuntimeError()
        except RuntimeError:
            logger.critical("Solver diverged (Inf detected)")
            if not self.params_solver.throw_error:
                return None
            raise RuntimeError("Failed solving: Inf found in solution")

        # Advance time
        self.iter += 1
        self.t = self.params_time.Tstart + self.iter * self.params_time.dt
        if self.params_solver.time_scheme != "cn":
            self.order = 2

        # Update fields
        self.fields.u_ = u_
        self.fields.p_ = p_
        self.fields.up_ = self._up_work
        self.fields.u_nn.assign(self.fields.u_n)
        self.fields.u_n.assign(u_)
        self.fields.p_n.assign(p_)

        # Measure
        self.y_meas = self.make_measurement(up=self.fields.up_)
        runtime = time.time() - t0

        if self._niter_multiple_of(self.iter, self.verbose):
            self.exporter.log_progress(
                self.iter,
                self.params_time.num_steps,
                self.t,
                self.params_time.Tfinal + self.params_time.Tstart,
                runtime,
            )

        # Log and export
        at_checkpoint = self._niter_multiple_of(self.iter, self.params_save.save_every)
        dE = self.compute_perturbation_energy() if self._niter_multiple_of(self.iter, self.params_save.energy_every) else np.nan
        self.exporter.log(
            u_ctrl=u_ctrl,
            y_meas=self.y_meas,
            dE=dE,
            t=self.t,
            runtime=runtime,
        )
        if at_checkpoint:
            self.exporter.export_xdmf(
                self.fields.u_n,
                self.fields.u_nn,
                self.fields.p_n,
                time=self.t,
                adjust_baseflow=1.0,
            )
            self.exporter.write_metadata()
            self.exporter.write_timeseries()

        return self.y_meas

    def write_timeseries(self) -> None:
        """Write the accumulated timeseries to CSV."""
        self.exporter.write_timeseries()

    @property
    def timeseries(self) -> pd.DataFrame:
        """Current timeseries as a DataFrame (built on access)."""
        return self.exporter.to_dataframe()

    # ── Solver helpers ────────────────────────────────────────────────────────

    def _make_solver(self, order: int | str) -> Any:
        """Return a MUMPS LU solver. Override to substitute a different linear solver."""
        return dolfin.LUSolver("mumps")

    def _solver_diverged(self, field: dolfin.Function) -> bool:
        """Return True if any MPI rank has a non-finite value in the velocity field."""
        local = not np.all(np.isfinite(field.vector().get_local()))
        return bool(dolfin.MPI.max(dolfin.MPI.comm_world, int(local)))

    def _niter_multiple_of(self, iter: int, divider: int) -> bool:
        """Return True when iter is a positive multiple of divider (False if divider is 0)."""
        return bool(divider and not iter % divider)

    # ── Energy ────────────────────────────────────────────────────────────────

    def compute_perturbation_energy(self) -> float:
        """Return ½‖u'‖²_L2, the kinetic energy of the current perturbation field."""
        return 0.5 * dolfin.norm(self.fields.u_, norm_type="L2", mesh=self.mesh) ** 2

    def compute_energy_field(
        self, export: bool = False, filename: Optional[Path | str] = None
    ) -> dolfin.Function:
        """Project u'·u' onto a P4 space and optionally write it to XDMF.

        Returns a scalar energy-density field on the P4 space.
        """
        # u_·u_ is degree 4 (product of two P2 fields) — P4 represents it exactly.
        P4 = dolfin.FunctionSpace(self.mesh, "CG", 4)
        Efield = projectm(dolfin.dot(self.fields.u_, self.fields.u_), P4)
        if export:
            write_xdmf(filename, Efield, "E")
        return Efield

    # ── Utilities ─────────────────────────────────────────────────────────────

    def merge(self, u: dolfin.Function, p: dolfin.Function) -> dolfin.Function:
        """Assign (u, p) into a new mixed-space function and return it."""
        up = dolfin.Function(self.W)
        self._function_assigner.assign(up, [u, p])
        return up

    def get_subdomain(self, name: str) -> dolfin.SubDomain | dolfin.CompiledSubDomain:
        """Look up a named boundary subdomain from the boundaries DataFrame."""
        return self.boundaries.loc[name].subdomain

    # ── Default IC / perturbation ─────────────────────────────────────────────

    def _default_steady_state_initial_guess(self) -> dolfin.UserExpression:
        """Return a uniform-flow expression at uinf as starting guess for the steady-state solver."""
        uinf = self.params_flow.uinf

        class _UniformFlow(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = uinf
                value[1] = 0.0
                value[2] = 0.0

            def value_shape(self):
                return (3,)

        return _UniformFlow()

    def _default_initial_perturbation(
        self, xloc: float = 0.0, yloc: float = 0.0, radius: float = 1.0
    ) -> dolfin.Function:
        """Return the default initial perturbation field (delegates to _perturbation_div0)."""
        return self._perturbation_div0(xloc, yloc, radius)

    def _perturbation_div0(
        self, xloc: float = 0.0, yloc: float = 0.0, radius: float = 1.0
    ) -> dolfin.Function:
        """Build a divergence-free Gaussian perturbation field merged with the base-flow pressure."""
        u_nodiv = get_div0_u(self, xloc=xloc, yloc=yloc, size=radius)
        p_default = projectm(self.fields.P0, self.P)
        return self.merge(u=u_nodiv, p=p_default)

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    def _make_boundaries(self) -> pd.DataFrame:
        """Return a DataFrame with a 'subdomain' column and boundary names as index."""
        pass

    @abstractmethod
    def _make_bcs(self) -> BoundaryConditions:
        """Return perturbation-field boundary conditions.

        The first entry of bcu MUST be the inlet BC — _make_BCs() replaces it
        with the full-field uniform profile.
        """
        pass
