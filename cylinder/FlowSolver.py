from __future__ import print_function
import FlowSolverParameters
import dolfin
from dolfin import dot, nabla_grad, dx, inner, div
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod

# from petsc4py import PETSc

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
        params_flow: FlowSolverParameters.Param_control,
        params_time: FlowSolverParameters.Param_time,
        params_save: FlowSolverParameters.Param_save,
        params_solver: FlowSolverParameters.Param_solver,
        params_mesh: FlowSolverParameters.Param_mesh,
        verbose: bool = True,
    ):
        self.params_flow = params_flow
        self.params_time = params_time
        self.params_save = params_save
        self.params_solver = params_solver
        self.params_mesh = params_mesh

        self.verbose = verbose
        self.params_time.Tf = self.params_time.num_steps * self.params_time.dt

        # Sensors
        self.params_flow.sensor_nr = self.params_flow.sensor_location.shape[0]
        # Initial state default
        self.initial_state = None

        # Save
        # params_save should contain savedir0
        self.define_paths()
        # shortcuts (self-explanatory)
        self.make_mesh()
        self.make_function_spaces()
        self.make_boundaries()
        self.make_actuator()
        # self.make_sensor()
        self.make_bcs()

        # for energy computation
        self.u_ = dolfin.Function(self.V)

    def define_paths(self):  # TODO move to superclass
        """Define attribute (dict) containing useful paths (save, etc.)"""
        logger.debug("Currently defining paths...")
        # Files location directory is params_save['savedir0']
        # dunnu touch below
        savedir0 = self.params_save.savedir0
        Tstart = self.params_time.Tstart  # start simulation from time...
        Trestartfrom = (
            self.params_time.Trestartfrom
        )  # use older files starting from time...

        def make_extension(T):
            return "_restart" + str(np.round(T, decimals=3)).replace(".", ",")

        file_start = make_extension(Tstart)
        file_restart = make_extension(Trestartfrom)

        ext_xdmf = ".xdmf"
        ext_csv = ".csv"

        savedir0 = self.params_save.savedir0

        filename_u0 = savedir0 / "steady" / ("u0" + ext_xdmf)
        filename_p0 = savedir0 / "steady" / ("p0" + ext_xdmf)

        filename_u = savedir0 / ("u" + file_restart + ext_xdmf)
        filename_uprev = savedir0 / ("uprev" + file_restart + ext_xdmf)
        filename_p = savedir0 / ("p" + file_restart + ext_xdmf)

        filename_u_restart = savedir0 / ("u" + file_start + ext_xdmf)
        filename_uprev_restart = savedir0 / ("uprev" + file_start + ext_xdmf)
        filename_p_restart = savedir0 / ("p" + file_start + ext_xdmf)

        filename_timeseries = savedir0 / ("timeseries1D" + file_start + ext_csv)

        self.paths = {
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
        # Set params
        # genmesh = self.genmesh
        # meshdir = self.paths["mesh"]  #'/stck/wjussiau/fenics-python/mesh/'
        # xinf = self.xinf  # 20 # 20 # 20
        # yinf = self.yinf  # 8 # 5 # 8
        # xinfa = self.xinfa  # -5 # -5 # -10
        # Working as follows:
        # if genmesh:
        #   if does not exist (with given params): generate with meshr
        #   and prepare to not read file (because mesh is already in memory)
        # else:
        #   set file name and prepare to read file
        # read file
        # readmesh = True
        # if genmesh:
        #     nx = self.nx  # 32
        #     meshname = "cylinder_" + str(nx) + ".xdmf"
        #     meshpath = meshdir / meshname  # os.path.join(meshdir, meshname)
        #     if not os.path.exists(meshpath) or self.remesh:
        #         if self.verbose:
        #             logger.debug("Mesh does not exist @: %s", meshpath)
        #             logger.debug("-- Creating mesh...")
        #         channel = dolfin.Rectangle(
        #             dolfin.Point(xinfa, -yinf), dolfin.Point(xinf, yinf)
        #         )
        #         cyl = dolfin.Circle(
        #             dolfin.Point(0.0, 0.0), self.d / 2, segments=self.segments
        #         )
        #         domain = channel - cyl
        #         mesh = dolfin.generate_mesh(domain, nx)
        #         with dolfin.XDMFFile(dolfin.MPI.comm_world, str(meshpath)) as fm:
        #             fm.write(mesh)
        #         readmesh = False
        # else:

        # readmesh = True

        # meshname = self.meshname

        # if mesh was not generated on the fly, read file
        # if readmesh:
        mesh = dolfin.Mesh(dolfin.MPI.comm_world)
        meshpath = (
            self.params_mesh.meshpath / self.params_mesh.meshname
        )  # os.path.join(meshdir, meshname)
        if self.verbose:
            logger.debug("Mesh exists @: %s", meshpath)
            logger.debug("--- Reading mesh...")
        with dolfin.XDMFFile(dolfin.MPI.comm_world, str(meshpath)) as fm:
            fm.read(mesh)
        # mesh = Mesh(dolfin.MPI.comm_world, meshpath) # if xml

        if self.verbose:
            logger.debug("Mesh has: %d cells" % (mesh.num_cells()))

        # assign mesh & facet normals
        self.mesh = mesh
        self.n = dolfin.FacetNormal(mesh)

    def make_function_spaces(self):  # TODO move to superclass
        """Define function spaces (u, p) = (CG2, CG1)"""
        # dolfin.Function spaces on mesh
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)  # was 'P'
        Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)  # was 'P'
        We = dolfin.MixedElement([Ve, Pe])
        self.V = dolfin.FunctionSpace(self.mesh, Ve)
        self.P = dolfin.FunctionSpace(self.mesh, Pe)
        self.W = dolfin.FunctionSpace(self.mesh, We)
        if self.verbose:
            logger.debug(
                "Function Space [V(CG2), P(CG1)] has: %d DOFs" % (self.W.dim())
            )

    def init_time_stepping(self):  # TODO move to superclass
        """Create varitional functions/forms & flush files & define u(0), p(0)"""
        t = self.params_time.Tstart
        self.t = t
        self.iter = 0

        # solutions
        u_ = dolfin.Function(self.V)
        p_ = dolfin.Function(self.P)

        # if not restart
        if self.params_time.Tstart == 0:
            # first order temporal integration
            self.order = 1

            # Set initial state up in W
            initial_up = dolfin.Function(self.W)

            # No initial state given -> base flow
            if self.initial_state is None:
                initial_up = dolfin.Function(self.W)
                # if not self.perturbations:
                #     initial_up.vector()[:] += self.up0.vector()[:]
            else:
                initial_up = self.initial_state

            # Impulse or state perturbation @ div0
            # Impulse if self.init_pert is inf
            if np.isinf(self.params_solver.init_pert):
                # not sure this would work in parallel
                initial_up.vector()[:] += self.get_B().reshape((-1,))
            else:
                udiv0 = flu2.get_div0_u(self, xloc=2, yloc=0, size=0.5)
                fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
                pert0 = dolfin.Function(self.W)
                fa.assign(pert0, [udiv0, self.p0])
                initial_up.vector()[:] += (
                    self.params_solver.init_pert * pert0.vector()[:]
                )

            initial_up.vector().apply("insert")
            up1 = initial_up

            # Split up to u, p
            fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            u1 = dolfin.Function(self.V)
            p1 = dolfin.Function(self.P)
            fa.assign([u1, p1], up1)

            # this is the initial state
            # if self.perturbations:
            bcs = self.bc_p["bcu"]  # bcs for perturbation formulation
            # else:
            # bcs = self.bc["bcu"]  # bcs for classic formulation
            u_n = flu.projectm(v=u1, V=self.V, bcs=bcs)
            u_nn = u_n.copy(deepcopy=True)
            p_n = flu.projectm(self.p0, self.P)

            u_ = u_n.copy(deepcopy=True)
            p_ = p_n.copy(deepcopy=True)

            # Flush files and save steady state as time_step 0
            if self.params_save.save_every:
                # if not self.perturbations:
                #     flu.write_xdmf(
                #         self.paths["u_restart"],
                #         u_n,
                #         "u",
                #         time_step=0.0,
                #         append=False,
                #         write_mesh=True,
                #     )
                #     flu.write_xdmf(
                #         self.paths["uprev_restart"],
                #         u_nn,
                #         "u_n",
                #         time_step=0.0,
                #         append=False,
                #         write_mesh=True,
                #     )
                #     flu.write_xdmf(
                #         self.paths["p_restart"],
                #         p_n,
                #         "p",
                #         time_step=0.0,
                #         append=False,
                #         write_mesh=True,
                #     )
                # else:
                u_n_save = dolfin.Function(self.V)
                p_n_save = dolfin.Function(self.P)
                u_n_save.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
                p_n_save.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
                flu.write_xdmf(
                    self.paths["u_restart"],
                    u_n_save,
                    "u",
                    time_step=0.0,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["uprev_restart"],
                    u_n_save,
                    "u_n",
                    time_step=0.0,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["p_restart"],
                    p_n_save,
                    "p",
                    time_step=0.0,
                    append=False,
                    write_mesh=True,
                )

        else:
            # find index to load saved data
            idxstart = (
                -1
                if (self.params_time.Tstart == -1)
                else int(
                    np.floor(
                        (self.params_time.Tstart - self.params_time.Trestartfrom)
                        / self.params_time.dt_old
                        / self.params_save.save_every_old
                    )
                )
            )
            # second order temporal integration
            self.order = self.params_time.restart_order  # 2
            # assign previous solution
            # here: subtract base flow if perturbation
            # if perturbations : read u_n, subtract u0, save
            # if not: read u_n, write u_n
            u_n = dolfin.Function(self.V)
            u_nn = dolfin.Function(self.V)
            p_n = dolfin.Function(self.P)
            # pdb.set_trace()
            flu.read_xdmf(self.paths["u"], u_n, "u", counter=idxstart)
            flu.read_xdmf(self.paths["uprev"], u_nn, "u_n", counter=idxstart)
            flu.read_xdmf(self.paths["p"], p_n, "p", counter=idxstart)

            flu.read_xdmf(self.paths["u"], u_, "u", counter=idxstart)
            flu.read_xdmf(self.paths["p"], p_, "p", counter=idxstart)

            # write in new file as first time step
            # important to do this before subtracting base flow (if perturbations)
            if self.params_save.save_every:
                flu.write_xdmf(
                    self.paths["u_restart"],
                    u_n,
                    "u",
                    time_step=self.params_time.Tstart,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["uprev_restart"],
                    u_nn,
                    "u_n",
                    time_step=self.params_time.Tstart,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["p_restart"],
                    p_n,
                    "p",
                    time_step=self.params_time.Tstart,
                    append=False,
                    write_mesh=True,
                )
            # if perturbations, remove base flow from loaded file
            # because one prefers to write complete flow (not just perturbations)
            # if self.perturbations:
            u_n.vector()[:] = u_n.vector()[:] - self.u0.vector()[:]
            u_nn.vector()[:] = u_nn.vector()[:] - self.u0.vector()[:]
            p_n.vector()[:] = p_n.vector()[:] - self.p0.vector()[:]
            u_.vector()[:] = u_.vector()[:] - self.u0.vector()[:]
            p_.vector()[:] = p_.vector()[:] - self.p0.vector()[:]

        if self.verbose and flu.MpiUtils.get_rank() == 0:
            logger.info(
                "Starting or restarting from time: %f with temporal scheme order: %d",
                self.params_time.Tstart,
                self.order,
            )

        # Assign fields
        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n

        # Compute things on x(0)
        fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
        up_n = dolfin.Function(self.W)
        fa.assign(up_n, [u_n, p_n])
        self.y_meas0 = self.make_measurement(mixed_field=up_n)
        self.y_meas = self.y_meas0
        # not valid in perturbations formulation
        cl1, cd1 = self.compute_force_coefficients(u_n, p_n)

        # Make time series pd.DataFrame
        y_meas_str = ["y_meas_" + str(i + 1) for i in range(self.params_flow.sensor_nr)]
        colnames = ["time", "u_ctrl"] + y_meas_str + ["dE", "runtime"]  # "cl", "cd"
        empty_data = np.zeros((self.params_time.num_steps + 1, len(colnames)))
        timeseries = pd.DataFrame(columns=colnames, data=empty_data)
        # u_ctrl = dolfin.Constant(0)
        timeseries.loc[0, "time"] = self.params_time.Tstart
        self.assign_measurement_to_dataframe(
            df=timeseries,
            y_meas=self.y_meas0,
            index=0,
            sensor_nr=self.params_flow.sensor_nr,
        )

        dEb = self.compute_energy()
        timeseries.loc[0, "dE"] = dEb
        self.timeseries = timeseries

    def make_solvers(self):  # TODO could be utils
        """Define solvers"""
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def set_initial_state(
        self, x0: dolfin.Function = None
    ):  # TODO could move to superclass
        """Define initial state and assign to self.initial_state
        x0: dolfin.Function(self.W)
        dolfin.Function needs to be called before self.init_time_stepping()"""
        self.initial_state = x0

    def step(self, u_ctrl: float) -> None:
        """Simulate system with perturbation formulation,
        possibly an actuation value, and a shift
        initial_up may be set as self.get_B() to compute impulse response"""
        iRe = dolfin.Constant(1 / self.params_flow.Re)
        k = dolfin.Constant(self.params_time.dt)

        v, q = dolfin.TestFunctions(self.W)
        up = dolfin.TrialFunction(self.W)
        u, p = dolfin.split(up)
        up_ = dolfin.Function(self.W)
        u_, p_ = dolfin.split(up_)
        u0 = self.u0

        if self.params_flow.is_eq_nonlinear:  # nonlinear
            b0_1 = 1  # order 1
            b0_2, b1_2 = 2, -1  # order 2
        else:  # linear, remove (u'.nabla)(u')
            b0_1 = b0_2 = b1_2 = 0

        # init with self.attr (from init_time_stepping)
        u_nn = self.u_nn
        u_n = self.u_n
        p_n = self.p_n

        # This step is handled with init_time_stepping for IPCS formulation
        if not hasattr(self, "assemblers_p"):  # make forms
            if self.verbose:
                logger.debug("Perturbations forms DO NOT exist: create...")

            shift = dolfin.Constant(self.params_flow.shift)
            # 1st order integration
            F1 = (
                dot((u - u_n) / k, v) * dx
                + dot(dot(u0, nabla_grad(u)), v) * dx
                + dot(dot(u, nabla_grad(u0)), v) * dx
                + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + dolfin.Constant(b0_1) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
                - p * div(v) * dx
                - div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u_1, u_2 but not p (ok like dis)

            # 2nd order integration
            F2 = (
                dot((3 * u - 4 * u_n + u_nn) / (2 * k), v) * dx
                + dot(dot(u0, nabla_grad(u)), v) * dx
                + dot(dot(u, nabla_grad(u0)), v) * dx
                + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + dolfin.Constant(b0_2) * dot(dot(u_n, nabla_grad(u_n)), v) * dx
                + dolfin.Constant(b1_2) * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
                - p * div(v) * dx
                - div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u_1, u_2 but not p (ok like dis)

            # Extract
            a1 = dolfin.lhs(F1)
            L1 = dolfin.rhs(F1)
            a2 = dolfin.lhs(F2)
            L2 = dolfin.rhs(F2)

            sysAssmb1 = dolfin.SystemAssembler(a1, L1, self.bc_p["bcu"])
            sysAssmb2 = dolfin.SystemAssembler(a2, L2, self.bc_p["bcu"])
            Ap1, Ap2 = dolfin.Matrix(), dolfin.Matrix()

            S = [dolfin.LUSolver("mumps") for i in range(2)]
            for assemblr, solver, A in zip([sysAssmb1, sysAssmb2], S, [Ap1, Ap2]):
                assemblr.assemble(A)
                solver.set_operator(A)

            self.bs_p = dolfin.Vector()  # create dolfin.rhs
            self.assemblers_p = {1: sysAssmb1, 2: sysAssmb2}
            self.solvers_p = {1: S[0], 2: S[1]}

            # save perturbation and full solution
            self.u_full = dolfin.Function(self.V)
            self.u_n_full = dolfin.Function(self.V)
            self.p_full = dolfin.Function(self.P)

        # time
        t0i = time.time()

        # control
        self.actuator_expression.ampl = u_ctrl

        # Assign system of eqs
        assembler = self.assemblers_p[self.order]
        solver = self.solvers_p[self.order]

        if (
            not self.params_solver.throw_error
        ):  # used for optimization -> return error code
            try:
                assembler.assemble(self.bs_p)  # assemble dolfin.rhs
                solver.solve(up_.vector(), self.bs_p)  # solve Ax=b
                u_, p_ = up_.split(deepcopy=True)
                # Patch: solve sometimes does not see it failed...
                if not np.isfinite(u_.vector().get_local()[0]):
                    logger.critical("Solver diverged, inf found")
                    raise RuntimeError("Inf found in solution")
            except RuntimeError:
                # Usually Krylov solver exploding return a RuntimeError
                # See: error_on_nonconvergence (but need to catch error somehow)
                logger.critical("Solver error --- Exiting step()...")
                return -1  # -1 is error code
        else:  # used for debugging -> show error message
            assembler.assemble(self.bs_p)  # assemble dolfin.rhs
            solver.solve(up_.vector(), self.bs_p)  # solve Ax=b
            u_, p_ = up_.split(deepcopy=True)
            if not np.isfinite(u_.vector().get_local()[0]):
                logger.critical("Solver diverged, inf found")
                raise RuntimeError("Inf found in solution")

        # Assign new
        u_nn.assign(u_n)
        u_n.assign(u_)
        p_n.assign(p_)

        # Update time
        self.iter += 1
        self.t = (
            self.params_time.Tstart + (self.iter) * self.params_time.dt
        )  # better accuracy than t+=dt

        # Assign to self
        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n
        self.up_ = up_

        # Goto order 2 next time
        self.order = 2

        # Measurement
        self.y_meas = self.make_measurement()

        tfi = time.time()
        if self.verbose and (
            not self.iter % self.verbose
        ):  # print every 1 if verbose is bool
            flu2.print_progress(self, runtime=tfi - t0i)

        # Log timeseries
        dE = self.compute_energy()  # on perturbation field
        self.log_timeseries(
            u_ctrl=u_ctrl,
            y_meas=self.y_meas,
            dE=dE,
            t=self.t,
            runtime=tfi - t0i,
        )
        # Save
        if self.params_save.save_every and not self.iter % self.params_save.save_every:
            self.u_full.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
            self.u_n_full.vector()[:] = u_nn.vector()[:] + self.u0.vector()[:]
            self.p_full.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
            if self.verbose:
                logger.debug("saving to files %s" % (self.params_save.savedir0))
            flu.write_xdmf(
                self.paths["u_restart"],
                self.u_full,
                "u",
                time_step=self.t,
                append=True,
                write_mesh=False,
            )
            flu.write_xdmf(
                self.paths["uprev_restart"],
                self.u_n_full,
                "u_n",
                time_step=self.t,
                append=True,
                write_mesh=False,
            )
            flu.write_xdmf(
                self.paths["p_restart"],
                self.p_full,
                "p",
                time_step=self.t,
                append=True,
                write_mesh=False,
            )
            # this is asynchronous and calls process 0?
            self.write_timeseries()

        return 0

    # Steady state
    def load_steady_state(self, assign: bool = True):  # TODO move to utils???
        u0 = dolfin.Function(self.V)
        p0 = dolfin.Function(self.P)
        flu.read_xdmf(self.paths["u0"], u0, "u")
        flu.read_xdmf(self.paths["p0"], p0, "p")

        # Assign u0, p0 >>> up0
        fa_VP2W = dolfin.FunctionAssigner(self.W, [self.V, self.P])
        up0 = dolfin.Function(self.W)
        fa_VP2W.assign(up0, [u0, p0])

        if assign:
            self.u0 = u0  # full field (u+upert)
            self.p0 = p0
            self.up0 = up0
            self.y_meas_steady = self.make_measurement(mixed_field=up0)

            # assign steady energy
            self.Eb = (
                1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=self.mesh) ** 2
            )  # same as <up, Q@up>
        return u0, p0, up0

    def compute_steady_state(self, method="newton", u_ctrl=0.0, **kwargs):
        """Compute flow steady state with given steady control"""
        # Save old control value, just in case
        actuation_ampl_old = self.actuator_expression.ampl
        # Set control value to prescribed u_ctrl
        self.actuator_expression.ampl = u_ctrl

        # If start is zero (i.e. not restart): compute
        # Note : could add a flag 'compute_steady_state' to compute or read...
        if self.params_time.Tstart == 0:  # and compute_steady_state
            # Solve
            if method == "newton":
                up0 = self.compute_steady_state_newton(**kwargs)
            else:
                up0 = self.compute_steady_state_picard(**kwargs)

            # assign up0, u0, p0 and write
            fa_W2VP = dolfin.FunctionAssigner([self.V, self.P], self.W)
            u0 = dolfin.Function(self.V)
            p0 = dolfin.Function(self.P)
            fa_W2VP.assign([u0, p0], up0)

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
                logger.debug("Stored base flow in: %s", self.params_save.savedir0)

            self.y_meas_steady = self.make_measurement(mixed_field=up0)

        # If IC is not zero: read steady state (should exist - should check though...)
        else:
            u0, p0, up0 = self.load_steady_state(assign=True)

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
        # if initial_guess is None:
        #    print('- Newton solver without initial guess')
        up_ = self.up_
        # u_, p_ = self.u_, self.p_
        # Solver param
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
        self.make_form_mixed_steady()
        iRe = dolfin.Constant(1 / self.params_flow.Re)

        # for residual computation
        bcu_inlet0 = self.bc_p["bcu"][0]
        bcu0 = self.bc["bcu"] + [bcu_inlet0]

        # define forms
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

            # show_max(u, 'u')
            res = dolfin.assemble(dolfin.action(ap, up1))
            [bc.apply(res) for bc in bcu0]
            res_norm = dolfin.norm(res) / dolfin.sqrt(ndof)
            if self.verbose:
                logger.info(
                    "Picard iteration: {0}/{1}, residual: {2}".format(
                        i + 1, max_iter, res_norm
                    )
                )
            if res_norm < tol:
                if self.verbose:
                    logger.info("Residual norm lower than tolerance {0}".format(tol))
                break

        return up1

    # Dataframe utility
    def make_y_dataframe_column_name(self, sensor_nr: int):
        """Return column names of different measurements y_meas_i"""
        return ["y_meas_" + str(i + 1) for i in range(sensor_nr)]

    def assign_measurement_to_dataframe(
        self, df: pd.DataFrame, y_meas: float, index: int, sensor_nr: int
    ):
        """Assign measurement (array y_meas) to DataFrame at index
        Essentially convert array (y_meas) to separate columns (y_meas_i)"""
        y_meas_str = self.make_y_dataframe_column_name(sensor_nr)
        for i_meas, name_meas in enumerate(y_meas_str):
            df.loc[index, name_meas] = y_meas[i_meas]

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
            u_ctrl  # careful here: log the command that was applied at time t (iter-1) to get time t+dt (iter)
        )
        # replace above line for several measurements
        self.assign_measurement_to_dataframe(
            df=self.timeseries,
            y_meas=y_meas,
            index=self.iter,
            sensor_nr=self.params_flow.sensor_nr,
        )
        self.timeseries.loc[self.iter, "dE"] = dE
        # self.timeseries.loc[self.iter, "cl"], self.timeseries.loc[self.iter, "cd"] = (
        #     cl,
        #     cd,
        # )
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    # General utility # could go outside class because
    # one might want to compute energy of an arbitrary velocity fieldÒ
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
    def make_boundaries(self):
        pass

    @abstractmethod
    def make_bcs(self):
        pass

    @abstractmethod
    def make_actuator(self):
        pass

    @abstractmethod
    def make_measurement(self):
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
        # f = self.actuator_expression
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
