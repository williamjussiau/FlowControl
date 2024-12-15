from __future__ import print_function
import dolfin
from dolfin import dot, nabla_grad, dx, inner, div
import numpy as np
import os
import pandas as pd
import time
# from petsc4py import PETSc

import pdb  # noqa: F401
import logging


import utils_flowsolver as flu
import utils_extract as flu2


logger = logging.getLogger(__name__)
FORMAT = (
    "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class AbstractFlowSolver:
    def __init__(
        self,
        params_flow,
        params_time,
        params_save,
        params_solver,
        params_mesh,
        verbose=True,
    ):
        # Probably bad practice
        # Unwrap all dictionaries into self.attribute
        alldict = {
            **params_flow,
            **params_time,
            **params_save,
            **params_solver,
            **params_mesh,
        }
        for key, item in alldict.items():  # all input dicts
            setattr(self, key, item)  # set corresponding attribute

        self.verbose = verbose
        # Parameters
        self.r = self.d / 2
        self.nu = self.uinf * self.d / self.Re  # dunnu touch
        # Time
        self.Tf = self.num_steps * self.dt  # final time
        # Sensors
        self.sensor_nr = self.sensor_location.shape[0]
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
        savedir0 = self.savedir0
        Tstart = self.Tstart  # start simulation from time...
        Trestartfrom = self.Trestartfrom  # use older files starting from time...

        def make_extension(T):
            return "_restart" + str(np.round(T, decimals=3)).replace(".", ",")

        file_start = make_extension(Tstart)
        file_restart = make_extension(Trestartfrom)

        ext_xdmf = ".xdmf"
        ext_csv = ".csv"

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
            "mesh": self.meshpath,
        }

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
            logger.debug("Function Space [V(CG2), P(CG1)] has: %d DOFs" % (self.W.dim()))

    def make_mesh(self):  # TODO move to superclass
        """Define mesh
        params_mesh contains either name of existing mesh
        or geometry parameters: xinf, yinf, xinfa, nx..."""
        # Set params
        genmesh = self.genmesh
        meshdir = self.paths["mesh"]  #'/stck/wjussiau/fenics-python/mesh/'
        xinf = self.xinf  # 20 # 20 # 20
        yinf = self.yinf  # 8 # 5 # 8
        xinfa = self.xinfa  # -5 # -5 # -10
        # Working as follows:
        # if genmesh:
        #   if does not exist (with given params): generate with meshr
        #   and prepare to not read file (because mesh is already in memory)
        # else:
        #   set file name and prepare to read file
        # read file
        readmesh = True
        if genmesh:
            nx = self.nx  # 32
            meshname = "cylinder_" + str(nx) + ".xdmf"
            meshpath = meshdir / meshname  # os.path.join(meshdir, meshname)
            if not os.path.exists(meshpath) or self.remesh:
                if self.verbose:
                    logger.debug("Mesh does not exist @: %s", meshpath)
                    logger.debug("-- Creating mesh...")
                channel = dolfin.Rectangle(
                    dolfin.Point(xinfa, -yinf), dolfin.Point(xinf, yinf)
                )
                cyl = dolfin.Circle(
                    dolfin.Point(0.0, 0.0), self.d / 2, segments=self.segments
                )
                domain = channel - cyl
                mesh = dolfin.generate_mesh(domain, nx)
                with dolfin.XDMFFile(dolfin.MPI.comm_world, str(meshpath)) as fm:
                    fm.write(mesh)
                readmesh = False
        else:
            meshname = self.meshname

        # if mesh was not generated on the fly, read file
        if readmesh:
            mesh = dolfin.Mesh(dolfin.MPI.comm_world)
            meshpath = meshdir / meshname  # os.path.join(meshdir, meshname)
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

    def init_time_stepping(self):  # TODO move to superclass
        """Create varitional functions/forms & flush files & define u(0), p(0)"""
        # Trial and test functions ####################################################
        # W = self.W

        # Define expressions used in variational forms
        # iRe = dolfin.Constant(1 / self.Re)
        # II = dolfin.Identity(2)
        # k = dolfin.Constant(self.dt)
        ##############################################################################

        t = self.Tstart
        self.t = t
        self.iter = 0

        # solutions
        u_ = dolfin.Function(self.V)
        p_ = dolfin.Function(self.P)

        # if not restart
        if self.Tstart == 0:
            # first order temporal integration
            self.order = 1

            # Set initial state up in W
            initial_up = dolfin.Function(self.W)

            # No initial state given -> base flow
            if self.initial_state is None:
                initial_up = dolfin.Function(self.W)
                if not self.perturbations:
                    initial_up.vector()[:] += self.up0.vector()[:]
            else:
                initial_up = self.initial_state

            # Impulse or state perturbation @ div0
            # Impulse if self.init_pert is inf
            if np.isinf(self.init_pert):
                # not sure this would work in parallel
                initial_up.vector()[:] += self.get_B().reshape((-1,))
            else:
                udiv0 = flu2.get_div0_u(self, xloc=2, yloc=0, size=0.5)
                fa = dolfin.FunctionAssigner(self.W, [self.V, self.P])
                pert0 = dolfin.Function(self.W)
                fa.assign(pert0, [udiv0, self.p0])
                initial_up.vector()[:] += self.init_pert * pert0.vector()[:]

            initial_up.vector().apply("insert")
            up1 = initial_up

            # Split up to u, p
            fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            u1 = dolfin.Function(self.V)
            p1 = dolfin.Function(self.P)
            fa.assign([u1, p1], up1)

            # this is the initial state
            if self.perturbations:
                bcs = self.bc_p["bcu"]  # bcs for perturbation formulation
            else:
                bcs = self.bc["bcu"]  # bcs for classic formulation
            u_n = flu.projectm(v=u1, V=self.V, bcs=bcs)
            u_nn = u_n.copy(deepcopy=True)
            p_n = flu.projectm(self.p0, self.P)

            u_ = u_n.copy(deepcopy=True)
            p_ = p_n.copy(deepcopy=True)

            # Flush files and save steady state as time_step 0
            if self.save_every:
                if not self.perturbations:
                    flu.write_xdmf(
                        self.paths["u_restart"],
                        u_n,
                        "u",
                        time_step=0.0,
                        append=False,
                        write_mesh=True,
                    )
                    flu.write_xdmf(
                        self.paths["uprev_restart"],
                        u_nn,
                        "u_n",
                        time_step=0.0,
                        append=False,
                        write_mesh=True,
                    )
                    flu.write_xdmf(
                        self.paths["p_restart"],
                        p_n,
                        "p",
                        time_step=0.0,
                        append=False,
                        write_mesh=True,
                    )
                else:
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
                if (self.Tstart == -1)
                else int(
                    np.floor(
                        (self.Tstart - self.Trestartfrom)
                        / self.dt_old
                        / self.save_every_old
                    )
                )
            )
            # second order temporal integration
            self.order = self.restart_order  # 2
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
            if self.save_every:
                flu.write_xdmf(
                    self.paths["u_restart"],
                    u_n,
                    "u",
                    time_step=self.Tstart,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["uprev_restart"],
                    u_nn,
                    "u_n",
                    time_step=self.Tstart,
                    append=False,
                    write_mesh=True,
                )
                flu.write_xdmf(
                    self.paths["p_restart"],
                    p_n,
                    "p",
                    time_step=self.Tstart,
                    append=False,
                    write_mesh=True,
                )
            # if perturbations, remove base flow from loaded file
            # because one prefers to write complete flow (not just perturbations)
            if self.perturbations:
                u_n.vector()[:] = u_n.vector()[:] - self.u0.vector()[:]
                u_nn.vector()[:] = u_nn.vector()[:] - self.u0.vector()[:]
                p_n.vector()[:] = p_n.vector()[:] - self.p0.vector()[:]
                u_.vector()[:] = u_.vector()[:] - self.u0.vector()[:]
                p_.vector()[:] = p_.vector()[:] - self.p0.vector()[:]

        if self.verbose and flu.MpiUtils.get_rank() == 0:
            logger.info(
                "Starting or restarting from time: %f with temporal scheme order: %d",
                self.Tstart,
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
        y_meas_str = ["y_meas_" + str(i + 1) for i in range(self.sensor_nr)]
        colnames = ["time", "u_ctrl"] + y_meas_str + ["dE", "cl", "cd", "runtime"]
        empty_data = np.zeros((self.num_steps + 1, len(colnames)))
        ts1d = pd.DataFrame(columns=colnames, data=empty_data)
        # u_ctrl = dolfin.Constant(0)
        ts1d.loc[0, "time"] = self.Tstart
        self.assign_measurement_to_dataframe(
            df=ts1d, y_meas=self.y_meas0, index=0, sensor_nr=self.sensor_nr
        )
        ts1d.loc[0, "cl"], ts1d.loc[0, "cd"] = cl1, cd1
        if self.compute_norms:
            dEb = self.compute_energy()
        else:
            dEb = 0
        ts1d.loc[0, "dE"] = dEb
        self.timeseries = ts1d

    def make_solvers(self):  # TODO could be utils
        """Define solvers"""
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def set_initial_state(self, x0=None):  # TODO could move to superclass
        """Define initial state and assign to self.initial_state
        x0: dolfin.Function(self.W)
        dolfin.Function needs to be called before self.init_time_stepping()"""
        self.initial_state = x0

    def step_perturbation(self, u_ctrl=0.0, shift=0.0, NL=True):
        """Simulate system with perturbation formulation,
        possibly an actuation value, and a shift
        initial_up may be set as self.get_B() to compute impulse response"""
        iRe = dolfin.Constant(1 / self.Re)
        k = dolfin.Constant(self.dt)

        v, q = dolfin.TestFunctions(self.W)
        up = dolfin.TrialFunction(self.W)
        u, p = dolfin.split(up)
        up_ = dolfin.Function(self.W)
        u_, p_ = dolfin.split(up_)
        u0 = self.u0

        if NL:  # nonlinear
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

            shift = dolfin.Constant(shift)
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

        if not self.throw_error:  # used for optimization -> return error code
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
        self.t = self.Tstart + (self.iter) * self.dt  # better accuracy than t+=dt

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
        # self.e_meas = self.y_meas

        tfi = time.time()
        if self.verbose and (
            not self.iter % self.verbose
        ):  # print every 1 if verbose is bool
            flu2.print_progress(self, runtime=tfi - t0i)

        # Log timeseries
        # Be careful: cl, cd, norms etc. are in perturbation formulation (miss u0, p0)
        # perturbation energy wrt base flow, here u_ = u_pert
        if self.compute_norms:
            dE = self.compute_energy()
            # dE = norm(self.u_, norm_type='L2', mesh=self.mesh) / self.Eb
            self.u_full.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
            self.p_full.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
            cl, cd = self.compute_force_coefficients(self.u_full, self.p_full)
            # cl, cd = 0, 1
        else:
            dE = -1
            cl = 0
            cd = 1
        self.log_timeseries(
            u_ctrl=u_ctrl,
            y_meas=self.y_meas,
            norm_u=0,  # norm(u_, norm_type='L2', mesh=self.mesh),
            norm_p=0,  # norm(p_, norm_type='L2', mesh=self.mesh),
            dE=dE,
            cl=cl,
            cd=cd,
            t=self.t,
            runtime=tfi - t0i,
        )
        # Save
        if self.save_every and not self.iter % self.save_every:
            self.u_full.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
            self.u_n_full.vector()[:] = u_nn.vector()[:] + self.u0.vector()[:]
            self.p_full.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
            if self.verbose:
                logger.debug("saving to files %s" % (self.savedir0))
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
    def load_steady_state(self, assign=True):  # TODO move to utils???
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
        if self.Tstart == 0:  # and compute_steady_state
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
            if self.save_every:
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
                logger.debug("Stored base flow in: %s", self.savedir0)

            self.y_meas_steady = self.make_measurement(mixed_field=up0)

        # If start is not zero: read steady state (should exist - should check though...)
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

    def compute_steady_state_newton(self, max_iter=25, initial_guess=None):
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

    # Dataframe utility
    def make_y_dataframe_column_name(self, sensor_nr):
        """Return column names of different measurements y_meas_i"""
        return ["y_meas_" + str(i + 1) for i in range(sensor_nr)]

    def assign_measurement_to_dataframe(self, df, y_meas, index, sensor_nr):
        """Assign measurement (array y_meas) to DataFrame at index
        Essentially convert array (y_meas) to separate columns (y_meas_i)"""
        y_meas_str = self.make_y_dataframe_column_name(sensor_nr)
        for i_meas, name_meas in enumerate(y_meas_str):
            df.loc[index, name_meas] = y_meas[i_meas]

    def write_timeseries(self):
        """Write pandas DataFrame to file"""
        if flu.MpiUtils.get_rank() == 0:
            # zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths["timeseries"], sep=",", index=False)

    def log_timeseries(
        self, u_ctrl, y_meas, norm_u, norm_p, dE, cl, cd, t, runtime
    ):  # TODO move to superclass
        """Fill timeseries table with data"""
        self.timeseries.loc[self.iter - 1, "u_ctrl"] = (
            u_ctrl  # careful here: log the command that was applied at time t (iter-1) to get time t+dt (iter)
        )
        # replace above line for several measurements
        self.assign_measurement_to_dataframe(
            df=self.timeseries, y_meas=y_meas, index=self.iter, sensor_nr=self.sensor_nr
        )
        self.timeseries.loc[self.iter, "dE"] = dE
        self.timeseries.loc[self.iter, "cl"], self.timeseries.loc[self.iter, "cd"] = (
            cl,
            cd,
        )
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    # General utility
    def compute_energy(self):  # TODO superclass
        """Compute energy of perturbation flow
        OPTIONS REMOVED FROM PREVIOUS VERSION:
        on full/restricted domain      (default:full=True)
        minus base flow                (default:diff=False)
        normalized by base flow energy (default:normalize=False)"""
        dE = 1 / 2 * dolfin.norm(self.u_, norm_type="L2", mesh=self.mesh) ** 2
        return dE

    def compute_energy_field(self, export=False, filename=None):  # TODO superclass
        """Compute field dot(u, u) to see spatial location of perturbation kinetic energy
        Perturbation formulation only"""
        Efield = dot(self.u_, self.u_)
        # Note: E = 1/2 * assemble(Efield * fs.dx)
        Efield = flu.projectm(Efield, self.P)  # project to deg 1
        if export:
            flu.write_xdmf(filename, Efield, "E")
        return Efield

    # Abstract methods (to be reimplemented for each case)
    def make_boundaries():
        raise NotImplementedError()

    def make_bcs():
        raise NotImplementedError()

    def make_actuator():
        raise NotImplementedError()

    def make_measurement():
        raise NotImplementedError()
    
    # TODO possible to do 1 implementation for all?
    # eg by linking to BC, that are implemented by user
    def make_form_mixed_steady():
        raise NotImplementedError()


