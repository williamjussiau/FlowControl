"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for let around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0


----------------------------------------------------------------------
Equations were made non-dimensional
----------------------------------------------------------------------
"""

from __future__ import print_function
import dolfin
from dolfin import dot, inner, grad, nabla_grad, sym, div, curl, dx
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import os
import time
import pandas as pd
import sympy as sp
import scipy.sparse as spr
# from petsc4py import PETSc

import pdb  # noqa: F401
import logging

from pathlib import Path

import importlib
import utils_flowsolver as flu

importlib.reload(flu)

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

class FlowSolver:
    """Base class for calculating flow
    Is instantiated with several structures (dicts) containing parameters
    See method .step and main for time-stepping (possibly actuated)
    Contain methods for frequency-response computation"""

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

    def define_paths(self):
        """Define attribute (dict) containing useful paths (save, etc.)"""
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

    def make_mesh(self):
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
                    logger.info("Mesh does not exist @: %s", meshpath)
                    logger.info("-- Creating mesh...")
                channel = Rectangle(
                    dolfin.Point(xinfa, -yinf), dolfin.Point(xinf, yinf)
                )
                cyl = Circle(dolfin.Point(0.0, 0.0), self.d / 2, segments=self.segments)
                domain = channel - cyl
                mesh = generate_mesh(domain, nx)
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
                logger.info("Mesh exists @: %s", meshpath)
                logger.info("--- Reading mesh...")
            with dolfin.XDMFFile(dolfin.MPI.comm_world, str(meshpath)) as fm:
                fm.read(mesh)
            # mesh = Mesh(dolfin.MPI.comm_world, meshpath) # if xml

        if self.verbose:
            logger.info("Mesh has: %d cells" % (mesh.num_cells()))

        # assign mesh & facet normals
        self.mesh = mesh
        self.n = dolfin.FacetNormal(mesh)

    def make_function_spaces(self):
        """Define function spaces (u, p) = (CG2, CG1)"""
        # dolfin.Function spaces on mesh
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)  # was 'P'
        Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)  # was 'P'
        We = dolfin.MixedElement([Ve, Pe])
        self.V = dolfin.FunctionSpace(self.mesh, Ve)
        self.P = dolfin.FunctionSpace(self.mesh, Pe)
        self.W = dolfin.FunctionSpace(self.mesh, We)
        if self.verbose:
            logger.info("Function Space [V(CG2), P(CG1)] has: %d DOFs" % (self.W.dim()))

    def make_boundaries(self):
        """Define boundaries (inlet, outlet, walls, cylinder, actuator)"""
        MESH_TOL = dolfin.DOLFIN_EPS
        # Define as compiled subdomains
        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinfa, MESH_TOL)",
            xinfa=self.xinfa,
            MESH_TOL=MESH_TOL,
        )
        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinf, MESH_TOL)",
            xinf=self.xinf,
            MESH_TOL=MESH_TOL,
        )
        ## Walls
        walls = dolfin.CompiledSubDomain(
            "on_boundary && \
                (near(x[1], -yinf, MESH_TOL) ||   \
                 near(x[1], yinf, MESH_TOL))",
            yinf=self.yinf,
            MESH_TOL=MESH_TOL,
        )

        ## Cylinder
        delta = (
            self.actuator_angular_size * dolfin.pi / 180
        )  # angular size of acutator, in rad
        theta_tol = 1 * dolfin.pi / 180

        # Compiled subdomains
        # define them as strings
        # increased speed but decreased readability
        def near_cpp(x, x0):
            return "near({x}, {x0}, MESH_TOL)".format(x=x, x0=x0)

        def between_cpp(x, xmin, xmax):
            return "{x}>={xmin} && {x}<={xmax}".format(xmin=xmin, x=x, xmax=xmax)

        close_to_cylinder_cpp = (
            between_cpp("x[0]", "-d/2", "d/2")
            + " && "
            + between_cpp("x[1]", "-d/2", "d/2")
        )
        theta_cpp = "atan2(x[1], x[0])"
        cone_up_cpp = between_cpp(
            theta_cpp + "-pi/2", "-delta/2 - theta_tol", "delta/2 + theta_tol"
        )
        cone_lo_cpp = between_cpp(
            theta_cpp + "+pi/2", "-delta/2 - theta_tol", "delta/2 + theta_tol"
        )
        cone_ri_cpp = between_cpp(
            theta_cpp, "-pi/2+delta/2 - theta_tol", "pi/2-delta/2 + theta_tol"
        )
        cone_le_cpp = (
            "("
            + between_cpp(theta_cpp, "-pi", "-pi/2-delta/2 + theta_tol")
            + ")"
            + " || "
            + "("
            + between_cpp(theta_cpp, "pi/2+delta/2 - theta_tol", "pi")
            + ")"
        )

        cylinder = dolfin.CompiledSubDomain(
            "on_boundary"
            + " && "
            + close_to_cylinder_cpp
            + " && "
            + "("
            + cone_le_cpp
            + " ||  "
            + cone_ri_cpp
            + ")",
            d=self.d,
            delta=delta,
            theta_tol=theta_tol,
        )
        actuator_up = dolfin.CompiledSubDomain(
            "on_boundary" + " && " + close_to_cylinder_cpp + " && " + cone_up_cpp,
            d=self.d,
            delta=delta,
            theta_tol=theta_tol,
        )
        actuator_lo = dolfin.CompiledSubDomain(
            "on_boundary" + " && " + close_to_cylinder_cpp + " && " + cone_lo_cpp,
            d=self.d,
            delta=delta,
            theta_tol=theta_tol,
        )

        # end

        # Whole cylinder for integration
        cylinder_whole = dolfin.CompiledSubDomain(
            "on_boundary && (x[0]*x[0] + x[1]*x[1] <= d+0.1)",
            d=self.d,
            MESH_TOL=MESH_TOL,
        )
        self.cylinder_intg = cylinder_whole
        ################################## todo

        # assign boundaries as pd.DataFrame
        boundaries_list = [
            "inlet",
            "outlet",
            "walls",
            "cylinder",
            "actuator_up",
            "actuator_lo",
        ]  # ,
        #'cylinder_whole']
        boundaries_df = pd.DataFrame(
            index=boundaries_list,
            data={
                "subdomain": [inlet, outlet, walls, cylinder, actuator_up, actuator_lo]
            },
        )  # ,
        # cylinder_whole]})
        self.actuator_angular_size_rad = delta
        self.boundaries = boundaries_df

    ##        ###############################################################################

    def make_actuator(self):
        """Define actuator on boundary
        Could be defined as volume actuator some day"""

        L = self.r * np.tan(self.actuator_angular_size_rad / 2)
        nsig = 2  # self.nsig_actuator
        actuator_bc = dolfin.Expression(
            ["0", "(x[0]>=L || x[0] <=-L) ? 0 : ampl*-1*(x[0]+L)*(x[0]-L) / (L*L)"],
            element=self.V.ufl_element(),
            ampl=1,
            L=L,
            den=(L / nsig) ** 2,
            nsig=nsig,
        )

        self.actuator_expression = actuator_bc

    def make_bcs(self):
        """Define boundary conditions"""
        # Boundary markers
        boundary_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        cell_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        # Boundary indices
        INLET_IDX = 0
        OUTLET_IDX = 1
        WALLS_IDX = 2
        CYLINDER_IDX = 3
        CYLINDER_ACTUATOR_UP_IDX = 4
        CYLINDER_ACTUATOR_LO_IDX = 5

        boundaries_idx = [
            INLET_IDX,
            OUTLET_IDX,
            WALLS_IDX,
            CYLINDER_IDX,
            CYLINDER_ACTUATOR_UP_IDX,
            CYLINDER_ACTUATOR_LO_IDX,
        ]

        # Mark boundaries (for dolfin.DirichletBC farther)
        for i, boundary_index in enumerate(boundaries_idx):
            self.boundaries.iloc[i].subdomain.mark(boundary_markers, boundary_index)
            self.boundaries.iloc[i].subdomain.mark(cell_markers, boundary_index)

        # Measures (e.g. subscript ds(INLET_IDX))
        ds = dolfin.Measure("ds", domain=self.mesh, subdomain_data=boundary_markers)
        dx = dolfin.Measure("dx", domain=self.mesh, subdomain_data=cell_markers)

        # assign all
        self.dx = dx
        self.ds = ds
        self.boundary_markers = boundary_markers  # dolfin.MeshFunction
        self.cell_markers = cell_markers  # dolfin.MeshFunction
        # complete boundaries pd.DataFrame
        self.boundaries["idx"] = boundaries_idx

        # create zeroBC for perturbation formulation
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["walls"].subdomain,
        )
        bcu_cylinder = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cylinder"].subdomain,
        )
        bcu_actuation_up = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_up"].subdomain,
        )
        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_lo"].subdomain,
        )
        bcu_p = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]

        self.bc_p = {"bcu": bcu_p, "bcp": []}  # log perturbation bcs

    def load_steady_state(self, assign=True):
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


        # Make BCs (full ns formulation)
        # inlet : u = uinf, v = 0
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((self.uinf, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        # walls : v = 0
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["walls"].subdomain,
        )
        # cylinder : (u,v)=(0,0)
        bcu_cylinder = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cylinder"].subdomain,
        )
        # actuators : (u,v)=(0,va)
        bcu_actuation_up = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_up"].subdomain,
        )
        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_lo"].subdomain,
        )
        bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]

        bcp_outlet = dolfin.DirichletBC(
            self.W.sub(1), dolfin.Constant(0), self.boundaries.loc["outlet"].subdomain
        )
        bcp = [bcp_outlet]

        self.bc = {"bcu": bcu, "bcp": bcp}


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
                logger.info("Stored base flow in: %s", self.savedir0)

            self.y_meas_steady = self.make_measurement(mixed_field=up0)

        # If start is not zero: read steady state (should exist - should check though...)
        else:
            u0, p0, up0 = self.load_steady_state(assign=True)

        # Compute lift & drag
        cl, cd = self.compute_force_coefficients(u0, p0)
        # cl, cd = 0, 1
        if self.verbose:
            logger.info("Lift coefficient is: cl = %f", cl)
            logger.info("Drag coefficient is: cd = %f", cd)

        # Set old actuator amplitude
        self.actuator_expression.ampl = actuation_ampl_old

        # assign steady state
        self.up0 = up0
        self.u0 = u0
        self.p0 = p0
        # assign steady cl, cd
        self.cl0 = cl
        self.cd0 = cd
        # assign steady energy
        self.Eb = (
            1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=self.mesh) ** 2
        )  # same as <up, Q@up>

    def make_form_mixed_steady(self, initial_guess=None):
        """Make nonlinear forms for steady state computation, in mixed element space.
        Can be used to assign self.F0 and compute state spaces matrices."""
        v, q = dolfin.TestFunctions(self.W)
        if initial_guess is None:
            up_ = dolfin.Function(self.W)
        else:
            up_ = initial_guess
        u_, p_ = dolfin.split(up_)  # not deep copy, we need the link
        iRe = dolfin.Constant(1 / self.Re)
        f = self.actuator_expression
        # Problem
        F0 = (
            dot(dot(u_, nabla_grad(u_)), v) * dx
            + iRe * inner(nabla_grad(u_), nabla_grad(v)) * dx
            - p_ * div(v) * dx
            - q * div(u_) * dx
            - dot(f, v) * dx
        )
        self.F0 = F0
        self.up_ = up_
        self.u_ = u_
        self.p_ = p_

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

    def compute_steady_state_picard(self, max_iter=10, tol=1e-14):
        """Compute steady state with fixed-point iteration
        Should have a larger convergence radius than Newton method
        if initialization is bad in Newton method (and it is)
        TODO: residual not 0 if u_ctrl not 0 (see bc probably)"""
        iRe = dolfin.Constant(1 / self.Re)

        # for residual computation
        bcu_inlet0 = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
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

    def stress_tensor(self, u, p):
        """Compute stress tensor (for lift & drag)"""
        return 2.0 * self.nu * (sym(grad(u))) - p * dolfin.Identity(
            p.geometric_dimension()
        )

    def compute_force_coefficients(self, u, p, enable=True):
        """Compute lift & drag coefficients
        For testing purposes, I added an argument enable
        To compute Cl, Cd, just put enable=True in this code"""
        if enable:
            sigma = self.stress_tensor(u, p)
            Fo = -dot(sigma, self.n)

            # integration surfaces names
            surfaces_names = ["cylinder", "actuator_up", "actuator_lo"]
            # integration surfaces indices
            surfaces_idx = [self.boundaries.loc[nm].idx for nm in surfaces_names]

            # define drag & lift expressions
            # sum symbolic forces
            drag_sym = sum(
                [Fo[0] * self.ds(int(sfi)) for sfi in surfaces_idx]
            )  # (forced int)
            lift_sym = sum(
                [Fo[1] * self.ds(int(sfi)) for sfi in surfaces_idx]
            )  # (forced int)
            # integrate sum of symbolic forces
            lift = dolfin.assemble(lift_sym)
            drag = dolfin.assemble(drag_sym)

            # define force coefficients by normalizing
            cd = drag / (1 / 2 * self.uinf**2 * self.d)
            cl = lift / (1 / 2 * self.uinf**2 * self.d)
        else:
            cl, cd = 0, 1
        return cl, cd

    def compute_vorticity(self, u=None):
        """Compute vorticity field of given velocity field u"""
        if u is None:
            u = self.u_
        # should probably project on space of order n-1 --> self.P
        vorticity = flu.projectm(curl(u), V=self.V.sub(0).collapse())
        return vorticity

    def compute_divergence(self, u=None):
        """Compute divergence field of given velocity field u"""
        if u is None:
            u = self.u_
        divergence = flu.projectm(div(u), self.P)
        return divergence

    # Initial perturbations ######################################################
    class localized_perturbation_u(dolfin.UserExpression):
        """Perturbation localized in disk
        Use: u = dolfin.interpolate(localized_perturbation_u(), self.V)
        or something like that"""

        def eval(self, value, x):
            if (x[0] - -2.5) ** 2 + (x[1] - 0.1) ** 2 <= 1:
                value[0] = 0.05
                value[1] = 0.05
            else:
                value[0] = 0
                value[1] = 0

        def value_shape(self):
            return (2,)

    class random_perturbation_u(dolfin.UserExpression):
        """Perturbation in the whole volume, random
        Use: see localized_perturbation_u"""

        def eval(self, value, x):
            value[0] = 0.1 * np.random.randn()
            value[1] = 0

        def value_shape(self):
            return (2,)

    def get_div0_u(self):
        """Create velocity field with zero divergence"""
        # V = self.V
        P = self.P

        # Define courant function
        x0 = 2
        y0 = 0
        # nsig = 5
        sigm = 0.5
        xm, ym = sp.symbols("x[0], x[1]")
        rr = (xm - x0) ** 2 + (ym - y0) ** 2
        fpsi = 0.25 * sp.exp(-1 / 2 * rr / sigm**2)
        # Piecewise does not work too well
        # fpsi = sp.Piecewise(   (sp.exp(-1/2 * rr / sigm**2),
        # rr <= nsig**2 * sigm**2), (0, True) )
        dfx = fpsi.diff(xm, 1)
        dfy = fpsi.diff(ym, 1)

        # Take derivatives
        # psi = dolfin.Expression(sp.ccode(fpsi), element=V.ufl_element())
        dfx_expr = dolfin.Expression(sp.ccode(dfx), element=P.ufl_element())
        dfy_expr = dolfin.Expression(sp.ccode(dfy), element=P.ufl_element())

        # Check
        # psiproj = flu.projectm(psi, P)
        # flu.write_xdmf('psi.xdmf', psiproj, 'psi')
        # flu.write_xdmf('psi_dx.xdmf', flu.projectm(dfx_expr, P), 'psidx')
        # flu.write_xdmf('psi_dy.xdmf', flu.projectm(dfy_expr, P), 'psidy')

        # Make velocity field
        upsi = flu.projectm(dolfin.as_vector([dfy_expr, -dfx_expr]), self.V)
        return upsi

    def get_div0_u_random(self, sigma=0.1, seed=0):
        """Create random velocity field with zero divergence"""
        # CG2 scalar
        P2 = self.V.sub(0).collapse()

        # Make scalar potential field in CG2 (scalar)
        a0 = dolfin.Function(P2)
        np.random.seed(seed)
        a0.vector()[:] += sigma * np.random.randn(a0.vector()[:].shape[0])

        # Take curl, then by definition div(u0)=div(curl(a0))=0
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 1)
        V1 = dolfin.FunctionSpace(self.mesh, Ve)

        u0 = flu.projectm(curl(a0), V1)

        ##divu0 = flu.projectm(div(u0), self.P)
        return u0

    def set_initial_state(self, x0=None):
        """Define initial state and assign to self.initial_state
        x0: dolfin.Function(self.W)
        dolfin.Function needs to be called before self.init_time_stepping()"""
        self.initial_state = x0

    def init_time_stepping(self):
        """Create varitional functions/forms & flush files & define u(0), p(0)"""
        # Trial and test functions ####################################################
        # W = self.W

        # Define expressions used in variational forms
        iRe = dolfin.Constant(1 / self.Re)
        II = dolfin.Identity(2)
        k = dolfin.Constant(self.dt)
        ##############################################################################

        t = self.Tstart
        self.t = t
        self.iter = 0

        # function spaces
        V = self.V
        P = self.P
        # trial and test functions
        u = dolfin.TrialFunction(V)
        v = dolfin.TestFunction(V)
        p = dolfin.TrialFunction(P)
        q = dolfin.TestFunction(P)
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
                udiv0 = self.get_div0_u()
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
        colnames = (
            ["time", "u_ctrl"]
            + y_meas_str
            + ["dE", "cl", "cd", "runtime"]
        )
        empty_data = np.zeros((self.num_steps + 1, len(colnames)))
        ts1d = pd.DataFrame(columns=colnames, data=empty_data)
        # u_ctrl = dolfin.Constant(0)
        ts1d.loc[0, "time"] = self.Tstart
        self.assign_measurement_to_dataframe(df=ts1d, y_meas=self.y_meas0, index=0)
        ts1d.loc[0, "cl"], ts1d.loc[0, "cd"] = cl1, cd1
        if self.compute_norms:
            dEb = self.compute_energy()
        else:
            dEb = 0
        ts1d.loc[0, "dE"] = dEb
        self.timeseries = ts1d

    def log_timeseries(self, u_ctrl, y_meas, norm_u, norm_p, dE, cl, cd, t, runtime):
        """Fill timeseries table with data"""
        self.timeseries.loc[self.iter - 1, "u_ctrl"] = (
            u_ctrl  # careful here: log the command that was applied at time t (iter-1) to get time t+dt (iter)
        )
        # replace above line for several measurements
        self.assign_measurement_to_dataframe(
            df=self.timeseries, y_meas=y_meas, index=self.iter
        )
        self.timeseries.loc[self.iter, "dE"] = dE
        self.timeseries.loc[self.iter, "cl"], self.timeseries.loc[self.iter, "cd"] = (
            cl,
            cd,
        )
        self.timeseries.loc[self.iter, "time"] = t
        self.timeseries.loc[self.iter, "runtime"] = runtime

    def print_progress(self, runtime):
        """Single line to print progress"""
        logger.info(
            "--- iter: %5d/%5d --- time: %3.3f/%3.2f --- elapsed %5.5f ---"
            % (self.iter, self.num_steps, self.t, self.Tf + self.Tstart, runtime)
        )

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
                logger.info("Perturbations forms DO NOT exist: create...")

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
            self.print_progress(runtime=tfi - t0i)

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
                logger.info("saving to files %s" % (self.savedir0))
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

    def prepare_restart(self, Trestart):
        """Prepare restart: set Tstart, redefine paths & files"""
        self.Tstart = Trestart
        self.define_paths()
        self.init_time_stepping()

    def make_solvers(self):
        """Define solvers"""
        # other possibilities: dolfin.KrylovSolver("bicgstab", "jacobi")
        # then solverparam = solver.paramters
        # solverparam[""]=...
        return dolfin.LUSolver("mumps")

    def make_measurement(self, field=None, mixed_field=None):
        """Perform measurement and assign"""
        ns = self.sensor_nr
        y_meas = np.zeros((ns,))

        for isensor in range(ns):
            xs_i = self.sensor_location[isensor]
            ts_i = self.sensor_type[isensor]

            # no mixed field (u,v,p) is given
            if field is not None:
                idx_dim = 0 if ts_i == "u" else 1
                y_meas_i = flu.MpiUtils.peval(field, xs_i)[idx_dim]
            else:
                if mixed_field is None:
                    # depending on sensor type, eval attribute field
                    if ts_i == "u":
                        y_meas_i = flu.MpiUtils.peval(self.u_, xs_i)[0]
                    else:
                        if ts_i == "v":
                            y_meas_i = flu.MpiUtils.peval(self.u_, xs_i)[1]
                        else:  # sensor_type=='p':
                            y_meas_i = flu.MpiUtils.peval(self.p_, xs_i)
                else:
                    # some mixed field in W = (u, v, p) is given
                    # eval and take index corresponding to sensor
                    sensor_types = dict(u=0, v=1, p=2)
                    y_meas_i = flu.MpiUtils.peval(mixed_field, xs_i)[sensor_types[ts_i]]

            y_meas[isensor] = y_meas_i
        return y_meas

    def assign_measurement_to_dataframe(self, df, y_meas, index):
        """Assign measurement (array y_meas) to DataFrame at index
        Essentially convert array (y_meas) to separate columns (y_meas_i)"""
        y_meas_str = self.make_y_dataframe_column_name()
        for i_meas, name_meas in enumerate(y_meas_str):
            df.loc[index, name_meas] = y_meas[i_meas]

    def make_y_dataframe_column_name(self):
        """Return column names of different measurements y_meas_i"""
        return ["y_meas_" + str(i + 1) for i in range(self.sensor_nr)]

    def write_timeseries(self):
        """Write pandas DataFrame to file"""
        if flu.MpiUtils.get_rank() == 0:
            # zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths["timeseries"], sep=",", index=False)

    def compute_energy(self):
        """Compute energy of perturbation flow
        OPTIONS REMOVED FROM PREVIOUS VERSION:
        on full/restricted domain      (default:full=True)
        minus base flow                (default:diff=False)
        normalized by base flow energy (default:normalize=False)"""
        dE = 1 / 2 * dolfin.norm(self.u_, norm_type="L2", mesh=self.mesh) ** 2
        return dE

    def compute_energy_field(self, export=False, filename=None):
        """Compute field dot(u, u) to see spatial location of perturbation kinetic energy
        Perturbation formulation only"""
        Efield = dot(self.u_, self.u_)
        # Note: E = 1/2 * assemble(Efield * fs.dx)
        Efield = flu.projectm(Efield, self.P)  # project to deg 1
        if export:
            flu.write_xdmf(filename, Efield, "E")
        return Efield

    def get_Dxy(self):
        """Get derivation matrices Dx, Dy in V space
        such that Dx*u = u.dx(0), Dy*u = u.dx(1)"""
        u = dolfin.TrialFunction(self.V)
        ut = dolfin.TestFunction(self.V)
        Dx = dolfin.assemble(inner(u.dx(0), ut) * self.dx)
        Dy = dolfin.assemble(inner(u.dx(1), ut) * self.dx)
        return Dx, Dy

    def get_A(self, perturbations=True, shift=0.0, timeit=True, up_0=None):
        """Get state-space dynamic matrix A linearized around some field up_0"""
        logger.info("Computing jacobian A...")

        if timeit:
            t0 = time.time()

        Jac = dolfin.PETScMatrix()
        v, q = dolfin.TestFunctions(self.W)
        iRe = dolfin.Constant(1 / self.Re)
        shift = dolfin.Constant(shift)

        if up_0 is None:
            up_ = self.up0  # base flow
        else:
            up_ = up_0
        u_, p_ = up_.split()

        if perturbations:  # perturbation equations linearized
            up = dolfin.TrialFunction(self.W)
            u, p = dolfin.split(up)
            dF0 = (
                -dot(dot(u_, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u, v but not p
            # create zeroBC for perturbation formulation
            bcu_inlet = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.boundaries.loc["inlet"].subdomain,
            )
            bcu_walls = dolfin.DirichletBC(
                self.W.sub(0).sub(1),
                dolfin.Constant(0),
                self.boundaries.loc["walls"].subdomain,
            )
            bcu_cylinder = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.boundaries.loc["cylinder"].subdomain,
            )
            bcu_actuation_up = dolfin.DirichletBC(
                self.W.sub(0),
                self.actuator_expression,
                self.boundaries.loc["actuator_up"].subdomain,
            )
            bcu_actuation_lo = dolfin.DirichletBC(
                self.W.sub(0),
                self.actuator_expression,
                self.boundaries.loc["actuator_lo"].subdomain,
            )
            bcu = [
                bcu_inlet,
                bcu_walls,
                bcu_cylinder,
                bcu_actuation_up,
                bcu_actuation_lo,
            ]
            self.actuator_expression.ampl = 0.0
            bcs = bcu
            # self.bc_p = {'bcu': bcu, 'bcp': []} # ???
        else:  # full ns + derivative
            F0 = (
                -dot(dot(u_, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u_), nabla_grad(v)) * dx
                + p_ * div(v) * dx
                + q * div(u_) * dx
                - shift * dot(u_, v) * dx
            )
            # prepare derivation
            du = dolfin.TrialFunction(self.W)
            dF0 = dolfin.derivative(F0, up_, du=du)
            # import pdb
            # pdb.set_trace()
            ## shift
            # dF0 = dF0 - shift*dot(u_,v)*dx
            # bcs
            self.actuator_expression.ampl = 0.0
            bcs = self.bc["bcu"]

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]

        if timeit:
            logger.info("Elapsed time: %f", time.time() - t0)

        return Jac

    def get_B(self, export=False, timeit=True):
        """Get actuation matrix B"""
        logger.info("Computing actuation matrix B...")

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.actuator_expression.ampl
        self.actuator_expression.ampl = 1.0

        # Method 1
        # restriction of actuation of boundary
        class RestrictFunction(dolfin.UserExpression):
            def __init__(self, boundary, fun, **kwargs):
                self.boundary = boundary
                self.fun = fun
                super(RestrictFunction, self).__init__(**kwargs)

            def eval(self, values, x):
                values[0] = 0
                values[1] = 0
                values[2] = 0
                if self.boundary.inside(x, True):
                    evalval = self.fun(x)
                    values[0] = evalval[0]
                    values[1] = evalval[1]

            def value_shape(self):
                return (3,)

        Bi = []
        for actuator_name in ["actuator_up", "actuator_lo"]:
            actuator_restricted = RestrictFunction(
                boundary=self.boundaries.loc[actuator_name].subdomain,
                fun=self.actuator_expression,
            )
            actuator_restricted = dolfin.interpolate(actuator_restricted, self.W)
            # actuator_restricted = flu.projectm(actuator_restricted, self.W)
            Bi.append(actuator_restricted)

        # this is supposedly B
        B_all_actuator = flu.projectm(sum(Bi), self.W)
        # get vector
        B = B_all_actuator.vector().get_local()
        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(
            B, eliminate_zeros=True, eliminate_under=1e-14
        ).toarray()
        B = B.T  # vertical B

        if export:
            fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            vv = dolfin.Function(self.V)
            pp = dolfin.Function(self.P)
            ww = dolfin.Function(self.W)
            ww.assign(B_all_actuator)
            fa.assign([vv, pp], ww)
            flu.write_xdmf("B.xdmf", vv, "B")

        self.actuator_expression.ampl = actuator_ampl_old

        if timeit:
            logger.info("Elapsed time: %f", time.time() - t0)

        return B

    def get_C(self, timeit=True, check=False):
        """Get measurement matrix C"""
        logger.info("Computing measurement matrix C...")

        if timeit:
            t0 = time.time()

        fspace = self.W
        uvp = dolfin.Function(fspace)
        uvp_vec = uvp.vector()
        dofmap = fspace.dofmap()

        ndof = fspace.dim()
        ns = self.sensor_nr
        C = np.zeros((ns, ndof))

        idof_old = 0
        # xs = self.sensor_location
        # Iteratively put each DOF at 1
        # And evaluate measurement on said DOF
        for idof in dofmap.dofs():
            uvp_vec[idof] = 1
            if idof_old > 0:
                uvp_vec[idof_old] = 0
            idof_old = idof
            C[:, idof] = self.make_measurement(mixed_field=uvp)
            # mixed_field permits p sensor

        # check:
        if check:
            for i in range(ns):
                sensor_types = dict(u=0, v=1, p=2)
                logger.debug(
                    "True probe: ",
                    self.up0(self.sensor_location[i])[
                        sensor_types[self.sensor_type[0]]
                    ],
                )
                logger.debug(
                    "\t with fun:", self.make_measurement(mixed_field=self.up0)
                )
                logger.debug("\t with C@x: ", C[i] @ self.up0.vector().get_local())

        if timeit:
            logger.info("Elapsed time: %f", time.time() - t0)

        return C

    def get_matrices_lifting(self, A, C, Q):
        """Return matrices A, B, C, Q resulting form lifting transform (Barbagallo et al. 2009)
        See get_Hw_lifting for details"""
        # Steady field with rho=1: S1
        logger.info("Computing steady actuated field...")
        self.actuator_expression.ampl = 1.0
        S1 = self.compute_steady_state_newton()
        S1v = S1.vector()

        # Q*S1 (as vector)
        sz = self.W.dim()
        QS1v = dolfin.Vector(S1v.copy())
        QS1v.set_local(
            np.zeros(
                sz,
            )
        )
        QS1v.apply("insert")
        Q.mult(S1v, QS1v)  # QS1v = Q * S1v

        # Bl = [Q*S1; -1]
        Bl = np.hstack((QS1v.get_local(), -1))  # stack -1
        Bl = np.atleast_2d(Bl).T  # as column

        # Cl = [C, 0]
        Cl = np.hstack((C, np.atleast_2d(0)))

        # Ql = diag(Q, 1)
        Qsp = flu.dense_to_sparse(Q)
        Qlsp = flu.spr.block_diag((Qsp, 1))

        # Al = diag(A, 0)
        Asp = flu.dense_to_sparse(A)
        Alsp = flu.spr.block_diag((Asp, 0))

        return Alsp, Bl, Cl, Qlsp

    def get_mass_matrix(self, sparse=False, volume=True, uvp=False):
        """Compute the mass matrix associated to
        spatial discretization"""
        logger.info("Computing mass matrix Q...")
        up = dolfin.TrialFunction(self.W)
        vq = dolfin.TestFunction(self.W)

        M = dolfin.PETScMatrix()
        # volume integral or surface integral (unused)
        dOmega = self.dx if volume else self.ds

        mf = sum([up[i] * vq[i] for i in range(2 + uvp)]) * dOmega  # sum u, v but not p
        dolfin.assemble(mf, tensor=M)

        if sparse:
            return flu.dense_to_sparse(M)
        return M

    def get_block_identity(self, sparse=False):
        """Compute the block-identity associated to
        the time-continuous, space-continuous formulation:
        E*dot(x) = A*x >>> E = blk(I, I, 0), 0 being on p dofs"""
        dof_idx = flu.get_subspace_dofs(self.W)
        sz = self.W.dim()
        diagE = np.zeros(sz)
        diagE[np.hstack([dof_idx[kk] for kk in ["u", "v"]])] = 1.0
        E = spr.diags(diagE, 0)
        if sparse:
            return E
        # cast
        return flu.sparse_to_petscmat(E)

    def check_mass_matrix(self, up=0, vq=0, random=True):
        """Given two vectors u, v (Functions on self.W),
        compute assemble(dot(u,v)*dx) v.s. u.local().T @ Q @ v.local()
        The result should be the same"""
        if random:
            logger.info("Creating random vectors")

            def createrandomfun():
                up = dolfin.Function(self.W)
                up.vector().set_local((np.random.randn(self.W.dim(), 1)))
                up.vector().apply("insert")
                return up

            up = createrandomfun()
            vq = createrandomfun()

        fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
        u = dolfin.Function(self.V)  # velocity only
        p = dolfin.Function(self.P)
        v = dolfin.Function(self.V)
        q = dolfin.Function(self.P)
        fa.assign([u, p], up)
        fa.assign([v, q], vq)

        # True integral of velocities
        d1 = dolfin.assemble(dot(u, v) * self.dx)

        # Discretized dot product (scipy)
        Q = self.get_mass_matrix(sparse=True)
        d2 = up.vector().get_local().T @ Q @ vq.vector().get_local()
        ## Note: u.T @ Qv = (Qv).T @ u
        # d2 = (Q @ v.vector().get_local()).T @ u.vector().get_local()

        # Discretized dot product (petsc)
        QQ = self.get_mass_matrix(sparse=False)
        uu = dolfin.Vector(up.vector())
        vv = dolfin.Vector(vq.vector())
        ww = dolfin.Vector(up.vector())  # intermediate result
        QQ.mult(vv, ww)  # ww = QQ*vv
        d3 = uu.inner(ww)

        return {"integral": d1, "dot_scipy": d2, "dot_petsc": d3}
###############################################################################
###############################################################################
############################ END CLASS DEFINITION #############################
###############################################################################
###############################################################################


###############################################################################
###############################################################################
############################     RUN EXAMPLE      #############################
###############################################################################
###############################################################################
if __name__ == "__main__":
    t000 = time.time()

    logger.info("Trying to instantiate FlowSolver...")
    params_flow = {
        "Re": 100.0,
        "uinf": 1.0,
        "d": 1.0,
        "sensor_location": np.array([[3, 0]]),  # sensor
        "sensor_type": ["v"],  # u, v, p only >> reimplement make_measurement
        "actuator_angular_size": 10,  # actuator angular size
    }
    params_time = {
        "dt": 0.005,
        "Tstart": 0,
        "num_steps": 10,
        "Tc": 0,
        "Trestartfrom": 0,
    }
    params_save = {
        "save_every": 5,
        "save_every_old": 100,
        "savedir0": Path(__file__).parent / "data_output",
        "compute_norms": True,
    }
    params_solver = {
        "throw_error": True,
        "perturbations": True,  #######
        "NL": True,  ################# NL=False only works with perturbations=True
        "init_pert": 1,
    }  # initial perturbation amplitude, np.inf=impulse (sequential only?)

    # o1
    params_mesh = {
        "genmesh": False,
        "remesh": False,
        "nx": 1,
        "meshpath": Path(__file__).parent / "data_input",
        "meshname": "O1.xdmf",
        "xinf": 20,  # 50, # 20
        "xinfa": -10,  # -30, # -5
        "yinf": 10,  # 30, # 8
        "segments": 540,
    }

    fs = FlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        verbose=True,
    )
    # alltimes = pd.DataFrame(columns=['1', '2', '3'], index=['assemble', 'solve'], data=np.zeros((2,3)))
    logger.info("__init__(): successful!")

    logger.info("Compute steady state...")
    u_ctrl_steady = 0.0
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=u_ctrl_steady)
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0
    )
    fs.load_steady_state(assign=True)

    logger.info("Init time-stepping")
    np.random.seed(42)
    sigma = 0.1
    # x0 = dolfin.Function(fs.W)
    # x0.vector()[:] += sigma * np.random.randn(x0.vector()[:].shape[0])
    # x0.vector()[:] /= np.linalg.norm(x0.vector()[:])
    # fs.set_initial_state(x0=x0)
    fs.init_time_stepping()

    logger.info("Step several times")
    sspath = Path(__file__).parent / "data_input"
    G = flu.read_ss(sspath / "sysid_o16_d=3_ssest.mat")
    Kss = flu.read_ss(sspath / "Kopt_reduced13.mat")

    x_ctrl = np.zeros((Kss.A.shape[0],))

    y_steady = 0 if fs.perturbations else fs.y_meas_steady  # reference measurement
    u_ctrl = 0
    u_ctrl0 = 1e-2
    tlen = 0.15
    tpeak = 1
    for i in range(fs.num_steps):
        # compute control
        if fs.t >= fs.Tc:
            # mpi broadcast sensor
            y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
            # compute error relative to base flow
            y_meas_err = y_steady - y_meas
            # wrapper around control toolbox
            u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)

        # open loop
        # u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        # closed loop
        u_ctrl += u_ctrl0 * np.exp(-1 / 2 * (fs.t - tpeak) ** 2 / tlen**2)

        fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)

    if fs.num_steps > 3:
        logger.info("Total time is: %f", time.time() - t000)
        logger.info("Iteration 1 time     --- %f", fs.timeseries.loc[1, "runtime"])
        logger.info("Iteration 2 time     --- %f", fs.timeseries.loc[2, "runtime"])
        logger.info(
            "Mean iteration time  --- %f", np.mean(fs.timeseries.loc[3:, "runtime"])
        )
        logger.info(
            "Time/iter/dof        --- %f",
            np.mean(fs.timeseries.loc[3:, "runtime"]) / fs.W.dim(),
        )
    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.user])

    fs.write_timeseries()
    logger.info(fs.timeseries)

    logger.info("Last two lines of the printed timetable should look like this:")
    logger.info(
        "9   0.045  1.634545  0.132531     0.0     0.0  0.000347 -3.385638  1.143313  0.159566"
    )
    logger.info(
        "10  0.050  0.000000  0.132341     0.0     0.0  0.000353 -3.107742  1.142722  0.143971"
    )

    logger.info('Checking utilitary functions')
    fs.get_A()
    #fs.get_B()
    #fs.get_C()
    fs.get_mass_matrix()
    fs.get_div0_u()
    fs.get_block_identity()


# TODO
# remove all references to full ns formulation
# -> remove bc relative to full ns....
# -> for base flow, we need full ns and bcs!!
# so we could keep 2 functions:
    # make fullns + bcs
    # make pertns + bcs_pert
    # but then use only pertns fo time stepping

# TODO 
# MIMO support
    
# TODO
# extract usecase specifics (eg cl, cd)
    
# TODO
# move all utilitary functions to utils*.py, then we sort

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
