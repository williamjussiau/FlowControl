"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

# from __future__ import print_function

import AbstractFlowSolver
import dolfin
from dolfin import dot, inner, nabla_grad, div, dx
import numpy as np
import time
import pandas as pd

import logging

from pathlib import Path

import utils_flowsolver as flu
import utils_extract as flu2


# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CylinderFlowSolver(AbstractFlowSolver.AbstractFlowSolver):
    """Base class for calculating flow
    Is instantiated with several structures (dicts) containing parameters
    See method .step and main for time-stepping (possibly actuated)
    Contain methods for frequency-response computation"""

    def __init__(self, **kwargs):  # redundant def here
        super().__init__(**kwargs)

    # Specifics to redefine for each case
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

    # Steady state
    def compute_steady_state(
        self, method="newton", u_ctrl=0.0, **kwargs
    ):  # TODO useless overriding
        """Overriding is useless, should do an additional method"""
        super().compute_steady_state(method, u_ctrl, **kwargs)
        # assign steady cl, cd
        cl, cd = self.compute_force_coefficients(self.u0, self.p0)

        self.cl0 = cl
        self.cd0 = cd
        if self.verbose:
            logger.info("Lift coefficient is: cl = %f", cl)
            logger.info("Drag coefficient is: cd = %f", cd)

    # Utility
    def get_A(
        self, perturbations=True, shift=0.0, timeit=True, up_0=None
    ):  # TODO idk, merge with make_mixed_form?
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

    def get_B(self, export=False, timeit=True):  # TODO keep here
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

    def get_C(self, timeit=True, check=False):  # TODO keep here
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

    # Additional, case-specific utility
    def compute_force_coefficients(self, u, p, enable=True):  # keep this one in here
        """Compute lift & drag coefficients
        For testing purposes, I added an argument enable
        To compute Cl, Cd, just put enable=True in this code"""
        if enable:
            sigma = flu2.stress_tensor(self.nu, u, p)
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
        "restart_order": 2,
    }
    params_save = {
        "save_every": 5,
        "save_every_old": 5,
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

    fs = CylinderFlowSolver(
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
    fs.init_time_stepping()

    logger.info("Step several times")
    sspath = Path(__file__).parent / "data_input"
    G = flu.read_ss(sspath / "sysid_o16_d=3_ssest.mat")
    Kss = flu.read_ss(sspath / "Kopt_reduced13.mat")

    x_ctrl = np.zeros((Kss.A.shape[0],))

    u_ctrl = 0
    u_ctrl0 = 1e-2
    tlen = 0.15
    tpeak = 1
    for i in range(fs.num_steps):
        # compute control
        # mpi broadcast sensor
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        # compute error relative to base flow
        y_meas_err = -y_meas
        # wrapper around control toolbox
        u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.dt)
        # step
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
    # dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.user])

    fs.write_timeseries()
    # logger.info(fs.timeseries)

    # Try restart
    params_time_restart = {
        "dt": 0.005,
        "dt_old": 0.005,
        "Tstart": 0.05,
        "num_steps": 10,
        "Tc": 0,
        "Trestartfrom": 0,
        "restart_order": 2,
    }
    fs_restart = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time_restart,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        verbose=True,
    )

    fs_restart.load_steady_state(assign=True)
    fs_restart.init_time_stepping()

    for i in range(fs_restart.num_steps):
        # compute control
        # mpi broadcast sensor
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        # compute error relative to base flow
        y_meas_err = -y_meas
        # wrapper around control toolbox
        u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs_restart.dt)
        # step
        fs_restart.step_perturbation(u_ctrl=u_ctrl, NL=fs_restart.NL, shift=0.0)

    fs_restart.write_timeseries()

    logger.info("Checking utilitary functions")
    fs.get_A()

    logger.info("Testing max(u) and mean(u)...")
    u_max_ref = 1.6345453902677856
    u_mean_ref = -0.0009997385060749036
    u_max = flu.apply_fun(fs.u_, np.max)
    u_mean = flu.apply_fun(fs.u_, np.mean)

    logger.info(f"umax: {u_max} // {u_max_ref}")
    logger.info(f"umean: {u_mean} // {u_mean_ref}")

    assert np.isclose(u_max, u_max_ref)
    assert np.isclose(u_mean, u_mean_ref)

    logger.info("End with success")

# TODO
# harmonize fullns / pertns
# eg u_full shoud only exist when saving
# still a lot of cleaning to do
# compute_norms -> rm
# TODO
# MIMO support
# TODO
# sort utility functions from utils..._flowsolver, _extract, _debug
# TODO
# take both kinds of actuation -> probably need if statements
# TODO split timeseries
# remove cl, cd ; + 2nd timeseries with case-specific data
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
